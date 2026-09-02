"""Trivial and classical baselines, run on the exact FlowCon-X splits.

The purpose is adversarial: if a decision tree on a single column, or a
random forest on five NetFlow statistics, lands within a few points of the
full model, then the reported result is a property of the dataset rather
than of the architecture, and the evaluation protocol has to change before
anything is written up.

Every model here is fitted on the same manifest-defined train indices and
scored on the same test indices as the neural model, and every deviation
from the original paper's configuration is recorded in the result JSON.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..metrics import classification_report, top_confusions
from .tabular import FAMILY_CITATIONS, build_features, parse_flows

try:  # pragma: no cover - exercised implicitly by the audit run
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    xgb = None
    HAS_XGBOOST = False


@dataclass
class ModelSpec:
    """How one baseline is fitted, and how that differs from its paper."""

    name: str
    kind: str
    params: Dict[str, object] = field(default_factory=dict)
    scale: bool = False
    max_train_rows: Optional[int] = None
    deviation: str = ""


# Each feature family is paired with the classifier its originating paper
# used, where one is specified. Deviations are recorded, never silent.
MODEL_SPECS: Dict[str, ModelSpec] = {
    "protocol_only": ModelSpec(
        name="decision_tree_d4",
        kind="decision_tree",
        params={"max_depth": 4},
        deviation="Port-only probe adapted to protocol number; the canonical CSV retains no port.",
    ),
    "condition_only": ModelSpec(
        name="decision_tree_d4",
        kind="decision_tree",
        params={"max_depth": 4},
        deviation="Diagnostic probe, not a published baseline.",
    ),
    # Identifier probes. Deeper trees than the protocol probe, because a
    # single hashed identifier column needs depth to isolate values, and an
    # under-powered probe would understate the shortcut it is looking for.
    "port_only": ModelSpec(name="decision_tree_d12", kind="decision_tree", params={"max_depth": 12}),
    "sni_only": ModelSpec(name="decision_tree_d20", kind="decision_tree", params={"max_depth": 20}),
    "server_ip_only": ModelSpec(name="decision_tree_d20", kind="decision_tree", params={"max_depth": 20}),
    "server_asn_only": ModelSpec(name="decision_tree_d16", kind="decision_tree", params={"max_depth": 16}),
    "capture_id_only": ModelSpec(name="decision_tree_d20", kind="decision_tree", params={"max_depth": 20}),
    "five_stat": ModelSpec(name="random_forest", kind="random_forest", params={"n_estimators": 200}),
    "first10_sizes": ModelSpec(name="gradient_boosted_trees", kind="gbt"),
    "first20_sizes": ModelSpec(name="gradient_boosted_trees", kind="gbt"),
    "size_histogram": ModelSpec(name="gradient_boosted_trees", kind="gbt"),
    "flow_meta": ModelSpec(name="gradient_boosted_trees", kind="gbt"),
    "cumul": ModelSpec(
        name="svm_rbf",
        kind="svm_rbf",
        scale=True,
        # RBF-SVM training is superlinear in the number of support vectors, and
        # under a hard split protocol the data is less separable, so the support
        # set grows and an unbounded fit does not terminate in reasonable time.
        # 8k stratified rows keeps every protocol affordable at the cost of
        # understating CUMUL, which is the safe direction for a baseline.
        max_train_rows=8000,
        params={"cache_size": 1000},
        deviation=(
            "Panchenko et al. fit an RBF-SVM with a grid search over C and gamma on the full "
            "training set. We fit an RBF-SVM with scikit-learn defaults on a stratified subsample "
            "of at most 8,000 rows (see subsampled_train_rows). Both deviations can only understate "
            "CUMUL, so it is a lower bound on what this feature set achieves."
        ),
    ),
    "appscanner": ModelSpec(name="random_forest", kind="random_forest", params={"n_estimators": 200}),
    "kfp": ModelSpec(
        name="random_forest",
        kind="random_forest",
        params={"n_estimators": 300},
        deviation=(
            "Hayes and Danezis use random-forest leaf vectors followed by a k-NN stage. We report the "
            "random forest directly, which is the standard simplification in the literature."
        ),
    ),
}


def _build_model(spec: ModelSpec, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    if spec.kind == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, class_weight="balanced", **spec.params)
    if spec.kind == "random_forest":
        return RandomForestClassifier(random_state=seed, class_weight="balanced", n_jobs=-1, **spec.params)
    if spec.kind == "svm_rbf":
        return SVC(kernel="rbf", class_weight="balanced", random_state=seed, **spec.params)
    if spec.kind == "gbt":
        if HAS_XGBOOST:
            return xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
                **spec.params,
            )
        return HistGradientBoostingClassifier(random_state=seed, **spec.params)
    raise ValueError(f"Unknown model kind {spec.kind!r}")


def _subsample(y: np.ndarray, indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Stratified subsample of ``indices`` down to at most ``limit`` rows."""
    if len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    labels = y[indices]
    keep: List[int] = []
    classes, counts = np.unique(labels, return_counts=True)
    quota = {int(c): max(1, int(round(limit * count / len(indices)))) for c, count in zip(classes, counts)}
    for cls in classes:
        pool = indices[labels == cls]
        take = min(len(pool), quota[int(cls)])
        keep.extend(rng.choice(pool, size=take, replace=False).tolist())
    return np.asarray(sorted(keep))


def run_family(
    df: pd.DataFrame,
    family: str,
    y: np.ndarray,
    class_names: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    flows=None,
    bootstrap: bool = True,
    feature_cache: Optional[Dict[str, tuple]] = None,
) -> Dict[str, object]:
    """Fit one baseline family and return its full metric block."""
    spec = MODEL_SPECS[family]
    # Feature construction is a Python loop over every flow and costs about as
    # much as fitting the model, so it is built once per family and reused
    # across split protocols.
    if feature_cache is not None and family in feature_cache:
        features, feature_names = feature_cache[family]
    else:
        features, feature_names = build_features(df, family, flows)
        if feature_cache is not None:
            feature_cache[family] = (features, feature_names)
    if features.shape[1] == 0:
        return {
            "family": family,
            "status": "unavailable",
            "description": FAMILY_CITATIONS.get(family, ""),
            "reason": (
                "The column this probe reads is absent or carries a single value in this dataset, so "
                "the probe would degenerate to the majority classifier. Reporting that number would "
                "read as 'this identifier does not help' when it means 'this dataset has no such field'."
            ),
        }

    fit_idx = train_idx
    subsampled = None
    if spec.max_train_rows is not None:
        fit_idx = _subsample(y, train_idx, spec.max_train_rows, seed)
        if len(fit_idx) < len(train_idx):
            subsampled = int(len(fit_idx))

    x_train, x_test = features[fit_idx], features[test_idx]
    if spec.scale:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

    # Under a strict split protocol a class can be absent from train entirely
    # (an app-disjoint split can remove every app of a service). XGBoost
    # requires a contiguous 0..k-1 label space, so fit on a remapped space and
    # map predictions back to the global class indices.
    present = np.unique(y[fit_idx])
    remap = {int(label): i for i, label in enumerate(present)}
    inverse = np.asarray(sorted(remap, key=lambda label: remap[label]), dtype=np.int64)
    y_fit = np.asarray([remap[int(label)] for label in y[fit_idx]], dtype=np.int64)

    model = _build_model(spec, seed)
    started = time.perf_counter()
    model.fit(x_train, y_fit)
    fit_seconds = time.perf_counter() - started
    predictions = inverse[np.asarray(model.predict(x_test), dtype=np.int64)]

    labels = np.arange(len(class_names))
    report = classification_report(predictions, y[test_idx], labels=labels, bootstrap=bootstrap, seed=seed)
    report["per_class_f1"] = {class_names[int(k)]: v for k, v in report["per_class_f1"].items()}
    report["support"] = {class_names[int(k)]: v for k, v in report["support"].items()}
    report["labels"] = list(class_names)

    out: Dict[str, object] = {
        "family": family,
        "status": "ok",
        "description": FAMILY_CITATIONS.get(family, ""),
        "model": spec.name,
        "n_features": int(features.shape[1]),
        "feature_names": feature_names if len(feature_names) <= 32 else feature_names[:32] + ["..."],
        "n_train": int(len(fit_idx)),
        "n_test": int(len(test_idx)),
        "classes_absent_from_train": [
            class_names[int(label)] for label in range(len(class_names)) if label not in remap
        ],
        "fit_seconds": round(fit_seconds, 3),
        "seed": seed,
        "metrics": report,
        "top_confusions": [
            {**item, "true": class_names[int(item["true"])], "predicted": class_names[int(item["predicted"])]}
            for item in top_confusions(predictions, y[test_idx], labels=labels, k=5)
        ],
    }
    if spec.deviation:
        out["deviation_from_original"] = spec.deviation
    if subsampled is not None:
        out["subsampled_train_rows"] = subsampled
    if hasattr(model, "feature_importances_") and features.shape[1] <= 64:
        importances = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(-importances)[:10]
        out["top_features"] = [
            {"name": feature_names[i], "importance": float(importances[i])} for i in order if importances[i] > 0
        ]
    return out


def run_all_families(
    df: pd.DataFrame,
    y: np.ndarray,
    class_names: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    families: Optional[Sequence[str]] = None,
    seed: int = 42,
    bootstrap: bool = True,
    feature_cache: Optional[Dict[str, tuple]] = None,
) -> Dict[str, Dict[str, object]]:
    selected = list(families) if families else list(MODEL_SPECS)
    flows = parse_flows(df)
    results: Dict[str, Dict[str, object]] = {}
    for family in selected:
        results[family] = run_family(
            df,
            family,
            y,
            class_names,
            train_idx,
            test_idx,
            seed=seed,
            flows=flows,
            bootstrap=bootstrap,
            feature_cache=feature_cache,
        )
    return results


def majority_class_report(
    y: np.ndarray, class_names: Sequence[str], train_idx: np.ndarray, test_idx: np.ndarray
) -> Dict[str, object]:
    """The floor every other number must be read against."""
    majority = int(np.bincount(y[train_idx], minlength=len(class_names)).argmax())
    predictions = np.full(len(test_idx), majority, dtype=np.int64)
    labels = np.arange(len(class_names))
    report = classification_report(predictions, y[test_idx], labels=labels, bootstrap=False)
    report["per_class_f1"] = {class_names[int(k)]: v for k, v in report["per_class_f1"].items()}
    report["support"] = {class_names[int(k)]: v for k, v in report["support"].items()}
    report["labels"] = list(class_names)
    return {
        "family": "majority_class",
        "status": "ok",
        "description": "Always predict the most frequent training class.",
        "model": "constant",
        "predicted_class": class_names[majority],
        "metrics": report,
    }
