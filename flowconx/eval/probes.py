"""Representation probing: how much nuisance information survives in z_flow.

This replaces the CIST score, which the audit showed was maximised by a
constant encoder and had no stated null model (AUDIT.md 5, M1). The protocol
here is the standard one:

1. Freeze the embedding.
2. Train a probe -- linear, and a 2-layer MLP -- to predict the nuisance
   variable from it.
3. Report probe accuracy against the majority-class floor, against the same
   probe on raw features, and alongside downstream task accuracy.

Driving nuisance information out is only interesting if task accuracy
survives, so both numbers are always reported together and neither is quoted
alone.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..metrics import balanced_accuracy, macro_f1


def _fit_probe(kind: str, train_x, train_y, test_x, seed: int):
    if kind == "linear":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, n_jobs=-1)
    elif kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        model = MLPClassifier(
            hidden_layer_sizes=(128,), max_iter=300, random_state=seed, early_stopping=True, n_iter_no_change=10
        )
    else:
        raise ValueError(f"Unknown probe kind {kind!r}")
    model.fit(train_x, train_y)
    return np.asarray(model.predict(test_x))


def probe_target(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    kinds=("linear", "mlp"),
) -> Dict[str, object]:
    """Probe accuracy for one target, with the floor that makes it readable."""
    classes = np.unique(np.concatenate([train_y, test_y]))
    if classes.size < 2:
        return {"status": "skipped", "reason": f"target has {classes.size} distinct value(s)"}
    counts = np.bincount(test_y, minlength=int(classes.max()) + 1)
    majority = float(counts.max() / max(counts.sum(), 1))
    out: Dict[str, object] = {
        "status": "ok",
        "n_classes": int(classes.size),
        "majority_class_accuracy": majority,
        "probes": {},
    }
    for kind in kinds:
        predictions = _fit_probe(kind, train_x, train_y, test_x, seed)
        accuracy = float(np.mean(predictions == test_y))
        out["probes"][kind] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1(predictions, test_y, classes),
            "balanced_accuracy": balanced_accuracy(predictions, test_y, classes),
            # The quantity to read: how far above chance the probe got. Zero
            # means the embedding carries no linearly (or MLP-) decodable
            # information about the nuisance beyond the class prior.
            "above_majority": accuracy - majority,
        }
    return out


def evaluate_nuisance_probes(
    embeddings: Dict[str, np.ndarray],
    splits: Dict[str, object],
    config,
    label_space,
    max_rows: int = 8000,
) -> Dict[str, object]:
    """Probe the deployed embedding, and the raw features, for the nuisance."""
    rng = np.random.default_rng(config.seed)

    def sample(side: str):
        n = len(splits[side].labels)
        if n <= max_rows:
            return np.arange(n)
        return np.sort(rng.choice(n, size=max_rows, replace=False))

    train_idx, test_idx = sample("train"), sample("test")
    results: Dict[str, object] = {
        "status": "ok",
        "nuisance_source": label_space.nuisance_source,
        "adversarial_weight": config.loss.lambda_adversarial,
        "targets": {},
    }
    if label_space.nuisance_source == "none":
        results["note"] = (
            "No adversarial nuisance is configured for this run, so the probe measures how much "
            "nuisance information the embedding retains without any pressure to remove it. That is "
            "the reference point the ablation compares against."
        )

    views = {
        "z_flow": (embeddings["train"][train_idx], embeddings["test"][test_idx]),
        "raw_flow_features": (
            splits["train"].features.flow_features[train_idx],
            splits["test"].features.flow_features[test_idx],
        ),
    }
    targets = {
        "nuisance": (splits["train"].nuisance_labels[train_idx], splits["test"].nuisance_labels[test_idx]),
        "task": (splits["train"].labels[train_idx], splits["test"].labels[test_idx]),
    }
    for target_name, (train_y, test_y) in targets.items():
        results["targets"][target_name] = {
            view: probe_target(train_x, train_y, test_x, test_y, config.seed)
            for view, (train_x, test_x) in views.items()
        }

    # The trade-off number the paper needs: nuisance removed vs task retained.
    nuisance_probe = results["targets"]["nuisance"].get("z_flow", {})
    task_probe = results["targets"]["task"].get("z_flow", {})
    if nuisance_probe.get("status") == "ok" and task_probe.get("status") == "ok":
        results["tradeoff"] = {
            "nuisance_above_majority_mlp": nuisance_probe["probes"]["mlp"]["above_majority"],
            "task_macro_f1_mlp": task_probe["probes"]["mlp"]["macro_f1"],
            "reading": (
                "Nuisance leakage is the first number and task performance the second. A method that "
                "drives the first to zero while collapsing the second has not achieved invariance, it "
                "has destroyed the representation."
            ),
        }
    return results
