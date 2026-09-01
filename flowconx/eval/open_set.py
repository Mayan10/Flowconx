"""Open-set evaluation: rejecting applications never seen in training.

This is where a metric-trained deployed embedding should genuinely beat a
softmax classifier, and it is a claim an ET-BERT-style fine-tuned classifier
cannot easily make: its output layer has one logit per training class and no
notion of "none of these".

Scoring rule: distance to the nearest class prototype in the deployed
embedding. Three baselines are scored on the *same* embedding so the
comparison isolates the rejection rule rather than the representation:
maximum softmax probability from a linear probe, an energy score over the
same probe's logits, and Mahalanobis distance to the nearest class Gaussian.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .closed_set import class_prototypes, l2_normalize


def auroc(scores_known: np.ndarray, scores_unknown: np.ndarray) -> float:
    """P(score of a known > score of an unknown), via the rank statistic.

    Higher score must mean "more likely known" for every scorer passed in.
    """
    if scores_known.size == 0 or scores_unknown.size == 0:
        return float("nan")
    combined = np.concatenate([scores_known, scores_unknown])
    order = np.argsort(combined, kind="stable")
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Average ranks over ties, or a scorer with many identical scores is
    # rewarded or punished arbitrarily.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(counts))
    np.add.at(tie_sum, inverse, ranks)
    ranks = (tie_sum / counts)[inverse]
    n_known = len(scores_known)
    rank_sum = ranks[:n_known].sum()
    return float((rank_sum - n_known * (n_known + 1) / 2) / (n_known * len(scores_unknown)))


def fpr_at_tpr(scores_known: np.ndarray, scores_unknown: np.ndarray, target_tpr: float = 0.95) -> float:
    """Unknown-accept rate at the threshold that keeps `target_tpr` of knowns."""
    if scores_known.size == 0 or scores_unknown.size == 0:
        return float("nan")
    threshold = float(np.quantile(scores_known, 1.0 - target_tpr))
    return float(np.mean(scores_unknown >= threshold))


def oscr_curve(
    scores_known: np.ndarray,
    correct_known: np.ndarray,
    scores_unknown: np.ndarray,
    n_points: int = 100,
) -> List[Dict[str, float]]:
    """Open-set classification rate: correct-known accuracy vs unknown-accept rate.

    A model can buy a low false-accept rate by rejecting everything, so the
    curve reports accuracy *among accepted knowns that were also classified
    correctly*, which is the quantity a deployment actually cares about.
    """
    if scores_known.size == 0 or scores_unknown.size == 0:
        return []
    thresholds = np.quantile(np.concatenate([scores_known, scores_unknown]), np.linspace(0.0, 1.0, n_points))
    out: List[Dict[str, float]] = []
    for threshold in thresholds:
        accepted = scores_known >= threshold
        ccr = float(np.mean(accepted & correct_known))
        fpr = float(np.mean(scores_unknown >= threshold))
        out.append({"threshold": float(threshold), "ccr": ccr, "fpr_unknown": fpr})
    return out


def _prototype_scores(train_x, train_y, test_x, n_classes) -> tuple:
    prototypes = class_prototypes(train_x, train_y, n_classes)
    present = np.flatnonzero(np.linalg.norm(prototypes, axis=1) > 0)
    similarity = l2_normalize(test_x) @ prototypes[present].T
    best = similarity.max(axis=1)
    predicted = present[np.argmax(similarity, axis=1)]
    return best, predicted


def _softmax_and_energy(train_x, train_y, test_x, seed: int):
    from sklearn.linear_model import LogisticRegression

    probe = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=-1)
    probe.fit(train_x, train_y)
    logits = probe.decision_function(test_x)
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    msp = probabilities.max(axis=1)
    # Energy score (Liu et al., NeurIPS 2020); negated so that higher still
    # means "more likely known", matching the convention of every other scorer.
    energy = -(-np.log(np.exp(shifted).sum(axis=1)) - logits.max(axis=1))
    return msp, energy, np.asarray(probe.predict(test_x))


def _mahalanobis(train_x, train_y, test_x, n_classes: int) -> np.ndarray:
    """Negative minimum Mahalanobis distance under a shared covariance."""
    dim = train_x.shape[1]
    means = np.zeros((n_classes, dim))
    centred = []
    for cls in range(n_classes):
        mask = train_y == cls
        if not np.any(mask):
            continue
        means[cls] = train_x[mask].mean(axis=0)
        centred.append(train_x[mask] - means[cls])
    if not centred:
        return np.zeros(len(test_x))
    pooled = np.concatenate(centred, axis=0)
    covariance = np.cov(pooled, rowvar=False) + np.eye(dim) * 1e-3
    precision = np.linalg.pinv(covariance)
    present = np.flatnonzero(np.linalg.norm(means, axis=1) > 0)
    distances = np.empty((len(test_x), len(present)))
    for j, cls in enumerate(present):
        delta = test_x - means[cls]
        distances[:, j] = np.einsum("ij,jk,ik->i", delta, precision, delta)
    return -distances.min(axis=1)


def evaluate_open_set(
    train_x: np.ndarray,
    train_split,
    test_x: np.ndarray,
    test_split,
    config,
    label_space,
    unknown_apps: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Score every rejection rule on the same embedding."""
    unknown = [a.lower() for a in (unknown_apps if unknown_apps is not None else config.data.unknown_apps)]
    if not unknown:
        return {
            "status": "skipped",
            "reason": "no data.unknown_apps configured; open-set needs held-out applications",
        }
    if "app" not in test_split.frame.columns:
        return {"status": "skipped", "reason": "dataset has no app column"}

    is_unknown = test_split.frame["app"].astype(str).str.lower().isin(unknown).to_numpy()
    if not is_unknown.any() or is_unknown.all():
        return {
            "status": "skipped",
            "reason": f"held-out apps {unknown} give {int(is_unknown.sum())} unknown of {len(is_unknown)} test rows",
        }

    n_classes = label_space.n_classes
    proto_score, proto_pred = _prototype_scores(train_x, train_split.labels, test_x, n_classes)
    msp, energy, probe_pred = _softmax_and_energy(train_x, train_split.labels, test_x, config.seed)
    mahalanobis = _mahalanobis(train_x, train_split.labels, test_x, n_classes)

    known = ~is_unknown
    truth = test_split.labels
    scorers = {
        "prototype_cosine": (proto_score, proto_pred == truth),
        "softmax_msp": (msp, probe_pred == truth),
        "energy": (energy, probe_pred == truth),
        "mahalanobis": (mahalanobis, probe_pred == truth),
    }

    results: Dict[str, object] = {
        "status": "ok",
        "unknown_apps": unknown,
        "n_known_test": int(known.sum()),
        "n_unknown_test": int(is_unknown.sum()),
        "scorers": {},
    }
    for name, (score, correct) in scorers.items():
        results["scorers"][name] = {
            "auroc": auroc(score[known], score[is_unknown]),
            "fpr_at_95tpr": fpr_at_tpr(score[known], score[is_unknown], 0.95),
            "closed_set_accuracy_on_knowns": float(np.mean(correct[known])),
            "oscr": oscr_curve(score[known], correct[known], score[is_unknown]),
        }
    return results
