"""Shared classification metrics.

Plain accuracy on this dataset is not informative: the largest service class
outnumbers the smallest by roughly three orders of magnitude, so a classifier
that never predicts the rare class still scores well. Every headline number
in the paper is therefore macro-F1 and balanced accuracy, with per-class F1
reported alongside so that a reader can see which class carries the loss.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Callers pass either a list of class names or an ``np.arange`` of class ids.
# ``Sequence`` alone excludes ndarray, which every internal caller uses.
Labels = Union[Sequence[Any], np.ndarray]


def _labels(*arrays: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate([np.asarray(a).ravel() for a in arrays if len(a)])) if arrays else np.zeros(0)


def confusion_matrix(pred: np.ndarray, target: np.ndarray, labels: Optional[Labels] = None) -> np.ndarray:
    order = np.asarray(labels) if labels is not None else _labels(pred, target)
    index = {label: i for i, label in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=np.int64)
    for t, p in zip(np.asarray(target).ravel(), np.asarray(pred).ravel()):
        if t in index and p in index:
            matrix[index[t], index[p]] += 1
    return matrix


def per_class_f1(pred: np.ndarray, target: np.ndarray, labels: Optional[Labels] = None) -> Dict[str, float]:
    order = np.asarray(labels) if labels is not None else _labels(pred, target)
    out: Dict[str, float] = {}
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    for label in order:
        tp = int(np.sum((pred == label) & (target == label)))
        fp = int(np.sum((pred == label) & (target != label)))
        fn = int(np.sum((pred != label) & (target == label)))
        denom = 2 * tp + fp + fn
        out[str(label)] = float(2 * tp / denom) if denom else 0.0
    return out


def per_class_support(target: np.ndarray, labels: Optional[Labels] = None) -> Dict[str, int]:
    order = np.asarray(labels) if labels is not None else _labels(target)
    target = np.asarray(target).ravel()
    return {str(label): int(np.sum(target == label)) for label in order}


def macro_f1(pred: np.ndarray, target: np.ndarray, labels: Optional[Labels] = None) -> float:
    scores = per_class_f1(pred, target, labels)
    return float(np.mean(list(scores.values()))) if scores else 0.0


def balanced_accuracy(pred: np.ndarray, target: np.ndarray, labels: Optional[Labels] = None) -> float:
    """Mean per-class recall. Equals accuracy only when classes are balanced."""
    order = np.asarray(labels) if labels is not None else _labels(pred, target)
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    recalls: List[float] = []
    for label in order:
        support = int(np.sum(target == label))
        if support == 0:
            continue
        recalls.append(float(np.sum((pred == label) & (target == label)) / support))
    return float(np.mean(recalls)) if recalls else 0.0


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target).ravel()
    return float(np.mean(np.asarray(pred).ravel() == target)) if target.size else 0.0


def bootstrap_ci(
    pred: np.ndarray,
    target: np.ndarray,
    statistic: str = "macro_f1",
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    labels: Optional[Labels] = None,
) -> Dict[str, float]:
    """Percentile bootstrap CI over the test set for one statistic."""
    fn = {"macro_f1": macro_f1, "balanced_accuracy": balanced_accuracy, "accuracy": lambda p, t, _labels=None: accuracy(p, t)}[
        statistic
    ]
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    order = np.asarray(labels) if labels is not None else _labels(pred, target)
    if target.size == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "n_resamples": 0}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    n = target.size
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        draws[i] = fn(pred[idx], target[idx], order)
    return {
        "point": float(fn(pred, target, order)),
        "lo": float(np.percentile(draws, 100 * alpha / 2)),
        "hi": float(np.percentile(draws, 100 * (1 - alpha / 2))),
        "n_resamples": int(n_resamples),
    }


def classification_report(
    pred: np.ndarray,
    target: np.ndarray,
    labels: Optional[Labels] = None,
    bootstrap: bool = True,
    seed: int = 0,
) -> Dict[str, object]:
    """The full metric block written into every ``metrics.json``."""
    order = np.asarray(labels) if labels is not None else _labels(pred, target)
    report: Dict[str, object] = {
        "accuracy": accuracy(pred, target),
        "macro_f1": macro_f1(pred, target, order),
        "balanced_accuracy": balanced_accuracy(pred, target, order),
        "per_class_f1": per_class_f1(pred, target, order),
        "support": per_class_support(target, order),
        "labels": [str(label) for label in order],
        "confusion_matrix": confusion_matrix(pred, target, order).tolist(),
    }
    if bootstrap:
        report["macro_f1_ci95"] = bootstrap_ci(pred, target, "macro_f1", seed=seed, labels=order)
        report["balanced_accuracy_ci95"] = bootstrap_ci(pred, target, "balanced_accuracy", seed=seed, labels=order)
    return report


def top_confusions(
    pred: np.ndarray, target: np.ndarray, labels: Optional[Labels] = None, k: int = 10
) -> List[Dict[str, Any]]:
    """The k most frequent off-diagonal (true, predicted) pairs."""
    order = np.asarray(labels) if labels is not None else _labels(pred, target)
    matrix = confusion_matrix(pred, target, order)
    pairs: List[Dict[str, Any]] = []
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j and matrix[i, j] > 0:
                support = int(matrix[i].sum())
                pairs.append(
                    {
                        "true": str(order[i]),
                        "predicted": str(order[j]),
                        "count": int(matrix[i, j]),
                        "rate": float(matrix[i, j] / support) if support else 0.0,
                    }
                )
    pairs.sort(key=lambda item: -item["count"])
    return pairs[:k]
