"""Few-shot enrollment of new applications.

The architecture's distinguishing property is that the deployed embedding is
trained with the same metric objective used at inference, so a new class can
be added by averaging k embeddings into a prototype -- k forward passes and a
vector write. A softmax classifier needs a new output unit and a fine-tuning
run. This module measures the accuracy side of that trade; ``eval/cost.py``
measures the cost side.

The encoder is frozen throughout. Nothing here trains.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..metrics import balanced_accuracy, macro_f1
from .closed_set import class_prototypes, l2_normalize


def enroll_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Prototypes built from at most ``shots`` examples per class."""
    selected = np.zeros(len(labels), dtype=bool)
    for cls in range(n_classes):
        pool = np.flatnonzero(labels == cls)
        if pool.size == 0:
            continue
        take = min(shots, pool.size)
        selected[rng.choice(pool, size=take, replace=False)] = True
    return class_prototypes(embeddings[selected], labels[selected], n_classes)


def predict_from_prototypes(prototypes: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    present = np.flatnonzero(np.linalg.norm(prototypes, axis=1) > 0)
    if present.size == 0:
        return np.zeros(len(test_x), dtype=np.int64)
    return present[np.argmax(l2_normalize(test_x) @ prototypes[present].T, axis=1)]


def enrollment_curve(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    n_classes: int,
    shots: Sequence[int],
    repeats: int = 5,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Accuracy against the number of enrolled examples per class.

    Repeated with different draws because at k=1 the result depends heavily on
    which single flow was drawn; the spread is reported, not hidden.
    """
    labels = np.arange(n_classes)
    curve: List[Dict[str, object]] = []
    for k in shots:
        scores_f1: List[float] = []
        scores_bal: List[float] = []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed * 1000 + k * 10 + repeat)
            prototypes = enroll_prototypes(train_x, train_y, n_classes, k, rng)
            predictions = predict_from_prototypes(prototypes, test_x)
            scores_f1.append(macro_f1(predictions, test_y, labels))
            scores_bal.append(balanced_accuracy(predictions, test_y, labels))
        curve.append(
            {
                "shots": int(k),
                "repeats": int(repeats),
                "macro_f1_mean": float(np.mean(scores_f1)),
                "macro_f1_std": float(np.std(scores_f1)),
                "balanced_accuracy_mean": float(np.mean(scores_bal)),
                "balanced_accuracy_std": float(np.std(scores_bal)),
            }
        )
    return curve


def evaluate_few_shot(
    train_x: np.ndarray,
    train_split,
    test_x: np.ndarray,
    test_split,
    config,
    label_space,
) -> Dict[str, object]:
    n_classes = label_space.n_classes
    shots = [k for k in config.eval.few_shot_k if k > 0]
    result: Dict[str, object] = {
        "status": "ok",
        "encoder": "frozen",
        "shots": shots,
        "service_enrollment": enrollment_curve(
            train_x, train_split.labels, test_x, test_split.labels, n_classes, shots, seed=config.seed
        ),
    }

    # The same curve at application granularity, which is the harder and more
    # deployment-relevant question: enrolling a new *app*, not a new category.
    if label_space.apps and len(label_space.apps) > 1:
        shared = np.intersect1d(np.unique(train_split.app_labels), np.unique(test_split.app_labels))
        if shared.size > 1:
            remap = {int(a): i for i, a in enumerate(shared)}
            train_mask = np.isin(train_split.app_labels, shared)
            test_mask = np.isin(test_split.app_labels, shared)
            result["app_enrollment"] = enrollment_curve(
                train_x[train_mask],
                np.asarray([remap[int(a)] for a in train_split.app_labels[train_mask]]),
                test_x[test_mask],
                np.asarray([remap[int(a)] for a in test_split.app_labels[test_mask]]),
                len(shared),
                shots,
                seed=config.seed,
            )
            result["app_enrollment_n_apps"] = int(shared.size)
        else:
            result["app_enrollment_skipped"] = (
                f"only {int(shared.size)} application(s) appear in both train and test under this split"
            )
    return result
