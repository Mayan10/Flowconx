"""Closed-set evaluation from a trained embedding.

Four heads are supported and all four are scored on the same embedding, so
that "the model is good" and "this particular classifier is good" stay
separable. The prototype head is the one the deployment story depends on: it
is the only one that supports enrolling a new class without retraining.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..metrics import classification_report, top_confusions


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def cosine_matrix(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(a if b is None else b).T


def class_prototypes(embeddings: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Mean embedding per class, L2-normalised. Empty classes stay at zero."""
    prototypes = np.zeros((n_classes, embeddings.shape[1]), dtype=np.float64)
    for cls in range(n_classes):
        mask = labels == cls
        if np.any(mask):
            prototypes[cls] = embeddings[mask].mean(axis=0)
    return l2_normalize(prototypes)


def prototype_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, n_classes: int
) -> np.ndarray:
    prototypes = class_prototypes(train_x, train_y, n_classes)
    present = np.flatnonzero(np.linalg.norm(prototypes, axis=1) > 0)
    if present.size == 0:
        return np.zeros(len(test_x), dtype=np.int64)
    similarity = l2_normalize(test_x) @ prototypes[present].T
    return present[np.argmax(similarity, axis=1)]


def knn_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int = 5, block: int = 2048
) -> np.ndarray:
    """Cosine k-NN, blocked so a large test set does not allocate n x m at once."""
    train_n = l2_normalize(train_x)
    test_n = l2_normalize(test_x)
    k = min(k, len(train_y))
    out = np.zeros(len(test_n), dtype=np.int64)
    n_classes = int(train_y.max()) + 1 if len(train_y) else 1
    for start in range(0, len(test_n), block):
        similarity = test_n[start : start + block] @ train_n.T
        top = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        neighbours = train_y[top]
        counts = np.zeros((neighbours.shape[0], n_classes), dtype=np.int32)
        for column in range(neighbours.shape[1]):
            np.add.at(counts, (np.arange(neighbours.shape[0]), neighbours[:, column]), 1)
        out[start : start + block] = counts.argmax(axis=1)
    return out


def linear_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=-1)
    model.fit(train_x, train_y)
    return np.asarray(model.predict(test_x))


def svm_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.svm import LinearSVC

    model = LinearSVC(C=1.0, dual="auto", max_iter=5000, class_weight="balanced", random_state=seed)
    model.fit(train_x, train_y)
    return np.asarray(model.predict(test_x))


def evaluate_heads(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    class_names: Sequence[str],
    heads: Sequence[str] = ("knn", "prototype"),
    k: int = 5,
    seed: int = 0,
    bootstrap_resamples: int = 1000,
) -> Dict[str, Dict[str, object]]:
    """Score every requested head on the same embedding."""
    labels = np.arange(len(class_names))
    predictors = {
        "knn": lambda: knn_predict(train_x, train_y, test_x, k=k),
        "prototype": lambda: prototype_predict(train_x, train_y, test_x, len(class_names)),
        "linear": lambda: linear_predict(train_x, train_y, test_x, seed=seed),
        "svm": lambda: svm_predict(train_x, train_y, test_x, seed=seed),
    }
    results: Dict[str, Dict[str, object]] = {}
    for head in heads:
        if head not in predictors:
            raise ValueError(f"Unknown classifier head {head!r}")
        predictions = predictors[head]()
        report = classification_report(
            predictions, test_y, labels=labels, bootstrap=bootstrap_resamples > 0, seed=seed
        )
        report["per_class_f1"] = {class_names[int(k_)]: v for k_, v in report["per_class_f1"].items()}
        report["support"] = {class_names[int(k_)]: v for k_, v in report["support"].items()}
        report["labels"] = list(class_names)
        report["top_confusions"] = [
            {**item, "true": class_names[int(item["true"])], "predicted": class_names[int(item["predicted"])]}
            for item in top_confusions(predictions, test_y, labels=labels, k=8)
        ]
        results[head] = report
    return results


def embedding_geometry(embeddings: np.ndarray, labels: np.ndarray, max_rows: int = 4000, seed: int = 0) -> Dict[str, float]:
    """Mean intra- and inter-class cosine, on a seeded subsample.

    The original implementation was a Python double loop over every pair,
    which is why the previous pipeline needed evaluation caps. This is a
    vectorised estimate and reports the sample size it used.
    """
    rng = np.random.default_rng(seed)
    if len(labels) > max_rows:
        idx = np.sort(rng.choice(len(labels), size=max_rows, replace=False))
        embeddings, labels = embeddings[idx], labels[idx]
    similarity = cosine_matrix(embeddings)
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), k=1)
    intra = similarity[same & upper]
    inter = similarity[~same & upper]
    return {
        "intra_class_cosine": float(intra.mean()) if intra.size else 0.0,
        "inter_class_cosine": float(inter.mean()) if inter.size else 0.0,
        "separation": float(intra.mean() - inter.mean()) if intra.size and inter.size else 0.0,
        "n_sampled": int(len(labels)),
    }


def stack_predictions(results: Dict[str, Dict[str, object]]) -> List[str]:
    return sorted(results)
