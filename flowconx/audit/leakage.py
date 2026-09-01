"""Leakage probes: what a split protocol failed to keep apart.

Each probe returns a structured verdict rather than a bare boolean, so that
``tests/test_leakage.py`` can assert on it and the audit report can print it.
A probe whose precondition is missing returns ``status='unavailable'`` with
the reason -- it must never quietly pass.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# Columns whose value, if it appears on both sides of a split, means flows
# from the same conversation or the same capture straddle train and test.
IDENTITY_COLUMNS = ("flow_id", "capture_id", "five_tuple", "client_ip", "server_ip", "origin")


def _verdict(name: str, passed: Optional[bool], **details: object) -> Dict[str, object]:
    status = "unavailable" if passed is None else ("pass" if passed else "FAIL")
    return {"check": name, "status": status, **details}


def check_index_disjoint(indices: Mapping[str, np.ndarray]) -> Dict[str, object]:
    """No row index may appear on two sides."""
    overlaps: Dict[str, int] = {}
    sides = list(indices)
    for i, a in enumerate(sides):
        for b in sides[i + 1 :]:
            shared = np.intersect1d(indices[a], indices[b])
            if shared.size:
                overlaps[f"{a}&{b}"] = int(shared.size)
    return _verdict("row_index_disjoint", not overlaps, overlaps=overlaps)


def check_column_disjoint(
    df: pd.DataFrame, indices: Mapping[str, np.ndarray], column: str
) -> Dict[str, object]:
    """No value of ``column`` may span train and test."""
    if column not in df.columns or df[column].isna().all():
        return _verdict(
            f"{column}_disjoint",
            None,
            reason=f"column {column!r} is not present in the table",
        )
    values = df[column].astype(str).to_numpy()
    train = set(values[indices["train"]].tolist())
    test = set(values[indices["test"]].tolist())
    shared = sorted(train & test)
    return _verdict(
        f"{column}_disjoint",
        not shared,
        n_shared=len(shared),
        n_train_values=len(train),
        n_test_values=len(test),
        examples=shared[:10],
    )


def _row_hashes(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    payload = df[list(columns)].astype(str).agg("\x1f".join, axis=1)
    return payload.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()).to_numpy()


def check_exact_duplicates(
    df: pd.DataFrame, indices: Mapping[str, np.ndarray], columns: Optional[Sequence[str]] = None
) -> Dict[str, object]:
    """No byte-identical observation may appear on both sides."""
    cols = list(columns) if columns else [c for c in df.columns if c not in ("flow_id",)]
    hashes = _row_hashes(df, cols)
    train = set(hashes[indices["train"]].tolist())
    test_hashes = hashes[indices["test"]]
    shared_mask = np.asarray([h in train for h in test_hashes])
    n_shared = int(shared_mask.sum())
    return _verdict(
        "no_exact_duplicates_across_splits",
        n_shared == 0,
        n_test_rows_duplicated_in_train=n_shared,
        fraction_of_test=float(n_shared / max(len(test_hashes), 1)),
        columns_hashed=cols,
    )


def check_near_duplicates(
    features: np.ndarray,
    indices: Mapping[str, np.ndarray],
    threshold: float = 0.999,
    max_rows_per_side: int = 8000,
    seed: int = 0,
) -> Dict[str, object]:
    """No test row may have a train row at cosine similarity above ``threshold``.

    Evaluated on a seeded subsample of each side because the exact
    computation is quadratic; the subsample size is reported so the estimate
    can be read for what it is.
    """
    rng = np.random.default_rng(seed)

    def take(side: str) -> np.ndarray:
        idx = indices[side]
        if len(idx) <= max_rows_per_side:
            return idx
        return np.sort(rng.choice(idx, size=max_rows_per_side, replace=False))

    train_idx, test_idx = take("train"), take("test")
    if train_idx.size == 0 or test_idx.size == 0:
        return _verdict("no_near_duplicates_across_splits", None, reason="a split side is empty")

    # Standardise per column first. Cosine similarity on raw AppScanner
    # features is dominated by the byte-sum and variance columns, whose
    # magnitudes are orders larger than the rest, and would report every pair
    # as a near duplicate regardless of what the flows actually did.
    reference = features[np.concatenate([train_idx, test_idx])].astype(np.float64)
    centre = reference.mean(axis=0)
    scale = np.maximum(reference.std(axis=0), 1e-9)

    def normalize(matrix: np.ndarray) -> np.ndarray:
        standardised = (matrix.astype(np.float64) - centre) / scale
        norms = np.linalg.norm(standardised, axis=1, keepdims=True)
        return standardised / np.maximum(norms, 1e-12)

    train = normalize(features[train_idx])
    test = normalize(features[test_idx])
    hits = 0
    best = -1.0
    for start in range(0, len(test), 512):
        block = test[start : start + 512] @ train.T
        maxima = block.max(axis=1)
        hits += int(np.sum(maxima >= threshold))
        best = max(best, float(maxima.max()))
    return _verdict(
        "no_near_duplicates_across_splits",
        hits == 0,
        threshold=threshold,
        standardised=True,
        n_test_rows_with_near_duplicate=hits,
        fraction_of_sampled_test=float(hits / len(test_idx)),
        max_cosine_observed=best,
        sampled_train_rows=int(train_idx.size),
        sampled_test_rows=int(test_idx.size),
    )


def check_label_not_in_declared_inputs(
    declared_inputs: Sequence[str], forbidden: Sequence[str] = IDENTITY_COLUMNS + ("app", "service")
) -> Dict[str, object]:
    """The model config must not declare an identifier or the label as an input."""
    overlap = sorted(set(declared_inputs) & set(forbidden))
    return _verdict(
        "label_and_identifiers_excluded_from_inputs",
        not overlap,
        declared_inputs=list(declared_inputs),
        forbidden_present=overlap,
    )


def check_nuisance_label_derivable(
    df: pd.DataFrame, nuisance: str = "condition", sources: Sequence[str] = ("flow iat mean", "flow iat std")
) -> Dict[str, object]:
    """Is the declared nuisance variable a deterministic function of model inputs?

    If it is, an adversarial head that removes it is removing a function of
    the very features the task classifier depends on, and the invariance
    claim is circular rather than empirical.
    """
    if nuisance not in df.columns:
        return _verdict("nuisance_not_deterministic_in_inputs", None, reason=f"no {nuisance!r} column")
    available = [c for c in sources if c in df.columns]
    if not available:
        return _verdict(
            "nuisance_not_deterministic_in_inputs", None, reason=f"none of {list(sources)} present"
        )
    from ..features import infer_condition

    if {"flow iat mean", "flow iat std"}.issubset(df.columns):
        reconstructed = [
            infer_condition(float(m), float(s), 0.0)
            for m, s in zip(
                pd.to_numeric(df["flow iat mean"], errors="coerce").fillna(0.0),
                pd.to_numeric(df["flow iat std"], errors="coerce").fillna(0.0),
            )
        ]
        agreement = float(np.mean(np.asarray(reconstructed) == df[nuisance].astype(str).to_numpy()))
    else:
        agreement = float("nan")
    return _verdict(
        "nuisance_not_deterministic_in_inputs",
        bool(agreement < 0.99) if agreement == agreement else None,
        nuisance=nuisance,
        reconstruction_agreement=agreement,
        reconstructed_from=["flow iat mean", "flow iat std"],
        note=(
            "Agreement near 1.0 means the nuisance label is a thresholding of two model input "
            "features, not an independently measured network condition."
        ),
    )


def run_all_checks(
    df: pd.DataFrame,
    indices: Mapping[str, np.ndarray],
    features: Optional[np.ndarray] = None,
    declared_inputs: Optional[Sequence[str]] = None,
    near_duplicate_threshold: float = 0.999,
    seed: int = 0,
    flow_id_synthesized: bool = False,
) -> Dict[str, object]:
    checks: List[Dict[str, object]] = [check_index_disjoint(indices)]
    for column in IDENTITY_COLUMNS:
        verdict = check_column_disjoint(df, indices, column)
        if column == "flow_id" and flow_id_synthesized:
            # A synthesized flow_id is unique by construction, so this check
            # cannot fail and proves nothing. Say so rather than banking a pass.
            verdict["note"] = (
                "flow_id was synthesized from row content because the table carries none; this "
                "check is vacuous here. The exact-duplicate probe is the load-bearing one."
            )
            verdict["vacuous"] = True
        checks.append(verdict)
    checks.append(check_exact_duplicates(df, indices))
    if features is not None:
        checks.append(check_near_duplicates(features, indices, near_duplicate_threshold, seed=seed))
    if declared_inputs is not None:
        checks.append(check_label_not_in_declared_inputs(declared_inputs))
    checks.append(check_nuisance_label_derivable(df))
    failures = [c["check"] for c in checks if c["status"] == "FAIL"]
    unavailable = [c["check"] for c in checks if c["status"] == "unavailable"]
    return {
        "checks": checks,
        "n_failed": len(failures),
        "failed": failures,
        "unavailable": unavailable,
        "verdict": "FAIL" if failures else ("INCOMPLETE" if unavailable else "PASS"),
    }
