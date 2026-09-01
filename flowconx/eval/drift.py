"""Temporal drift and prototype re-enrollment.

CESNET-QUIC22 spans four consecutive weeks, so a model trained on the earliest
data can be scored week by week going forward. The question a deployment asks
is not "does accuracy fall" -- it always does -- but "how fast, and how
cheaply can it be restored". Prototype re-enrollment answers the second half:
the encoder stays frozen and only the class prototypes are refreshed.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..metrics import balanced_accuracy, macro_f1
from .closed_set import class_prototypes
from .few_shot import enroll_prototypes, predict_from_prototypes


def _period_of(frame) -> np.ndarray:
    """The time bucket each row belongs to, coarsest available."""
    if "capture_id" in frame.columns:
        capture = frame["capture_id"].astype(str)
        if capture.str.contains("/").all():
            return capture.str.split("/").str[0].to_numpy()
    if "flow_start_ts" in frame.columns:
        import pandas as pd

        stamps = pd.to_datetime(frame["flow_start_ts"], unit="s", errors="coerce")
        return stamps.dt.strftime("%Y-W%V").fillna("unknown").to_numpy()
    return np.full(len(frame), "unknown")


def evaluate_drift(
    train_x: np.ndarray,
    train_split,
    test_x: np.ndarray,
    test_split,
    config,
    label_space,
) -> Dict[str, object]:
    periods = _period_of(test_split.frame)
    unique = sorted(set(periods.tolist()))
    if len(unique) < 2:
        return {
            "status": "skipped",
            "reason": (
                f"the test split spans {len(unique)} time period(s); temporal drift needs at least two. "
                "Use data.split_protocol=temporal on a dataset with a real timeline."
            ),
        }

    n_classes = label_space.n_classes
    labels = np.arange(n_classes)
    baseline = class_prototypes(train_x, train_split.labels, n_classes)
    train_periods = sorted(set(_period_of(train_split.frame).tolist()))

    curve: List[Dict[str, object]] = []
    for period in unique:
        mask = periods == period
        predictions = predict_from_prototypes(baseline, test_x[mask])
        entry: Dict[str, object] = {
            "period": period,
            "n_flows": int(mask.sum()),
            "macro_f1": macro_f1(predictions, test_split.labels[mask], labels),
            "balanced_accuracy": balanced_accuracy(predictions, test_split.labels[mask], labels),
        }
        # Re-enrollment: refresh prototypes from k labelled flows of this
        # period, then re-score the rest of it. The encoder is untouched.
        for k in (5, 25, 100):
            rng = np.random.default_rng(config.seed * 977 + k)
            pool = np.flatnonzero(mask)
            if pool.size <= k * n_classes:
                continue
            refreshed = enroll_prototypes(test_x[pool], test_split.labels[pool], n_classes, k, rng)
            held = pool  # scored on the whole period; the k enrolled flows are
            # a small fraction of it and are reported alongside.
            entry[f"macro_f1_after_{k}shot_reenroll"] = macro_f1(
                predict_from_prototypes(refreshed, test_x[held]), test_split.labels[held], labels
            )
        curve.append(entry)

    first = curve[0]["macro_f1"]
    degradation = [
        {"period": entry["period"], "relative_to_first": float(entry["macro_f1"] - first)} for entry in curve
    ]
    return {
        "status": "ok",
        "train_periods": train_periods,
        "test_periods": unique,
        "curve": curve,
        "degradation": degradation,
        "total_drop_macro_f1": float(curve[-1]["macro_f1"] - first),
    }
