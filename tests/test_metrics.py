"""Metric-suite tests.

The point of these is narrow: assert that the headline metrics behave
differently from accuracy on an imbalanced set, since that difference is the
reason the paper switched to them.
"""

from __future__ import annotations

import numpy as np

from flowconx.labels import RareClassPolicy, apply_rare_class_policy
from flowconx.metrics import (
    accuracy,
    balanced_accuracy,
    bootstrap_ci,
    classification_report,
    macro_f1,
    per_class_f1,
    top_confusions,
)


def test_perfect_prediction_scores_one():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert accuracy(y, y) == 1.0
    assert macro_f1(y, y) == 1.0
    assert balanced_accuracy(y, y) == 1.0


def test_majority_predictor_on_imbalanced_data_looks_good_only_on_accuracy():
    """99 rows of class 0, 1 row of class 1. Always predicting 0 scores 0.99."""
    target = np.array([0] * 99 + [1])
    pred = np.zeros(100, dtype=int)
    assert accuracy(pred, target) == 0.99
    # Macro-F1 and balanced accuracy both refuse to be impressed.
    assert macro_f1(pred, target) < 0.51
    assert balanced_accuracy(pred, target) == 0.5


def test_per_class_f1_covers_classes_absent_from_predictions():
    target = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0])
    scores = per_class_f1(pred, target, labels=[0, 1, 2])
    assert set(scores) == {"0", "1", "2"}
    assert scores["1"] == 0.0
    assert scores["2"] == 0.0


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 4, 400)
    pred = np.where(rng.random(400) < 0.75, target, rng.integers(0, 4, 400))
    ci = bootstrap_ci(pred, target, "macro_f1", n_resamples=200, seed=0)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["hi"] - ci["lo"] > 0.0


def test_bootstrap_ci_is_deterministic_for_a_seed():
    rng = np.random.default_rng(1)
    target = rng.integers(0, 3, 200)
    pred = rng.integers(0, 3, 200)
    first = bootstrap_ci(pred, target, "macro_f1", n_resamples=100, seed=5)
    second = bootstrap_ci(pred, target, "macro_f1", n_resamples=100, seed=5)
    assert first == second


def test_classification_report_confusion_matrix_is_consistent():
    target = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 2, 2])
    report = classification_report(pred, target, bootstrap=False)
    matrix = np.asarray(report["confusion_matrix"])
    assert matrix.sum() == len(target)
    assert matrix.trace() == int(np.sum(pred == target))


def test_top_confusions_reports_the_worst_pair_first():
    target = np.array([0] * 10 + [1] * 10)
    pred = np.array([1] * 8 + [0] * 2 + [1] * 10)
    pairs = top_confusions(pred, target, k=3)
    assert pairs[0]["true"] == "0" and pairs[0]["predicted"] == "1"
    assert pairs[0]["count"] == 8


def test_rare_class_policy_modes_are_exhaustive_and_recorded(synthetic_frame):
    import pandas as pd

    rare = synthetic_frame.iloc[:3].copy()
    rare["service"] = "xr_interactive"
    frame = pd.concat([synthetic_frame, rare], ignore_index=True)

    dropped, report = apply_rare_class_policy(frame, RareClassPolicy(mode="drop", min_class_count=10))
    assert report["action"] == "dropped"
    assert "xr_interactive" not in set(dropped["service"])
    assert report["rows_dropped"] == 3

    merged, report = apply_rare_class_policy(
        frame, RareClassPolicy(mode="merge_into_parent", min_class_count=10)
    )
    assert report["action"] == "merged"
    assert "xr_interactive" not in set(merged["service"])
    assert len(merged) == len(frame)

    kept, report = apply_rare_class_policy(frame, RareClassPolicy(mode="keep_and_report", min_class_count=10))
    assert report["action"] == "kept"
    assert "warning" in report
    assert len(kept) == len(frame)
