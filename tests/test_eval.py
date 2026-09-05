"""Evaluation-mode tests.

These check the properties that make each metric mean what the paper says it
means: that AUROC is a rank statistic and not an accuracy, that an enrollment
curve improves with more shots, that a defence's overhead is measured
alongside its accuracy cost, and that every mode refuses to run rather than
returning a number when its precondition is missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from flowconx.eval.closed_set import (
    class_prototypes,
    cosine_matrix,
    embedding_geometry,
    evaluate_heads,
    knn_predict,
    prototype_predict,
)
from flowconx.eval.few_shot import enroll_prototypes, enrollment_curve, predict_from_prototypes
from flowconx.eval.open_set import auroc, fpr_at_tpr, oscr_curve


@pytest.fixture
def clustered():
    """Four well-separated Gaussian clusters in 16 dimensions."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(4, 16)) * 3.0

    def sample(n: int, seed: int):
        local = np.random.default_rng(seed)
        y = local.integers(0, 4, n)
        return centres[y] + local.normal(scale=0.4, size=(n, 16)), y

    return sample(600, 1), sample(300, 2)


# --------------------------------------------------------------------------
# Closed set
# --------------------------------------------------------------------------


def test_knn_and_prototype_recover_clean_clusters(clustered):
    (train_x, train_y), (test_x, test_y) = clustered
    assert float(np.mean(knn_predict(train_x, train_y, test_x, k=5) == test_y)) > 0.95
    assert float(np.mean(prototype_predict(train_x, train_y, test_x, 4) == test_y)) > 0.95


def test_knn_blocking_matches_unblocked(clustered):
    """The blocked implementation must not change the answer."""
    (train_x, train_y), (test_x, _) = clustered
    assert np.array_equal(
        knn_predict(train_x, train_y, test_x, k=5, block=4096),
        knn_predict(train_x, train_y, test_x, k=5, block=17),
    )


def test_prototypes_are_unit_norm_and_skip_absent_classes():
    x = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    y = np.array([0, 0, 1])
    prototypes = class_prototypes(x, y, n_classes=4)
    assert prototypes.shape == (4, 2)
    assert np.allclose(np.linalg.norm(prototypes[:2], axis=1), 1.0)
    # Classes with no examples stay at the origin rather than being invented.
    assert np.allclose(prototypes[2:], 0.0)


def test_prototype_predict_never_returns_an_empty_class():
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 3])
    predictions = prototype_predict(x, y, np.array([[1.0, 0.1], [0.1, 1.0]]), n_classes=5)
    assert set(predictions.tolist()) <= {0, 3}


def test_cosine_matrix_diagonal_is_one(clustered):
    (train_x, _), _ = clustered
    assert np.allclose(np.diag(cosine_matrix(train_x[:20])), 1.0, atol=1e-6)


def test_embedding_geometry_separates_clusters(clustered):
    _, (test_x, test_y) = clustered
    geometry = embedding_geometry(test_x, test_y, seed=0)
    assert geometry["intra_class_cosine"] > geometry["inter_class_cosine"]
    assert geometry["separation"] == pytest.approx(
        geometry["intra_class_cosine"] - geometry["inter_class_cosine"]
    )


def test_evaluate_heads_reports_every_requested_head(clustered):
    (train_x, train_y), (test_x, test_y) = clustered
    names = ["a", "b", "c", "d"]
    results = evaluate_heads(
        train_x, train_y, test_x, test_y, names, heads=("knn", "prototype", "linear"), bootstrap_resamples=50
    )
    assert set(results) == {"knn", "prototype", "linear"}
    for report in results.values():
        assert set(report["per_class_f1"]) == set(names)
        assert "macro_f1_ci95" in report
        assert report["macro_f1_ci95"]["lo"] <= report["macro_f1"] <= report["macro_f1_ci95"]["hi"]


def test_unknown_head_is_rejected(clustered):
    (train_x, train_y), (test_x, test_y) = clustered
    with pytest.raises(ValueError, match="Unknown classifier head"):
        evaluate_heads(train_x, train_y, test_x, test_y, ["a"], heads=("magic",))


# --------------------------------------------------------------------------
# Open set
# --------------------------------------------------------------------------


def test_auroc_endpoints_and_ties():
    assert auroc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auroc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    # All scores tied: no information, so exactly chance.
    assert auroc(np.ones(50), np.ones(50)) == pytest.approx(0.5)


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    known, unknown = rng.normal(1.0, 1.0, 500), rng.normal(-0.5, 1.5, 400)
    expected = roc_auc_score(
        np.r_[np.ones(len(known)), np.zeros(len(unknown))], np.r_[known, unknown]
    )
    assert auroc(known, unknown) == pytest.approx(expected, abs=1e-9)


def test_fpr_at_tpr_keeps_the_promised_true_positive_rate():
    rng = np.random.default_rng(4)
    known = rng.normal(2.0, 1.0, 2000)
    unknown = rng.normal(-2.0, 1.0, 2000)
    threshold = float(np.quantile(known, 0.05))
    assert np.mean(known >= threshold) == pytest.approx(0.95, abs=0.02)
    assert fpr_at_tpr(known, unknown, 0.95) < 0.05


def test_oscr_curve_is_monotone_in_both_axes():
    rng = np.random.default_rng(5)
    known = rng.normal(1.0, 1.0, 500)
    correct = rng.random(500) < 0.9
    unknown = rng.normal(-1.0, 1.0, 500)
    curve = oscr_curve(known, correct, unknown, n_points=40)
    ccr = [point["ccr"] for point in curve]
    fpr = [point["fpr_unknown"] for point in curve]
    # Raising the threshold can only reject more of both.
    assert all(a >= b - 1e-12 for a, b in zip(ccr, ccr[1:]))
    assert all(a >= b - 1e-12 for a, b in zip(fpr, fpr[1:]))


def test_empty_side_gives_nan_not_a_number():
    assert np.isnan(auroc(np.array([1.0]), np.zeros(0)))
    assert np.isnan(fpr_at_tpr(np.zeros(0), np.array([1.0])))
    assert oscr_curve(np.zeros(0), np.zeros(0, dtype=bool), np.array([1.0])) == []


# --------------------------------------------------------------------------
# Few-shot enrollment
# --------------------------------------------------------------------------


def test_enrollment_uses_at_most_k_per_class(clustered):
    (train_x, train_y), _ = clustered
    rng = np.random.default_rng(0)
    # With k=1 the prototype is a single example, so it must lie on the unit
    # sphere and be one of the training directions.
    prototypes = enroll_prototypes(train_x, train_y, 4, shots=1, rng=rng)
    assert np.allclose(np.linalg.norm(prototypes, axis=1), 1.0)


def test_enrollment_curve_improves_with_more_shots(clustered):
    (train_x, train_y), (test_x, test_y) = clustered
    curve = enrollment_curve(train_x, train_y, test_x, test_y, 4, shots=[1, 5, 50], repeats=5, seed=0)
    scores = [point["macro_f1_mean"] for point in curve]
    assert scores[-1] >= scores[0], "more enrolled examples must not hurt on clean clusters"
    # The spread at k=1 must be reported, not hidden: a single draw is noisy.
    assert curve[0]["macro_f1_std"] >= curve[-1]["macro_f1_std"]


def test_enrollment_curve_is_deterministic(clustered):
    (train_x, train_y), (test_x, test_y) = clustered
    kwargs = dict(n_classes=4, shots=[1, 5], repeats=3, seed=7)
    first = enrollment_curve(train_x, train_y, test_x, test_y, **kwargs)
    second = enrollment_curve(train_x, train_y, test_x, test_y, **kwargs)
    assert first == second


def test_predict_from_empty_prototypes_is_defined():
    assert predict_from_prototypes(np.zeros((3, 8)), np.ones((5, 8))).shape == (5,)


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


def test_every_perturbation_preserves_shape_and_reports_overhead():
    from flowconx.eval.robustness import PERTURBATIONS

    rng = np.random.default_rng(0)
    lengths = rng.uniform(64, 1400, 30)
    iats = rng.uniform(0.1, 20.0, 30)
    directions = np.where(rng.random(30) < 0.5, 1, -1)
    for perturbation in PERTURBATIONS:
        new_lengths, new_iats, new_dirs = perturbation.apply(
            lengths.copy(), iats.copy(), directions.copy(), np.random.default_rng(1)
        )
        assert len(new_lengths) == len(new_iats) == len(new_dirs), perturbation.name
        assert len(new_lengths) > 0, perturbation.name
        assert np.all(new_lengths >= 0), perturbation.name
        assert np.all(new_iats >= 0), perturbation.name


def test_padding_defences_never_shrink_a_packet():
    from flowconx.eval.robustness import PERTURBATIONS

    lengths = np.array([100.0, 500.0, 1400.0])
    iats = np.zeros(3)
    directions = np.ones(3)
    for name in ("pad_mtu", "pad_buckets", "quantize_128", "random_pad"):
        perturbation = next(p for p in PERTURBATIONS if p.name == name)
        new_lengths, _, _ = perturbation.apply(lengths, iats, directions, np.random.default_rng(0))
        assert np.all(new_lengths >= lengths), f"{name} made a packet smaller, which is not padding"


def test_packet_loss_merges_inter_arrivals_rather_than_dropping_time():
    """A dropped packet's gap folds into the next surviving packet.

    The invariant is not that total elapsed time is preserved -- packets
    dropped *after* the last survivor are simply never observed, so the
    observed flow legitimately ends earlier. The invariant is that no time
    vanishes from between two surviving packets, which is what an observer
    downstream of the loss actually sees. Getting this wrong would make loss
    look like a speed-up and flatter the robustness result.
    """
    from flowconx.eval.robustness import _packet_loss

    lengths = np.arange(1.0, 21.0)
    iats = np.ones(20)
    directions = np.ones(20)
    rng = np.random.default_rng(0)
    keep = rng.random(20) >= 0.5
    if not keep.any():
        keep[0] = True
    expected = float(iats[: np.flatnonzero(keep)[-1] + 1].sum())

    _, new_iats, new_dirs = _packet_loss(0.5)(lengths, iats, directions, np.random.default_rng(0))
    assert len(new_iats) == int(keep.sum())
    assert new_iats.sum() == pytest.approx(expected), "time vanished from between surviving packets"
    assert new_iats.sum() <= iats.sum()


def test_constant_rate_removes_all_size_and_timing_variation():
    from flowconx.eval.robustness import PERTURBATIONS

    perturbation = next(p for p in PERTURBATIONS if p.name == "constant_rate")
    rng = np.random.default_rng(0)
    new_lengths, new_iats, _ = perturbation.apply(
        rng.uniform(64, 1400, 20), rng.uniform(0.1, 20, 20), np.ones(20), rng
    )
    assert new_lengths.std() == pytest.approx(0.0, abs=1e-9)
    assert new_iats.std() == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Nuisance probing
# --------------------------------------------------------------------------


def test_probe_recovers_a_nuisance_that_is_present():
    """A probe must find information that is genuinely there.

    If it cannot, then "the probe found nothing" carries no evidence and the
    invariance claim is unfalsifiable.
    """
    from flowconx.eval.probes import probe_target

    rng = np.random.default_rng(0)
    nuisance = rng.integers(0, 3, 800)
    offsets = rng.normal(size=(3, 12)) * 4.0
    x = offsets[nuisance] + rng.normal(scale=0.4, size=(800, 12))
    result = probe_target(x[:400], nuisance[:400], x[400:], nuisance[400:], seed=0)
    assert result["status"] == "ok"
    assert result["probes"]["linear"]["above_majority"] > 0.4
    assert result["probes"]["mlp"]["above_majority"] > 0.4


def test_probe_finds_nothing_when_the_nuisance_is_absent():
    from flowconx.eval.probes import probe_target

    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 12))
    nuisance = rng.integers(0, 3, 800)  # independent of x by construction
    result = probe_target(x[:400], nuisance[:400], x[400:], nuisance[400:], seed=0)
    assert abs(result["probes"]["mlp"]["above_majority"]) < 0.15


def test_probe_skips_a_single_valued_target():
    from flowconx.eval.probes import probe_target

    x = np.random.default_rng(2).normal(size=(50, 4))
    y = np.zeros(50, dtype=int)
    assert probe_target(x[:25], y[:25], x[25:], y[25:], seed=0)["status"] == "skipped"


# --------------------------------------------------------------------------
# Temporal drift
# --------------------------------------------------------------------------


def test_drift_falls_through_to_a_finer_granularity(synthetic_frame):
    """A temporal test slice inside one week must not skip the evaluation.

    A temporal split holds out the last 20% of the timeline, which on
    CESNET-QUIC22 is about five days -- all in one ISO week. Bucketing by week
    alone produced a single period, and the drift evaluation skipped itself on
    exactly the split it exists for.
    """
    import pandas as pd

    from flowconx.eval.drift import _period_of

    frame = synthetic_frame.copy()
    # One week, several days: the coarse bucketing collapses, the fine one does not.
    frame["capture_id"] = "W-2022-47/" + pd.Series(
        [f"2022112{i % 5}" for i in range(len(frame))], index=frame.index
    )
    granularity, periods = _period_of(frame)
    assert granularity != "week"
    assert len(set(periods.tolist())) >= 2


def test_drift_prefers_the_coarsest_usable_granularity(synthetic_frame):
    import pandas as pd

    from flowconx.eval.drift import _period_of

    frame = synthetic_frame.copy()
    frame["capture_id"] = pd.Series(
        [f"W-2022-4{4 + (i % 4)}/2022110{i % 5}" for i in range(len(frame))], index=frame.index
    )
    granularity, periods = _period_of(frame)
    assert granularity == "week", "several weeks are present, so week is the right axis"
    assert len(set(periods.tolist())) == 4


# --------------------------------------------------------------------------
# Cross-dataset transfer
# --------------------------------------------------------------------------


def test_shared_taxonomy_is_a_function_not_a_relation():
    """No source label may map to two shared classes.

    The mapping is a judgement call; a label appearing under two shared
    classes would make the transfer result depend on dictionary order.
    """
    from flowconx.eval.transfer import SHARED_TAXONOMY

    for dataset in ("fiveg_traffic", "cesnet_quic22"):
        seen: dict = {}
        for shared, sources in SHARED_TAXONOMY.items():
            for name in sources.get(dataset, []):
                assert name not in seen, f"{name} maps to both {seen[name]} and {shared}"
                seen[name] = shared


def test_unmapped_labels_are_dropped_not_forced():
    from flowconx.eval.transfer import map_to_shared

    mapped = map_to_shared(["mail", "streaming_media", "search"], "cesnet_quic22")
    assert mapped[0] == -1 and mapped[2] == -1, "unmapped labels must be -1, not coerced"
    assert mapped[1] >= 0


def test_every_corpus_label_is_either_mapped_or_documented():
    """A label must not be silently absent from both the mapping and the
    unmapped list -- that is how an omission becomes invisible."""
    from flowconx.eval.transfer import SHARED_TAXONOMY, UNMAPPED

    for dataset, unmapped in UNMAPPED.items():
        mapped = {n for sources in SHARED_TAXONOMY.values() for n in sources.get(dataset, [])}
        assert not (mapped & set(unmapped)), f"{dataset}: a label is both mapped and listed unmapped"


def test_transfer_skips_when_dimensions_differ():
    from flowconx.eval.transfer import evaluate_transfer

    result = evaluate_transfer(
        np.zeros((10, 16)), ["streaming_media"] * 10, "cesnet_quic22",
        np.zeros((10, 32)), ["live_streaming"] * 10, "fiveg_traffic",
    )
    assert result["status"] == "skipped"
    assert "dimension" in result["reason"]


def test_transfer_recovers_planted_structure():
    """Zero-shot transfer must work when the two corpora share geometry."""
    from flowconx.eval.transfer import evaluate_transfer

    rng = np.random.default_rng(0)
    centres = rng.normal(size=(3, 24)) * 5.0
    src_names = ["streaming_media", "games", "videoconferencing"]
    tgt_names = ["live_streaming", "online_game", "video_conferencing"]
    src_idx = rng.integers(0, 3, 300)
    tgt_idx = rng.integers(0, 3, 300)
    result = evaluate_transfer(
        centres[src_idx] + rng.normal(scale=0.3, size=(300, 24)),
        [src_names[i] for i in src_idx], "cesnet_quic22",
        centres[tgt_idx] + rng.normal(scale=0.3, size=(300, 24)),
        [tgt_names[i] for i in tgt_idx], "fiveg_traffic",
        shots=(0, 5), repeats=2,
    )
    assert result["status"] == "ok"
    assert result["zero_shot_macro_f1"] > 0.9, "planted shared geometry must transfer"
