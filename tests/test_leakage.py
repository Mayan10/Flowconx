"""Leakage tests. These run in CI and are allowed to fail the build.

The contract they enforce:

1. A split is a partition. No flow, capture session, server or 5-tuple may
   appear on two sides.
2. No exact or near-duplicate observation may straddle train and test.
3. Nothing that identifies the row or encodes the label may reach a feature
   extractor as an input.
4. A split protocol that cannot be honoured must raise, not silently degrade.
"""

from __future__ import annotations

import numpy as np
import pytest

from flowconx.audit import leakage
from flowconx.audit.splits import (
    SPLIT_PROTOCOLS,
    SplitUnavailable,
    build_split,
    ensure_flow_ids,
    indices_from_manifest,
)
from flowconx.audit.tabular import FORBIDDEN_INPUT_COLUMNS, FEATURE_FAMILIES, build_features
from flowconx.data.schema import MODEL_INPUT_COLUMNS, PROVENANCE_COLUMNS

STRICT_PROTOCOLS = ("session_disjoint", "server_disjoint", "app_disjoint", "origin_disjoint", "temporal")


def _split(frame, protocol, seed=0):
    return build_split(frame, protocol, seed=seed, val_fraction=0.1, test_fraction=0.2)


# --------------------------------------------------------------------------
# 1. A split is a partition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", SPLIT_PROTOCOLS)
def test_split_is_a_partition(synthetic_frame, protocol):
    _, indices = _split(synthetic_frame, protocol)
    covered = np.concatenate([indices[side] for side in ("train", "val", "test")])
    assert len(covered) == len(synthetic_frame), f"{protocol} lost or duplicated rows"
    assert len(np.unique(covered)) == len(covered), f"{protocol} placed a row on two sides"
    assert leakage.check_index_disjoint(indices)["status"] == "pass"


@pytest.mark.parametrize("protocol", SPLIT_PROTOCOLS)
def test_no_flow_id_spans_two_splits(synthetic_frame, protocol):
    frame, _ = ensure_flow_ids(synthetic_frame)
    _, indices = _split(frame, protocol)
    verdict = leakage.check_column_disjoint(frame, indices, "flow_id")
    assert verdict["status"] == "pass", verdict


@pytest.mark.parametrize("protocol", ("session_disjoint",))
def test_no_capture_session_spans_train_and_test(synthetic_frame, protocol):
    _, indices = _split(synthetic_frame, protocol)
    verdict = leakage.check_column_disjoint(synthetic_frame, indices, "capture_id")
    assert verdict["status"] == "pass", verdict


def test_no_server_ip_spans_train_and_test(synthetic_frame):
    _, indices = _split(synthetic_frame, "server_disjoint")
    verdict = leakage.check_column_disjoint(synthetic_frame, indices, "server_ip")
    assert verdict["status"] == "pass", verdict


def test_random_split_does_leak_capture_sessions(synthetic_frame):
    """The contrast that motivates the whole protocol change.

    A stratified random split over flows puts the same capture session on
    both sides. This test asserts that it *does*, so that the session-disjoint
    result is known to be measuring something different rather than being an
    identical split under another name.
    """
    _, indices = _split(synthetic_frame, "random_flow")
    verdict = leakage.check_column_disjoint(synthetic_frame, indices, "capture_id")
    assert verdict["status"] == "FAIL", "random_flow unexpectedly produced session-disjoint splits"
    assert verdict["n_shared"] > 0


# --------------------------------------------------------------------------
# 2. Duplicates
# --------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", STRICT_PROTOCOLS)
def test_no_exact_duplicates_across_splits(synthetic_frame, protocol):
    _, indices = _split(synthetic_frame, protocol)
    verdict = leakage.check_exact_duplicates(synthetic_frame, indices)
    assert verdict["status"] == "pass", verdict


@pytest.mark.parametrize("protocol", ("session_disjoint",))
def test_no_near_duplicates_across_splits(synthetic_frame, protocol):
    _, indices = _split(synthetic_frame, protocol)
    features, _ = build_features(synthetic_frame, "appscanner")
    verdict = leakage.check_near_duplicates(features, indices, threshold=0.99999)
    assert verdict["status"] == "pass", verdict


def test_duplicate_row_is_detected(synthetic_frame):
    """A row copied from train into test must trip the duplicate check."""
    import pandas as pd

    frame = pd.concat([synthetic_frame, synthetic_frame.iloc[[0]]], ignore_index=True)
    indices = {
        "train": np.asarray([0]),
        "val": np.asarray([], dtype=int),
        "test": np.asarray([len(frame) - 1]),
    }
    verdict = leakage.check_exact_duplicates(frame, indices, columns=[c for c in frame.columns if c != "flow_id"])
    assert verdict["status"] == "FAIL", verdict


# --------------------------------------------------------------------------
# 3. Identifiers and labels never become inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(FEATURE_FAMILIES))
def test_feature_family_exposes_no_identifier(synthetic_frame, family):
    _, names = build_features(synthetic_frame, family)
    assert not FORBIDDEN_INPUT_COLUMNS.intersection(names), f"{family} exposes an identifier"


def test_declared_model_inputs_exclude_provenance_and_labels():
    verdict = leakage.check_label_not_in_declared_inputs(MODEL_INPUT_COLUMNS)
    assert verdict["status"] == "pass", verdict
    assert not set(MODEL_INPUT_COLUMNS) & set(PROVENANCE_COLUMNS)


def test_label_leak_is_detected():
    verdict = leakage.check_label_not_in_declared_inputs(["packet_lengths", "service"])
    assert verdict["status"] == "FAIL"
    assert verdict["forbidden_present"] == ["service"]


# --------------------------------------------------------------------------
# 4. Unavailable protocols raise
# --------------------------------------------------------------------------


def test_missing_provenance_raises_rather_than_degrading(synthetic_frame):
    legacy = synthetic_frame.drop(columns=["capture_id", "flow_start_ts", "server_ip", "origin"])
    for protocol in ("session_disjoint", "temporal", "server_disjoint", "origin_disjoint"):
        with pytest.raises(SplitUnavailable):
            _split(legacy, protocol)
    # random_flow and app_disjoint still work; that is the whole problem.
    for protocol in ("random_flow", "app_disjoint"):
        _split(legacy, protocol)


# --------------------------------------------------------------------------
# 5. Manifests are reproducible
# --------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", SPLIT_PROTOCOLS)
def test_manifest_round_trips(synthetic_frame, protocol):
    manifest, indices = _split(synthetic_frame, protocol)
    recovered = indices_from_manifest(synthetic_frame, manifest)
    for side in ("train", "val", "test"):
        assert np.array_equal(np.sort(recovered[side]), np.sort(indices[side]))


@pytest.mark.parametrize("protocol", SPLIT_PROTOCOLS)
def test_split_is_deterministic_for_a_seed(synthetic_frame, protocol):
    first, _ = _split(synthetic_frame, protocol, seed=7)
    second, _ = _split(synthetic_frame, protocol, seed=7)
    assert first.checksums == second.checksums


def test_nuisance_condition_is_flagged_as_derivable(synthetic_frame):
    """`condition` is a threshold on two model inputs, and must be flagged.

    This is not a bug in the splitter; it is a property of the label
    definition. The test pins it so that the day someone replaces the
    heuristic with a measured condition, this test fails and the paper's
    invariance claim gets revisited.
    """
    verdict = leakage.check_nuisance_label_derivable(synthetic_frame)
    assert verdict["status"] == "FAIL", verdict
    assert verdict["reconstruction_agreement"] > 0.99


# --------------------------------------------------------------------------
# Real data, when it is present locally
# --------------------------------------------------------------------------


def test_real_csv_provenance_status(real_frame):
    """Documents which strict protocols the committed CSV can support.

    The committed CSV carries no provenance, so this asserts the *known*
    state. It flips to a pass for the strict protocols once the preparers are
    re-run with the provenance columns they now emit.
    """
    from flowconx.audit.splits import available_protocols

    availability = available_protocols(real_frame)
    assert availability["random_flow"] is True
    assert availability["app_disjoint"] is True
    if not availability["session_disjoint"]:
        pytest.xfail(
            "The committed CSV predates the provenance columns, so session-disjoint evaluation is "
            "impossible on it. Regenerate with scripts/prepare_*.py to lift this."
        )
    _, indices = _split(real_frame, "session_disjoint")
    assert leakage.check_column_disjoint(real_frame, indices, "capture_id")["status"] == "pass"


def test_manifest_survives_a_gzip_round_trip(synthetic_frame, tmp_path):
    """Large manifests are committed gzipped; loading must be transparent."""
    from flowconx.audit.splits import load_manifest, write_manifest

    manifest, _ = _split(synthetic_frame, "session_disjoint")
    for name in ("manifest.json", "manifest.json.gz"):
        path = write_manifest(manifest, tmp_path / name)
        assert path.exists()
        reloaded = load_manifest(path)
        assert reloaded.checksums == manifest.checksums
        assert reloaded.train == manifest.train
