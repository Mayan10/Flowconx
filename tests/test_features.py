"""Feature-extraction tests.

Two properties are load-bearing:

1. **Determinism.** The same rows must produce byte-identical tensors, or the
   determinism claim for the whole pipeline is false at its first step.
2. **No constants, no identifiers.** Every per-packet channel must actually
   vary within a flow, and no identifier column may reach the tensors. Both
   were violated by the extractor this one replaced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowconx.data.schema import MODEL_INPUT_COLUMNS, PROVENANCE_COLUMNS
from flowconx.features.packet import (
    FLOW_FEATURE_NAMES,
    PACKET_FEATURE_NAMES,
    build_features,
    build_packet_features,
    parse_float_series,
    truncate,
)


def test_parse_float_series_handles_separators_and_junk():
    assert np.allclose(parse_float_series("1;2;3"), [1, 2, 3])
    assert np.allclose(parse_float_series("1,2,3"), [1, 2, 3])
    assert np.allclose(parse_float_series("1|2|3"), [1, 2, 3])
    assert np.allclose(parse_float_series("1;;2; ;3"), [1, 2, 3])
    assert np.allclose(parse_float_series("1;oops;3"), [1, 3])
    assert parse_float_series("").size == 0
    assert parse_float_series(None).size == 0
    assert parse_float_series(float("nan")).size == 0


def test_packet_features_are_not_constant_within_a_flow():
    """The defect this extractor exists to fix.

    The previous version broadcast 11 of its 16 channels as flow-level
    constants. Here every channel must vary for a flow whose packets vary.
    """
    rng = np.random.default_rng(0)
    lengths = rng.uniform(64, 1400, 24)
    iats = rng.uniform(0.5, 40.0, 24)
    directions = np.where(rng.random(24) < 0.5, 1, -1)
    features = build_packet_features(lengths, iats, directions, 24)
    assert features is not None
    for i, name in enumerate(PACKET_FEATURE_NAMES):
        assert features[:, i].std() > 0, f"channel {name} is constant within a flow"


def test_packet_features_are_deterministic():
    lengths = np.array([100.0, 200.0, 300.0])
    iats = np.array([0.0, 5.0, 10.0])
    directions = np.array([1, -1, 1])
    a = build_packet_features(lengths, iats, directions, 8)
    b = build_packet_features(lengths, iats, directions, 8)
    assert np.array_equal(a, b)


def test_direction_zero_is_treated_as_forward():
    features = build_packet_features(np.array([100.0]), np.array([0.0]), np.array([0]), 4)
    assert features[0, PACKET_FEATURE_NAMES.index("direction")] == 1.0


def test_empty_flow_returns_none_rather_than_a_synthetic_sequence():
    """No fabrication. The previous pipeline generated a label-seeded sequence."""
    assert build_packet_features(np.zeros(0), np.zeros(0), np.zeros(0), 8) is None


def test_build_features_drops_and_counts_empty_rows(synthetic_frame):
    frame = synthetic_frame.copy()
    frame.loc[frame.index[:3], "packet_lengths"] = ""
    frame.loc[frame.index[:3], "iat_values"] = ""
    frame.loc[frame.index[:3], "directions"] = ""
    bundle = build_features(frame, max_packets=16)
    assert bundle.n_dropped == 3
    assert len(bundle) == len(frame) - 3
    assert len(bundle.kept_index) == len(bundle)


def test_build_features_is_deterministic(synthetic_frame):
    a = build_features(synthetic_frame, max_packets=16)
    b = build_features(synthetic_frame, max_packets=16)
    assert np.array_equal(a.packet_seq, b.packet_seq)
    assert np.array_equal(a.flow_features, b.flow_features)
    assert np.array_equal(a.packet_mask, b.packet_mask)


def test_mask_marks_exactly_the_padding(synthetic_frame):
    bundle = build_features(synthetic_frame, max_packets=40)
    for row in range(min(20, len(bundle))):
        real = ~bundle.packet_mask[row]
        # Padding must be a suffix: no real packet may follow a padded slot.
        assert not np.any(np.diff(real.astype(int)) > 0), "mask is not a contiguous prefix"
        assert np.allclose(bundle.packet_seq[row][bundle.packet_mask[row]], 0.0)


def test_features_are_finite(synthetic_frame):
    bundle = build_features(synthetic_frame, max_packets=16)
    assert np.isfinite(bundle.packet_seq).all()
    assert np.isfinite(bundle.flow_features).all()


def test_truncate_is_a_prefix(synthetic_frame):
    full = build_features(synthetic_frame, max_packets=24)
    short = truncate(full, 5)
    assert short.observed_packets == 5
    assert np.array_equal(short.packet_seq, full.packet_seq[:, :5])
    assert np.array_equal(short.packet_mask, full.packet_mask[:, :5])


def test_observed_packets_matches_truncation(synthetic_frame):
    """Reading fewer packets must equal reading many and slicing."""
    direct = build_features(synthetic_frame, max_packets=24, observed_packets=6)
    sliced = truncate(build_features(synthetic_frame, max_packets=24), 6)
    assert np.array_equal(direct.packet_seq, sliced.packet_seq)


def test_observed_packets_cannot_exceed_max(synthetic_frame):
    with pytest.raises(ValueError, match="observed_packets"):
        build_features(synthetic_frame, max_packets=8, observed_packets=16)


def test_no_identifier_reaches_the_feature_names():
    names = set(PACKET_FEATURE_NAMES) | set(FLOW_FEATURE_NAMES)
    assert not names & set(PROVENANCE_COLUMNS)
    assert "protocol" not in names, "protocol is provenance; it encoded the export format"
    assert not names & {"app", "service", "sni", "server_ip", "server_port", "capture_id"}


def test_declared_model_inputs_exclude_all_provenance():
    assert not set(MODEL_INPUT_COLUMNS) & set(PROVENANCE_COLUMNS)


def test_identical_flows_give_identical_features():
    row = {
        "packet_lengths": "100;200;300;400",
        "iat_values": "0;1;2;3",
        "directions": "1;-1;1;-1",
        "total packets": 4,
        "total fwd packets": 2,
        "total backward packets": 2,
        "packet length mean": 250.0,
        "packet length std": 111.8,
        "flow iat mean": 1.5,
        "flow iat std": 1.118,
        "flow duration": 6.0,
        "flow bytes/s": 166666.0,
        "flow packets/s": 666.0,
    }
    frame = pd.DataFrame([row, row])
    bundle = build_features(frame, max_packets=8)
    assert np.array_equal(bundle.packet_seq[0], bundle.packet_seq[1])
    assert np.array_equal(bundle.flow_features[0], bundle.flow_features[1])
