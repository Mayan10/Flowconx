"""Deterministic feature extraction from the canonical flow CSV.

This replaces the original extractor, which had two defects the audit called
out (AUDIT.md 1.4):

* eleven of its sixteen packet channels were flow-level scalars broadcast
  identically across all 128 positions, so most of the transformer's input
  width carried a constant;
* the "network condition" series was a constant vector plus Gaussian noise,
  with the declared nuisance label quantised into channel 7 -- the model was
  handed the very variable the adversarial head was supposed to remove.

Here the two views are genuinely different. The **sequence view** is strictly
per-packet: nothing that is constant within a flow appears in it. The
**context view** is the flow-level statistics vector. Cross-attention between
them is then a real fusion of two descriptions rather than a fusion of a
sequence with a copy of its own summary.

Nothing is synthesised. If a flow has no packet series, it is dropped and
counted, rather than having one generated from a label-seeded RNG (L7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Per-packet channels. Every one of these varies within a flow.
PACKET_FEATURE_NAMES: List[str] = [
    "size_norm",          # packet size / 1500
    "log_iat",            # log1p(inter-arrival ms) / 10
    "direction",          # +1 client->server, -1 server->client
    "signed_size",        # size_norm * direction
    "log_cum_bytes",      # log1p(bytes seen so far) / 16   -- causal
    "log_cum_time",       # log1p(ms elapsed so far) / 12   -- causal
    "size_delta",         # (size - previous size) / 1500
    "burst_position",     # index within the current same-direction run / 16
]

# Flow-level context. Deliberately excludes anything identifying: no address,
# no port, no SNI, no capture id.
FLOW_FEATURE_NAMES: List[str] = [
    "log_n_packets",
    "log_n_bytes",
    "log_duration_ms",
    "forward_ratio",
    "log_mean_size",
    "log_std_size",
    "log_mean_iat",
    "log_std_iat",
    "log_bytes_per_s",
    "log_packets_per_s",
]

PKT_FEATURE_DIM = len(PACKET_FEATURE_NAMES)
FLOW_FEATURE_DIM = len(FLOW_FEATURE_NAMES)

MAX_PACKET_SIZE = 1500.0


@dataclass
class FeatureBundle:
    """Model inputs plus the bookkeeping needed to trace them back to rows."""

    packet_seq: np.ndarray      # (n, T, PKT_FEATURE_DIM) float32
    packet_mask: np.ndarray     # (n, T) bool, True where padded
    flow_features: np.ndarray   # (n, FLOW_FEATURE_DIM) float32
    kept_index: np.ndarray      # positions in the source frame that survived
    n_dropped: int
    observed_packets: int
    packet_feature_names: List[str]
    flow_feature_names: List[str]

    def __len__(self) -> int:
        return int(self.packet_seq.shape[0])

    def describe(self) -> Dict[str, object]:
        return {
            "n_flows": len(self),
            "n_dropped_no_sequence": int(self.n_dropped),
            "observed_packets": int(self.observed_packets),
            "packet_features": list(self.packet_feature_names),
            "flow_features": list(self.flow_feature_names),
        }


def parse_float_series(text: object) -> np.ndarray:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.zeros(0, dtype=np.float64)
    raw = str(text).replace("|", ";").replace(",", ";")
    out: List[float] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return np.asarray(out, dtype=np.float64)


def _safe_log(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(values, 0.0, None))


def build_packet_features(
    lengths: np.ndarray,
    iats: np.ndarray,
    directions: np.ndarray,
    observed_packets: int,
) -> Optional[np.ndarray]:
    """Per-packet feature matrix for one flow, or None if it has no packets."""
    n = min(len(lengths), len(iats), len(directions), observed_packets)
    if n == 0:
        return None
    lengths = np.clip(lengths[:n].astype(np.float64), 0.0, 65535.0)
    iats = np.clip(iats[:n].astype(np.float64), 0.0, None)
    directions = np.sign(directions[:n].astype(np.float64))
    directions[directions == 0.0] = 1.0

    out = np.zeros((n, PKT_FEATURE_DIM), dtype=np.float32)
    size_norm = lengths / MAX_PACKET_SIZE
    out[:, 0] = size_norm
    out[:, 1] = _safe_log(iats) / 10.0
    out[:, 2] = directions
    out[:, 3] = size_norm * directions

    # Causal by construction. An earlier version normalised these by the
    # window total, which means the value at packet 1 depended on packets the
    # classifier had not seen yet -- harmless for full-flow scoring, but it
    # silently inflated the early-classification curve, which is precisely the
    # claim that has to hold. Normalising by a fixed constant instead makes
    # truncating the sequence exactly equal to observing fewer packets, and
    # tests/test_features.py asserts that equality.
    out[:, 4] = _safe_log(np.cumsum(lengths)) / 16.0
    out[:, 5] = _safe_log(np.cumsum(iats)) / 12.0

    out[1:, 6] = np.diff(size_norm)

    # Position within the current run of same-direction packets. Bursts are
    # the structure every classical fingerprinting method keys on, so it is
    # given to the model explicitly rather than left to be inferred.
    burst = np.zeros(n, dtype=np.float64)
    run = 0.0
    previous = directions[0]
    for i in range(n):
        run = run + 1.0 if directions[i] == previous else 0.0
        previous = directions[i]
        burst[i] = run
    out[:, 7] = np.clip(burst, 0.0, 16.0) / 16.0
    return out


def build_flow_features(row: Mapping[str, object], n_packets_observed: int) -> np.ndarray:
    out = np.zeros(FLOW_FEATURE_DIM, dtype=np.float32)
    n_packets = float(row.get("total packets", n_packets_observed) or n_packets_observed)
    mean_size = float(row.get("packet length mean", 0.0) or 0.0)
    std_size = float(row.get("packet length std", 0.0) or 0.0)
    mean_iat = float(row.get("flow iat mean", 0.0) or 0.0)
    std_iat = float(row.get("flow iat std", 0.0) or 0.0)
    duration = float(row.get("flow duration", 0.0) or 0.0)
    bytes_per_s = float(row.get("flow bytes/s", 0.0) or 0.0)
    packets_per_s = float(row.get("flow packets/s", 0.0) or 0.0)
    fwd = float(row.get("total fwd packets", 0.0) or 0.0)
    bwd = float(row.get("total backward packets", 0.0) or 0.0)

    out[0] = np.log1p(max(n_packets, 0.0)) / 12.0
    out[1] = np.log1p(max(n_packets * mean_size, 0.0)) / 20.0
    out[2] = np.log1p(max(duration, 0.0)) / 16.0
    out[3] = fwd / max(fwd + bwd, 1.0)
    out[4] = np.log1p(max(mean_size, 0.0)) / 8.0
    out[5] = np.log1p(max(std_size, 0.0)) / 8.0
    out[6] = np.log1p(max(mean_iat, 0.0)) / 10.0
    out[7] = np.log1p(max(std_iat, 0.0)) / 10.0
    out[8] = np.log1p(max(bytes_per_s, 0.0)) / 20.0
    out[9] = np.log1p(max(packets_per_s, 0.0)) / 12.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_features(
    df: pd.DataFrame,
    max_packets: int = 32,
    observed_packets: Optional[int] = None,
) -> FeatureBundle:
    """Vectorise a canonical frame into model inputs.

    ``observed_packets`` truncates the sequence at inference *and* training
    time without re-reading the CSV, which is what the input-budget sweep and
    the early-classification curve both need.
    """
    budget = int(observed_packets or max_packets)
    if budget > max_packets:
        raise ValueError(f"observed_packets ({budget}) exceeds max_packets ({max_packets})")

    lengths_col = df["packet_lengths"].to_numpy(dtype=object)
    iats_col = df["iat_values"].to_numpy(dtype=object)
    dirs_col = df["directions"].to_numpy(dtype=object)
    records = df.to_dict("records")

    packet_seq = np.zeros((len(df), budget, PKT_FEATURE_DIM), dtype=np.float32)
    packet_mask = np.ones((len(df), budget), dtype=bool)
    flow_features = np.zeros((len(df), FLOW_FEATURE_DIM), dtype=np.float32)
    kept: List[int] = []
    dropped = 0

    for i in range(len(df)):
        features = build_packet_features(
            parse_float_series(lengths_col[i]),
            parse_float_series(iats_col[i]),
            parse_float_series(dirs_col[i]),
            budget,
        )
        if features is None:
            dropped += 1
            continue
        n = features.shape[0]
        packet_seq[i, :n] = features
        packet_mask[i, :n] = False
        flow_features[i] = build_flow_features(records[i], n)
        kept.append(i)

    keep = np.asarray(kept, dtype=int)
    return FeatureBundle(
        packet_seq=packet_seq[keep],
        packet_mask=packet_mask[keep],
        flow_features=flow_features[keep],
        kept_index=keep,
        n_dropped=dropped,
        observed_packets=budget,
        packet_feature_names=list(PACKET_FEATURE_NAMES),
        flow_feature_names=list(FLOW_FEATURE_NAMES),
    )


def truncate(bundle: FeatureBundle, observed_packets: int) -> FeatureBundle:
    """A view of ``bundle`` with fewer observed packets, for early-decision curves."""
    budget = min(observed_packets, bundle.observed_packets)
    return FeatureBundle(
        packet_seq=bundle.packet_seq[:, :budget].copy(),
        packet_mask=bundle.packet_mask[:, :budget].copy(),
        flow_features=bundle.flow_features.copy(),
        kept_index=bundle.kept_index.copy(),
        n_dropped=bundle.n_dropped,
        observed_packets=budget,
        packet_feature_names=list(bundle.packet_feature_names),
        flow_feature_names=list(bundle.flow_feature_names),
    )


def assert_no_identifier_leak(names: Sequence[str], forbidden: Sequence[str]) -> None:
    overlap = sorted(set(names) & set(forbidden))
    if overlap:
        raise ValueError(f"Feature names include identifier columns: {overlap}")


def feature_summary() -> Tuple[List[str], List[str]]:
    return list(PACKET_FEATURE_NAMES), list(FLOW_FEATURE_NAMES)
