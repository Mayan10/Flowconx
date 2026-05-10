"""Feature extraction helpers for flow-level CSV rows."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import MAX_PACKETS, NET_TIMESTEPS, PKT_FEAT_DIM, NET_FEAT_DIM


PACKET_LENGTH_ALIASES = [
    "packet_lengths",
    "pkt_lengths",
    "lengths",
    "packet_size_series",
    "pkt_size_series",
]

IAT_ALIASES = [
    "iat_values",
    "iats",
    "inter_arrival_times",
    "inter_packet_times",
    "time_delta_series",
]

DIRECTION_ALIASES = [
    "directions",
    "direction_series",
    "dir_series",
]


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def row_get(row: Mapping[str, object], names: Sequence[str], default: float = 0.0) -> float:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name.strip().lower())
        if value is None or value == "":
            continue
        try:
            if isinstance(value, str):
                value = value.replace(",", "")
            number = float(value)
            if math.isfinite(number):
                return number
        except (TypeError, ValueError):
            continue
    return default


def row_text(row: Mapping[str, object], names: Sequence[str]) -> Optional[str]:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name.strip().lower())
        if value is not None and str(value).strip():
            return str(value)
    return None


def parse_series(value: Optional[str], dtype: str = "float") -> Optional[np.ndarray]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    for sep in [";", "|", " "]:
        text = text.replace(sep, ",")
    parts = [part for part in text.split(",") if part != ""]
    if not parts:
        return None
    try:
        if dtype == "int":
            return np.asarray([int(float(part)) for part in parts], dtype=np.int64)
        return np.asarray([float(part) for part in parts], dtype=np.float32)
    except ValueError:
        return None


def pad_or_trim(array: np.ndarray, length: int, width: int) -> np.ndarray:
    out = np.zeros((length, width), dtype=np.float32)
    if array.size == 0:
        return out
    rows = min(length, array.shape[0])
    cols = min(width, array.shape[1])
    out[:rows, :cols] = array[:rows, :cols]
    return out


def _protocol_flags(protocol: float) -> Tuple[float, float, float]:
    protocol_int = int(protocol)
    tcp = 1.0 if protocol_int == 6 else 0.0
    udp = 1.0 if protocol_int == 17 else 0.0
    quic = udp
    return tcp, udp, quic


def _make_directions(count: int, fwd_packets: float, bwd_packets: float, rng: np.random.Generator) -> np.ndarray:
    total = max(fwd_packets + bwd_packets, 1.0)
    fwd_prob = min(max(fwd_packets / total, 0.05), 0.95)
    dirs = rng.choice([1.0, -1.0], size=count, p=[fwd_prob, 1.0 - fwd_prob])
    return dirs.astype(np.float32)


def packet_sequence_from_row(
    row: Mapping[str, object],
    max_packets: int = MAX_PACKETS,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Build a packet sequence from true packet series or aggregate flow columns."""
    if rng is None:
        rng = np.random.default_rng(stable_seed(row_text(row, ["app", "label", "service"]) or "flow"))

    lengths = parse_series(row_text(row, PACKET_LENGTH_ALIASES))
    iats = parse_series(row_text(row, IAT_ALIASES))
    directions = parse_series(row_text(row, DIRECTION_ALIASES), dtype="int")

    fwd_packets = row_get(row, ["total fwd packets", "tot fwd pkts", "fwd packets", "fwd_pkt_count"], 32.0)
    bwd_packets = row_get(row, ["total backward packets", "total bwd packets", "bwd packets", "bwd_pkt_count"], 16.0)
    pkt_count = int(max(4, min(max_packets, row_get(row, ["packet count", "tot pkts", "total packets"], fwd_packets + bwd_packets))))

    if lengths is None:
        mean_len = row_get(row, ["packet length mean", "pkt len mean", "average packet size", "avg pkt size"], 600.0)
        std_len = row_get(row, ["packet length std", "pkt len std", "packet length variance"], 180.0)
        if std_len > 2000.0:
            std_len = math.sqrt(std_len)
        lengths = rng.normal(mean_len, max(std_len, 1.0), size=pkt_count).clip(40.0, 1500.0).astype(np.float32)
    if iats is None:
        mean_iat = row_get(row, ["flow iat mean", "iat mean", "flow_iat_mean"], 20.0)
        std_iat = row_get(row, ["flow iat std", "iat std", "flow_iat_std"], max(mean_iat * 0.2, 1.0))
        iats = rng.normal(mean_iat, max(std_iat, 1.0), size=len(lengths)).clip(0.0).astype(np.float32)
    if directions is None:
        directions = _make_directions(len(lengths), fwd_packets, bwd_packets, rng)

    count = min(max_packets, len(lengths), len(iats), len(directions))
    lengths = lengths[:count].astype(np.float32)
    iats = iats[:count].astype(np.float32)
    directions = directions[:count].astype(np.float32)

    duration = row_get(row, ["flow duration", "duration", "flow_duration"], float(np.sum(iats)))
    bytes_per_s = row_get(row, ["flow bytes/s", "flow bytes per s", "bytes_per_second"], float(np.sum(lengths) / max(duration, 1.0)))
    packets_per_s = row_get(row, ["flow packets/s", "flow packets per s", "packets_per_second"], float(count / max(duration, 1.0)))
    fwd_ratio = float(np.mean(directions > 0.0))
    bwd_ratio = 1.0 - fwd_ratio
    protocol = row_get(row, ["protocol", "proto"], 6.0)
    tcp, udp, quic = _protocol_flags(protocol)
    syn = row_get(row, ["syn flag count", "syn"], 0.0)
    ack = row_get(row, ["ack flag count", "ack"], 0.0)
    rst = row_get(row, ["rst flag count", "rst"], 0.0)

    cumulative_time = np.cumsum(iats)
    total_time = max(float(cumulative_time[-1]) if len(cumulative_time) else 1.0, 1.0)
    burst_index = np.floor((cumulative_time / total_time) * 8.0) / 8.0

    seq = np.zeros((count, PKT_FEAT_DIM), dtype=np.float32)
    seq[:, 0] = lengths / 1500.0
    seq[:, 1] = np.log1p(iats) / 10.0
    seq[:, 2] = directions
    seq[:, 3] = math.log1p(max(packets_per_s, 0.0)) / 10.0
    seq[:, 4] = math.log1p(max(bytes_per_s, 0.0)) / 20.0
    seq[:, 5] = fwd_ratio
    seq[:, 6] = bwd_ratio
    seq[:, 7] = tcp
    seq[:, 8] = udp
    seq[:, 9] = quic
    seq[:, 10] = min(syn, 16.0) / 16.0
    seq[:, 11] = min(ack, 64.0) / 64.0
    seq[:, 12] = min(rst, 16.0) / 16.0
    seq[:, 13] = math.log1p(max(duration, 0.0)) / 20.0
    seq[:, 14] = np.clip(burst_index, 0.0, 1.0)
    seq[:, 15] = np.arange(count, dtype=np.float32) / max(count - 1, 1)
    return pad_or_trim(seq, max_packets, PKT_FEAT_DIM)


def network_series_from_row(
    row: Mapping[str, object],
    timesteps: int = NET_TIMESTEPS,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Build a network condition series from explicit or proxy columns."""
    if rng is None:
        rng = np.random.default_rng(stable_seed(row_text(row, ["app", "label", "service"]) or "net"))

    rtt = row_get(row, ["rtt_ms", "rtt", "mean_rtt", "flow iat mean"], 35.0)
    jitter = row_get(row, ["jitter_ms", "jitter", "flow iat std"], 5.0)
    loss = row_get(row, ["loss_rate", "packet_loss", "loss"], 0.0)
    retrans = row_get(row, ["retransmissions", "retrans", "retransmit_count"], 0.0)
    throughput = row_get(row, ["throughput_mbps", "throughput", "flow bytes/s"], 5.0)
    uplink_ratio = row_get(row, ["uplink_ratio", "down/up ratio", "down_up_ratio"], 0.5)
    queue_delay = row_get(row, ["queue_delay_ms", "queue_delay", "buffer_delay"], max(jitter * 0.4, 1.0))
    condition_hint = condition_to_index(infer_condition(rtt, jitter, loss))

    t = np.linspace(0.0, 1.0, timesteps, dtype=np.float32)
    drift = rng.normal(0.0, 0.03, size=(timesteps, NET_FEAT_DIM)).astype(np.float32)
    series = np.zeros((timesteps, NET_FEAT_DIM), dtype=np.float32)
    series[:, 0] = np.log1p(max(rtt, 0.0)) / 8.0
    series[:, 1] = np.log1p(max(jitter, 0.0)) / 6.0
    series[:, 2] = np.clip(loss, 0.0, 1.0)
    series[:, 3] = np.log1p(max(retrans, 0.0)) / 8.0
    series[:, 4] = np.log1p(max(throughput, 0.0)) / 16.0
    series[:, 5] = np.clip(uplink_ratio, 0.0, 1.0)
    series[:, 6] = np.log1p(max(queue_delay, 0.0)) / 6.0
    series[:, 7] = condition_hint / 3.0
    series += drift * (0.5 + t[:, None])
    return np.clip(series, -5.0, 5.0).astype(np.float32)


def infer_condition(rtt_ms: float, jitter_ms: float, loss_rate: float) -> str:
    score = 0
    if rtt_ms > 80.0:
        score += 1
    if rtt_ms > 180.0:
        score += 1
    if jitter_ms > 25.0:
        score += 1
    if loss_rate > 0.01:
        score += 1
    if loss_rate > 0.04:
        score += 1
    if score <= 0:
        return "good"
    if score <= 2:
        return "moderate"
    if score <= 4:
        return "degraded"
    return "bad"


def condition_to_index(condition: str) -> int:
    return {"good": 0, "moderate": 1, "degraded": 2, "bad": 3}.get(condition, 4)


def augment_network_condition(
    pkt_seq: np.ndarray,
    net_series: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Create a same-service counterfactual with changed network condition."""
    pkt_aug = np.array(pkt_seq, copy=True)
    net_aug = np.array(net_series, copy=True)
    rtt_scale = float(rng.choice([0.7, 1.0, 1.8, 3.2]))
    jitter_add = float(rng.choice([0.0, 0.1, 0.25, 0.55]))
    loss_add = float(rng.choice([0.0, 0.005, 0.02, 0.06]))

    pkt_aug[:, 1] = np.clip(pkt_aug[:, 1] * rtt_scale + rng.normal(0.0, 0.015, size=pkt_aug.shape[0]), 0.0, 10.0)
    net_aug[:, 0] = np.clip(net_aug[:, 0] * rtt_scale, 0.0, 5.0)
    net_aug[:, 1] = np.clip(net_aug[:, 1] + jitter_add, 0.0, 5.0)
    net_aug[:, 2] = np.clip(net_aug[:, 2] + loss_add, 0.0, 1.0)

    approx_rtt = float(np.expm1(net_aug[:, 0].mean() * 8.0))
    approx_jitter = float(np.expm1(net_aug[:, 1].mean() * 6.0))
    approx_loss = float(net_aug[:, 2].mean())
    return pkt_aug.astype(np.float32), net_aug.astype(np.float32), infer_condition(approx_rtt, approx_jitter, approx_loss)

