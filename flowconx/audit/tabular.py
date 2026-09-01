"""Trivial and classical feature families for the shortcut audit.

Everything here reads the canonical flow CSV directly. It deliberately does
*not* go through :mod:`flowconx.features`, because the point of the audit is
to ask what a reviewer could get with almost no modelling at all -- if a
decision tree on one column gets close to the full network, the dataset is
the story.

Each family is declared with the exact columns it is allowed to touch, so
that ``tests/test_leakage.py`` can assert that identifier-like fields never
silently become inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Fields that must never be handed to a model as an input feature: they
# identify the row or its provenance rather than describing its behaviour.
FORBIDDEN_INPUT_COLUMNS = frozenset(
    {"flow_id", "app", "service", "capture_id", "origin", "server_ip", "client_ip", "sni", "server_name"}
)

# Sequence columns in the canonical schema, stored as ';'-separated strings.
SEQUENCE_COLUMNS = ("packet_lengths", "iat_values", "directions")

# Flow-level scalar columns in the canonical schema.
SCALAR_COLUMNS = (
    "rtt_ms",
    "jitter_ms",
    "loss_rate",
    "total packets",
    "total fwd packets",
    "total backward packets",
    "packet length mean",
    "packet length std",
    "flow iat mean",
    "flow iat std",
    "flow duration",
    "flow bytes/s",
    "flow packets/s",
    "protocol",
)


def parse_series(text: object, dtype=np.float64) -> np.ndarray:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.zeros(0, dtype=dtype)
    raw = str(text).replace("|", ";").replace(",", ";")
    parts = [part for part in raw.split(";") if part.strip()]
    if not parts:
        return np.zeros(0, dtype=dtype)
    try:
        return np.asarray([float(part) for part in parts], dtype=dtype)
    except ValueError:
        return np.zeros(0, dtype=dtype)


@dataclass
class ParsedFlow:
    lengths: np.ndarray
    iats: np.ndarray
    directions: np.ndarray

    @property
    def signed(self) -> np.ndarray:
        n = min(len(self.lengths), len(self.directions))
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        return self.lengths[:n] * np.sign(self.directions[:n])


def parse_flows(df: pd.DataFrame) -> List[ParsedFlow]:
    lengths = df["packet_lengths"].map(parse_series).tolist() if "packet_lengths" in df else []
    iats = df["iat_values"].map(parse_series).tolist() if "iat_values" in df else []
    dirs = df["directions"].map(parse_series).tolist() if "directions" in df else []
    n = len(df)
    empty = np.zeros(0)
    return [
        ParsedFlow(
            lengths=lengths[i] if i < len(lengths) else empty,
            iats=iats[i] if i < len(iats) else empty,
            directions=dirs[i] if i < len(dirs) else empty,
        )
        for i in range(n)
    ]


def _safe(values: np.ndarray, fn: Callable[[np.ndarray], float], default: float = 0.0) -> float:
    if values.size == 0:
        return default
    try:
        out = float(fn(values))
    except (ValueError, FloatingPointError):
        return default
    return out if np.isfinite(out) else default


def _moments(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """The AppScanner statistic block: 18 summary statistics of one series."""
    stats = {
        f"{prefix}_min": _safe(values, np.min),
        f"{prefix}_max": _safe(values, np.max),
        f"{prefix}_mean": _safe(values, np.mean),
        f"{prefix}_mad": _safe(values, lambda v: float(np.mean(np.abs(v - np.mean(v))))),
        f"{prefix}_std": _safe(values, np.std),
        f"{prefix}_var": _safe(values, np.var),
        f"{prefix}_skew": _safe(values, lambda v: float(((v - v.mean()) ** 3).mean() / max(v.std() ** 3, 1e-9))),
        f"{prefix}_kurt": _safe(values, lambda v: float(((v - v.mean()) ** 4).mean() / max(v.std() ** 4, 1e-9))),
        f"{prefix}_sum": _safe(values, np.sum),
        f"{prefix}_n": float(values.size),
    }
    for q in (10, 20, 30, 40, 50, 60, 70, 80):
        stats[f"{prefix}_p{q}"] = _safe(values, lambda v, q=q: float(np.percentile(v, q)))
    return stats


# --------------------------------------------------------------------------
# Feature families
# --------------------------------------------------------------------------


def family_protocol_only(df: pd.DataFrame, flows: Sequence[ParsedFlow]) -> Tuple[np.ndarray, List[str]]:
    """The 'destination port only' probe, adapted to our schema.

    The canonical CSV retains no port number, so the closest single-column
    identifier available is the IP protocol number. This is a *weaker* probe
    than port-only would be; the AUDIT records that as a limitation.
    """
    _ = flows
    values = pd.to_numeric(df.get("protocol", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(-1.0)
    return values.to_numpy(dtype=np.float64).reshape(-1, 1), ["protocol"]


def family_five_stat(df: pd.DataFrame, flows: Sequence[ParsedFlow]) -> Tuple[np.ndarray, List[str]]:
    """Packet count, byte count, duration, mean IAT, mean packet size."""
    packets = pd.to_numeric(df.get("total packets", 0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    mean_len = pd.to_numeric(df.get("packet length mean", 0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    duration = pd.to_numeric(df.get("flow duration", 0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    mean_iat = pd.to_numeric(df.get("flow iat mean", 0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    total_bytes = packets * mean_len
    matrix = np.column_stack([packets, total_bytes, duration, mean_iat, mean_len])
    return matrix, ["n_packets", "n_bytes", "duration_ms", "mean_iat_ms", "mean_pkt_len"]


def _first_n(flows: Sequence[ParsedFlow], n: int) -> np.ndarray:
    out = np.zeros((len(flows), n), dtype=np.float64)
    for i, flow in enumerate(flows):
        signed = flow.signed[:n]
        out[i, : len(signed)] = signed
    return out


def family_first10_sizes(df: pd.DataFrame, flows: Sequence[ParsedFlow]) -> Tuple[np.ndarray, List[str]]:
    _ = df
    return _first_n(flows, 10), [f"signed_size_{i}" for i in range(10)]


def family_first20_sizes(df: pd.DataFrame, flows: Sequence[ParsedFlow]) -> Tuple[np.ndarray, List[str]]:
    _ = df
    return _first_n(flows, 20), [f"signed_size_{i}" for i in range(20)]


_SIZE_BINS = np.array([0, 64, 128, 192, 256, 384, 512, 768, 1024, 1280, 1400, 1500, 65536], dtype=np.float64)


def family_size_histogram(df: pd.DataFrame, flows: Sequence[ParsedFlow]) -> Tuple[np.ndarray, List[str]]:
    _ = df
    n_bins = len(_SIZE_BINS) - 1
    out = np.zeros((len(flows), n_bins * 2), dtype=np.float64)
    for i, flow in enumerate(flows):
        n = min(len(flow.lengths), len(flow.directions))
        if n == 0:
            continue
        lengths = flow.lengths[:n]
        dirs = np.sign(flow.directions[:n])
        for offset, mask in ((0, dirs > 0), (n_bins, dirs <= 0)):
            selected = lengths[mask]
            if selected.size:
                counts, _ = np.histogram(selected, bins=_SIZE_BINS)
                out[i, offset : offset + n_bins] = counts / max(n, 1)
    names = [f"fwd_bin_{i}" for i in range(n_bins)] + [f"bwd_bin_{i}" for i in range(n_bins)]
    return out, names


_CUMUL_POINTS = 100


def family_cumul(df: pd.DataFrame, flows: Sequence[ParsedFlow]) -> Tuple[np.ndarray, List[str]]:
    """CUMUL (Panchenko et al., NDSS 2016).

    Four count/volume features followed by ``_CUMUL_POINTS`` linearly
    interpolated samples of the cumulative signed-size curve.
    """
    _ = df
    out = np.zeros((len(flows), 4 + _CUMUL_POINTS), dtype=np.float64)
    grid = np.linspace(0.0, 1.0, _CUMUL_POINTS)
    for i, flow in enumerate(flows):
        signed = flow.signed
        if signed.size == 0:
            continue
        incoming = signed[signed < 0]
        outgoing = signed[signed > 0]
        out[i, 0] = incoming.size
        out[i, 1] = outgoing.size
        out[i, 2] = float(np.abs(incoming).sum())
        out[i, 3] = float(outgoing.sum())
        cumulative = np.cumsum(signed)
        positions = np.linspace(0.0, 1.0, cumulative.size)
        out[i, 4:] = np.interp(grid, positions, cumulative) if cumulative.size > 1 else cumulative[0]
    names = ["n_in", "n_out", "bytes_in", "bytes_out"] + [f"cumul_{i}" for i in range(_CUMUL_POINTS)]
    return out, names


def family_appscanner(df: pd.DataFrame, flows: Sequence[ParsedFlow]) -> Tuple[np.ndarray, List[str]]:
    """AppScanner (Taylor et al., EuroS&P 2016): statistics over size series.

    The original computes 18 statistics over the incoming, outgoing and
    combined packet-size series (54 features). We add the same block over the
    inter-arrival series because our schema retains it.
    """
    _ = df
    rows: List[Dict[str, float]] = []
    for flow in flows:
        n = min(len(flow.lengths), len(flow.directions))
        lengths = flow.lengths[:n]
        dirs = np.sign(flow.directions[:n]) if n else np.zeros(0)
        block: Dict[str, float] = {}
        block.update(_moments(lengths[dirs > 0] if n else np.zeros(0), "out_len"))
        block.update(_moments(lengths[dirs <= 0] if n else np.zeros(0), "in_len"))
        block.update(_moments(lengths, "all_len"))
        block.update(_moments(flow.iats, "all_iat"))
        rows.append(block)
    if not rows:
        return np.zeros((0, 0)), []
    names = list(rows[0].keys())
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
    return matrix, names


def family_kfp(df: pd.DataFrame, flows: Sequence[ParsedFlow]) -> Tuple[np.ndarray, List[str]]:
    """k-fingerprinting (Hayes & Danezis, USENIX Security 2016), core block.

    The published feature set is ~175 features built from packet counts,
    ordering, concentration in fixed-size windows and burst structure. We
    implement the count, ordering, concentration and burst groups; the
    HTML-marker and per-site timing groups do not transfer to our schema and
    are documented as omitted in AUDIT.md.
    """
    _ = df
    n_conc = 20
    n_order = 20
    width = 12 + n_conc + n_order
    out = np.zeros((len(flows), width), dtype=np.float64)
    for i, flow in enumerate(flows):
        n = min(len(flow.lengths), len(flow.directions))
        if n == 0:
            continue
        dirs = np.sign(flow.directions[:n])
        iats = flow.iats[: len(flow.iats)]
        total = float(n)
        n_in = float(np.sum(dirs <= 0))
        n_out = float(np.sum(dirs > 0))
        out[i, 0] = total
        out[i, 1] = n_in
        out[i, 2] = n_out
        out[i, 3] = n_in / max(total, 1.0)
        out[i, 4] = n_out / max(total, 1.0)
        out[i, 5] = _safe(iats, np.sum)
        out[i, 6] = _safe(iats, np.mean)
        out[i, 7] = _safe(iats, np.std)
        out[i, 8] = _safe(iats, np.max)
        # Burst structure: maximal runs of same-direction packets.
        changes = np.flatnonzero(np.diff(dirs)) + 1
        runs = np.diff(np.concatenate([[0], changes, [n]]))
        out[i, 9] = float(runs.size)
        out[i, 10] = _safe(runs.astype(np.float64), np.mean)
        out[i, 11] = _safe(runs.astype(np.float64), np.max)
        # Concentration of outgoing packets in equal-count windows.
        window = max(int(np.ceil(n / n_conc)), 1)
        for w in range(n_conc):
            chunk = dirs[w * window : (w + 1) * window]
            out[i, 12 + w] = float(np.sum(chunk > 0)) if chunk.size else 0.0
        # Cumulative outgoing count sampled at n_order positions.
        cumulative_out = np.cumsum(dirs > 0).astype(np.float64)
        positions = np.linspace(0.0, 1.0, cumulative_out.size)
        grid = np.linspace(0.0, 1.0, n_order)
        out[i, 12 + n_conc :] = (
            np.interp(grid, positions, cumulative_out) if cumulative_out.size > 1 else cumulative_out[0]
        )
    names = (
        [
            "n_total",
            "n_in",
            "n_out",
            "frac_in",
            "frac_out",
            "iat_sum",
            "iat_mean",
            "iat_std",
            "iat_max",
            "n_bursts",
            "burst_mean",
            "burst_max",
        ]
        + [f"conc_{i}" for i in range(n_conc)]
        + [f"order_{i}" for i in range(n_order)]
    )
    return out, names


def family_flow_meta(df: pd.DataFrame, flows: Sequence[ParsedFlow]) -> Tuple[np.ndarray, List[str]]:
    """Every flow-level scalar the aggregator wrote, and nothing sequential.

    This is the 'what does the summary row alone give you' probe. It is the
    single most informative audit row, because a NetFlow-style collector in a
    real deployment produces exactly this and nothing more.
    """
    _ = flows
    columns = [c for c in SCALAR_COLUMNS if c in df.columns]
    matrix = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) for c in columns]
    )
    return matrix, list(columns)


def family_condition_only(df: pd.DataFrame, flows: Sequence[ParsedFlow]) -> Tuple[np.ndarray, List[str]]:
    """The declared nuisance variable, used alone as a predictor of the label.

    If the nuisance condition predicts the task label well above chance, the
    adversarial condition-removal head is being asked to delete task-relevant
    information, and the invariance claim needs restating.
    """
    _ = flows
    if "condition" not in df.columns:
        return np.zeros((len(df), 0)), []
    codes = pd.Categorical(df["condition"].astype(str)).codes.astype(np.float64)
    return codes.reshape(-1, 1), ["condition"]


FEATURE_FAMILIES: Dict[str, Callable[[pd.DataFrame, Sequence[ParsedFlow]], Tuple[np.ndarray, List[str]]]] = {
    "protocol_only": family_protocol_only,
    "condition_only": family_condition_only,
    "five_stat": family_five_stat,
    "first10_sizes": family_first10_sizes,
    "first20_sizes": family_first20_sizes,
    "size_histogram": family_size_histogram,
    "flow_meta": family_flow_meta,
    "cumul": family_cumul,
    "appscanner": family_appscanner,
    "kfp": family_kfp,
}

# Human-readable provenance for each family, used verbatim in the audit report.
FAMILY_CITATIONS: Dict[str, str] = {
    "protocol_only": "Single-column identifier probe (port-only analogue; our schema retains no port).",
    "condition_only": "The declared nuisance variable used alone as a predictor.",
    "five_stat": "Five flow-summary statistics, the classic NetFlow-style baseline.",
    "first10_sizes": "First 10 signed packet sizes.",
    "first20_sizes": "First 20 signed packet sizes.",
    "size_histogram": "Directional packet-size histogram.",
    "flow_meta": "All flow-level scalars present in the canonical CSV.",
    "cumul": "CUMUL, Panchenko et al., NDSS 2016.",
    "appscanner": "AppScanner, Taylor et al., IEEE EuroS&P 2016.",
    "kfp": "k-fingerprinting, Hayes and Danezis, USENIX Security 2016 (count/order/concentration/burst groups).",
}


def build_features(
    df: pd.DataFrame,
    family: str,
    flows: Optional[Sequence[ParsedFlow]] = None,
) -> Tuple[np.ndarray, List[str]]:
    if family not in FEATURE_FAMILIES:
        raise ValueError(f"Unknown feature family {family!r}. Known: {sorted(FEATURE_FAMILIES)}")
    parsed = flows if flows is not None else parse_flows(df)
    matrix, names = FEATURE_FAMILIES[family](df, parsed)
    leaked = FORBIDDEN_INPUT_COLUMNS.intersection(names)
    if leaked:
        raise ValueError(f"Feature family {family!r} exposes identifier columns as inputs: {sorted(leaked)}")
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), names
