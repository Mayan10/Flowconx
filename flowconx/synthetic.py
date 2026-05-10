"""Synthetic hackathon dataset generator for end-to-end testing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SERVICE_APPS = {
    "streaming": ["youtube", "netflix", "prime_video"],
    "gaming": ["valorant", "league_of_legends", "roblox"],
    "conferencing": ["zoom", "teams", "google_meet"],
    "bulk_transfer": ["file_download", "ftp"],
    "browsing": ["web_browsing"],
    "xr_interactive": ["cloud_vr", "vr_video", "ar_session"],
}


SERVICE_PROFILES = {
    "streaming": {"length": 1050, "iat": 18, "fwd": 0.82, "rtt": 45, "jitter": 8, "loss": 0.003},
    "gaming": {"length": 190, "iat": 12, "fwd": 0.55, "rtt": 35, "jitter": 5, "loss": 0.002},
    "conferencing": {"length": 520, "iat": 15, "fwd": 0.56, "rtt": 55, "jitter": 12, "loss": 0.006},
    "bulk_transfer": {"length": 1240, "iat": 7, "fwd": 0.9, "rtt": 70, "jitter": 10, "loss": 0.004},
    "browsing": {"length": 430, "iat": 35, "fwd": 0.68, "rtt": 50, "jitter": 14, "loss": 0.003},
    "xr_interactive": {"length": 760, "iat": 9, "fwd": 0.7, "rtt": 28, "jitter": 4, "loss": 0.001},
}


CONDITION_PROFILES = {
    "good": {"rtt_scale": 0.7, "jitter_scale": 0.7, "loss_add": 0.0},
    "moderate": {"rtt_scale": 1.1, "jitter_scale": 1.2, "loss_add": 0.004},
    "degraded": {"rtt_scale": 2.0, "jitter_scale": 2.3, "loss_add": 0.015},
    "bad": {"rtt_scale": 3.5, "jitter_scale": 4.0, "loss_add": 0.05},
}


def _series_to_text(values: np.ndarray, precision: int = 3) -> str:
    return ";".join(f"{float(value):.{precision}f}" for value in values)


def _make_flow(service: str, app: str, condition: str, rng: np.random.Generator) -> Dict[str, object]:
    profile = SERVICE_PROFILES[service]
    condition_profile = CONDITION_PROFILES[condition]
    packet_count = int(rng.integers(42, 128))
    base_length = profile["length"] * rng.normal(1.0, 0.08)
    base_iat = profile["iat"] * rng.normal(1.0, 0.12)

    if service == "xr_interactive":
        frame = np.sin(np.linspace(0, 9 * np.pi, packet_count))
        lengths = base_length + 260 * np.maximum(frame, 0) + rng.normal(0, 80, packet_count)
        iats = base_iat + 4 * np.maximum(-frame, 0) + rng.normal(0, 2, packet_count)
    elif service == "streaming":
        frame = np.sin(np.linspace(0, 5 * np.pi, packet_count))
        lengths = base_length + 180 * np.maximum(frame, 0) + rng.normal(0, 120, packet_count)
        iats = base_iat + rng.normal(0, 4, packet_count)
    elif service == "gaming":
        lengths = base_length + rng.normal(0, 60, packet_count)
        iats = base_iat + rng.normal(0, 2, packet_count)
    else:
        lengths = base_length + rng.normal(0, 140, packet_count)
        iats = base_iat + rng.normal(0, max(base_iat * 0.3, 1), packet_count)

    lengths = np.clip(lengths, 48, 1450)
    iats = np.clip(iats * condition_profile["rtt_scale"], 0.2, 500)
    directions = rng.choice([1, -1], packet_count, p=[profile["fwd"], 1.0 - profile["fwd"]])

    rtt = profile["rtt"] * condition_profile["rtt_scale"] * rng.normal(1.0, 0.08)
    jitter = profile["jitter"] * condition_profile["jitter_scale"] * rng.normal(1.0, 0.12)
    loss = min(max(profile["loss"] + condition_profile["loss_add"] + rng.normal(0, 0.001), 0.0), 0.2)
    duration = float(np.sum(iats))
    bytes_total = float(np.sum(lengths))
    fwd_packets = int(np.sum(directions > 0))
    bwd_packets = packet_count - fwd_packets
    flow_bytes_s = bytes_total / max(duration / 1000.0, 0.001)
    flow_packets_s = packet_count / max(duration / 1000.0, 0.001)

    return {
        "app": app,
        "service": service,
        "condition": condition,
        "packet_lengths": _series_to_text(lengths, 2),
        "iat_values": _series_to_text(iats, 3),
        "directions": ";".join(str(int(value)) for value in directions),
        "total packets": packet_count,
        "total fwd packets": fwd_packets,
        "total backward packets": bwd_packets,
        "packet length mean": float(np.mean(lengths)),
        "packet length std": float(np.std(lengths)),
        "flow iat mean": float(np.mean(iats)),
        "flow iat std": float(np.std(iats)),
        "flow duration": duration,
        "flow bytes/s": flow_bytes_s,
        "flow packets/s": flow_packets_s,
        "rtt_ms": float(rtt),
        "jitter_ms": float(jitter),
        "loss_rate": float(loss),
        "throughput_mbps": flow_bytes_s * 8.0 / 1_000_000.0,
        "uplink_ratio": float(fwd_packets / max(packet_count, 1)),
        "protocol": 17 if service in {"streaming", "gaming", "xr_interactive", "conferencing"} else 6,
        "syn flag count": 1 if service in {"bulk_transfer", "browsing"} else 0,
        "ack flag count": int(packet_count * 0.6),
        "rst flag count": 0,
    }


def generate_synthetic_dataframe(
    flows_per_app: int = 80,
    include_xr: bool = True,
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for service, apps in SERVICE_APPS.items():
        if service == "xr_interactive" and not include_xr:
            continue
        for app in apps:
            for _ in range(flows_per_app):
                condition = str(rng.choice(list(CONDITION_PROFILES.keys()), p=[0.45, 0.3, 0.18, 0.07]))
                rows.append(_make_flow(service, app, condition, rng))
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def write_synthetic_csv(
    output_path: str | Path,
    flows_per_app: int = 80,
    include_xr: bool = True,
    seed: int = 123,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_dataframe(flows_per_app=flows_per_app, include_xr=include_xr, seed=seed)
    df.to_csv(path, index=False)
    return path

