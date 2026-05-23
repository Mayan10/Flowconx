#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowconx.config import infer_service, normalize_label
from flowconx.features import infer_condition


CANONICAL_COLUMNS = [
    "app",
    "service",
    "condition",
    "packet_lengths",
    "iat_values",
    "directions",
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
]


USE_COLUMNS = [
    "APP",
    "CATEGORY",
    "DURATION",
    "BYTES",
    "BYTES_REV",
    "PACKETS",
    "PACKETS_REV",
    "PROTOCOL",
    "PPI",
]


CATEGORY_TO_SERVICE = {
    "streaming_media": "streaming",
    "music": "streaming",
    "games": "gaming",
    "videoconferencing": "conferencing",
    "file_sharing": "bulk_transfer",
    "search": "browsing",
    "social": "browsing",
    "blogs_news": "browsing",
    "e_commerce": "browsing",
    "mail": "browsing",
    "advertising": "browsing",
    "authentication_services": "browsing",
    "analytics_telemetry": "browsing",
    "information_systems": "browsing",
    "instant_messaging": "browsing",
    "other_services_and_apis": "browsing",
    "antivirus": "browsing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a balanced full-month CESNET-QUIC22 training CSV.")
    parser.add_argument("--input", required=True, help="Path to the cesnet-quic22 folder.")
    parser.add_argument("--output", required=True, help="Output canonical CSV path.")
    parser.add_argument("--rows-per-service", type=int, default=15000)
    parser.add_argument("--chunk-rows", type=int, default=200000)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def service_from_category(value: object) -> str:
    token = normalize_label(value)
    return CATEGORY_TO_SERVICE.get(token, infer_service(token))


def input_files(root: Path) -> List[Path]:
    return sorted(root.rglob("flows-*.csv.gz"))


def series_text(values: Iterable[float], precision: int) -> str:
    return ";".join(f"{float(value):.{precision}f}" for value in values)


def parse_ppi(value: object) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 3:
        return None
    try:
        iats = np.asarray(parsed[0], dtype=np.float64)
        dirs = np.asarray(parsed[1], dtype=np.int64)
        lengths = np.asarray(parsed[2], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    count = min(len(iats), len(dirs), len(lengths))
    if count == 0:
        return None
    return iats[:count], dirs[:count], lengths[:count]


def numeric(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def row_to_canonical(row: pd.Series) -> Optional[Dict[str, object]]:
    parsed = parse_ppi(row.get("PPI"))
    if parsed is None:
        return None
    iats, dirs, lengths = parsed
    app = normalize_label(row.get("APP"))
    service = service_from_category(row.get("CATEGORY"))
    duration_ms = max(numeric(row.get("DURATION")) * 1000.0, float(np.sum(iats)))
    bytes_fwd = numeric(row.get("BYTES"))
    bytes_rev = numeric(row.get("BYTES_REV"))
    packets_fwd = int(numeric(row.get("PACKETS")))
    packets_rev = int(numeric(row.get("PACKETS_REV")))
    total_packets = max(packets_fwd + packets_rev, len(lengths), 1)
    total_bytes = max(bytes_fwd + bytes_rev, float(np.sum(lengths)))
    rtt = float(np.mean(iats))
    jitter = float(np.std(iats))
    length_mean = total_bytes / total_packets
    length_std = float(np.std(lengths))
    condition = infer_condition(rtt, jitter, 0.0)
    protocol = int(numeric(row.get("PROTOCOL"), 17.0))
    return {
        "app": app,
        "service": service,
        "condition": condition,
        "packet_lengths": series_text(np.clip(lengths, 0, 65535), 2),
        "iat_values": series_text(np.clip(iats, 0, 1_000_000), 4),
        "directions": ";".join(str(int(value)) for value in dirs),
        "rtt_ms": rtt,
        "jitter_ms": jitter,
        "loss_rate": 0.0,
        "total packets": total_packets,
        "total fwd packets": packets_fwd,
        "total backward packets": packets_rev,
        "packet length mean": length_mean,
        "packet length std": length_std,
        "flow iat mean": rtt,
        "flow iat std": jitter,
        "flow duration": duration_ms,
        "flow bytes/s": total_bytes / max(duration_ms / 1000.0, 0.001),
        "flow packets/s": total_packets / max(duration_ms / 1000.0, 0.001),
        "protocol": protocol,
    }


def prune_reservoir(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    return frame.nsmallest(limit, "_sample_key").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    root = Path(args.input)
    files = input_files(root)
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        raise FileNotFoundError(f"No flows-*.csv.gz files found under {root}")

    rng = np.random.default_rng(args.seed)
    reservoirs: Dict[str, List[pd.DataFrame]] = {}
    scanned = 0

    for file_idx, path in enumerate(files, start=1):
        file_rows = 0
        for chunk in pd.read_csv(path, usecols=USE_COLUMNS, chunksize=args.chunk_rows, compression="infer"):
            file_rows += len(chunk)
            scanned += len(chunk)
            services = chunk["CATEGORY"].map(service_from_category)
            random_keys = rng.random(len(chunk))
            chunk = chunk.assign(service_for_sampling=services.to_numpy(), _sample_key=random_keys)
            for service, group in chunk.groupby("service_for_sampling"):
                if service == "unknown":
                    continue
                local = group.nsmallest(min(len(group), args.rows_per_service), "_sample_key")
                reservoirs.setdefault(service, []).append(local)
                total = sum(len(part) for part in reservoirs[service])
                if total > args.rows_per_service * 3:
                    reservoirs[service] = [prune_reservoir(pd.concat(reservoirs[service], ignore_index=True), args.rows_per_service)]
        print(f"[{file_idx}/{len(files)}] {path}: scanned {file_rows} rows")

    selected_parts = []
    for service, parts in sorted(reservoirs.items()):
        selected = prune_reservoir(pd.concat(parts, ignore_index=True), args.rows_per_service)
        selected_parts.append(selected)
        print(f"selected {len(selected)} rows for {service}")

    selected_rows = pd.concat(selected_parts, ignore_index=True)
    selected_rows = selected_rows.sort_values("_sample_key").reset_index(drop=True)
    canonical_rows = []
    for _, row in selected_rows.iterrows():
        canonical = row_to_canonical(row)
        if canonical is not None:
            canonical_rows.append(canonical)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(canonical_rows, columns=CANONICAL_COLUMNS)
    out.to_csv(output, index=False)
    print(f"Scanned {scanned} CESNET rows")
    print(f"Saved {len(out)} flows to {output}")
    print(out["service"].value_counts().to_string())


if __name__ == "__main__":
    main()
