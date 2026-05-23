#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
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


KNOWN_APPS = {
    "youtube_live": "youtube_live",
    "youtube": "youtube",
    "netflix": "netflix",
    "amazon_prime": "amazon_prime",
    "zoom": "zoom",
    "ms_teams": "ms_teams",
    "google_meet": "google_meet",
    "roblox": "roblox",
    "zepeto": "zepeto",
    "battleground": "battleground",
    "teamfight_tactics": "teamfight_tactics",
    "geforce_now": "geforce_now",
    "kt_gamebox": "kt_gamebox",
    "naver_now": "naver_now",
    "afreecatv": "afreecatv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Kaggle 5G Traffic Datasets into FlowCon-X format.")
    parser.add_argument("--input", required=True, help="Path to data/5G_Traffic_Datasets.")
    parser.add_argument("--output", required=True, help="Output canonical CSV path.")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--max-packets", type=int, default=128, help="Packet sequence length stored per flow row.")
    parser.add_argument("--chunk-rows", type=int, default=250000, help="Rows per pandas chunk.")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--flush-windows", type=int, default=512, help="Number of completed windows to flush at once.")
    return parser.parse_args()


def csv_files(root: Path, limit: Optional[int]) -> List[Path]:
    files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    return files[:limit] if limit else files


def app_from_path(path: Path) -> str:
    text = "/".join(part.lower() for part in path.parts)
    for token, app in KNOWN_APPS.items():
        if token in text:
            return app
    return normalize_label(path.parent.name)


def protocol_to_number(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"udp", "quic", "dns", "rtp", "17"} or text.startswith("quic"):
        return 17
    if text in {"tcp", "tls", "ssl", "http", "https", "6"} or text.startswith("tcp") or text.startswith("tls"):
        return 6
    try:
        return int(float(text))
    except ValueError:
        return 0


def is_private_ip(value: object) -> bool:
    text = str(value)
    return (
        text.startswith("10.")
        or text.startswith("192.168.")
        or text.startswith("172.16.")
        or text.startswith("172.17.")
        or text.startswith("172.18.")
        or text.startswith("172.19.")
        or text.startswith("172.2")
        or text.startswith("172.30.")
        or text.startswith("172.31.")
    )


def private_ip_mask(values: pd.Series) -> np.ndarray:
    text = values.astype(str)
    return (
        text.str.startswith("10.")
        | text.str.startswith("192.168.")
        | text.str.startswith("172.16.")
        | text.str.startswith("172.17.")
        | text.str.startswith("172.18.")
        | text.str.startswith("172.19.")
        | text.str.startswith("172.2")
        | text.str.startswith("172.30.")
        | text.str.startswith("172.31.")
    ).to_numpy(dtype=bool)


def direction_array(source: pd.Series, destination: pd.Series) -> np.ndarray:
    src_private = private_ip_mask(source)
    dst_private = private_ip_mask(destination)
    return np.where(src_private & ~dst_private, 1, np.where(dst_private & ~src_private, -1, 1)).astype(np.int64)


def timestamp_array(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() >= max(1, len(values) // 2):
        return numeric.to_numpy(dtype=np.float64)
    cleaned = values.astype(str).str.replace(r"[^\x00-\x7F]+", "", regex=True).str.strip()
    parsed = pd.to_datetime(cleaned, errors="coerce")
    return parsed.astype("int64").to_numpy(dtype=np.float64) / 1_000_000_000.0


def series_text(values: Iterable[float], precision: int) -> str:
    return ";".join(f"{float(value):.{precision}f}" for value in values)


@dataclass
class WindowStats:
    app: str
    service: str
    max_packets: int
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    prev_ts: Optional[float] = None
    total_packets: int = 0
    fwd_packets: int = 0
    bwd_packets: int = 0
    length_sum: float = 0.0
    length_sq_sum: float = 0.0
    iat_sum: float = 0.0
    iat_sq_sum: float = 0.0
    packet_lengths: List[float] = field(default_factory=list)
    iat_values: List[float] = field(default_factory=list)
    directions: List[int] = field(default_factory=list)
    protocols: Counter = field(default_factory=Counter)

    def add_batch(self, timestamps: np.ndarray, lengths: np.ndarray, packet_directions: np.ndarray, protocols: np.ndarray) -> None:
        if len(timestamps) == 0:
            return
        if self.first_ts is None:
            self.first_ts = float(timestamps[0])
            prepend = timestamps[0]
        else:
            prepend = float(self.prev_ts)
        iats = np.maximum(0.0, np.diff(timestamps, prepend=prepend) * 1000.0)
        self.last_ts = float(timestamps[-1])
        self.prev_ts = float(timestamps[-1])
        self.total_packets += int(len(lengths))
        self.fwd_packets += int(np.sum(packet_directions > 0))
        self.bwd_packets += int(np.sum(packet_directions <= 0))
        self.length_sum += float(np.sum(lengths))
        self.length_sq_sum += float(np.sum(lengths * lengths))
        self.iat_sum += float(np.sum(iats))
        self.iat_sq_sum += float(np.sum(iats * iats))
        protocol_values, protocol_counts = np.unique(protocols, return_counts=True)
        for value, count in zip(protocol_values.tolist(), protocol_counts.tolist()):
            self.protocols[int(value)] += int(count)

        remaining = self.max_packets - len(self.packet_lengths)
        if remaining > 0:
            take = min(remaining, len(lengths))
            self.packet_lengths.extend(float(value) for value in lengths[:take])
            self.iat_values.extend(float(value) for value in iats[:take])
            self.directions.extend(int(value) for value in packet_directions[:take])

    def add_packet(self, timestamp: float, length: float, packet_direction: int, protocol: int) -> None:
        if self.first_ts is None:
            self.first_ts = timestamp
            iat_ms = 0.0
        else:
            iat_ms = max(0.0, (timestamp - float(self.prev_ts)) * 1000.0)
        self.last_ts = timestamp
        self.prev_ts = timestamp
        self.total_packets += 1
        if packet_direction > 0:
            self.fwd_packets += 1
        else:
            self.bwd_packets += 1
        self.length_sum += length
        self.length_sq_sum += length * length
        self.iat_sum += iat_ms
        self.iat_sq_sum += iat_ms * iat_ms
        self.protocols[protocol] += 1
        if len(self.packet_lengths) < self.max_packets:
            self.packet_lengths.append(length)
            self.iat_values.append(iat_ms)
            self.directions.append(packet_direction)

    def to_row(self) -> Dict[str, object]:
        total = max(self.total_packets, 1)
        duration_ms = max(0.0, (float(self.last_ts or 0.0) - float(self.first_ts or 0.0)) * 1000.0)
        length_mean = self.length_sum / total
        iat_mean = self.iat_sum / total
        length_var = max(0.0, self.length_sq_sum / total - length_mean * length_mean)
        iat_var = max(0.0, self.iat_sq_sum / total - iat_mean * iat_mean)
        length_std = math.sqrt(length_var)
        iat_std = math.sqrt(iat_var)
        condition = infer_condition(iat_mean, iat_std, 0.0)
        protocol = self.protocols.most_common(1)[0][0] if self.protocols else 0
        return {
            "app": self.app,
            "service": self.service,
            "condition": condition,
            "packet_lengths": series_text(self.packet_lengths or [0.0], 2),
            "iat_values": series_text(self.iat_values or [0.0], 4),
            "directions": ";".join(str(value) for value in (self.directions or [1])),
            "rtt_ms": iat_mean,
            "jitter_ms": iat_std,
            "loss_rate": 0.0,
            "total packets": self.total_packets,
            "total fwd packets": self.fwd_packets,
            "total backward packets": self.bwd_packets,
            "packet length mean": length_mean,
            "packet length std": length_std,
            "flow iat mean": iat_mean,
            "flow iat std": iat_std,
            "flow duration": duration_ms,
            "flow bytes/s": self.length_sum / max(duration_ms / 1000.0, 0.001),
            "flow packets/s": self.total_packets / max(duration_ms / 1000.0, 0.001),
            "protocol": protocol,
        }


def flush_completed(windows: Dict[int, WindowStats], current_window: int, writer: csv.DictWriter, keep_recent: int) -> int:
    flushed = 0
    cutoff = current_window - keep_recent
    for key in sorted(list(windows.keys())):
        if key >= cutoff:
            continue
        writer.writerow(windows.pop(key).to_row())
        flushed += 1
    return flushed


def process_file(path: Path, writer: csv.DictWriter, args: argparse.Namespace) -> int:
    app = app_from_path(path)
    service = infer_service(app)
    windows: Dict[int, WindowStats] = {}
    rows_written = 0
    start_ts: Optional[float] = None
    last_window = 0
    keep_recent = 2

    try:
        reader = pd.read_csv(
            path,
            usecols=["Time", "Source", "Destination", "Protocol", "Length"],
            chunksize=args.chunk_rows,
            encoding="utf-8-sig",
            encoding_errors="replace",
            on_bad_lines="skip",
        )
    except ValueError as exc:
        if "Usecols do not match columns" not in str(exc):
            raise
        reader = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["Time", "Source", "Destination", "Length"],
            usecols=[0, 1, 2, 3],
            chunksize=args.chunk_rows,
            encoding="utf-8-sig",
            encoding_errors="replace",
            on_bad_lines="skip",
        )
    for chunk in reader:
        timestamps = timestamp_array(chunk["Time"])
        lengths = pd.to_numeric(chunk["Length"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(timestamps) & np.isfinite(lengths) & (timestamps > 0)
        if not np.any(valid):
            continue
        timestamps = timestamps[valid]
        lengths = np.clip(lengths[valid], 0.0, 65535.0)
        packet_directions = direction_array(chunk["Source"], chunk["Destination"])[valid]
        if "Protocol" in chunk:
            protocols = chunk["Protocol"].map(protocol_to_number).to_numpy(dtype=np.int64)[valid]
        else:
            protocols = np.zeros(len(chunk), dtype=np.int64)[valid]
        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        lengths = lengths[order]
        packet_directions = packet_directions[order]
        protocols = protocols[order]
        if start_ts is None:
            start_ts = float(timestamps[0])
        window_ids = ((timestamps - start_ts) // max(args.window_seconds, 0.1)).astype(np.int64)
        last_window = max(last_window, int(window_ids[-1]))

        for window_id in np.unique(window_ids):
            window_key = int(window_id)
            stats = windows.get(window_key)
            if stats is None:
                stats = WindowStats(app=app, service=service, max_packets=args.max_packets)
                windows[window_key] = stats
            mask = window_ids == window_key
            stats.add_batch(timestamps[mask], lengths[mask], packet_directions[mask], protocols[mask])
        if len(windows) >= args.flush_windows:
            rows_written += flush_completed(windows, last_window, writer, keep_recent)

    for key in sorted(windows):
        writer.writerow(windows[key].to_row())
        rows_written += 1
    return rows_written


def main() -> None:
    args = parse_args()
    root = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = csv_files(root, args.limit_files)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")

    total = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        for idx, path in enumerate(files, start=1):
            rows = process_file(path, writer, args)
            total += rows
            print(f"[{idx}/{len(files)}] {path}: wrote {rows} flows")

    frame = pd.read_csv(output, usecols=["app", "service", "condition"])
    print(f"Saved {total} flows to {output}")
    print(frame.value_counts().head(30).to_string())


if __name__ == "__main__":
    main()
