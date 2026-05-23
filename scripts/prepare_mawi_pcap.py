#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


Endpoint = Tuple[bytes, int]
FlowKey = Tuple[int, Endpoint, Endpoint]
WindowKey = Tuple[int, FlowKey]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a MAWI pcap trace into compact FlowCon-X background flows.")
    parser.add_argument("--input", required=True, help="Path to a MAWI .pcap file.")
    parser.add_argument("--output", required=True, help="Output canonical CSV path.")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--max-packets", type=int, default=128)
    parser.add_argument("--max-flows", type=int, default=20000)
    parser.add_argument("--max-packets-to-scan", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def series_text(values: Iterable[float], precision: int) -> str:
    return ";".join(f"{float(value):.{precision}f}" for value in values)


@dataclass
class FlowStats:
    protocol: int
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

    def add(self, timestamp: float, length: int, direction: int) -> None:
        if self.first_ts is None:
            self.first_ts = timestamp
            iat_ms = 0.0
        else:
            iat_ms = max(0.0, (timestamp - float(self.prev_ts)) * 1000.0)
        self.last_ts = timestamp
        self.prev_ts = timestamp
        self.total_packets += 1
        if direction > 0:
            self.fwd_packets += 1
        else:
            self.bwd_packets += 1
        length_value = float(max(0, min(length, 65535)))
        self.length_sum += length_value
        self.length_sq_sum += length_value * length_value
        self.iat_sum += iat_ms
        self.iat_sq_sum += iat_ms * iat_ms
        if len(self.packet_lengths) < self.max_packets:
            self.packet_lengths.append(length_value)
            self.iat_values.append(iat_ms)
            self.directions.append(direction)

    def to_row(self) -> Dict[str, object]:
        total = max(self.total_packets, 1)
        duration_ms = max(0.0, (float(self.last_ts or 0.0) - float(self.first_ts or 0.0)) * 1000.0)
        length_mean = self.length_sum / total
        iat_mean = self.iat_sum / total
        length_std = math.sqrt(max(0.0, self.length_sq_sum / total - length_mean * length_mean))
        iat_std = math.sqrt(max(0.0, self.iat_sq_sum / total - iat_mean * iat_mean))
        return {
            "app": "mawi_background",
            "service": "unknown",
            "condition": infer_condition(iat_mean, iat_std, 0.0),
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
            "protocol": self.protocol,
        }


def pcap_endian_and_scale(header: bytes) -> Tuple[str, float]:
    magic = header[:4]
    if magic == b"\xd4\xc3\xb2\xa1":
        return "<", 1_000_000.0
    if magic == b"\xa1\xb2\xc3\xd4":
        return ">", 1_000_000.0
    if magic == b"\x4d\x3c\xb2\xa1":
        return "<", 1_000_000_000.0
    if magic == b"\xa1\xb2\x3c\x4d":
        return ">", 1_000_000_000.0
    raise ValueError("Unsupported pcap magic number.")


def parse_packet(packet: bytes, orig_len: int) -> Optional[Tuple[int, Endpoint, Endpoint, int]]:
    if len(packet) < 14:
        return None
    offset = 14
    eth_type = int.from_bytes(packet[12:14], "big")
    while eth_type in {0x8100, 0x88A8} and len(packet) >= offset + 4:
        eth_type = int.from_bytes(packet[offset + 2 : offset + 4], "big")
        offset += 4

    if eth_type == 0x0800:
        if len(packet) < offset + 20:
            return None
        ihl = (packet[offset] & 0x0F) * 4
        proto = packet[offset + 9]
        src = packet[offset + 12 : offset + 16]
        dst = packet[offset + 16 : offset + 20]
        transport = offset + ihl
    elif eth_type == 0x86DD:
        if len(packet) < offset + 40:
            return None
        proto = packet[offset + 6]
        src = packet[offset + 8 : offset + 24]
        dst = packet[offset + 24 : offset + 40]
        transport = offset + 40
        while proto in {0, 43, 60} and len(packet) >= transport + 8:
            next_proto = packet[transport]
            ext_len = (packet[transport + 1] + 1) * 8
            transport += ext_len
            proto = next_proto
        if proto == 44:
            return None
    else:
        return None

    if proto not in {6, 17} or len(packet) < transport + 4:
        return None
    src_port = int.from_bytes(packet[transport : transport + 2], "big")
    dst_port = int.from_bytes(packet[transport + 2 : transport + 4], "big")
    src_ep = (src, src_port)
    dst_ep = (dst, dst_port)
    if src_ep <= dst_ep:
        return proto, src_ep, dst_ep, 1
    return proto, dst_ep, src_ep, -1


def add_to_reservoir(heap: List[Tuple[float, int, Dict[str, object]]], row: Dict[str, object], rng: np.random.Generator, max_flows: int, counter: int) -> int:
    key = float(rng.random())
    item = (-key, counter, row)
    if len(heap) < max_flows:
        heapq.heappush(heap, item)
    elif key < -heap[0][0]:
        heapq.heapreplace(heap, item)
    return counter + 1


def flush_old_windows(active: Dict[WindowKey, FlowStats], current_window: int, keep_recent: int, heap: List[Tuple[float, int, Dict[str, object]]], rng: np.random.Generator, max_flows: int, counter: int) -> int:
    cutoff = current_window - keep_recent
    for key in list(active.keys()):
        window_id, _ = key
        if window_id >= cutoff:
            continue
        counter = add_to_reservoir(heap, active.pop(key).to_row(), rng, max_flows, counter)
    return counter


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    active: Dict[WindowKey, FlowStats] = {}
    heap: List[Tuple[float, int, Dict[str, object]]] = []
    counter = 0
    packets = 0
    accepted = 0
    first_ts: Optional[float] = None
    current_window = 0
    keep_recent = 2

    with Path(args.input).open("rb") as handle:
        header = handle.read(24)
        endian, scale = pcap_endian_and_scale(header)
        rec = struct.Struct(endian + "IIII")
        while True:
            rec_header = handle.read(16)
            if len(rec_header) < 16:
                break
            ts_sec, ts_frac, incl_len, orig_len = rec.unpack(rec_header)
            packet = handle.read(incl_len)
            if len(packet) < incl_len:
                break
            packets += 1
            if args.max_packets_to_scan and packets > args.max_packets_to_scan:
                break
            timestamp = ts_sec + ts_frac / scale
            if first_ts is None:
                first_ts = timestamp
            window_id = int((timestamp - first_ts) // max(args.window_seconds, 0.1))
            current_window = max(current_window, window_id)
            parsed = parse_packet(packet, orig_len)
            if parsed is None:
                continue
            proto, ep_a, ep_b, direction = parsed
            key = (window_id, (proto, ep_a, ep_b))
            stats = active.get(key)
            if stats is None:
                stats = FlowStats(protocol=proto, max_packets=args.max_packets)
                active[key] = stats
            stats.add(timestamp, orig_len, direction)
            accepted += 1
            if packets % 1_000_000 == 0:
                counter = flush_old_windows(active, current_window, keep_recent, heap, rng, args.max_flows, counter)
                print(f"packets={packets} accepted={accepted} reservoir={len(heap)} active={len(active)}")

    for key in list(active.keys()):
        counter = add_to_reservoir(heap, active.pop(key).to_row(), rng, args.max_flows, counter)

    rows = [item[2] for item in heap]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    out.to_csv(output, index=False)
    print(f"Scanned {packets} packets, accepted {accepted} TCP/UDP packets")
    print(f"Saved {len(out)} MAWI background flows to {output}")
    print(out["condition"].value_counts().to_string())


if __name__ == "__main__":
    main()
