"""Turning a source record into one canonical Flow row.

The `condition` column deserves a note. The previous pipeline defined it as
``infer_condition(iat_mean, iat_std, 0.0)`` and simultaneously wrote those two
statistics as model inputs, so the declared nuisance variable was a threshold
on the model's own features and the audit reconstructed it with agreement
1.000 (AUDIT.md 3, L5). That is not a network condition; it is a relabelling
of the flow's timing.

Here `condition` is derived from properties of the *capture context* rather
than the flow's own timing where the source supports it, and where it does
not, it is written as ``unknown`` and the loader records that it is
unavailable. Nothing downstream may treat an `unknown` condition as a measured
one.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from .schema import CANONICAL_COLUMNS, make_flow_id


def series_text(values: Sequence[float], precision: int) -> str:
    return ";".join(f"{float(v):.{precision}f}" for v in values)


def int_series_text(values: Sequence[int]) -> str:
    return ";".join(str(int(v)) for v in values)


def canonical_row(
    *,
    origin: str,
    capture_id: str,
    index: int,
    app: str,
    service: str,
    lengths: Sequence[float],
    iats: Sequence[float],
    directions: Sequence[int],
    flow_start_ts: float,
    n_packets: int,
    n_forward: int,
    n_backward: int,
    packet_length_mean: float,
    packet_length_std: float,
    flow_iat_mean: float,
    flow_iat_std: float,
    flow_duration_ms: float,
    flow_bytes_per_s: float,
    flow_packets_per_s: float,
    protocol: int,
    condition: str = "unknown",
    vantage: str = "",
    client_ip: str = "",
    server_ip: str = "",
    server_port: Optional[int] = None,
    sni: str = "",
    dst_asn: Optional[int] = None,
    rtt_ms: float = 0.0,
    jitter_ms: float = 0.0,
    loss_rate: float = 0.0,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "flow_id": make_flow_id(origin, capture_id, index),
        "origin": origin,
        "capture_id": capture_id,
        "vantage": vantage,
        "flow_start_ts": round(float(flow_start_ts), 6),
        "client_ip": client_ip,
        "server_ip": server_ip,
        "server_port": "" if server_port is None else int(server_port),
        "sni": sni,
        "dst_asn": "" if dst_asn is None else int(dst_asn),
        "app": app,
        "service": service,
        "condition": condition,
        "packet_lengths": series_text(lengths or [0.0], 2),
        "iat_values": series_text(iats or [0.0], 4),
        "directions": int_series_text(directions or [1]),
        "rtt_ms": round(float(rtt_ms), 6),
        "jitter_ms": round(float(jitter_ms), 6),
        "loss_rate": round(float(loss_rate), 6),
        "total packets": int(n_packets),
        "total fwd packets": int(n_forward),
        "total backward packets": int(n_backward),
        "packet length mean": round(float(packet_length_mean), 6),
        "packet length std": round(float(packet_length_std), 6),
        "flow iat mean": round(float(flow_iat_mean), 6),
        "flow iat std": round(float(flow_iat_std), 6),
        "flow duration": round(float(flow_duration_ms), 6),
        "flow bytes/s": round(float(flow_bytes_per_s), 6),
        "flow packets/s": round(float(flow_packets_per_s), 6),
        "protocol": int(protocol),
    }
    missing = [c for c in CANONICAL_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"canonical_row is missing columns: {missing}")
    return row


def row_from_segment(
    segment,
    *,
    origin: str,
    capture_id: str,
    index: int,
    app: str,
    service: str,
    condition: str = "unknown",
    vantage: str = "",
    client_ip: str = "",
    sni: str = "",
    dst_asn: Optional[int] = None,
) -> Dict[str, object]:
    """Canonical row from a :class:`~flowconx.data.flow_builder.FlowSegment`."""
    stats = segment.summary()
    return canonical_row(
        origin=origin,
        capture_id=capture_id,
        index=index,
        app=app,
        service=service,
        lengths=segment.lengths,
        iats=segment.iats,
        directions=segment.directions,
        flow_start_ts=segment.first_ts,
        n_packets=segment.n_packets,
        n_forward=segment.n_forward,
        n_backward=segment.n_backward,
        packet_length_mean=stats["packet length mean"],
        packet_length_std=stats["packet length std"],
        flow_iat_mean=stats["flow iat mean"],
        flow_iat_std=stats["flow iat std"],
        flow_duration_ms=stats["flow duration"],
        flow_bytes_per_s=stats["flow bytes/s"],
        flow_packets_per_s=stats["flow packets/s"],
        protocol=segment.protocol,
        condition=condition,
        vantage=vantage,
        client_ip=client_ip,
        server_ip=segment.server_ip,
        server_port=segment.server_port,
        sni=sni,
        dst_asn=dst_asn,
        # `rtt_ms` and `jitter_ms` are *not* aliases of the IAT statistics any
        # more. No RTT is measurable from these captures, so they are zero and
        # the loader reports the condition as unknown.
        rtt_ms=0.0,
        jitter_ms=0.0,
        loss_rate=0.0,
    )
