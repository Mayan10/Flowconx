"""Packet-to-flow aggregation shared by the packet-level sources.

One :class:`FlowSegment` is a bounded conversation segment in the NetFlow
sense: packets between the same endpoint pair over the same transport, cut by
an idle timeout (the conversation went quiet) or an active timeout (it has run
long enough that it should be reported). Both cuts are necessary. Without an
idle timeout a capture's flow table grows without bound; without an active
timeout a thirty-minute streaming session becomes a single row and the dataset
collapses to one observation per capture.

Segments from one capture are correlated by construction, which is exactly why
``capture_id`` exists and why session-disjoint splitting is the honest
protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterator, List, Optional, Tuple


@dataclass
class FlowSegment:
    """A finished conversation segment, ready to become one canonical row."""

    key: Hashable
    protocol: int
    server_ip: str
    server_port: Optional[int]
    first_ts: float
    last_ts: float
    n_packets: int
    n_forward: int
    n_backward: int
    bytes_total: float
    bytes_forward: float
    bytes_backward: float
    lengths: List[float]
    iats: List[float]
    directions: List[int]
    length_sum: float
    length_sq_sum: float
    iat_sum: float
    iat_sq_sum: float
    cut_reason: str

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.last_ts - self.first_ts) * 1000.0)

    def summary(self) -> Dict[str, float]:
        """Flow-level scalars, computed from streaming accumulators."""
        n = max(self.n_packets, 1)
        length_mean = self.length_sum / n
        iat_mean = self.iat_sum / n
        length_var = max(0.0, self.length_sq_sum / n - length_mean * length_mean)
        iat_var = max(0.0, self.iat_sq_sum / n - iat_mean * iat_mean)
        duration_s = max(self.duration_ms / 1000.0, 1e-3)
        return {
            "packet length mean": length_mean,
            "packet length std": math.sqrt(length_var),
            "flow iat mean": iat_mean,
            "flow iat std": math.sqrt(iat_var),
            "flow duration": self.duration_ms,
            "flow bytes/s": self.bytes_total / duration_s,
            "flow packets/s": self.n_packets / duration_s,
        }


@dataclass
class _ActiveFlow:
    protocol: int
    server_ip: str
    max_packets: int
    server_port: Optional[int] = None
    first_ts: float = 0.0
    last_ts: float = 0.0
    n_packets: int = 0
    n_forward: int = 0
    n_backward: int = 0
    bytes_forward: float = 0.0
    bytes_backward: float = 0.0
    length_sum: float = 0.0
    length_sq_sum: float = 0.0
    iat_sum: float = 0.0
    iat_sq_sum: float = 0.0
    lengths: List[float] = field(default_factory=list)
    iats: List[float] = field(default_factory=list)
    directions: List[int] = field(default_factory=list)

    def add(self, timestamp: float, length: float, direction: int, server_port: Optional[int]) -> None:
        if self.n_packets == 0:
            self.first_ts = timestamp
            iat_ms = 0.0
        else:
            iat_ms = max(0.0, (timestamp - self.last_ts) * 1000.0)
        self.last_ts = timestamp
        self.n_packets += 1
        if direction > 0:
            self.n_forward += 1
            self.bytes_forward += length
        else:
            self.n_backward += 1
            self.bytes_backward += length
        self.length_sum += length
        self.length_sq_sum += length * length
        self.iat_sum += iat_ms
        self.iat_sq_sum += iat_ms * iat_ms
        if self.server_port is None and server_port is not None:
            # The transport-layer rows that carry ports are usually the
            # handshake at the start; take the first one we see.
            self.server_port = server_port
        if len(self.lengths) < self.max_packets:
            self.lengths.append(length)
            self.iats.append(iat_ms)
            self.directions.append(direction)

    def finish(self, key: Hashable, cut_reason: str) -> FlowSegment:
        return FlowSegment(
            key=key,
            protocol=self.protocol,
            server_ip=self.server_ip,
            server_port=self.server_port,
            first_ts=self.first_ts,
            last_ts=self.last_ts,
            n_packets=self.n_packets,
            n_forward=self.n_forward,
            n_backward=self.n_backward,
            bytes_total=self.bytes_forward + self.bytes_backward,
            bytes_forward=self.bytes_forward,
            bytes_backward=self.bytes_backward,
            lengths=list(self.lengths),
            iats=list(self.iats),
            directions=list(self.directions),
            length_sum=self.length_sum,
            length_sq_sum=self.length_sq_sum,
            iat_sum=self.iat_sum,
            iat_sq_sum=self.iat_sq_sum,
            cut_reason=cut_reason,
        )


class FlowAccumulator:
    """Streaming packet-to-flow aggregator with idle and active timeouts."""

    def __init__(
        self,
        max_packets: int = 128,
        idle_timeout_s: float = 64.0,
        active_timeout_s: float = 300.0,
        sweep_every: int = 20000,
    ) -> None:
        self.max_packets = max_packets
        self.idle_timeout_s = idle_timeout_s
        self.active_timeout_s = active_timeout_s
        self.sweep_every = sweep_every
        self._active: Dict[Hashable, _ActiveFlow] = {}
        self._packets_seen = 0
        self._now = 0.0

    def add(
        self,
        key: Hashable,
        timestamp: float,
        length: float,
        direction: int,
        protocol: int,
        server_ip: str,
        server_port: Optional[int] = None,
    ) -> Iterator[FlowSegment]:
        """Add one packet; yields any segments the addition closed."""
        self._packets_seen += 1
        # Captures are in timestamp order, but a stray out-of-order row must
        # not rewind the clock and expire everything.
        self._now = max(self._now, timestamp)

        flow = self._active.get(key)
        if flow is not None and timestamp - flow.first_ts >= self.active_timeout_s:
            yield flow.finish(key, "active_timeout")
            del self._active[key]
            flow = None
        if flow is not None and timestamp - flow.last_ts >= self.idle_timeout_s:
            yield flow.finish(key, "idle_timeout")
            del self._active[key]
            flow = None
        if flow is None:
            flow = _ActiveFlow(protocol=protocol, server_ip=server_ip, max_packets=self.max_packets)
            self._active[key] = flow
        flow.add(timestamp, length, direction, server_port)

        if self._packets_seen % self.sweep_every == 0:
            yield from self._sweep()

    def _sweep(self) -> Iterator[FlowSegment]:
        """Expire flows the stream has moved past. Bounds the flow table."""
        expired: List[Tuple[Hashable, str]] = []
        for key, flow in self._active.items():
            if self._now - flow.last_ts >= self.idle_timeout_s:
                expired.append((key, "idle_timeout"))
            elif self._now - flow.first_ts >= self.active_timeout_s:
                expired.append((key, "active_timeout"))
        for key, reason in expired:
            yield self._active.pop(key).finish(key, reason)

    def flush_all(self) -> Iterator[FlowSegment]:
        """Close every remaining flow at end of capture."""
        for key in sorted(self._active, key=str):
            yield self._active[key].finish(key, "end_of_capture")
        self._active.clear()

    @property
    def n_active(self) -> int:
        return len(self._active)
