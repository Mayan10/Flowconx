"""5G Traffic Datasets loader: packet CSVs to conversation-level flows.

The original preparer bucketed packets into fixed 10-second wall-clock windows
over an entire capture file and aggregated every conversation in the window
into one row (AUDIT.md 1.2). That is not a flow, and it made the unit of
observation differ from CESNET's. This loader replaces it.

What a row is here
------------------

One row is one **conversation segment**: all packets exchanged between the
capture's client host and one server host over one transport, bounded by an
idle timeout and an active timeout in the NetFlow tradition. A long streaming
session therefore yields several rows, all carrying the same ``capture_id``,
which is what session-disjoint splitting groups on.

Labels come from the dataset's own directory taxonomy and nothing else:
``5G_Traffic_Datasets/<Service>/<App>/<App>_<n>.csv`` gives ``service`` and
``app`` directly. No substring matching, no hand-written alias table -- both
were sources of mislabelling in the previous pipeline (AUDIT.md 3, L9).

Known deviations from an ideal extraction are recorded in ``PROTOCOL.md``.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .archives import ArchiveMember, list_members, text_stream
from .flow_builder import FlowAccumulator, FlowSegment

ORIGIN = "5g_traffic"

# The dataset's own top-level directories are the service label space. Six
# classes, defined by the people who collected the traffic.
SERVICE_FROM_DIRECTORY: Dict[str, str] = {
    "Game_Streaming": "game_streaming",
    "Online_Game": "online_game",
    "Live_Streaming": "live_streaming",
    "Stored_Streaming": "stored_streaming",
    "Metaverse": "metaverse",
    "Video_Conferencing": "video_conferencing",
}

# Wireshark names the topmost dissected layer, so the Protocol column holds
# things like "TLSv1.2" or "GQUIC" rather than the transport. A census over
# the whole archive (see PROTOCOL.md) finds 40 distinct layer names, most of
# them a long tail of exotic dissectors -- RakNet, DB-LSP, Pathport, MPEG TS,
# R-GOOSE -- that fire on UDP payloads. Roblox captures are 100% RakNet, which
# is Roblox's UDP game protocol, so dropping the tail would silently discard
# an entire application.
#
# The rule is therefore: an explicit list of TCP-based layers, and everything
# else that is not explicitly non-IP is treated as UDP.
TCP_LAYERS = frozenset(
    {
        "TCP",
        "TLSV1",
        "TLSV1.1",
        "TLSV1.2",
        "TLSV1.3",
        "SSL",
        "SSLV2",
        "SSLV3",
        "HTTP",
        "HTTP/JSON",
        "HTTP/XML",
        "HTTP2",
        "THRIFT",
        "ELASTICSEARCH",
        "FMTP",
        "H1",
    }
)

# Layers that are not a client-server transport conversation at all.
SKIP_LAYERS = frozenset({"ICMP", "ICMPV6", "ARP", "IGMP", "LLDP", "STP", "STP/RSTP"})

# "58632  >  443 [SYN] ..." on transport rows. Higher-layer rows carry
# application text instead, so ports are recovered opportunistically.
PORT_PATTERN = re.compile(r"^\s*(\d{1,5})\s*(?:>|&gt;|→)\s*(\d{1,5})\b")

_PRIVATE_PREFIXES = ("10.", "192.168.", "127.", "169.254.", "192.0.0.")


def is_private(address: str) -> bool:
    if address.startswith(_PRIVATE_PREFIXES):
        return True
    if address.startswith("172."):
        try:
            second = int(address.split(".", 2)[1])
        except (IndexError, ValueError):
            return False
        return 16 <= second <= 31
    return address.startswith("fe80:") or address.startswith("fd") or address == "::1"


def transport_of(layer: str) -> Optional[int]:
    """IP protocol number for a Wireshark layer name, or None to skip the row."""
    token = layer.strip().upper()
    if token in SKIP_LAYERS:
        return None
    if token in TCP_LAYERS:
        return 6
    return 17


class TimestampParser:
    """Parse ``YYYY-MM-DD HH:MM:SS.ffffff`` fast enough for 370M rows.

    ``datetime.strptime`` costs roughly 10 us per call, which is minutes of
    wall clock per capture. The date changes at most a couple of times per
    file, so it is parsed once and cached, and the time of day is arithmetic
    on the digits.
    """

    def __init__(self) -> None:
        self._date_cache: Dict[str, float] = {}

    def __call__(self, text: str) -> Optional[float]:
        if len(text) < 19 or text[4] != "-" or text[10] not in " T":
            return None
        day = text[:10]
        base = self._date_cache.get(day)
        if base is None:
            from datetime import datetime, timezone

            try:
                base = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                return None
            self._date_cache[day] = base
        try:
            hours = int(text[11:13])
            minutes = int(text[14:16])
            seconds = float(text[17:])
        except ValueError:
            return None
        return base + hours * 3600 + minutes * 60 + seconds


# ---------------------------------------------------------------------------
# Export formats
# ---------------------------------------------------------------------------
#
# The archive holds two Wireshark export variants. 67 captures are the
# documented CSV with a quoted header; the 8 Amazon_Prime captures are a
# headerless, tab-separated export with only four fields and a Korean-locale
# timestamp. Ignoring the second variant would drop the whole
# `stored_streaming`/Amazon Prime application, so both are supported and the
# manifest records which each capture used.

CSV_HEADER_PREFIX = '"No."'

# "Jun 21, 2022 00:03:06.471027000 대한민국 표준시" -- month name, then a
# nanosecond fraction, then a localised timezone name. Captured in Korea
# Standard Time, which is a fixed UTC+9 with no daylight saving.
KST_OFFSET_SECONDS = 9 * 3600
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_locale_timestamp(text: str) -> Optional[float]:
    """Parse the TSV variant's ``Mon DD, YYYY HH:MM:SS.fffffffff TZ`` stamp."""
    parts = text.strip().split()
    if len(parts) < 4:
        return None
    month = _MONTHS.get(parts[0][:3].lower())
    if month is None:
        return None
    try:
        from datetime import datetime, timezone

        day = int(parts[1].rstrip(","))
        year = int(parts[2])
        clock = parts[3]
        hours, minutes, rest = clock.split(":")
        seconds = float(rest[:15])  # tolerate nanosecond precision
        base = datetime(year, month, day, tzinfo=timezone.utc).timestamp()
    except (ValueError, IndexError):
        return None
    return base + int(hours) * 3600 + int(minutes) * 60 + seconds - KST_OFFSET_SECONDS


@dataclass(frozen=True)
class RowFormat:
    """How to pull (timestamp, source, destination, length, layer, info) out."""

    name: str
    delimiter: str
    has_header: bool
    time_at: int
    source_at: int
    destination_at: int
    length_at: int
    layer_at: Optional[int]
    info_at: Optional[int]
    width: int


CSV_FORMAT = RowFormat(
    name="wireshark_csv",
    delimiter=",",
    has_header=True,
    time_at=1,
    source_at=2,
    destination_at=3,
    length_at=5,
    layer_at=4,
    info_at=6,
    width=6,
)

TSV_FORMAT = RowFormat(
    name="wireshark_tsv_no_header",
    delimiter="\t",
    has_header=False,
    time_at=0,
    source_at=1,
    destination_at=2,
    length_at=3,
    layer_at=None,
    info_at=None,
    width=4,
)


def detect_format(member: ArchiveMember) -> RowFormat:
    """Sniff the first line of a capture to pick its export variant."""
    with text_stream(member) as handle:
        first = handle.readline()
    if first.startswith(CSV_HEADER_PREFIX) or first.startswith("No.,"):
        return CSV_FORMAT
    if "\t" in first:
        return TSV_FORMAT
    # Unrecognised: assume the documented CSV so the failure is visible in the
    # skip counters rather than silently producing an empty capture.
    return CSV_FORMAT


@dataclass
class FiveGConfig:
    archive: str = "data/raw/Archive.zip"
    max_packets: int = 128
    min_packets: int = 8
    idle_timeout_s: float = 64.0
    active_timeout_s: float = 300.0
    # No row budget by default. Parsing runs at roughly 400k rows/s, so the
    # whole 44 GB archive is a ~15 minute pass, and reading it all avoids the
    # bias a prefix cap introduces: the first N rows of a media capture are
    # its startup phase, which is not representative of the session.
    max_rows_per_capture: Optional[int] = None
    # Flows per capture are a *reservoir*, not a prefix, so the sample spans
    # the whole capture. Seeded per capture, so the selection is a pure
    # function of (seed, capture_id).
    max_flows_per_capture: Optional[int] = 6_000
    seed: int = 42
    capture_glob: str = "5G_Traffic_Datasets/"

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class CaptureStats:
    capture_id: str
    app: str
    service: str
    rows_read: int = 0
    rows_skipped: int = 0
    flows_emitted: int = 0
    flows_too_short: int = 0
    truncated: bool = False
    row_format: str = "unknown"
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    layers: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "capture_id": self.capture_id,
            "app": self.app,
            "service": self.service,
            "rows_read": self.rows_read,
            "rows_skipped": self.rows_skipped,
            "flows_emitted": self.flows_emitted,
            "flows_too_short": self.flows_too_short,
            "truncated": self.truncated,
            "row_format": self.row_format,
            "span_seconds": round((self.last_ts or 0.0) - (self.first_ts or 0.0), 3),
            "top_layers": dict(sorted(self.layers.items(), key=lambda kv: -kv[1])[:6]),
        }


def _capture_seed(seed: int, capture_id: str) -> int:
    """Stable per-capture seed; `hash()` is salted per process."""
    import hashlib

    return int(hashlib.sha256(f"{seed}:{capture_id}".encode("utf-8")).hexdigest()[:8], 16)


def label_from_member(member: ArchiveMember) -> Optional[Tuple[str, str]]:
    """``(service, app)`` from the member path, or None if it is not a capture."""
    parts = member.parts
    if len(parts) < 4 or parts[0] != "5G_Traffic_Datasets":
        return None
    service = SERVICE_FROM_DIRECTORY.get(parts[1])
    if service is None:
        return None
    return service, parts[2].lower()


def iter_captures(config: FiveGConfig) -> List[Tuple[ArchiveMember, str, str]]:
    out: List[Tuple[ArchiveMember, str, str]] = []
    for member in list_members(config.archive, suffixes=(".csv",)):
        label = label_from_member(member)
        if label is None:
            continue
        out.append((member, label[0], label[1]))
    return out


def extract_capture(
    member: ArchiveMember,
    service: str,
    app: str,
    config: FiveGConfig,
) -> Tuple[List[FlowSegment], CaptureStats]:
    """Stream one capture and return its conversation segments."""
    capture_id = member.name
    stats = CaptureStats(capture_id=capture_id, app=app, service=service)
    accumulator = FlowAccumulator(
        max_packets=config.max_packets,
        idle_timeout_s=config.idle_timeout_s,
        active_timeout_s=config.active_timeout_s,
    )
    parse_ts = TimestampParser()
    segments: List[FlowSegment] = []

    capacity = config.max_flows_per_capture
    rng = np.random.default_rng(_capture_seed(config.seed, capture_id))
    seen = 0

    def collect(finished: Iterator[FlowSegment]) -> bool:
        """Reservoir-sample finished segments. Always returns False.

        The return value exists so the caller can stop early; with a
        reservoir there is never a reason to, because every segment in the
        capture gets a fair chance of selection.
        """
        nonlocal seen
        for segment in finished:
            if segment.n_packets < config.min_packets:
                stats.flows_too_short += 1
                continue
            stats.flows_emitted += 1
            if capacity is None or len(segments) < capacity:
                segments.append(segment)
            else:
                # Algorithm R: replace a held segment with probability
                # capacity/seen, giving every segment the same inclusion
                # probability regardless of where it fell in the capture.
                position = int(rng.integers(0, seen + 1))
                if position < capacity:
                    segments[position] = segment
            seen += 1
        return False

    row_format = detect_format(member)
    stats.row_format = row_format.name
    parse_time = parse_ts if row_format is CSV_FORMAT else parse_locale_timestamp

    with text_stream(member) as handle:
        reader = csv.reader(handle, delimiter=row_format.delimiter)
        if row_format.has_header:
            try:
                next(reader)
            except StopIteration:
                return segments, stats
        for row in reader:
            if config.max_rows_per_capture and stats.rows_read >= config.max_rows_per_capture:
                stats.truncated = True
                break
            stats.rows_read += 1
            if len(row) <= row_format.width - 1:
                stats.rows_skipped += 1
                continue

            if row_format.layer_at is not None:
                layer = row[row_format.layer_at]
                protocol = transport_of(layer)
            else:
                # The TSV export carries no protocol column. The transport is
                # unrecoverable, so it is written as 0 and, because `protocol`
                # is provenance rather than a model input, nothing downstream
                # can turn that into a shortcut.
                layer = "unknown"
                protocol = 0
            stats.layers[layer] = stats.layers.get(layer, 0) + 1
            if protocol is None:
                stats.rows_skipped += 1
                continue

            timestamp = parse_time(row[row_format.time_at])
            if timestamp is None:
                stats.rows_skipped += 1
                continue
            try:
                length = float(row[row_format.length_at])
            except ValueError:
                stats.rows_skipped += 1
                continue

            source, destination = row[row_format.source_at], row[row_format.destination_at]
            src_private, dst_private = is_private(source), is_private(destination)
            if src_private == dst_private:
                # Both private (local chatter) or both public (no client side
                # to orient on). Neither is a client-server conversation.
                stats.rows_skipped += 1
                continue
            if src_private:
                client, server, direction = source, destination, 1
            else:
                client, server, direction = destination, source, -1

            server_port = None
            if row_format.info_at is not None and len(row) > row_format.info_at:
                match = PORT_PATTERN.match(row[row_format.info_at])
                if match:
                    ports = (int(match.group(1)), int(match.group(2)))
                    server_port = ports[1] if src_private else ports[0]

            if stats.first_ts is None:
                stats.first_ts = timestamp
            stats.last_ts = timestamp

            if collect(
                accumulator.add(
                    key=(protocol, client, server),
                    timestamp=timestamp,
                    length=length,
                    direction=direction,
                    protocol=protocol,
                    server_ip=server,
                    server_port=server_port,
                )
            ):
                break

    collect(accumulator.flush_all())
    return segments, stats
