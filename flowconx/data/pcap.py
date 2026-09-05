"""Minimal pcap and pcapng reader.

Pure Python, no external capture library, because a dependency on libpcap
bindings is a barrier for an artifact evaluator on a machine they do not
administer. It handles what the corpora in this project actually contain:
Ethernet-framed IPv4 and IPv6 carrying TCP or UDP, in classic pcap (both
endiannesses, microsecond and nanosecond) and in pcapng.

pcapng support matters here specifically because the ISCX captures are
documented inconsistently -- some sources say `.pcap`, others `.pcapng` -- and
a loader that assumes one and silently reads zero packets from the other is
worse than one that refuses.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Tuple

Endpoint = Tuple[str, int]

PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", 1_000_000.0),
    b"\xa1\xb2\xc3\xd4": (">", 1_000_000.0),
    b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000.0),
    b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000.0),
}
PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"


class UnsupportedCapture(ValueError):
    """Raised when a file is not a capture this reader understands.

    Deliberately an error rather than an empty iterator: a loader that yields
    nothing looks identical to a capture with no matching traffic, and that
    ambiguity has cost this project a day before.
    """


@dataclass(frozen=True)
class Packet:
    timestamp: float
    length: int
    protocol: int
    source: Endpoint
    destination: Endpoint


def format_ip(raw: bytes) -> str:
    import ipaddress

    try:
        return str(ipaddress.ip_address(raw))
    except ValueError:
        return ""


def parse_ethernet(frame: bytes, captured_length: int) -> Optional[Tuple[int, Endpoint, Endpoint]]:
    """(protocol, source, destination) for an Ethernet frame, or None to skip."""
    if len(frame) < 14:
        return None
    offset = 14
    ether_type = int.from_bytes(frame[12:14], "big")
    while ether_type in {0x8100, 0x88A8} and len(frame) >= offset + 4:
        ether_type = int.from_bytes(frame[offset + 2 : offset + 4], "big")
        offset += 4

    if ether_type == 0x0800:
        if len(frame) < offset + 20:
            return None
        header_length = (frame[offset] & 0x0F) * 4
        protocol = frame[offset + 9]
        source_ip = frame[offset + 12 : offset + 16]
        destination_ip = frame[offset + 16 : offset + 20]
        transport = offset + header_length
    elif ether_type == 0x86DD:
        if len(frame) < offset + 40:
            return None
        protocol = frame[offset + 6]
        source_ip = frame[offset + 8 : offset + 24]
        destination_ip = frame[offset + 24 : offset + 40]
        transport = offset + 40
        # Walk hop-by-hop, routing and destination-options headers.
        while protocol in {0, 43, 60} and len(frame) >= transport + 8:
            next_protocol = frame[transport]
            transport += (frame[transport + 1] + 1) * 8
            protocol = next_protocol
        if protocol == 44:  # fragment header: no reliable ports
            return None
    else:
        return None

    if protocol not in {6, 17} or len(frame) < transport + 4:
        return None
    source_port = int.from_bytes(frame[transport : transport + 2], "big")
    destination_port = int.from_bytes(frame[transport + 2 : transport + 4], "big")
    _ = captured_length
    return protocol, (format_ip(source_ip), source_port), (format_ip(destination_ip), destination_port)


def _iter_pcap(handle: BinaryIO, endian: str, scale: float, max_packets: Optional[int]) -> Iterator[Packet]:
    header_format = endian + "IIII"
    seen = 0
    while True:
        header = handle.read(16)
        if len(header) < 16:
            return
        seconds, fraction, captured, original = struct.unpack(header_format, header)
        frame = handle.read(captured)
        if len(frame) < captured:
            return
        seen += 1
        if max_packets and seen > max_packets:
            return
        parsed = parse_ethernet(frame, captured)
        if parsed is None:
            continue
        protocol, source, destination = parsed
        yield Packet(seconds + fraction / scale, original, protocol, source, destination)


def _iter_pcapng(handle: BinaryIO, max_packets: Optional[int]) -> Iterator[Packet]:
    """Enhanced Packet Blocks only; interface timestamp resolution honoured."""
    handle.seek(0)
    endian = "<"
    resolutions: dict[int, float] = {}
    seen = 0
    while True:
        head = handle.read(8)
        if len(head) < 8:
            return
        block_type = int.from_bytes(head[:4], "little")
        if block_type == 0x0A0D0D0A:  # Section Header Block: byte order magic
            byte_order = handle.read(4)
            endian = "<" if byte_order == b"\x4d\x3c\x2b\x1a" else ">"
            total = int.from_bytes(head[4:8], "little" if endian == "<" else "big")
            handle.seek(-16, 1)
            handle.read(total)
            resolutions = {}
            continue
        total = int.from_bytes(head[4:8], "little" if endian == "<" else "big")
        if total < 12:
            return
        body = handle.read(total - 12)
        handle.read(4)  # trailing length
        if block_type == 0x00000001 and len(body) >= 8:  # Interface Description
            resolutions[len(resolutions)] = 1_000_000.0
        elif block_type == 0x00000006 and len(body) >= 20:  # Enhanced Packet
            interface, high, low, captured, original = struct.unpack(endian + "IIIII", body[:20])
            frame = body[20 : 20 + captured]
            seen += 1
            if max_packets and seen > max_packets:
                return
            parsed = parse_ethernet(frame, captured)
            if parsed is None:
                continue
            protocol, source, destination = parsed
            timestamp = ((high << 32) | low) / resolutions.get(interface, 1_000_000.0)
            yield Packet(timestamp, original, protocol, source, destination)


def read_packets(path: str | Path, max_packets: Optional[int] = None) -> Iterator[Packet]:
    """Yield TCP/UDP packets from a pcap or pcapng file.

    Raises :class:`UnsupportedCapture` rather than yielding nothing, so a
    format mismatch is loud.
    """
    target = Path(path)
    with target.open("rb") as handle:
        magic = handle.read(4)
        if magic in PCAP_MAGICS:
            endian, scale = PCAP_MAGICS[magic]
            handle.read(20)  # rest of the global header
            yield from _iter_pcap(handle, endian, scale, max_packets)
            return
        if magic == PCAPNG_MAGIC:
            yield from _iter_pcapng(handle, max_packets)
            return
        raise UnsupportedCapture(
            f"{target} does not begin with a pcap or pcapng magic number (saw {magic!r}). "
            "If it is gzipped, decompress it first; if it is a Wireshark CSV export, use the "
            "5G Traffic loader instead."
        )


def capture_format(path: str | Path) -> str:
    """'pcap', 'pcapng', or 'unknown' -- for reporting what was actually read."""
    with Path(path).open("rb") as handle:
        magic = handle.read(4)
    if magic in PCAP_MAGICS:
        return "pcap"
    if magic == PCAPNG_MAGIC:
        return "pcapng"
    return "unknown"
