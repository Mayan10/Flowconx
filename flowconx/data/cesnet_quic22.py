"""CESNET-QUIC22 loader: 28 daily flow files, streamed from the zip.

Each row is already a bidirectional flow with a ``PPI`` field holding the
first ~30 packets as (IAT, direction, size) triples, plus the fields that make
strict evaluation possible: ``TIME_FIRST`` (four weeks of real timeline),
``SRC_IP``/``DST_IP``, ``DST_ASN``, ports, and ``QUIC_SNI``.

Those last four are retained as *provenance*, never as model inputs. They
exist so the audit can measure how much of the task a destination port or an
SNI string explains on its own -- the single most important comparison in the
whole evaluation, because if SNI alone solves it then nothing else in the
paper matters.

Sampling
--------

21 GB of doubly-compressed CSV is a several-hour read. Rows are therefore
taken by a **seeded per-day reservoir**, stratified by service, so that:

* every one of the 28 days contributes, which keeps the temporal split real;
* the sample is a deterministic function of ``(seed, rows_per_day_per_class)``
  and the scan order, and is recorded in the manifest;
* class balance is imposed at read time rather than by discarding later.
"""

from __future__ import annotations

import ast
import csv
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .archives import ArchiveMember, list_members, text_stream
from .canonical import canonical_row

ORIGIN = "cesnet_quic22"

# CESNET ships its own service taxonomy in the CATEGORY column. It is used
# verbatim: the previous pipeline replaced it with a hand-written mapping whose
# substring fallback mislabelled apps (AUDIT.md 3, L9).
CATEGORY_COLUMN = "CATEGORY"
APP_COLUMN = "APP"

USE_COLUMNS = (
    "SRC_IP",
    "DST_IP",
    "DST_ASN",
    "SRC_PORT",
    "DST_PORT",
    "PROTOCOL",
    "QUIC_SNI",
    "TIME_FIRST",
    "DURATION",
    "BYTES",
    "BYTES_REV",
    "PACKETS",
    "PACKETS_REV",
    "APP",
    "CATEGORY",
    "PPI",
    "PPI_LEN",
)


@dataclass
class CesnetConfig:
    archive: str = "data/raw/cesnet-quic22.zip"
    max_packets: int = 30
    min_packets: int = 4
    # Reservoir size per (day, service). 28 days x ~20 services x this many is
    # the upper bound on dataset size.
    rows_per_day_per_class: int = 400
    # Hard cap on rows scanned per day file, so a full pass is bounded. Rows
    # are read in file order, which is timestamp order within the day.
    max_rows_per_day: Optional[int] = None
    seed: int = 42
    min_class_rows: int = 500

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class DayStats:
    capture_id: str
    week: str
    date: str
    rows_read: int = 0
    rows_kept: int = 0
    rows_unparsable: int = 0
    rows_too_short: int = 0
    truncated: bool = False
    classes: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "capture_id": self.capture_id,
            "week": self.week,
            "date": self.date,
            "rows_read": self.rows_read,
            "rows_kept": self.rows_kept,
            "rows_unparsable": self.rows_unparsable,
            "rows_too_short": self.rows_too_short,
            "truncated": self.truncated,
            "classes": dict(sorted(self.classes.items())),
        }


def parse_ppi(value: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``[[iats],[directions],[sizes]]`` -> three aligned arrays."""
    if not value or value[0] != "[":
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError, MemoryError):
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 3:
        return None
    try:
        iats = np.asarray(parsed[0], dtype=np.float64)
        directions = np.asarray(parsed[1], dtype=np.int64)
        lengths = np.asarray(parsed[2], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    count = min(len(iats), len(directions), len(lengths))
    if count == 0:
        return None
    return iats[:count], directions[:count], lengths[:count]


def parse_timestamp(value: str) -> float:
    """CESNET writes ISO-8601 without a zone; it is UTC per the dataset README."""
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return 0.0


def numeric(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def normalize_category(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def iter_day_files(config: CesnetConfig) -> List[Tuple[ArchiveMember, str, str]]:
    """``(member, week, date)`` for each daily flow file, in date order."""
    out: List[Tuple[ArchiveMember, str, str]] = []
    for member in list_members(config.archive, suffixes=(".csv.gz",)):
        parts = member.parts
        if len(parts) < 4 or not member.stem.startswith("flows-"):
            continue
        week = parts[1]
        date = member.stem.removeprefix("flows-").removesuffix(".csv.gz")
        out.append((member, week, date))
    out.sort(key=lambda item: item[2])
    return out


class _Reservoir:
    """Seeded reservoir sample of fixed capacity, holding raw CSV rows."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.items: List[List[str]] = []
        self.seen = 0

    def offer(self, item: List[str]) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        # Classic Algorithm R: replace with probability capacity/seen.
        j = int(self.rng.integers(0, self.seen))
        if j < self.capacity:
            self.items[j] = item


def extract_day(
    member: ArchiveMember,
    week: str,
    date: str,
    config: CesnetConfig,
) -> Tuple[List[Dict[str, object]], DayStats]:
    """Stream one day file and return a stratified reservoir of canonical rows.

    ``PPI`` is a nested Python literal and ``ast.literal_eval`` on it costs
    about twenty times as much as reading the rest of the row -- 95% of the
    runtime in a naive implementation. The reservoir therefore stores raw
    field tuples and parses only the rows that survive selection, which turns
    a two-hour pass over the archive into a few minutes and lets every row of
    every day be *considered* rather than only a biased prefix.
    """
    capture_id = f"{week}/{date}"
    stats = DayStats(capture_id=capture_id, week=week, date=date)
    # Seeded per day so days are independent and the sample is a pure function
    # of (config.seed, date).
    rng = np.random.default_rng(_day_seed(config.seed, date))
    reservoirs: Dict[str, _Reservoir] = {}

    with text_stream(member) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], stats
        try:
            index_of = {name: header.index(name) for name in USE_COLUMNS}
        except ValueError as exc:
            raise ValueError(f"{member.name} is missing an expected column: {exc}") from exc
        category_at = index_of[CATEGORY_COLUMN]
        ppi_len_at = index_of["PPI_LEN"]
        width = len(header)

        for row in reader:
            if config.max_rows_per_day and stats.rows_read >= config.max_rows_per_day:
                stats.truncated = True
                break
            stats.rows_read += 1
            if len(row) != width:
                stats.rows_unparsable += 1
                continue
            service = normalize_category(row[category_at])
            if not service or service == "unknown":
                continue
            # PPI_LEN lets the length filter run without touching PPI itself.
            try:
                if int(float(row[ppi_len_at])) < config.min_packets:
                    stats.rows_too_short += 1
                    continue
            except ValueError:
                stats.rows_unparsable += 1
                continue
            bucket = reservoirs.get(service)
            if bucket is None:
                bucket = _Reservoir(config.rows_per_day_per_class, rng)
                reservoirs[service] = bucket
            bucket.offer(row)

    rows: List[Dict[str, object]] = []
    for service in sorted(reservoirs):
        for raw in reservoirs[service].items:
            record: Dict[str, str] = {name: raw[index_of[name]] for name in USE_COLUMNS}
            parsed = parse_ppi(record["PPI"])
            if parsed is None:
                stats.rows_unparsable += 1
                continue
            canonical = _to_canonical(record, parsed, capture_id, service, len(rows), config)
            if canonical is None:
                stats.rows_unparsable += 1
                continue
            rows.append(canonical)
            stats.classes[service] = stats.classes.get(service, 0) + 1
    stats.rows_kept = len(rows)
    return rows, stats


def _day_seed(seed: int, date: str) -> int:
    """Stable per-day seed. ``hash()`` is salted per process, so use SHA256."""
    import hashlib

    digest = hashlib.sha256(f"{seed}:{date}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _to_canonical(
    record: Dict[str, str],
    parsed: Tuple[np.ndarray, np.ndarray, np.ndarray],
    capture_id: str,
    service: str,
    index: int,
    config: CesnetConfig,
) -> Optional[Dict[str, object]]:
    iats, directions, lengths = parsed
    n = min(len(iats), config.max_packets)
    iats, directions, lengths = iats[:n], directions[:n], lengths[:n]

    packets_fwd = int(numeric(record.get("PACKETS")))
    packets_rev = int(numeric(record.get("PACKETS_REV")))
    total_packets = max(packets_fwd + packets_rev, int(n), 1)
    bytes_fwd = numeric(record.get("BYTES"))
    bytes_rev = numeric(record.get("BYTES_REV"))
    total_bytes = max(bytes_fwd + bytes_rev, float(lengths.sum()))
    duration_ms = max(numeric(record.get("DURATION")) * 1000.0, float(iats.sum()))
    duration_s = max(duration_ms / 1000.0, 1e-3)

    dst_asn = record.get("DST_ASN")
    try:
        asn = int(float(dst_asn)) if dst_asn else None
    except (TypeError, ValueError):
        asn = None
    port = record.get("DST_PORT")
    try:
        server_port = int(float(port)) if port else None
    except (TypeError, ValueError):
        server_port = None

    return canonical_row(
        origin=ORIGIN,
        capture_id=capture_id,
        index=index,
        app=(record.get(APP_COLUMN) or "unknown").strip().lower(),
        service=service,
        lengths=lengths.tolist(),
        iats=iats.tolist(),
        directions=directions.tolist(),
        flow_start_ts=parse_timestamp(record.get("TIME_FIRST") or ""),
        n_packets=total_packets,
        n_forward=packets_fwd,
        n_backward=packets_rev,
        packet_length_mean=total_bytes / total_packets,
        packet_length_std=float(lengths.std()),
        flow_iat_mean=float(iats.mean()),
        flow_iat_std=float(iats.std()),
        flow_duration_ms=duration_ms,
        flow_bytes_per_s=total_bytes / duration_s,
        flow_packets_per_s=total_packets / duration_s,
        protocol=int(numeric(record.get("PROTOCOL"), 17.0)),
        # No measured RTT, jitter or loss exists in this dataset. Writing the
        # IAT statistics into these columns is what made the nuisance label a
        # function of the model's inputs before; they stay zero and the
        # condition stays unknown.
        condition="unknown",
        client_ip=(record.get("SRC_IP") or "").strip(),
        server_ip=(record.get("DST_IP") or "").strip(),
        server_port=server_port,
        sni=(record.get("QUIC_SNI") or "").strip().lower(),
        dst_asn=asn,
    )


def iter_all_days(config: CesnetConfig) -> Iterator[Tuple[List[Dict[str, object]], DayStats]]:
    for member, week, date in iter_day_files(config):
        yield extract_day(member, week, date, config)


def filter_rare_services(rows: Sequence[Dict[str, object]], min_rows: int) -> Tuple[List[Dict[str, object]], List[str]]:
    """Drop services with too few rows to support a held-out estimate."""
    counts: Dict[str, int] = {}
    for row in rows:
        counts[str(row["service"])] = counts.get(str(row["service"]), 0) + 1
    dropped = sorted(name for name, count in counts.items() if count < min_rows)
    kept = [row for row in rows if str(row["service"]) not in dropped]
    return kept, dropped
