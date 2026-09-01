"""Build the canonical flow CSVs from the raw archives.

    python -m flowconx.data.prepare --source all

Reads directly from ``data/raw/*.zip``; nothing is ever expanded to disk. Each
run writes the dataset CSV plus a manifest recording the archive checksum, the
exact loader configuration, per-capture statistics and the resulting class
counts, so that a reviewer can tell what was read and what was skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .archives import sha256_of_file
from .canonical import row_from_segment
from .cesnet_quic22 import CesnetConfig, extract_day, filter_rare_services, iter_day_files
from .cesnet_quic22 import ORIGIN as CESNET_ORIGIN
from .fiveg_traffic import FiveGConfig, extract_capture, iter_captures
from .fiveg_traffic import ORIGIN as FIVEG_ORIGIN
from .schema import CANONICAL_COLUMNS

SOURCES = ("fiveg", "cesnet")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def environment_block() -> Dict[str, object]:
    versions = {}
    for module in ("numpy", "pandas"):
        try:
            versions[module] = __import__(module).__version__
        except (ImportError, AttributeError):
            versions[module] = "absent"
    return {
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "library_versions": versions,
    }


def write_rows(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def class_counts(rows: Sequence[Dict[str, object]], column: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row[column])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def prepare_fiveg(config: FiveGConfig, output: Path, checksum: bool = True) -> Dict[str, object]:
    captures = iter_captures(config)
    print(f"[5g_traffic] {len(captures)} captures")
    rows: List[Dict[str, object]] = []
    capture_stats: List[Dict[str, object]] = []
    started = time.perf_counter()
    for position, (member, service, app) in enumerate(captures, start=1):
        segments, stats = extract_capture(member, service, app, config)
        for segment in segments:
            rows.append(
                row_from_segment(
                    segment,
                    origin=FIVEG_ORIGIN,
                    capture_id=member.name,
                    index=len(rows),
                    app=app,
                    service=service,
                    # These captures record a single client host behind a 5G
                    # modem; the client address is constant per capture and
                    # carries no label information, but it is retained so the
                    # audit can verify that rather than assume it.
                    client_ip="",
                )
            )
        capture_stats.append(stats.as_dict())
        print(
            f"  [{position:>2}/{len(captures)}] {member.stem:<28} {service:<20} "
            f"rows={stats.rows_read:>9,} segments={stats.flows_emitted:>6,} "
            f"kept={len(segments):>5,} total={len(rows):>7,}"
        )
    elapsed = time.perf_counter() - started
    write_rows(rows, output)
    return {
        "origin": FIVEG_ORIGIN,
        "output": str(output),
        "n_rows": len(rows),
        "n_captures": len(captures),
        "config": config.as_dict(),
        "archive_sha256": sha256_of_file(config.archive) if checksum else "skipped",
        "service_counts": class_counts(rows, "service"),
        "app_counts": class_counts(rows, "app"),
        "capture_stats": capture_stats,
        "wall_clock_seconds": round(elapsed, 1),
        "environment": environment_block(),
    }


def prepare_cesnet(config: CesnetConfig, output: Path, checksum: bool = True) -> Dict[str, object]:
    days = iter_day_files(config)
    print(f"[cesnet_quic22] {len(days)} daily flow files, {days[0][2]} to {days[-1][2]}")
    rows: List[Dict[str, object]] = []
    day_stats: List[Dict[str, object]] = []
    started = time.perf_counter()
    for position, (member, week, date) in enumerate(days, start=1):
        day_rows, stats = extract_day(member, week, date, config)
        # flow_id is built from (origin, capture_id, index); re-index into the
        # global row space so identifiers stay unique across days.
        for offset, row in enumerate(day_rows):
            row["flow_id"] = _reindex(row["flow_id"], CESNET_ORIGIN, str(row["capture_id"]), len(rows) + offset)
        rows.extend(day_rows)
        day_stats.append(stats.as_dict())
        print(
            f"  [{position:>2}/{len(days)}] {week} {date} "
            f"read={stats.rows_read:>9,} kept={stats.rows_kept:>6,} total={len(rows):>7,}"
        )
    kept, dropped = filter_rare_services(rows, config.min_class_rows)
    if dropped:
        print(f"  dropped services below {config.min_class_rows} rows: {dropped}")
    elapsed = time.perf_counter() - started
    write_rows(kept, output)
    return {
        "origin": CESNET_ORIGIN,
        "output": str(output),
        "n_rows": len(kept),
        "n_rows_before_rare_filter": len(rows),
        "dropped_services": dropped,
        "n_days": len(days),
        "date_range": [days[0][2], days[-1][2]],
        "config": config.as_dict(),
        "archive_sha256": sha256_of_file(config.archive) if checksum else "skipped",
        "service_counts": class_counts(kept, "service"),
        "app_counts": class_counts(kept, "app"),
        "day_stats": day_stats,
        "wall_clock_seconds": round(elapsed, 1),
        "environment": environment_block(),
    }


def _reindex(_old: object, origin: str, capture_id: str, index: int) -> str:
    from .schema import make_flow_id

    return make_flow_id(origin, capture_id, index)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical flow CSVs from the raw archives.")
    parser.add_argument("--source", choices=list(SOURCES) + ["all"], default="all")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--manifest-dir", default="results/data")
    parser.add_argument("--fiveg-archive", default="data/raw/Archive.zip")
    parser.add_argument("--cesnet-archive", default="data/raw/cesnet-quic22.zip")
    parser.add_argument("--max-packets", type=int, default=128)
    parser.add_argument("--window-seconds", type=float, default=10.0, help="5G active timeout / window length.")
    parser.add_argument("--idle-seconds", type=float, default=10.0, help="5G idle timeout.")
    parser.add_argument("--min-packets", type=int, default=8)
    parser.add_argument("--max-rows-per-capture", type=int, default=None,
                        help="Row cap per capture. Default None reads every packet row.")
    parser.add_argument("--max-flows-per-capture", type=int, default=6000)
    parser.add_argument("--rows-per-day-per-class", type=int, default=400)
    parser.add_argument("--max-rows-per-day", type=int, default=None)
    parser.add_argument("--min-class-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-checksum", action="store_true", help="Skip hashing the archives (they are 3 GB and 21 GB).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    checksum = not args.no_checksum

    if args.source in ("fiveg", "all"):
        config = FiveGConfig(
            archive=args.fiveg_archive,
            max_packets=args.max_packets,
            min_packets=args.min_packets,
            idle_timeout_s=args.idle_seconds,
            active_timeout_s=args.window_seconds,
            max_rows_per_capture=args.max_rows_per_capture,
            max_flows_per_capture=args.max_flows_per_capture,
            seed=args.seed,
        )
        manifest = prepare_fiveg(config, output_dir / "fiveg_traffic.csv", checksum=checksum)
        (manifest_dir / "fiveg_traffic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[5g_traffic] wrote {manifest['n_rows']:,} rows -> {manifest['output']}")
        print(f"[5g_traffic] services: {manifest['service_counts']}")

    if args.source in ("cesnet", "all"):
        config = CesnetConfig(
            archive=args.cesnet_archive,
            max_packets=30,
            min_packets=4,
            rows_per_day_per_class=args.rows_per_day_per_class,
            max_rows_per_day=args.max_rows_per_day,
            seed=args.seed,
            min_class_rows=args.min_class_rows,
        )
        manifest = prepare_cesnet(config, output_dir / "cesnet_quic22.csv", checksum=checksum)
        (manifest_dir / "cesnet_quic22_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[cesnet_quic22] wrote {manifest['n_rows']:,} rows -> {manifest['output']}")
        print(f"[cesnet_quic22] services: {manifest['service_counts']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
