"""Phase 0 audit entry point.

    python -m flowconx.audit.run_audit --csv data/processed/<file>.csv

Builds every split protocol the table can support, writes the manifests to
``splits/``, runs the leakage probes and every trivial/classical baseline on
each split, and writes machine-readable results to ``results/audit/``.

Nothing here reads a checkpoint or trains a neural network. The audit has to
be cheap enough that it runs in CI on every commit.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..labels import RARE_CLASS_MODES, RareClassPolicy, apply_rare_class_policy, label_array
from . import leakage
from .baselines import MODEL_SPECS, majority_class_report, run_all_families
from .splits import (
    SPLIT_PROTOCOLS,
    SplitUnavailable,
    available_protocols,
    build_split,
    describe_provenance,
    ensure_flow_ids,
    write_manifest,
)
from .tabular import build_features, parse_flows


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def environment_block() -> Dict[str, object]:
    versions: Dict[str, str] = {}
    for module in ("numpy", "pandas", "sklearn", "scipy", "xgboost", "torch"):
        try:
            versions[module] = __import__(module).__version__
        except (ImportError, AttributeError):
            versions[module] = "absent"
    return {
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "library_versions": versions,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlowCon-X shortcut and leakage audit.")
    parser.add_argument("--csv", required=True, help="Canonical flow CSV to audit.")
    parser.add_argument("--label-column", default="service")
    parser.add_argument(
        "--protocols",
        nargs="*",
        default=list(SPLIT_PROTOCOLS),
        help="Split protocols to attempt. Unavailable ones are recorded, not skipped silently.",
    )
    parser.add_argument("--families", nargs="*", default=list(MODEL_SPECS))
    parser.add_argument("--rare-class-mode", default="drop", choices=list(RARE_CLASS_MODES))
    parser.add_argument("--min-class-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="Row cap, for smoke runs only.")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--results-dir", default="results/audit")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap CIs (faster smoke runs).")
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.999,
        help="Cosine similarity above which a train/test pair counts as a near duplicate.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv)
    dataset = csv_path.stem
    started = time.perf_counter()

    df = pd.read_csv(csv_path, nrows=args.limit)
    policy = RareClassPolicy(
        mode=args.rare_class_mode, min_class_count=args.min_class_count, label_column=args.label_column
    )
    df, class_report = apply_rare_class_policy(df, policy)

    # Record what the table actually carries *before* anything is synthesized,
    # so a legacy table is not reported as having provenance it never had.
    provenance = describe_provenance(df)
    availability = available_protocols(df)

    # Then materialise flow_id, so the disjointness probes run against the same
    # identifiers the manifests are written with. On a legacy table this is a
    # content hash, and the report says so.
    df, flow_id_synthesized = ensure_flow_ids(df)
    y, class_names = label_array(df, args.label_column)
    flows = parse_flows(df)
    # A single dense feature view used for the near-duplicate probe. AppScanner
    # is the widest purely-behavioural family, so two rows that match on it are
    # the same observation for any practical purpose.
    # One shared cache: feature matrices are reused across every split
    # protocol, and the near-duplicate probe reuses the AppScanner matrix.
    feature_cache: Dict[str, tuple] = {}
    feature_cache["appscanner"] = build_features(df, "appscanner", flows)
    dedup_features = feature_cache["appscanner"][0]

    summary: Dict[str, object] = {
        "dataset": dataset,
        "csv": str(csv_path),
        "n_rows": int(len(df)),
        "label_column": args.label_column,
        "classes": class_names,
        "class_handling": class_report,
        "provenance_columns_present": provenance,
        "flow_id_synthesized": flow_id_synthesized,
        "split_protocol_availability": availability,
        "environment": environment_block(),
        "config": vars(args),
        "splits": {},
    }

    results_root = Path(args.results_dir) / dataset
    results_root.mkdir(parents=True, exist_ok=True)

    for protocol in args.protocols:
        entry: Dict[str, object] = {"protocol": protocol}
        try:
            manifest, indices = build_split(
                df,
                protocol,
                seed=args.seed,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                label_column=args.label_column,
                dataset_path=csv_path,
            )
        except SplitUnavailable as exc:
            entry["status"] = "unavailable"
            entry["reason"] = str(exc)
            summary["splits"][protocol] = entry
            print(f"[{protocol}] UNAVAILABLE: {exc}")
            continue

        # Gzip large manifests; a 112k-row flow-ID list is ~4 MB uncompressed.
        suffix = ".json.gz" if len(df) > 20000 else ".json"
        manifest_path = Path(args.splits_dir) / dataset / f"{protocol}_seed{args.seed}{suffix}"
        write_manifest(manifest, manifest_path)
        entry["status"] = "ok"
        entry["manifest"] = str(manifest_path)
        entry["checksums"] = manifest.checksums
        entry["sizes"] = {side: int(len(idx)) for side, idx in indices.items()}
        entry["class_counts"] = manifest.class_counts
        entry["notes"] = manifest.notes

        entry["leakage"] = leakage.run_all_checks(
            df,
            indices,
            features=dedup_features,
            declared_inputs=["packet_lengths", "iat_values", "directions"],
            near_duplicate_threshold=args.near_duplicate_threshold,
            seed=args.seed,
            flow_id_synthesized=flow_id_synthesized,
        )
        print(f"[{protocol}] leakage verdict: {entry['leakage']['verdict']} "
              f"(failed: {entry['leakage']['failed']})")

        baselines = run_all_families(
            df,
            y,
            class_names,
            indices["train"],
            indices["test"],
            families=args.families,
            seed=args.seed,
            bootstrap=not args.no_bootstrap,
            feature_cache=feature_cache,
        )
        baselines["majority_class"] = majority_class_report(y, class_names, indices["train"], indices["test"])
        entry["baselines"] = {
            name: {
                "macro_f1": result.get("metrics", {}).get("macro_f1"),
                "balanced_accuracy": result.get("metrics", {}).get("balanced_accuracy"),
                "accuracy": result.get("metrics", {}).get("accuracy"),
                "status": result.get("status"),
            }
            for name, result in baselines.items()
        }

        split_dir = results_root / protocol
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "baselines.json").write_text(
            json.dumps({"environment": summary["environment"], "results": baselines}, indent=2), encoding="utf-8"
        )
        (split_dir / "leakage.json").write_text(json.dumps(entry["leakage"], indent=2), encoding="utf-8")
        summary["splits"][protocol] = entry

        for name, result in sorted(
            baselines.items(), key=lambda kv: -(kv[1].get("metrics", {}).get("macro_f1") or 0.0)
        ):
            metrics = result.get("metrics", {})
            print(
                f"  {name:18s} macro-F1={metrics.get('macro_f1', float('nan')):.4f} "
                f"bal-acc={metrics.get('balanced_accuracy', float('nan')):.4f} "
                f"acc={metrics.get('accuracy', float('nan')):.4f}"
            )

    summary["wall_clock_seconds"] = round(time.perf_counter() - started, 2)
    summary_path = results_root / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    write_markdown_summary(summary, results_root / "audit_summary.md")
    print(f"Wrote {results_root / 'audit_summary.md'}")
    return 0


def write_markdown_summary(summary: Dict[str, object], path: Path) -> None:
    """Human-readable comparison of every baseline against every split protocol."""
    splits = summary["splits"]
    ok_splits = [name for name, entry in splits.items() if entry.get("status") == "ok"]
    families: List[str] = []
    for name in ok_splits:
        for family in splits[name]["baselines"]:
            if family not in families:
                families.append(family)

    lines = [
        f"# Shortcut audit: `{summary['dataset']}`",
        "",
        f"- Rows after class handling: **{summary['n_rows']:,}**",
        f"- Classes: {', '.join(summary['classes'])}",
        f"- Rare-class mode: `{summary['class_handling']['mode']}` "
        f"(action: {summary['class_handling']['action']})",
        f"- Git commit: `{summary['environment']['git_commit'][:12]}`",
        "",
        "## Split protocol availability",
        "",
        "| Protocol | Available | Note |",
        "| --- | --- | --- |",
    ]
    for protocol, entry in splits.items():
        available = "yes" if entry.get("status") == "ok" else "**no**"
        note = entry.get("reason", "; ".join(entry.get("notes", [])) if entry.get("notes") else "")
        lines.append(f"| `{protocol}` | {available} | {str(note)[:180]} |")

    lines += ["", "## Baseline macro-F1 by split protocol", ""]
    header = "| Baseline | " + " | ".join(f"`{s}`" for s in ok_splits) + " |"
    lines += [header, "| --- |" + " --- |" * len(ok_splits)]
    for family in families:
        cells = []
        for split_name in ok_splits:
            value = splits[split_name]["baselines"].get(family, {}).get("macro_f1")
            cells.append(f"{value:.4f}" if isinstance(value, float) else "n/a")
        lines.append(f"| `{family}` | " + " | ".join(cells) + " |")

    lines += ["", "## Leakage verdicts", "", "| Protocol | Verdict | Failed checks |", "| --- | --- | --- |"]
    for protocol in ok_splits:
        verdict = splits[protocol]["leakage"]
        failed = ", ".join(f"`{c}`" for c in verdict["failed"]) or "—"
        lines.append(f"| `{protocol}` | **{verdict['verdict']}** | {failed} |")

    lines += [
        "",
        "Generated by `python -m flowconx.audit.run_audit`. Do not edit by hand.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
