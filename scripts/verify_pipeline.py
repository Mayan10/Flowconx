#!/usr/bin/env python3
"""End-to-end pipeline verification that needs no data.

    python scripts/verify_pipeline.py      # ~2 minutes, CPU only

Generates a small table in the canonical schema, then runs the whole pipeline
over it -- audit, split manifests, training, every evaluation mode, seed
aggregation, significance testing and paper-asset generation -- and asserts
that each stage produced what it claims to.

This exists because every other pipeline target needs the raw archives, which
are 3.2 GB and 21 GB and require registration. An artifact evaluator on a
clean machine could previously run the unit tests and nothing else. This is
the thing they can run to confirm the pipeline itself works, and it is the
first thing to run after `make setup`.

It verifies plumbing, not science: the synthetic data is separable by
construction and its numbers mean nothing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def step(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def run(command: List[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"FAILED: {' '.join(command)}")


def main(argv: Optional[List[str]] = None) -> int:
    _ = argv
    from scripts.audit_smoke import synthetic_table  # noqa: E402

    work = Path(tempfile.mkdtemp(prefix="flowconx-verify-"))
    try:
        data_dir = work / "data" / "processed"
        data_dir.mkdir(parents=True)
        csv = data_dir / "verify.csv"

        step("1/6  generate a table in the canonical schema")
        frame = synthetic_table()
        frame.to_csv(csv, index=False)
        print(f"  {len(frame):,} rows, {frame['service'].nunique()} classes, "
              f"{frame['capture_id'].nunique()} captures -> {csv}")

        step("2/6  audit: split protocols, identifier probes, leakage checks")
        run(
            [sys.executable, "-m", "flowconx.audit.run_audit", "--csv", str(csv),
             "--results-dir", str(work / "results" / "audit"), "--splits-dir", str(work / "splits"),
             "--no-bootstrap", "--seed", "0"],
            ROOT,
        )
        summary = json.loads((work / "results" / "audit" / "verify" / "audit_summary.json").read_text())
        reachable = [k for k, v in summary["splits"].items() if v.get("status") == "ok"]
        # The generated table carries every provenance column, so every
        # grouping protocol must be reachable on it. A protocol that silently
        # became unavailable is exactly what this catches.
        expected = {"random_flow", "session_disjoint", "temporal", "server_disjoint",
                    "client_disjoint", "vantage_disjoint", "app_disjoint", "origin_disjoint"}
        missing = expected - set(reachable)
        assert not missing, f"split protocols unreachable on a schema-complete table: {sorted(missing)}"
        print(f"  {len(reachable)} split protocols reachable: {', '.join(reachable)}")
        manifests = list((work / "splits").rglob("*.json*"))
        assert manifests, "no split manifests written"
        print(f"  {len(manifests)} split manifests written")

        step("3/6  train and evaluate, every evaluation mode on")
        config = work / "verify.yaml"
        config.write_text(
            "defaults: [%s]\n" % (ROOT / "configs" / "base.yaml")
            + "name: verify\n"
            + "description: Pipeline verification on generated data. Numbers are meaningless.\n"
            + "data:\n"
            + f"  dataset: verify\n  csv: {csv}\n  split_protocol: session_disjoint\n"
            + "  max_packets: 16\n  min_class_count: 20\n  unknown_apps: [streaming_app1]\n"
            + "train:\n  epochs: 2\n  stage1_epochs: 1\n  batch_size: 64\n  device: cpu\n"
            + "eval:\n  bootstrap_resamples: 20\n  classifier_heads: [knn, prototype]\n"
            + "  open_set: true\n  few_shot: true\n  drift: true\n  early_classification: true\n"
            + "  robustness: true\n  probes: true\n  cost: true\n"
            + "  few_shot_k: [1, 5]\n  early_packet_budgets: [1, 4]\n",
            encoding="utf-8",
        )
        for seed in (0, 1):
            run([sys.executable, "-m", "flowconx.run", "--config", str(config), "--seed", str(seed),
                 "--output-root", str(work / "results"), "--overwrite"], ROOT)

        metrics = json.loads(next((work / "results" / "verify").rglob("metrics.json")).read_text())
        for block in ("closed_set", "cost", "open_set", "few_shot", "early_classification",
                      "robustness", "probes", "provenance"):
            assert block in metrics, f"metrics.json is missing the {block!r} block"
        for block in ("open_set", "few_shot", "early_classification", "robustness", "probes"):
            status = metrics[block].get("status")
            assert status in ("ok", "skipped"), f"{block} reported status {status!r}"
            print(f"  {block:22s} {status}")
        assert metrics["provenance"]["git"]["commit"], "no git commit recorded"
        print(f"  provenance             commit {metrics['provenance']['git']['commit'][:12]}, "
              f"dirty={metrics['provenance']['git']['dirty']}")

        step("4/6  aggregate across seeds")
        run([sys.executable, "-m", "flowconx.analysis.aggregate", "--results", str(work / "results"),
             "--out", str(work / "results" / "aggregate.json")], ROOT)
        groups = json.loads((work / "results" / "aggregate.json").read_text())["groups"]
        assert groups and groups[0]["n_seeds"] == 2, "aggregation did not group the two seeds"
        print(f"  {len(groups)} group(s), {groups[0]['n_seeds']} seeds")

        step("5/6  significance testing")
        run([sys.executable, "-m", "flowconx.analysis.significance", "--results", str(work / "results"),
             "--out", str(work / "results" / "significance.json")], ROOT)
        significance = json.loads((work / "results" / "significance.json").read_text())
        assert "families" in significance
        print(f"  {len(significance['families'])} family/families, "
              f"{len(significance['skipped'])} comparison(s) skipped for want of seeds")

        step("6/6  paper assets")
        paper = work / "paper"
        (paper / "sections").mkdir(parents=True)
        shutil.copy(ROOT / "paper" / "references.bib", paper / "references.bib")
        run([sys.executable, "scripts/make_paper_assets.py", "--results", str(work / "results"),
             "--out", str(paper), "--datasets", "verify"], ROOT)
        tables = list((paper / "tables").glob("*.tex"))
        assert tables, "no tables generated"
        print(f"  {len(tables)} table(s) generated")

        print("\nPIPELINE VERIFIED: audit, splits, training, all evaluation modes, "
              "aggregation, significance and asset generation all ran end to end.")
        print("This checks plumbing, not science. The generated data is separable by "
              "construction and its numbers mean nothing.")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
