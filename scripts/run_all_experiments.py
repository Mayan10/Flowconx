#!/usr/bin/env python3
"""Run every experiment behind the paper, in dependency order.

    python scripts/run_all_experiments.py --seeds 0 1 2 3 4

Two things this does that a shell loop would not:

* **De-duplicates by config hash.** Two configs that differ only in their name
  are the same experiment; running both wastes GPU hours and produces two
  columns of the same number. The runner detects the collision and runs it
  once.
* **Never silently overwrites.** A run whose metrics.json already exists is
  skipped unless --overwrite is passed, so re-running the sweep after adding
  one ablation costs one ablation, and a result a table already cites cannot
  be replaced by accident.

Progress, failures and skips are all written to results/sweep_log.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowconx.experiment import load_config  # noqa: E402

# Order matters: the headline runs first, so that a sweep interrupted halfway
# has produced the numbers the paper cannot do without.
STAGES: Dict[str, List[str]] = {
    "headline": [
        "configs/cesnet_main.yaml",
        "configs/fiveg_main.yaml",
    ],
    "split_contrast": [
        "configs/cesnet_random_contrast.yaml",
        "configs/cesnet_temporal.yaml",
        "configs/cesnet_server_disjoint.yaml",
        "configs/fiveg_random_contrast.yaml",
        "configs/fiveg_temporal.yaml",
        "configs/fiveg_server_disjoint.yaml",
    ],
    "novelty": [
        "configs/fiveg_open_set.yaml",
    ],
    "ablations": sorted(str(p) for p in Path("configs/ablations").glob("*.yaml")),
    # The subset of the ablation family that can still bear on a surviving
    # claim. See ABLATIONS_CUT below and paper/RESULTS.md for what the full
    # family would add and why it was not run.
    "ablations_core": sorted(
        str(p)
        for p in Path("configs/ablations").glob("*.yaml")
        if p.stem in {
            # Does the dual encoder justify itself? The only rows that speak to
            # the architecture's one structural commitment.
            "classify_z_app",
            "classify_z_network",
            "classify_z_concat",
            "full",
            # Why does adversarial removal fail? C9 is refuted; this says
            # whether the weight is the reason or the mechanism is.
            "adv_weight_0p0",
            "adv_weight_0p01",
            "adv_weight_0p1",
            "adv_weight_0p5",
            "adv_weight_1p0",
            "no_adversarial",
        }
    ),
    # Ablations scored on the open-set metric rather than closed-set macro-F1.
    # Restricted to the four that could bear on C4; see the family's full.yaml.
    "ablations_openset": sorted(str(p) for p in Path("configs/ablations_openset").glob("*.yaml")),
    # The adversarial sweep on the dataset where the nuisance actually exists.
    # The closed-set sweep ran on CESNET, where it does not.
    "ablations_adv5g": sorted(str(p) for p in Path("configs/ablations_adv5g").glob("*.yaml")),
    # Baselines reuse the model's config for data, split and training budget,
    # and are dispatched through flowconx.baselines.run_baselines rather than
    # flowconx.run. Same results layout, so the aggregator needs no special
    # case.
    "baselines": [
        "configs/cesnet_main.yaml",
        "configs/fiveg_main.yaml",
    ],
}

# Stages whose configs go to the baseline runner instead of the model runner.
BASELINE_STAGES = {"baselines"}

# Ablations deliberately NOT run, and why. Recorded here rather than left as a
# silent gap: a table that omits rows without saying so reads as though the
# rows were run and were uninteresting.
#
# The closed-set ablation family was designed to explain *why* FlowCon-X wins.
# It does not win -- it loses to XGBoost on both corpora (0.783 vs 0.790 on
# CESNET, 0.547 vs 0.849 on 5G) -- so sweeping temperatures, margins, fusion
# variants and input budgets would measure internal variation in a result that
# is beaten either way. Cutting them takes the remaining sweep from ~22 h to
# ~6 h on one machine.
ABLATIONS_CUT = {
    "fusion variants": ["fusion_concat", "fusion_gated_sum", "fusion_late", "no_dual_encoder"],
    "loss-term ablations": [
        "no_flow_metric", "no_disentangle", "no_prototype", "contrastive_only",
        "margin_only", "joint_training",
    ],
    "temperature sweep": ["temperature_0p03", "temperature_0p05", "temperature_0p1", "temperature_0p2"],
    "margin sweep": ["margin_0p05", "margin_0p1", "margin_0p3", "margin_0p4"],
    "input-budget sweep": ["packets_1", "packets_3", "packets_5", "packets_10", "packets_20", "packets_30"],
    "capacity control": ["capacity_matched_small"],
}
# `no_flow_metric` is cut *here* but is run in the open-set family, where it is
# the decisive test of the only surviving modelling claim.


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full experiment sweep.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--headline-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds for the headline configs. The brief asks for 10; defaults to --seeds.",
    )
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    parser.add_argument("--results", default="results")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List what would run, and stop.")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--set", action="append", default=[], help="Config override applied to every run.")
    return parser.parse_args(argv)


def plan(args: argparse.Namespace) -> List[Dict[str, object]]:
    """Every (config, seed) to run, with duplicates by settings removed."""
    seen_hash: Dict[str, str] = {}
    jobs: List[Dict[str, object]] = []
    for stage in args.stages:
        seeds = args.headline_seeds if (stage == "headline" and args.headline_seeds) else args.seeds
        for config_path in STAGES[stage]:
            if not Path(config_path).exists():
                continue
            config = load_config(config_path)
            # Namespaced by stage kind: the same config legitimately appears
            # once for the model and once for the baselines, and those are
            # different runs.
            digest = f"{'baseline' if stage in BASELINE_STAGES else 'model'}:{config.hash()}"
            if digest in seen_hash:
                jobs.append(
                    {
                        "stage": stage,
                        "config": config_path,
                        "status": "duplicate",
                        "duplicate_of": seen_hash[digest],
                        "config_hash": digest,
                    }
                )
                continue
            seen_hash[digest] = config_path
            for seed in seeds:
                jobs.append(
                    {
                        "stage": stage,
                        "config": config_path,
                        "seed": seed,
                        "config_hash": digest,
                        "experiment": config.name,
                        "dataset": config.data.dataset,
                        "split": config.data.split_protocol,
                        "status": "pending",
                    }
                )
    return jobs


def run_one(job: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    module = "flowconx.baselines.run_baselines" if job["stage"] in BASELINE_STAGES else "flowconx.run"
    command = [
        sys.executable,
        "-m",
        module,
        "--config",
        str(job["config"]),
        "--seed",
        str(job["seed"]),
        "--output-root",
        args.results,
    ]
    for override in args.set:
        command += ["--set", override]
    if args.overwrite:
        command.append("--overwrite")
    if args.save_model and module == "flowconx.run":
        command.append("--save-model")

    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = round(time.perf_counter() - started, 1)
    # Exit code 2 is the deliberate "metrics.json already exists" refusal, not
    # a failure: it is what makes re-running the sweep cheap.
    if completed.returncode == 2:
        return {**job, "status": "skipped_exists", "seconds": elapsed}
    if completed.returncode != 0:
        return {
            **job,
            "status": "failed",
            "returncode": completed.returncode,
            "seconds": elapsed,
            "stderr_tail": completed.stderr.strip().splitlines()[-15:],
        }
    summary = [line for line in completed.stdout.splitlines() if "macro-F1" in line]
    return {**job, "status": "ok", "seconds": elapsed, "summary": summary}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    jobs = plan(args)
    runnable = [j for j in jobs if j["status"] == "pending"]
    duplicates = [j for j in jobs if j["status"] == "duplicate"]

    print(f"{len(runnable)} runs planned across stages {args.stages}")
    for job in duplicates:
        print(f"  duplicate settings: {job['config']} == {job['duplicate_of']} (skipping)")
    if args.dry_run:
        by_stage: Dict[str, int] = defaultdict(int)
        for job in runnable:
            by_stage[str(job["stage"])] += 1
        for stage, count in by_stage.items():
            print(f"  {stage}: {count} runs")
        return 0

    log: List[Dict[str, object]] = list(duplicates)
    failures = 0
    for position, job in enumerate(runnable, start=1):
        label = f"{Path(str(job['config'])).stem} seed={job['seed']}"
        print(f"[{position}/{len(runnable)}] {label} ... ", end="", flush=True)
        result = run_one(job, args)
        log.append(result)
        print(f"{result['status']} ({result['seconds']}s)")
        if result["status"] == "failed":
            failures += 1
            for line in result.get("stderr_tail", []):
                print(f"      {line}")
            if not args.continue_on_error:
                break
        Path(args.results).mkdir(parents=True, exist_ok=True)
        Path(args.results, "sweep_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    ok = sum(1 for entry in log if entry.get("status") == "ok")
    skipped = sum(1 for entry in log if entry.get("status") == "skipped_exists")
    print(f"\n{ok} ran, {skipped} already present, {failures} failed")
    print(f"log: {Path(args.results, 'sweep_log.json')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
