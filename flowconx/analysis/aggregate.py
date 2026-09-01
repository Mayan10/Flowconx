"""Aggregate every run in ``results/`` into one table-ready structure.

    python -m flowconx.analysis.aggregate --results results --out results/aggregate.json

Walks ``results/<experiment>/<dataset>/<split>/seed<n>/metrics.json``, groups
by experiment, and reports mean +/- std across seeds together with the number
of seeds each cell rests on. A cell built from fewer seeds than the paper
claims is flagged rather than silently averaged, because "mean of 2 seeds"
printed next to "mean of 10" in the same column is a misrepresentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# The seed counts the brief commits to. Cells below these are marked.
HEADLINE_MIN_SEEDS = 10
STANDARD_MIN_SEEDS = 5

METRIC_KEYS = ("macro_f1", "balanced_accuracy", "accuracy")


def iter_runs(results_root: str | Path) -> List[Dict[str, Any]]:
    """Every metrics.json under the results root, with its path metadata."""
    root = Path(results_root)
    runs: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "closed_set" not in payload:
            continue
        relative = path.relative_to(root).parts
        if len(relative) < 5:
            continue
        runs.append(
            {
                "path": str(path),
                "experiment": relative[0],
                "dataset": relative[1],
                "split": relative[2],
                "seed": payload.get("seed"),
                "config_hash": payload.get("config_hash"),
                "metrics": payload,
            }
        )
    return runs


def _summarise(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "n_seeds": int(array.size),
    }


def aggregate(results_root: str | Path, heads: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    runs = iter_runs(results_root)
    groups: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        key = f"{run['experiment']}|{run['dataset']}|{run['split']}"
        group = groups.setdefault(
            key,
            {
                "experiment": run["experiment"],
                "dataset": run["dataset"],
                "split": run["split"],
                "config_hashes": set(),
                "seeds": [],
                "heads": {},
                "extras": {},
            },
        )
        group["config_hashes"].add(run["config_hash"])
        group["seeds"].append(run["seed"])
        for head, report in run["metrics"]["closed_set"].items():
            if heads and head not in heads:
                continue
            bucket = group["heads"].setdefault(head, {metric: [] for metric in METRIC_KEYS})
            for metric in METRIC_KEYS:
                if metric in report:
                    bucket[metric].append(float(report[metric]))
        _collect_extras(group["extras"], run["metrics"])

    out: Dict[str, Any] = {"n_runs": len(runs), "groups": []}
    for key in sorted(groups):
        group = groups[key]
        hashes = sorted(h for h in group["config_hashes"] if h)
        entry: Dict[str, Any] = {
            "experiment": group["experiment"],
            "dataset": group["dataset"],
            "split": group["split"],
            "seeds": sorted(s for s in group["seeds"] if s is not None),
            "n_seeds": len(group["seeds"]),
            "config_hash": hashes[0] if len(hashes) == 1 else hashes,
            "heads": {
                head: {metric: _summarise(values) for metric, values in metrics.items() if values}
                for head, metrics in group["heads"].items()
            },
            "extras": {name: _summarise(values) for name, values in group["extras"].items() if values},
        }
        if len(hashes) > 1:
            entry["warning_mixed_configs"] = (
                "Runs under this experiment name have different config hashes; they are not seeds of "
                "the same experiment and must not be averaged together."
            )
        if entry["n_seeds"] < STANDARD_MIN_SEEDS:
            entry["warning_few_seeds"] = (
                f"{entry['n_seeds']} seed(s). The paper's stated protocol is {STANDARD_MIN_SEEDS} "
                f"minimum and {HEADLINE_MIN_SEEDS} for the headline table."
            )
        out["groups"].append(entry)
    return out


def _collect_extras(bucket: Dict[str, List[float]], metrics: Dict[str, Any]) -> None:
    """Scalar summaries from the optional evaluation blocks, if present."""

    def push(name: str, value: object) -> None:
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            bucket.setdefault(name, []).append(float(value))

    push("n_parameters", metrics.get("training", {}).get("n_parameters"))
    push("train_seconds", metrics.get("training", {}).get("seconds"))
    cost = metrics.get("cost", {})
    push("latency_p50_ms", cost.get("end_to_end_batch1", {}).get("p50_ms"))
    push("latency_p95_ms", cost.get("end_to_end_batch1", {}).get("p95_ms"))
    push("latency_p99_ms", cost.get("end_to_end_batch1", {}).get("p99_ms"))
    push("throughput_flows_per_s", cost.get("throughput_batched", {}).get("flows_per_second"))
    push("model_size_bytes", cost.get("model_size_bytes"))
    for name, scorer in (metrics.get("open_set", {}) or {}).get("scorers", {}).items():
        push(f"open_set_auroc_{name}", scorer.get("auroc"))
        push(f"open_set_fpr95_{name}", scorer.get("fpr_at_95tpr"))
    drift = metrics.get("drift", {}) or {}
    push("drift_total_drop_macro_f1", drift.get("total_drop_macro_f1"))
    tradeoff = (metrics.get("probes", {}) or {}).get("tradeoff", {})
    push("probe_nuisance_above_majority", tradeoff.get("nuisance_above_majority_mlp"))
    push("probe_task_macro_f1", tradeoff.get("task_macro_f1_mlp"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate results across seeds.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/aggregate.json")
    parser.add_argument("--heads", nargs="*", default=None)
    args = parser.parse_args(argv)

    payload = aggregate(args.results, args.heads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{payload['n_runs']} runs in {len(payload['groups'])} experiment groups")
    for group in payload["groups"]:
        head = next(iter(group["heads"]), None)
        if head is None:
            continue
        stats = group["heads"][head].get("macro_f1", {})
        flag = "  [FEW SEEDS]" if "warning_few_seeds" in group else ""
        print(
            f"  {group['experiment']:<28} {group['dataset']:<16} {group['split']:<18} "
            f"{head}: macro-F1 {stats.get('mean', float('nan')):.4f} "
            f"+/- {stats.get('std', float('nan')):.4f}  (n={group['n_seeds']}){flag}"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
