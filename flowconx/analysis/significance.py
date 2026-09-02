"""Significance testing.

No table in the paper says "better" without a test behind it. This module
produces ``results/significance.json``, which the table generator reads; a
comparison absent from that file cannot be rendered as a win.

Three tests, each for the situation it is actually valid in:

* **McNemar** for two classifiers on the *same* test set. This is the right
  test for a paired comparison of predictions and it is what the encrypted
  traffic literature under-uses.
* **Wilcoxon signed-rank** across seeds, for aggregate comparisons where the
  unit of observation is a run rather than a test example.
* **Bootstrap** over the test set for a single system's confidence interval.

Multiple comparisons within an ablation family are corrected with
Holm-Bonferroni, and the correction is recorded in the output so that a reader
can see which family a p-value belongs to.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TestResult:
    test: str
    statistic: float
    p_value: float
    n: int
    effect_size: float
    effect_size_name: str
    detail: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray, exact_threshold: int = 25) -> TestResult:
    """Paired test of two classifiers on the same examples.

    Uses the exact binomial test when the discordant count is small, where the
    chi-square approximation is unreliable, and the continuity-corrected
    chi-square otherwise. Effect size is the odds ratio of the discordant
    pairs, which is what "how much better" means for a paired comparison.
    """
    from scipy import stats

    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError("McNemar needs predictions on the same test set")
    b = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))  # a wrong, b right
    n_discordant = b + c
    if n_discordant == 0:
        return TestResult(
            test="mcnemar",
            statistic=0.0,
            p_value=1.0,
            n=int(correct_a.size),
            effect_size=1.0,
            effect_size_name="odds_ratio",
            detail={"b_only_a_correct": b, "c_only_b_correct": c, "n_discordant": 0},
        )
    if n_discordant < exact_threshold:
        p_value = float(stats.binomtest(b, n_discordant, 0.5).pvalue)
        statistic = float(min(b, c))
        method = "exact_binomial"
    else:
        statistic = float((abs(b - c) - 1) ** 2 / n_discordant)
        p_value = float(stats.chi2.sf(statistic, df=1))
        method = "chi2_continuity_corrected"
    return TestResult(
        test="mcnemar",
        statistic=statistic,
        p_value=p_value,
        n=int(correct_a.size),
        effect_size=float(b / c) if c else float("inf"),
        effect_size_name="odds_ratio",
        detail={
            "b_only_a_correct": b,
            "c_only_b_correct": c,
            "n_discordant": n_discordant,
            "accuracy_a": float(correct_a.mean()),
            "accuracy_b": float(correct_b.mean()),
            "method": method,
        },
    )


def wilcoxon_across_seeds(scores_a: Sequence[float], scores_b: Sequence[float]) -> TestResult:
    """Paired test across seeds. Effect size is the rank-biserial correlation."""
    from scipy import stats

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Wilcoxon needs the same seeds for both systems")
    differences = a - b
    if np.allclose(differences, 0.0):
        return TestResult(
            test="wilcoxon",
            statistic=0.0,
            p_value=1.0,
            n=int(a.size),
            effect_size=0.0,
            effect_size_name="rank_biserial",
            detail={"mean_difference": 0.0, "note": "all differences are zero"},
        )
    if a.size < 6:
        # Wilcoxon on fewer than six pairs cannot reach p < 0.05 whatever the
        # data. Reporting a p-value would imply a power the design lacks.
        return TestResult(
            test="wilcoxon",
            statistic=float("nan"),
            p_value=float("nan"),
            n=int(a.size),
            effect_size=float(np.mean(differences)),
            effect_size_name="mean_difference",
            detail={
                "mean_difference": float(np.mean(differences)),
                "std_difference": float(np.std(differences)),
                "note": (
                    f"{a.size} seeds is too few for a Wilcoxon signed-rank test to reach significance "
                    "at any effect size; run at least 6 seeds, and 10 for the headline table."
                ),
            },
        )
    statistic, p_value = stats.wilcoxon(a, b)
    ranks = stats.rankdata(np.abs(differences))
    positive = float(np.sum(ranks[differences > 0]))
    total = float(np.sum(ranks))
    return TestResult(
        test="wilcoxon",
        statistic=float(statistic),
        p_value=float(p_value),
        n=int(a.size),
        effect_size=float(2 * positive / total - 1) if total else 0.0,
        effect_size_name="rank_biserial",
        detail={
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "mean_difference": float(np.mean(differences)),
            "std_difference": float(np.std(differences, ddof=1)) if a.size > 1 else 0.0,
        },
    )


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired Cohen's d, reported alongside every p-value.

    When the differences are numerically constant the ratio explodes to a
    meaningless magnitude, which reads in a table as a spectacular effect when
    it actually means "these seeds barely vary". Such cases return infinity
    with the correct sign, which a table generator can render as "n/a
    (constant difference)" rather than as 9.2e14.
    """
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if differences.size < 2:
        return 0.0
    mean = float(np.mean(differences))
    spread = float(np.std(differences, ddof=1))
    # Relative tolerance: a spread far below the scale of the mean difference
    # is numerical noise, not a real denominator.
    if spread <= max(1e-12, abs(mean) * 1e-9):
        return 0.0 if mean == 0.0 else float(np.inf if mean > 0 else -np.inf)
    return float(mean / spread)


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> List[Dict[str, object]]:
    """Holm-Bonferroni step-down correction over one family of comparisons.

    Returned in the input order, each entry carrying its adjusted threshold and
    whether it survives, so a table generator can mark significance without
    re-deriving the correction.
    """
    values = np.asarray(p_values, dtype=float)
    # NaN sorts last and, crucially, every comparison against it is False --
    # including `value > threshold` -- so a naive step-down marks an
    # underpowered comparison as *significant*. That is precisely the failure
    # the Wilcoxon refusal exists to prevent, so NaN is handled explicitly.
    order = np.argsort(np.where(np.isnan(values), np.inf, values), kind="stable")
    m = len(values)
    out: List[Dict[str, object]] = [{} for _ in range(m)]
    still_rejecting = True
    for rank, index in enumerate(order):
        threshold = alpha / (m - rank)
        if np.isnan(values[index]):
            still_rejecting = False
            rejected = False
        elif not still_rejecting or values[index] > threshold:
            still_rejecting = False
            rejected = False
        else:
            rejected = True
        out[int(index)] = {
            "p_value": float(values[index]),
            "undetermined": bool(np.isnan(values[index])),
            "rank": int(rank + 1),
            "adjusted_threshold": float(threshold),
            "significant": bool(rejected),
            "family_size": int(m),
            "alpha": float(alpha),
        }
    return out


def compare_family(
    comparisons: Sequence[Tuple[str, TestResult]],
    alpha: float = 0.05,
    family_name: str = "unnamed",
) -> Dict[str, object]:
    """Apply the correction across one family and package it for the tables."""
    corrected = holm_bonferroni([result.p_value for _, result in comparisons], alpha=alpha)
    return {
        "family": family_name,
        "correction": "holm_bonferroni",
        "alpha": alpha,
        "n_comparisons": len(comparisons),
        "comparisons": [
            {"name": name, **result.as_dict(), "correction": adjustment}
            for (name, result), adjustment in zip(comparisons, corrected)
        ],
    }


def load_seed_scores(
    results_root: str | Path,
    experiment: str,
    dataset: str,
    split: str,
    head: str = "prototype",
    metric: str = "macro_f1",
) -> Dict[int, float]:
    """Collect one metric across every seed of one experiment."""
    root = Path(results_root) / experiment / dataset / split
    scores: Dict[int, float] = {}
    for path in sorted(root.glob("seed*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = payload.get("closed_set", {}).get(head)
        if report is None:
            continue
        scores[int(payload.get("seed", -1))] = float(report[metric])
    return scores


def paired_seed_comparison(
    a_scores: Dict[int, float], b_scores: Dict[int, float]
) -> Optional[Tuple[TestResult, float, List[int]]]:
    """Wilcoxon plus Cohen's d over the seeds both systems were run at."""
    shared = sorted(set(a_scores) & set(b_scores))
    if len(shared) < 2:
        return None
    a = [a_scores[s] for s in shared]
    b = [b_scores[s] for s in shared]
    return wilcoxon_across_seeds(a, b), cohens_d(a, b), shared


# --------------------------------------------------------------------------
# CLI: build results/significance.json, which the table generator reads
# --------------------------------------------------------------------------

# Which experiments form one correction family. Holm-Bonferroni is applied
# within a family, so a family has to be a set of comparisons that were
# planned together -- not every comparison in the repository.
FAMILIES: Dict[str, Dict[str, object]] = {
    "component_ablations": {
        "reference": "ablation_full",
        "members": [
            "ablation_no_flow_metric",
            "ablation_no_dual_encoder",
            "ablation_fusion_concat",
            "ablation_fusion_gated_sum",
            "ablation_fusion_late",
            "ablation_no_adversarial",
            "ablation_no_disentangle",
            "ablation_no_prototype",
            "ablation_contrastive_only",
            "ablation_margin_only",
            "ablation_joint_training",
            "ablation_capacity_matched_small",
        ],
    },
    "embedding_choice": {
        "reference": "ablation_full",
        "members": ["ablation_classify_z_app", "ablation_classify_z_network", "ablation_classify_z_concat"],
    },
    "against_baselines": {
        "reference": "flowconx_main",
        "members": [
            "baseline_deeppacket_cnn",
            "baseline_fsnet",
            "baseline_lstm_attention",
            "baseline_mlp_stats",
        ],
    },
}


# Claim C1's test: does the split protocol change the number? Unlike the other
# families these compare *across* splits, so the pairing is by seed on two
# different (experiment, split) cells rather than within one.
SPLIT_CONTRASTS: List[Dict[str, str]] = [
    {
        "name": "random_flow_vs_session_disjoint",
        "reference_experiment": "flowconx_main",
        "reference_split": "session_disjoint",
        "member_experiment": "flowconx_random_contrast",
        "member_split": "random_flow",
    },
    {
        "name": "temporal_vs_session_disjoint",
        "reference_experiment": "flowconx_main",
        "reference_split": "session_disjoint",
        "member_experiment": "flowconx_temporal",
        "member_split": "temporal",
    },
    {
        "name": "server_disjoint_vs_session_disjoint",
        "reference_experiment": "flowconx_main",
        "reference_split": "session_disjoint",
        "member_experiment": "flowconx_server_disjoint",
        "member_split": "server_disjoint",
    },
]


def _head_for(experiment: str) -> str:
    """Baselines report under 'softmax'; the model reports under 'prototype'."""
    return "softmax" if experiment.startswith("baseline_") else "prototype"


def build_significance(
    results_root: str | Path,
    datasets: Sequence[str],
    splits: Sequence[str],
    alpha: float = 0.05,
    metric: str = "macro_f1",
) -> Dict[str, object]:
    """Every planned comparison, tested and corrected within its family."""
    out: Dict[str, object] = {
        "metric": metric,
        "alpha": alpha,
        "correction": "holm_bonferroni",
        "test": "wilcoxon_signed_rank_across_seeds",
        "families": [],
        "skipped": [],
    }
    for family_name, definition in FAMILIES.items():
        reference_name = str(definition["reference"])
        for dataset in datasets:
            for split in splits:
                reference = load_seed_scores(
                    results_root, reference_name, dataset, split, _head_for(reference_name), metric
                )
                if len(reference) < 2:
                    out["skipped"].append(
                        {
                            "family": family_name,
                            "dataset": dataset,
                            "split": split,
                            "reason": f"reference {reference_name} has {len(reference)} seed(s)",
                        }
                    )
                    continue
                comparisons: List[Tuple[str, TestResult]] = []
                metadata: List[Dict[str, object]] = []
                for member in definition["members"]:  # type: ignore[union-attr]
                    scores = load_seed_scores(results_root, member, dataset, split, _head_for(member), metric)
                    paired = paired_seed_comparison(reference, scores)
                    if paired is None:
                        out["skipped"].append(
                            {
                                "family": family_name,
                                "dataset": dataset,
                                "split": split,
                                "experiment": member,
                                "reason": f"{len(scores)} seed(s) shared with the reference",
                            }
                        )
                        continue
                    result, effect, shared = paired
                    comparisons.append((member, result))
                    metadata.append(
                        {
                            "experiment": member,
                            "dataset": dataset,
                            "split": split,
                            "reference": reference_name,
                            "cohens_d": effect,
                            "shared_seeds": shared,
                            "reference_mean": float(np.mean([reference[s] for s in shared])),
                            "member_mean": float(np.mean([scores[s] for s in shared])),
                        }
                    )
                if not comparisons:
                    continue
                family = compare_family(comparisons, alpha=alpha, family_name=f"{family_name}/{dataset}/{split}")
                for entry, extra in zip(family["comparisons"], metadata):  # type: ignore[index]
                    entry.update(extra)
                out["families"].append(family)

    # Split-protocol contrasts, one family per dataset.
    for dataset in datasets:
        comparisons: List[Tuple[str, TestResult]] = []
        metadata: List[Dict[str, object]] = []
        for contrast in SPLIT_CONTRASTS:
            reference = load_seed_scores(
                results_root,
                contrast["reference_experiment"],
                dataset,
                contrast["reference_split"],
                _head_for(contrast["reference_experiment"]),
                metric,
            )
            member = load_seed_scores(
                results_root,
                contrast["member_experiment"],
                dataset,
                contrast["member_split"],
                _head_for(contrast["member_experiment"]),
                metric,
            )
            paired = paired_seed_comparison(reference, member)
            if paired is None:
                out["skipped"].append(
                    {
                        "family": "split_protocol",
                        "dataset": dataset,
                        "contrast": contrast["name"],
                        "reason": f"{len(reference)} reference and {len(member)} member seed(s)",
                    }
                )
                continue
            result, effect, shared = paired
            comparisons.append((contrast["name"], result))
            metadata.append(
                {
                    "experiment": contrast["member_experiment"],
                    "dataset": dataset,
                    "split": contrast["member_split"],
                    "reference": f"{contrast['reference_experiment']}@{contrast['reference_split']}",
                    "cohens_d": effect,
                    "shared_seeds": shared,
                    "reference_mean": float(np.mean([reference[s] for s in shared])),
                    "member_mean": float(np.mean([member[s] for s in shared])),
                }
            )
        if comparisons:
            family = compare_family(comparisons, alpha=alpha, family_name=f"split_protocol/{dataset}")
            for entry, extra in zip(family["comparisons"], metadata):  # type: ignore[index]
                entry.update(extra)
            out["families"].append(family)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build results/significance.json.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/significance.json")
    parser.add_argument("--datasets", nargs="*", default=["cesnet_quic22", "fiveg_traffic"])
    parser.add_argument("--splits", nargs="*", default=["session_disjoint", "temporal", "random_flow"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--metric", default="macro_f1")
    args = parser.parse_args(argv)

    payload = build_significance(args.results, args.datasets, args.splits, args.alpha, args.metric)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    tested = sum(len(f["comparisons"]) for f in payload["families"])
    significant = sum(
        1 for f in payload["families"] for c in f["comparisons"] if c["correction"]["significant"]
    )
    print(f"{tested} comparisons tested, {significant} significant after Holm-Bonferroni at alpha={args.alpha}")
    for family in payload["families"]:
        print(f"\n{family['family']}")
        for comparison in family["comparisons"]:
            marker = "*" if comparison["correction"]["significant"] else " "
            print(
                f"  {marker} {comparison['name']:<34} "
                f"{comparison['member_mean']:.4f} vs {comparison['reference_mean']:.4f}  "
                f"p={comparison['p_value']:.4g}  d={comparison['cohens_d']:.2f}"
            )
    if payload["skipped"]:
        print(f"\n{len(payload['skipped'])} comparison(s) skipped for want of seeds:")
        for entry in payload["skipped"][:10]:
            print(f"  {entry.get('experiment', entry['family'])}: {entry['reason']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
