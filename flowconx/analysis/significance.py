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
    order = np.argsort(values, kind="stable")
    m = len(values)
    out: List[Dict[str, object]] = [{} for _ in range(m)]
    still_rejecting = True
    for rank, index in enumerate(order):
        threshold = alpha / (m - rank)
        if not still_rejecting or values[index] > threshold:
            still_rejecting = False
            rejected = False
        else:
            rejected = True
        out[int(index)] = {
            "p_value": float(values[index]),
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
