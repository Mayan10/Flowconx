"""Significance-testing tests.

The property that matters most here is negative: a comparison that could not
be tested must never be reported as significant. An underpowered or undefined
p-value silently becoming a star in a table is the failure mode that makes a
results section untrustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from flowconx.analysis.significance import (
    cohens_d,
    compare_family,
    holm_bonferroni,
    mcnemar,
    wilcoxon_across_seeds,
)


# --------------------------------------------------------------------------
# Holm-Bonferroni
# --------------------------------------------------------------------------


def test_holm_is_more_conservative_than_uncorrected():
    """0.04 is significant alone but must not survive a family of four."""
    corrected = holm_bonferroni([0.001, 0.02, 0.04, 0.3])
    assert [c["significant"] for c in corrected] == [True, False, False, False]
    assert corrected[0]["adjusted_threshold"] == pytest.approx(0.05 / 4)


def test_holm_step_down_operates_in_sorted_order():
    """The step-down walks p-values ascending, not in input order.

    Given [0.001, 0.5, 0.002] the procedure tests 0.001 against 0.05/3, then
    0.002 against 0.05/2, then 0.5 against 0.05. The first two survive and the
    last does not, so in *input* order the verdicts are [True, False, True].
    Reading the array positionally instead of by rank is an easy way to get
    this backwards.
    """
    corrected = holm_bonferroni([0.001, 0.5, 0.002])
    assert [c["significant"] for c in corrected] == [True, False, True]
    assert [c["rank"] for c in corrected] == [1, 3, 2]


def test_holm_does_not_resume_after_a_failure():
    """Once a rank fails, every larger p-value fails regardless of threshold."""
    corrected = holm_bonferroni([0.001, 0.03, 0.031])
    assert corrected[0]["significant"] is True
    # 0.03 fails against 0.05/2 = 0.025, so 0.031 must fail too even though it
    # would pass its own 0.05 threshold.
    assert corrected[1]["significant"] is False
    assert corrected[2]["significant"] is False
    assert corrected[2]["adjusted_threshold"] == pytest.approx(0.05)


def test_nan_p_value_is_never_significant():
    """The regression this test exists for.

    NaN compares False against everything, including `value > threshold`, so a
    naive step-down takes the reject branch and marks an *untestable*
    comparison as significant. That is exactly what the Wilcoxon refusal below
    six seeds is meant to prevent, so it must not be undone here.
    """
    corrected = holm_bonferroni([float("nan"), 0.001])
    assert corrected[0]["significant"] is False
    assert corrected[0]["undetermined"] is True
    assert corrected[1]["undetermined"] is False


def test_all_nan_family_yields_no_significance():
    corrected = holm_bonferroni([float("nan")] * 3)
    assert not any(c["significant"] for c in corrected)


def test_holm_preserves_input_order():
    corrected = holm_bonferroni([0.3, 0.001, 0.02])
    assert [c["p_value"] for c in corrected] == [0.3, 0.001, 0.02]
    assert corrected[1]["rank"] == 1


# --------------------------------------------------------------------------
# Wilcoxon
# --------------------------------------------------------------------------


def test_wilcoxon_refuses_below_six_seeds():
    """Fewer than six pairs cannot reach p < 0.05 at any effect size."""
    result = wilcoxon_across_seeds([0.9, 0.9, 0.9], [0.1, 0.1, 0.1])
    assert np.isnan(result.p_value)
    assert "too few" in result.detail["note"]


def test_wilcoxon_runs_at_six_seeds():
    result = wilcoxon_across_seeds([0.8, 0.81, 0.79, 0.82, 0.80, 0.83], [0.7, 0.71, 0.69, 0.72, 0.70, 0.73])
    assert not np.isnan(result.p_value)
    assert result.p_value < 0.05
    assert result.effect_size == pytest.approx(1.0)


def test_wilcoxon_on_identical_scores_is_not_significant():
    result = wilcoxon_across_seeds([0.5] * 8, [0.5] * 8)
    assert result.p_value == 1.0
    assert result.effect_size == 0.0


def test_wilcoxon_requires_matched_seeds():
    with pytest.raises(ValueError):
        wilcoxon_across_seeds([0.1, 0.2], [0.1, 0.2, 0.3])


# --------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------


def test_mcnemar_on_identical_predictions_is_not_significant():
    correct = np.random.default_rng(0).random(200) < 0.8
    result = mcnemar(correct, correct)
    assert result.p_value == 1.0
    assert result.detail["n_discordant"] == 0


def test_mcnemar_uses_the_exact_test_when_discordants_are_few():
    a = np.array([True] * 100)
    b = a.copy()
    b[:5] = False
    result = mcnemar(a, b)
    assert result.detail["method"] == "exact_binomial"
    assert result.detail["n_discordant"] == 5


def test_mcnemar_detects_a_real_difference():
    rng = np.random.default_rng(1)
    a = rng.random(2000) < 0.90
    b = rng.random(2000) < 0.80
    result = mcnemar(a, b)
    assert result.p_value < 0.01
    assert result.effect_size > 1.0


def test_mcnemar_requires_the_same_test_set():
    with pytest.raises(ValueError, match="same test set"):
        mcnemar(np.ones(10, dtype=bool), np.ones(9, dtype=bool))


# --------------------------------------------------------------------------
# Effect size
# --------------------------------------------------------------------------


def test_cohens_d_does_not_explode_on_constant_differences():
    """A constant difference is not an infinitely large effect in a table."""
    assert np.isinf(cohens_d([0.8, 0.81, 0.79], [0.75, 0.76, 0.74]))
    assert cohens_d([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_compare_family_packages_corrections_alongside_results():
    a = wilcoxon_across_seeds([0.8] * 8, [0.7] * 8)
    b = wilcoxon_across_seeds([0.8] * 8, [0.8] * 8)
    family = compare_family([("a", a), ("b", b)], family_name="demo")
    assert family["n_comparisons"] == 2
    assert {c["name"] for c in family["comparisons"]} == {"a", "b"}
    assert all("correction" in c for c in family["comparisons"])
