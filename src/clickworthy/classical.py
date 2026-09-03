# Stage 1: classical A/B tests, Wilson/Newcombe CIs, power/MDE, BH correction.
#
# Conventions (CLAUDE.md "Statistical conventions" section):
#   - CIs for single proportions: Wilson.
#   - CIs for differences of proportions: Newcombe.
#   - Multiple comparisons within a test: Benjamini-Hochberg.
#   - These are horse-race tests with no true control arm -- results are
#     described as "package A outperforms package B", never "treatment
#     beats control".

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_chisquare,
    test_proportions_2indep,
)

ALPHA = 0.05


def wilson_ci(count: int, nobs: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    low, high = proportion_confint(count, nobs, alpha=alpha, method="wilson")
    return low, high


def arm_summary(test_df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """headline, impressions, clicks, CTR, and Wilson CI per arm, sorted by CTR desc."""
    out = test_df[["headline", "impressions", "clicks"]].copy()
    out["ctr"] = out["clicks"] / out["impressions"]
    ci = out.apply(lambda r: wilson_ci(r["clicks"], r["impressions"], alpha), axis=1)
    out["wilson_low"], out["wilson_high"] = zip(*ci)
    return out.sort_values("ctr", ascending=False).reset_index(drop=True)


def two_proportion_test(
    count1: int, nobs1: int, count2: int, nobs2: int
) -> tuple[float, float]:
    """Score (Wald-type) two-proportion z-test. Returns (z_stat, p_value)."""
    stat, pval = test_proportions_2indep(
        count1, nobs1, count2, nobs2, method="score", compare="diff"
    )
    return stat, pval


def newcombe_diff_ci(
    count1: int, nobs1: int, count2: int, nobs2: int, alpha: float = ALPHA
) -> tuple[float, float]:
    """Newcombe (score-based) CI for p1 - p2."""
    low, high = confint_proportions_2indep(
        count1, nobs1, count2, nobs2, method="newcomb", compare="diff", alpha=alpha
    )
    return low, high


def lift(count1: int, nobs1: int, count2: int, nobs2: int) -> tuple[float, float]:
    """Absolute lift (p1 - p2) and relative lift ((p1 - p2) / p2) of arm 1 vs arm 2."""
    p1, p2 = count1 / nobs1, count2 / nobs2
    abs_lift = p1 - p2
    rel_lift = abs_lift / p2 if p2 > 0 else np.nan
    return abs_lift, rel_lift


def pairwise_tests(test_df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """All pairwise two-proportion tests within a test's arms: raw and
    BH-adjusted p-values, Newcombe diff CI, absolute/relative lift."""
    rows = []
    n = len(test_df)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = test_df.iloc[i], test_df.iloc[j]
            stat, p_raw = two_proportion_test(
                a["clicks"], a["impressions"], b["clicks"], b["impressions"]
            )
            diff_low, diff_high = newcombe_diff_ci(
                a["clicks"], a["impressions"], b["clicks"], b["impressions"], alpha
            )
            abs_lift, rel_lift = lift(
                a["clicks"], a["impressions"], b["clicks"], b["impressions"]
            )
            rows.append(
                {
                    "headline_a": a["headline"],
                    "headline_b": b["headline"],
                    "ctr_a": a["clicks"] / a["impressions"],
                    "ctr_b": b["clicks"] / b["impressions"],
                    "abs_lift": abs_lift,
                    "rel_lift": rel_lift,
                    "z_stat": stat,
                    "p_raw": p_raw,
                    "newcombe_diff_low": diff_low,
                    "newcombe_diff_high": diff_high,
                }
            )
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p_raw"], alpha=alpha, method="fdr_bh")[1]
    return out


def chi_square_across_arms(test_df: pd.DataFrame) -> tuple[float, float, int]:
    """Chi-square test of k independent proportions across all arms of one
    test. Returns (chi2_stat, p_value, dof); dof = n_arms - 1."""
    chi2_stat, p_value, _table = proportions_chisquare(
        test_df["clicks"].to_numpy(), test_df["impressions"].to_numpy()
    )
    dof = len(test_df) - 1
    return chi2_stat, p_value, dof
