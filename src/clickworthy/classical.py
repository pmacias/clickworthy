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
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_chisquare,
    test_proportions_2indep,
)

ALPHA = 0.05
POWER = 0.80


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
    """Two-proportion z-test using the score statistic (statsmodels
    method="score", i.e. the pooled-variance z-test, not the Wald form).
    Returns (z_stat, p_value)."""
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


def archive_wide_significance(packages_df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """One omnibus chi-square-across-arms p-value per test (via
    chi_square_across_arms), then BH-adjusted across the whole set of
    tests -- the archive-wide analogue of the within-test BH correction
    used in pairwise_tests. Tests are the unit of analysis (CLAUDE.md
    convention), so this asks "is there a real difference somewhere among
    this test's arms", not "is any specific pair significant".

    packages_df must have columns: clickability_test_id, clicks, impressions.
    Returns one row per test: clickability_test_id, n_arms, chi2, p_raw, p_bh.
    """
    rows = []
    for test_id, g in packages_df.groupby("clickability_test_id"):
        chi2_stat, p_raw, dof = chi_square_across_arms(g)
        rows.append(
            {
                "clickability_test_id": test_id,
                "n_arms": dof + 1,
                "chi2": chi2_stat,
                "p_raw": p_raw,
            }
        )
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p_raw"], alpha=alpha, method="fdr_bh")[1]
    return out


def split_half_winner_curse(
    packages_df: pd.DataFrame, rng: np.random.Generator, min_impressions: int = 100
) -> pd.DataFrame:
    """Empirical winner's-curse demonstration via random split-half.

    We don't have impression-level (viewer-level) data -- only per-arm
    aggregate clicks/impressions -- so a literal time-based split isn't
    available. Instead, each arm's impressions are split into two random
    halves by hypergeometric thinning: given clicks_k out of impressions_n,
    the number of those clicks landing in a random half-sized subsample of
    impressions follows Hypergeometric(N=impressions_n, K=clicks_k,
    n=impressions_n // 2). This is a valid random split under the same
    exchangeability assumption random assignment already requires (which
    viewer lands in "half A" vs "half B" is arbitrary), and lets selection
    (based on half A) and re-estimation (based on held-out half B) use
    independent data -- the textbook winner's-curse setup.

    For each test with every arm >= min_impressions: split all arms,
    pick the "winner" as the arm with the highest half-A CTR, then compare
    that winner's half-A CTR (the inflated, selection-based estimate) to
    its half-B CTR (an independent re-estimate) and to its true full-sample
    CTR. Returns one row per qualifying test.
    """
    rows = []
    for test_id, g in packages_df.groupby("clickability_test_id"):
        if (g["impressions"] < min_impressions).any():
            continue
        n_half = (g["impressions"] // 2).to_numpy()
        clicks = g["clicks"].to_numpy()
        impressions = g["impressions"].to_numpy()
        clicks_a = rng.hypergeometric(clicks, impressions - clicks, n_half)
        clicks_b = clicks - clicks_a
        impressions_b = impressions - n_half
        ctr_a = clicks_a / n_half
        ctr_b = clicks_b / impressions_b
        ctr_full = clicks / impressions

        winner_idx = np.argmax(ctr_a)
        rows.append(
            {
                "clickability_test_id": test_id,
                "n_arms": len(g),
                "winner_ctr_a": ctr_a[winner_idx],
                "winner_ctr_b": ctr_b[winner_idx],
                "winner_ctr_full": ctr_full[winner_idx],
            }
        )
    out = pd.DataFrame(rows)
    out["curse_gap"] = out["winner_ctr_a"] - out["winner_ctr_b"]
    return out


def simulate_aa_peeking(
    rng: np.random.Generator,
    n_sims: int,
    true_p: float,
    max_n_per_arm: int,
    peek_every: int,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Null A/A peeking simulation: two arms with the *same* true CTR
    (true_p), viewers arriving one at a time up to max_n_per_arm per arm.
    Every `peek_every` viewers per arm, run classical.two_proportion_test
    on the data accumulated so far and check whether p < alpha. Two stopping
    rules are compared per simulated trial:
      - "fixed_horizon": only the final p-value (at max_n_per_arm) counts.
      - "peeking": stop and declare "significant" the first time any peek's
        p-value crosses alpha (informal continuous monitoring, matching
        Upworthy's undocumented stopping rules per CLAUDE.md).
    Since true_p is identical in both arms, every "significant" result
    under either rule is a false positive by construction -- this isolates
    the alpha-inflation effect of peeking from any real effect.

    Returns one row per simulated trial: sim_id, fixed_horizon_reject (bool),
    peeking_reject (bool), n_peeks_to_reject (NaN if peeking never rejected).
    """
    rows = []
    for sim_id in range(n_sims):
        clicks_a = rng.binomial(1, true_p, max_n_per_arm)
        clicks_b = rng.binomial(1, true_p, max_n_per_arm)
        peeking_reject = False
        n_peeks_to_reject = np.nan
        peek_n = 0
        for n in range(peek_every, max_n_per_arm + 1, peek_every):
            peek_n += 1
            ca, cb = clicks_a[:n].sum(), clicks_b[:n].sum()
            if ca == 0 and cb == 0:
                # No clicks in either arm yet: the pooled proportion is 0, the
                # test statistic is undefined, and there is nothing to peek at.
                continue
            _, p = two_proportion_test(ca, n, cb, n)
            if not peeking_reject and p < alpha:
                peeking_reject = True
                n_peeks_to_reject = peek_n

        ca_final, cb_final = clicks_a.sum(), clicks_b.sum()
        _, p_final = two_proportion_test(ca_final, max_n_per_arm, cb_final, max_n_per_arm)
        rows.append(
            {
                "sim_id": sim_id,
                "fixed_horizon_reject": p_final < alpha,
                "peeking_reject": peeking_reject,
                "n_peeks_to_reject": n_peeks_to_reject,
            }
        )
    return pd.DataFrame(rows)


def mde_for_n(
    n_per_arm: float, baseline_p: float = 0.01, alpha: float = ALPHA, power: float = POWER
) -> tuple[float, float]:
    """Minimum detectable effect for a balanced two-proportion z-test with
    n_per_arm impressions per arm, at the given baseline CTR, alpha, and
    power (two-sided).

    Solves for Cohen's h via statsmodels' NormalIndPower (the standard
    two-proportion power formulation), then converts h back to an absolute/
    relative lift around baseline_p using the arcsine-variance-stabilizing
    transform (h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1))), since h alone isn't
    interpretable as a CTR difference without anchoring it to a baseline.

    Returns (mde_abs, mde_rel), lift of the better arm over baseline_p.
    """
    analysis = NormalIndPower()
    h = analysis.solve_power(
        effect_size=None,
        nobs1=n_per_arm,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    phi1 = 2 * np.arcsin(np.sqrt(baseline_p))
    phi2 = phi1 + abs(h)
    p2 = np.sin(phi2 / 2) ** 2
    mde_abs = p2 - baseline_p
    mde_rel = mde_abs / baseline_p
    return mde_abs, mde_rel


def chi_square_across_arms(test_df: pd.DataFrame) -> tuple[float, float, int]:
    """Chi-square test of k independent proportions across all arms of one
    test. Returns (chi2_stat, p_value, dof); dof = n_arms - 1.

    Degenerate case: if every arm has zero clicks, expected clicks are zero
    in every cell too (pooled CTR is 0), so the usual (obs-exp)^2/exp chi2
    statistic is 0/0 -- not a computation failure, but a test with no
    information to distinguish arms at all. Reported as chi2=0.0, p=1.0
    (observed matches expected exactly, trivially) rather than NaN, so it
    doesn't silently poison downstream archive-wide aggregation.
    """
    clicks = test_df["clicks"].to_numpy()
    impressions = test_df["impressions"].to_numpy()
    if clicks.sum() == 0:
        return 0.0, 1.0, len(test_df) - 1
    chi2_stat, p_value, _table = proportions_chisquare(clicks, impressions)
    dof = len(test_df) - 1
    return chi2_stat, p_value, dof
