# Portfolio-site figure: naive "declared winners" vs. the hierarchical model's
# partially-pooled view of the same comparison. Built on the Stage 2 Task 4
# settled full-exploratory-sample fit (tune=4000, target_accept=0.95,
# draws=4000) -- read from disk in chunks, never refit here.
#
# Definitions (fixed before looking at results):
#   - Per test, the "naive winner" and "runner-up" are the top two arms by raw
#     CTR (clicks / impressions).
#   - Naive verdict: classical.two_proportion_test (score z-test), unadjusted
#     p < 0.05 -> "declared winner". BH across all tests' top-two comparisons
#     is reported alongside for context.
#   - Pooled verdict: from posterior p_arm draws, P(p_winner > p_runner_up) and
#     relative lift p_winner / p_runner_up - 1 (posterior median, equal-tailed
#     95% credible interval). "Still clear" = P >= 0.95.

import gc

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from clickworthy.classical import ALPHA, lift, two_proportion_test

CLEAR_THRESHOLD = 0.95


def top_two_by_ctr(packages_df: pd.DataFrame) -> pd.DataFrame:
    """One row per test: its top two arms by raw CTR, with the naive verdict.

    `packages_df` must be in the same row order the model was fit on (the
    `arm_idx` columns returned index into it, and so into p_arm_dim_0) and
    have columns clickability_test_id, headline, impressions, clicks.
    Ties in CTR are broken by row order (stable sort)."""
    df = packages_df.reset_index(drop=True).copy()
    df["arm_idx"] = np.arange(len(df))
    df["ctr"] = df["clicks"] / df["impressions"]
    df = df.sort_values(["clickability_test_id", "ctr"], ascending=[True, False], kind="stable")

    rows = []
    for test_id, g in df.groupby("clickability_test_id", sort=True):
        w, r = g.iloc[0], g.iloc[1]
        _, p_raw = two_proportion_test(w["clicks"], w["impressions"], r["clicks"], r["impressions"])
        _, rel = lift(w["clicks"], w["impressions"], r["clicks"], r["impressions"])
        rows.append(
            {
                "clickability_test_id": test_id,
                "n_arms": len(g),
                "w_arm_idx": int(w["arm_idx"]),
                "r_arm_idx": int(r["arm_idx"]),
                "w_headline": w["headline"],
                "r_headline": r["headline"],
                "w_impressions": int(w["impressions"]),
                "r_impressions": int(r["impressions"]),
                "w_clicks": int(w["clicks"]),
                "r_clicks": int(r["clicks"]),
                "w_ctr": w["ctr"],
                "r_ctr": r["ctr"],
                "naive_lift": rel,
                "p_raw": p_raw,
            }
        )
    out = pd.DataFrame(rows)
    # A test whose top two arms both have zero clicks has an undefined score
    # statistic (NaN p); it is trivially not a declared winner.
    out["p_raw"] = out["p_raw"].fillna(1.0)
    out["p_bh"] = multipletests(out["p_raw"], alpha=ALPHA, method="fdr_bh")[1]
    out["declared"] = out["p_raw"] < ALPHA
    out["declared_bh"] = out["p_bh"] < ALPHA
    return out


def pooled_top_two(
    netcdf_path: str, w_arm_idx: np.ndarray, r_arm_idx: np.ndarray, chunk_size: int = 2304
) -> pd.DataFrame:
    """Posterior comparison of each test's winner vs. runner-up arm, read from
    a saved fit's p_arm in chunks along p_arm_dim_0 (never the whole ~2GB
    array at once -- CLAUDE.md's 16GB-machine convention).

    Returns one row per (w, r) pair, in input order: p_better = P(p_w > p_r),
    pooled_lift_median / _low / _high (relative lift p_w / p_r - 1, posterior
    median and equal-tailed 95% interval), and the 95% intervals for each
    arm's p (for plotting)."""
    import xarray as xr

    w_arm_idx, r_arm_idx = np.asarray(w_arm_idx), np.asarray(r_arm_idx)
    needed = np.union1d(w_arm_idx, r_arm_idx)
    draws = {}  # arm_idx -> flattened (chain * draw,) p_arm draws, needed arms only

    with xr.open_dataset(netcdf_path, group="posterior", engine="h5netcdf") as ds:
        n_arms = ds.sizes["p_arm_dim_0"]
        for start in range(0, n_arms, chunk_size):
            stop = min(start + chunk_size, n_arms)
            want = needed[(needed >= start) & (needed < stop)]
            if len(want) == 0:
                continue
            block = ds["p_arm"].isel(p_arm_dim_0=slice(start, stop)).values  # (chain, draw, k)
            for a in want:
                draws[int(a)] = block[:, :, a - start].reshape(-1).astype(np.float32)
            del block
            gc.collect()

    rows = []
    for w, r in zip(w_arm_idx, r_arm_idx):
        pw, pr = draws[int(w)].astype(np.float64), draws[int(r)].astype(np.float64)
        rel = pw / pr - 1
        lo, med, hi = np.percentile(rel, [2.5, 50, 97.5])
        rows.append(
            {
                "p_better": float((pw > pr).mean()),
                "pooled_lift_median": med,
                "pooled_lift_low": lo,
                "pooled_lift_high": hi,
                "w_p_low": np.percentile(pw, 2.5),
                "w_p_median": np.median(pw),
                "w_p_high": np.percentile(pw, 97.5),
                "r_p_low": np.percentile(pr, 2.5),
                "r_p_median": np.median(pr),
                "r_p_high": np.percentile(pr, 97.5),
            }
        )
    return pd.DataFrame(rows)


def exemplar_candidates(
    comparison: pd.DataFrame, max_headline_chars: int = 90, n: int = 5
) -> tuple[pd.DataFrame, float]:
    """Fixed selection rule for the figure's exemplar test (the "typical
    declared winner"): declared winner naively (p_raw < 0.05), any pooled
    verdict; winner and runner-up have DIFFERENT headline text, both under
    `max_headline_chars`; ranked by how close the pooled/naive lift ratio is
    to the median ratio over ALL declared winners (computed before the
    headline filters). Returns (top-n candidates, that median ratio)."""
    declared = comparison[comparison["declared"]].copy()
    declared["lift_ratio"] = declared["pooled_lift_median"] / declared["naive_lift"]
    median_ratio = declared["lift_ratio"].median()

    grp = declared[
        (declared["w_headline"] != declared["r_headline"])
        & (declared["w_headline"].str.len() < max_headline_chars)
        & (declared["r_headline"].str.len() < max_headline_chars)
    ].copy()
    grp["dist_to_median_ratio"] = (grp["lift_ratio"] - median_ratio).abs()
    return grp.sort_values("dist_to_median_ratio").head(n), median_ratio


# --- Figure -----------------------------------------------------------------
# Colors match the portfolio site (navy surface). Coral marks the pooled /
# model-based estimates; muted blue-grey marks the raw / naive ones.

SITE_RC = {
    "figure.facecolor": "#284a72", "axes.facecolor": "#284a72", "savefig.facecolor": "#284a72",
    "text.color": "#f3f6fa", "axes.labelcolor": "#c9d6e6",
    "xtick.color": "#8ea6c3", "ytick.color": "#8ea6c3",
    "axes.edgecolor": "#8ea6c3", "grid.color": "#3d6290",
}
ACCENT = "#ff9273"  # pooled
MUTED = "#a9b8cc"  # raw / naive
INK = "#f3f6fa"
INK_SECONDARY = "#c9d6e6"


def _fmt_p(p: float) -> str:
    """p-value as plain text: 'p = 0.007', or 'p = 2 × 10⁻⁶' when tiny."""
    if p >= 0.001:
        return f"p = {p:.3f}"
    mant, exp = f"{p:.0e}".split("e")
    sup = str(int(exp)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"p = {mant} × 10{sup}"


def _fmt_p_better(p: float, n_draws: int) -> str:
    """P(better) with the Monte Carlo floor made explicit: with n_draws draws,
    a value of 1.0 only means 'above 1 - 1/n_draws'."""
    if p >= 1 - 1 / n_draws:
        return "P(better) > 0.999"
    return f"P(better) = {p:.3f}"


def plot_site_figure(exemplar: pd.Series, declared: pd.DataFrame, n_draws: int):
    """Two-panel site figure. `exemplar` is one row of the comparison frame
    (top_two_by_ctr joined with pooled_top_two, plus lift_ratio); `declared`
    is every declared-winner row. Returns the matplotlib Figure."""
    import textwrap

    import matplotlib.pyplot as plt

    from clickworthy.classical import wilson_ci

    with plt.rc_context({**SITE_RC, "font.size": 11}):
        fig = plt.figure(figsize=(8, 6), dpi=200)
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.28,
                              left=0.04, right=0.97, top=0.79, bottom=0.10)
        ax_l, ax_r = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

        fig.suptitle("Winning versions won by about a quarter less than they seemed",
                     x=0.04, ha="left", y=0.965, fontsize=14, fontweight="bold")
        fig.text(0.04, 0.91,
                 "Naive A/B read vs. a hierarchical model pooling 3,823 Upworthy tests",
                 ha="left", fontsize=11, color=INK_SECONDARY)

        # Left: the exemplar's two headlines, raw vs. pooled 95% intervals.
        rows = [
            ("w", exemplar["w_headline"], exemplar["w_clicks"], exemplar["w_impressions"]),
            ("r", exemplar["r_headline"], exemplar["r_clicks"], exemplar["r_impressions"]),
        ]
        y_raw, y_pool, y_head = [3.55, 0.55], [3.05, 0.05], [4.1, 1.1]
        for i, (k, headline, clicks, impressions) in enumerate(rows):
            lo, hi = wilson_ci(clicks, impressions)
            ctr = clicks / impressions
            ax_l.plot([lo * 100, hi * 100], [y_raw[i]] * 2, color=MUTED, lw=2.5,
                      solid_capstyle="round")
            ax_l.plot(ctr * 100, y_raw[i], "o", color=MUTED, ms=7)
            ax_l.plot([exemplar[f"{k}_p_low"] * 100, exemplar[f"{k}_p_high"] * 100],
                      [y_pool[i]] * 2, color=ACCENT, lw=2.5, solid_capstyle="round")
            ax_l.plot(exemplar[f"{k}_p_median"] * 100, y_pool[i], "o", color=ACCENT, ms=7)
            label = "Winner" if k == "w" else "Runner-up"
            ax_l.text(0.0, y_head[i], f"{label} · {clicks:,} clicks / {impressions:,} views\n"
                      + "\n".join(textwrap.wrap(f"“{headline}”", 46)),
                      transform=ax_l.get_yaxis_transform(), va="bottom", ha="left",
                      fontsize=10.5, color=INK, linespacing=1.25)

        ax_l.set_ylim(-0.5, 6.9)
        ax_l.set_xlim(0.4, 2.4)
        ax_l.set_yticks([])
        ax_l.set_xlabel("Click-through rate (%)")
        ax_l.grid(axis="x", lw=0.6)
        for side in ("top", "right", "left"):
            ax_l.spines[side].set_visible(False)

        # Interval-type labels to the right of the runner-up's (lower-CTR) bars.
        x_lab = max(exemplar["r_p_high"],
                    wilson_ci(exemplar["r_clicks"], exemplar["r_impressions"])[1]) * 100 + 0.08
        ax_l.text(x_lab, y_raw[1], "raw CTR (95% interval)", va="center", fontsize=10,
                  color=INK_SECONDARY)
        ax_l.text(x_lab, y_pool[1], "after pooling (95% credible)", va="center", fontsize=10,
                  color=INK_SECONDARY)

        # Verdict annotations: colored swatch + ink text, top of panel.
        verdicts = [
            (MUTED, f"Naive: lift {exemplar['naive_lift']:+.0%}, {_fmt_p(exemplar['p_raw'])}"),
            (ACCENT, f"Pooled: lift {exemplar['pooled_lift_median']:+.0%}, "
                     f"{_fmt_p_better(exemplar['p_better'], n_draws)}"),
        ]
        for j, (color, txt) in enumerate(verdicts):
            yy = 0.97 - 0.065 * j
            ax_l.plot([0.0, 0.05], [yy] * 2, transform=ax_l.transAxes, color=color,
                      lw=3, clip_on=False, solid_capstyle="round")
            ax_l.text(0.07, yy, txt, transform=ax_l.transAxes, va="center", fontsize=10.5,
                      color=INK)

        # Right: every declared winner, naive vs. pooled lift.
        is_ex = declared["clickability_test_id"] == exemplar["clickability_test_id"]
        ax_r.scatter(declared.loc[~is_ex, "naive_lift"] * 100,
                     declared.loc[~is_ex, "pooled_lift_median"] * 100,
                     s=12, color=MUTED, alpha=0.55, linewidths=0)
        ax_r.scatter(exemplar["naive_lift"] * 100, exemplar["pooled_lift_median"] * 100,
                     s=70, color=ACCENT, edgecolors="#284a72", linewidths=1.5, zorder=3)
        ax_r.annotate("example\nat left",
                      (exemplar["naive_lift"] * 100, exemplar["pooled_lift_median"] * 100),
                      xytext=(18, 26), textcoords="offset points", fontsize=10, color=INK,
                      arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, lw=0.8))
        lim = 340
        ax_r.plot([0, lim], [0, lim], ls="--", color=INK_SECONDARY, lw=1)
        ax_r.text(0.97, 0.04, "dashed: y = x (no shrinkage)", transform=ax_r.transAxes,
                  ha="right", va="bottom", fontsize=10, color=INK_SECONDARY)
        ax_r.set_xlim(0, lim)
        ax_r.set_ylim(0, 220)
        ax_r.set_xlabel("Naive lift, raw CTR (%)")
        ax_r.set_ylabel("Pooled lift, posterior median (%)")
        ax_r.grid(lw=0.6)
        for side in ("top", "right"):
            ax_r.spines[side].set_visible(False)
        med_naive = declared["naive_lift"].median()
        med_pool = declared["pooled_lift_median"].median()
        ax_r.set_title(f"All {len(declared)} declared winners (p < 0.05)\n"
                       f"median lift {med_naive:+.0%} → {med_pool:+.0%}",
                       loc="left", fontsize=11, color=INK, pad=8, linespacing=1.4)
    return fig
