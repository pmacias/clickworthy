# Shared metrics + programmatic diagnostics checks (divergences, r_hat, ESS thresholds).
#
# Required checks (CLAUDE.md "Bayesian" section): zero divergences, r_hat < 1.01 on every
# parameter, bulk and tail ESS > 400. Checked programmatically here, not eyeballed off a
# trace/rank plot -- a fit failing any of these is a bug to fix before interpretation,
# not a result.

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd

R_HAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400


@dataclass
class DiagnosticsReport:
    """Pass/fail summary of CLAUDE.md's required post-fit diagnostics."""

    n_divergences: int
    max_r_hat: float
    min_ess_bulk: float
    min_ess_tail: float
    failing_params: pd.DataFrame  # empty if none fail
    passed: bool

    def summary(self) -> str:
        lines = [
            f"divergences: {self.n_divergences} "
            f"({'PASS' if self.n_divergences == 0 else 'FAIL'})",
            f"max r_hat: {self.max_r_hat:.4f} "
            f"({'PASS' if self.max_r_hat < R_HAT_THRESHOLD else 'FAIL'})",
            f"min ess_bulk: {self.min_ess_bulk:.0f} "
            f"({'PASS' if self.min_ess_bulk > ESS_THRESHOLD else 'FAIL'})",
            f"min ess_tail: {self.min_ess_tail:.0f} "
            f"({'PASS' if self.min_ess_tail > ESS_THRESHOLD else 'FAIL'})",
            f"overall: {'PASS' if self.passed else 'FAIL'}",
        ]
        return "\n".join(lines)


def check_diagnostics(
    idata: az.InferenceData, var_names: list[str] | None = None
) -> DiagnosticsReport:
    """Run CLAUDE.md's required post-fit diagnostics on a PyMC InferenceData: zero
    divergences, r_hat < 1.01 on every parameter, bulk/tail ESS > 400. `var_names`
    restricts which variables are checked (e.g. excluding large per-arm Deterministics
    not worth summarizing individually); defaults to everything in the posterior group.
    Returns a DiagnosticsReport with pass/fail per check and the specific parameters
    (if any) failing an r_hat or ESS threshold, for debugging."""
    n_divergences = int(idata.sample_stats["diverging"].sum())

    # round_to="none": az.summary's default 2-decimal rounding can round a real r_hat
    # of e.g. 1.0068 up to a displayed 1.01, which would misfire the < 1.01 threshold
    # check below on a value that doesn't actually violate it. Need full precision here.
    summary = az.summary(idata, var_names=var_names, kind="diagnostics", round_to="none")
    max_r_hat = summary["r_hat"].max()
    min_ess_bulk = summary["ess_bulk"].min()
    min_ess_tail = summary["ess_tail"].min()

    failing = summary[
        (summary["r_hat"] >= R_HAT_THRESHOLD)
        | (summary["ess_bulk"] <= ESS_THRESHOLD)
        | (summary["ess_tail"] <= ESS_THRESHOLD)
    ]

    passed = n_divergences == 0 and len(failing) == 0

    return DiagnosticsReport(
        n_divergences=n_divergences,
        max_r_hat=max_r_hat,
        min_ess_bulk=min_ess_bulk,
        min_ess_tail=min_ess_tail,
        failing_params=failing,
        passed=passed,
    )


def divergence_funnel_check(
    idata: az.InferenceData, scale_var: str = "sigma"
) -> pd.DataFrame:
    """Check whether NUTS divergences concentrate at small values of a hierarchical
    scale parameter -- the classic 'funnel' signature of a too-tight prior on that
    scale fighting its true value (non-centered parameterization is supposed to
    remove this, but the check should still be run, not assumed). Returns a table
    comparing the scale parameter's distribution on divergent vs. non-divergent
    draws; a lower mean/median on the divergent rows is the funnel signature."""
    diverging = idata.sample_stats["diverging"].values.flatten()
    scale_draws = idata.posterior[scale_var].values.flatten()

    if diverging.sum() == 0:
        return pd.DataFrame(
            {
                "divergent": [False],
                "n_draws": [len(scale_draws)],
                f"{scale_var}_mean": [scale_draws.mean()],
                f"{scale_var}_median": [np.median(scale_draws)],
            }
        )

    rows = []
    for is_div in [False, True]:
        mask = diverging == is_div
        rows.append(
            {
                "divergent": is_div,
                "n_draws": int(mask.sum()),
                f"{scale_var}_mean": scale_draws[mask].mean(),
                f"{scale_var}_median": np.median(scale_draws[mask]),
            }
        )
    return pd.DataFrame(rows)
