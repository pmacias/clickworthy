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


def check_diagnostics_from_netcdf(
    path: str, chunk_sizes: dict[str, int]
) -> DiagnosticsReport:
    """Chunked, disk-backed version of check_diagnostics for fits too large to hold fully
    in memory alongside other objects (e.g. a full-exploratory-sample fit's ~7GB posterior
    on a 16GB machine -- see CLAUDE.md's Bayesian conventions). Reads `path` (an
    idata.to_netcdf() file) via xarray + h5netcdf, processing each named variable in
    `chunk_sizes` (var_name -> chunk size along its `<var>_dim_0` dimension) in slices
    rather than loading the whole array at once. A variable with no `_dim_0` dimension
    (e.g. `sigma`, a scalar) is loaded whole regardless of chunk_sizes -- it's small at any
    archive scale -- and contributes its r_hat and bulk/tail ESS to the overall max/min
    exactly like every chunked variable. `failing_params` is left empty (not worth the
    memory cost of building a full per-parameter table this way); use
    `compare_params_from_netcdf` to inspect specific parameters if failures need
    identifying."""
    import gc

    import xarray as xr

    with xr.open_dataset(path, group="sample_stats", engine="h5netcdf") as ss:
        n_divergences = int(ss["diverging"].values.sum())

    max_r_hat = -np.inf
    min_ess_bulk, min_ess_tail = np.inf, np.inf

    with xr.open_dataset(path, group="posterior", engine="h5netcdf") as ds:
        for var in ds.data_vars:
            dim = f"{var}_dim_0"
            if dim not in ds[var].dims:
                sub = ds[var].load()
                max_r_hat = max(max_r_hat, az.rhat(sub)[var].item())
                min_ess_bulk = min(min_ess_bulk, az.ess(sub, method="bulk")[var].item())
                min_ess_tail = min(min_ess_tail, az.ess(sub, method="tail")[var].item())
                continue
            chunk_size = chunk_sizes.get(var, ds.sizes[dim])
            for start in range(0, ds.sizes[dim], chunk_size):
                idxs = list(range(start, min(start + chunk_size, ds.sizes[dim])))
                sub = ds[var].isel({dim: idxs}).load()
                max_r_hat = max(max_r_hat, az.rhat(sub)[var].values.max())
                min_ess_bulk = min(min_ess_bulk, az.ess(sub, method="bulk")[var].values.min())
                min_ess_tail = min(min_ess_tail, az.ess(sub, method="tail")[var].values.min())
                del sub
                gc.collect()

    passed = (
        n_divergences == 0
        and max_r_hat < R_HAT_THRESHOLD
        and min_ess_bulk > ESS_THRESHOLD
        and min_ess_tail > ESS_THRESHOLD
    )
    return DiagnosticsReport(
        n_divergences=n_divergences,
        max_r_hat=max_r_hat,
        min_ess_bulk=min_ess_bulk,
        min_ess_tail=min_ess_tail,
        failing_params=pd.DataFrame(),
        passed=passed,
    )


def compare_params_from_netcdf(
    paths_labels: list[tuple[str, str]], var_idx_map: dict[str, list[int]]
) -> pd.DataFrame:
    """Lazy, disk-backed r_hat/ESS comparison of specific named parameters across multiple
    saved fits (idata.to_netcdf() files) -- e.g. "does this parameter's diagnostic improve
    across configurations", without ever loading either fit's full posterior into memory
    (CLAUDE.md's Bayesian conventions: comparisons across saved fits use chunked/lazy reads,
    not multiple full InferenceData objects held at once). `var_idx_map` maps variable name
    -> list of indices along its `<var>_dim_0` dimension. `paths_labels` is a list of
    (path, label) pairs, one row-group per fit."""
    import xarray as xr

    rows = []
    for path, label in paths_labels:
        with xr.open_dataset(path, group="posterior", engine="h5netcdf") as ds:
            for var, idxs in var_idx_map.items():
                sub = ds[var].isel({f"{var}_dim_0": idxs}).load()
                for i, idx in enumerate(idxs):
                    one = sub.isel({f"{var}_dim_0": i})
                    rows.append(
                        {
                            "attempt": label,
                            "param": f"{var}[{idx}]",
                            "r_hat": az.rhat(one)[var].item(),
                            "ess_bulk": az.ess(one, method="bulk")[var].item(),
                            "ess_tail": az.ess(one, method="tail")[var].item(),
                        }
                    )
    return pd.DataFrame(rows).set_index("param")


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
