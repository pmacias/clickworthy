# Stage 2: Bayesian model builders.
#
# Two models live here:
#   - Task 1 (pedagogical warm-up): a per-arm Beta-Binomial model, fit
#     analytically via conjugacy -- no PyMC/MCMC needed, since the point is
#     to show the closed-form case before Task 2 introduces sampling.
#   - Tasks 2-4: the hierarchical binomial-logit model (prior predictive
#     simulator + PyMC builder), used on the ~500-test subsample and then
#     the full exploratory sample.
#
# Conventions (CLAUDE.md "Bayesian" section):
#   - Likelihood is on raw counts: clicks ~ Binomial(impressions, p).
#   - Priors are weakly informative, never flat, and stated explicitly.
#   - Fixed seeds for anything stochastic (see constants.py).

from dataclasses import dataclass

import numpy as np
from scipy import stats

from clickworthy.constants import SEED_BETA_BINOMIAL_PPC, SEED_HIERARCHICAL_PRIOR_PRED


@dataclass
class BetaBinomialPosterior:
    """Conjugate Beta-Binomial posterior for a single arm."""

    prior_alpha: float
    prior_beta: float
    clicks: int
    impressions: int

    @property
    def post_alpha(self) -> float:
        return self.prior_alpha + self.clicks

    @property
    def post_beta(self) -> float:
        return self.prior_beta + (self.impressions - self.clicks)

    @property
    def mean(self) -> float:
        return self.post_alpha / (self.post_alpha + self.post_beta)

    def credible_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """Equal-tailed posterior credible interval for p."""
        low, high = stats.beta.ppf(
            [alpha / 2, 1 - alpha / 2], self.post_alpha, self.post_beta
        )
        return low, high


def fit_beta_binomial(
    clicks: int, impressions: int, prior_alpha: float, prior_beta: float
) -> BetaBinomialPosterior:
    """Analytic Beta-Binomial posterior for one arm: Beta(prior_alpha,
    prior_beta) prior, Binomial(impressions, p) likelihood -> Beta(
    prior_alpha + clicks, prior_beta + impressions - clicks) posterior."""
    return BetaBinomialPosterior(prior_alpha, prior_beta, clicks, impressions)


def sample_prior_predictive(
    prior_alpha: float,
    prior_beta: float,
    impressions: int,
    n_draws: int = 2000,
    seed: int = SEED_BETA_BINOMIAL_PPC,
) -> np.ndarray:
    """Simulate fake click counts from the prior alone: draw p ~ Beta(prior),
    then clicks ~ Binomial(impressions, p). Returns an array of simulated
    click counts, length n_draws."""
    rng = np.random.default_rng(seed)
    p_draws = rng.beta(prior_alpha, prior_beta, size=n_draws)
    return rng.binomial(impressions, p_draws)


def sample_posterior_predictive(
    posterior: BetaBinomialPosterior,
    n_draws: int = 2000,
    seed: int = SEED_BETA_BINOMIAL_PPC,
) -> np.ndarray:
    """Simulate fake click counts from the posterior predictive: draw
    p ~ Beta(posterior), then clicks ~ Binomial(impressions, p)."""
    rng = np.random.default_rng(seed)
    p_draws = rng.beta(posterior.post_alpha, posterior.post_beta, size=n_draws)
    return rng.binomial(posterior.impressions, p_draws)


# Task 2 onward: hierarchical model. A NumPy prior-predictive simulator first
# (so the prior can be checked without building the PyMC graph), then the PyMC
# model builder itself. Both share the structure below; the notebook supplies
# the concrete prior constants (MU_MEAN, MU_SD, SIGMA_SCALE).
#
# Structure (CLAUDE.md "Bayesian" section):
#   clicks_arm ~ Binomial(impressions_arm, p_arm)
#   logit(p_arm) = mu_test + delta_arm
#   mu_test ~ Normal(mu_mean, mu_sd), independent per test (deliberately UNPOOLED)
#   delta_arm = sigma * z_arm, z_arm ~ Normal(0, 1)      (non-centered)
#   sigma ~ HalfNormal(sigma_scale), single GLOBAL value shared by every arm


def sample_hierarchical_prior_predictive(
    test_idx: np.ndarray,
    impressions: np.ndarray,
    mu_mean: float,
    mu_sd: float,
    sigma_scale: float,
    n_worlds: int = 2000,
    seed: int = SEED_HIERARCHICAL_PRIOR_PRED,
) -> np.ndarray:
    """Simulate `n_worlds` prior-predictive datasets for the hierarchical
    model above. `test_idx` is a 0..n_tests-1 integer array (one entry per
    arm) mapping each arm to its test; `impressions` is the matching
    per-arm impressions array.

    Each simulated world draws one independent mu_test per distinct test
    (unpooled), one GLOBAL sigma, and one z_arm per arm (non-centered:
    delta_arm = sigma * z_arm), then clicks_arm ~ Binomial(impressions_arm,
    p_arm). Returns an array of shape (n_worlds, n_arms) of simulated click
    counts."""
    rng = np.random.default_rng(seed)
    n_tests = test_idx.max() + 1
    n_arms = len(impressions)

    mu_test_draws = rng.normal(mu_mean, mu_sd, size=(n_worlds, n_tests))
    sigma_draws = np.abs(rng.normal(0.0, sigma_scale, size=n_worlds))  # HalfNormal
    z_draws = rng.normal(0.0, 1.0, size=(n_worlds, n_arms))

    mu_per_arm = mu_test_draws[:, test_idx]
    delta_arm = sigma_draws[:, None] * z_draws
    p_arm = 1.0 / (1.0 + np.exp(-(mu_per_arm + delta_arm)))
    return rng.binomial(impressions[None, :], p_arm)


def build_hierarchical_model(
    test_idx: np.ndarray,
    impressions: np.ndarray,
    clicks: np.ndarray,
    mu_mean: float,
    mu_sd: float,
    sigma_scale: float,
) -> "pm.Model":
    """PyMC build of the hierarchical model specified and prior-predictive-checked
    in the notebook: independent per-test mu_test (unpooled), single global sigma,
    non-centered delta_arm. `test_idx` is a 0..n_tests-1 integer array (one entry
    per arm) mapping each arm to its test; `impressions`/`clicks` are the matching
    per-arm observed data."""
    import pymc as pm

    n_tests = int(test_idx.max()) + 1
    n_arms = len(impressions)

    with pm.Model() as model:
        mu_test = pm.Normal("mu_test", mu=mu_mean, sigma=mu_sd, shape=n_tests)
        sigma = pm.HalfNormal("sigma", sigma=sigma_scale)
        z_arm = pm.Normal("z_arm", mu=0.0, sigma=1.0, shape=n_arms)
        delta_arm = pm.Deterministic("delta_arm", sigma * z_arm)
        p_arm = pm.Deterministic(
            "p_arm", pm.math.invlogit(mu_test[test_idx] + delta_arm)
        )
        pm.Binomial("clicks_obs", n=impressions, p=p_arm, observed=clicks)

    return model
