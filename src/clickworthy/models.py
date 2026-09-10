# Stage 2: PyMC model builders (hierarchical models start in Task 2).
#
# Task 1 (pedagogical warm-up, this file's only content so far): a
# per-arm Beta-Binomial model, fit analytically via conjugacy -- no
# PyMC/MCMC needed here, since the point is to show the closed-form
# case before Task 2 introduces sampling.
#
# Conventions (CLAUDE.md "Bayesian" section):
#   - Likelihood is on raw counts: clicks ~ Binomial(impressions, p).
#   - Priors are weakly informative, never flat, and stated explicitly.
#   - Fixed seeds for anything stochastic (see constants.py).

from dataclasses import dataclass

import numpy as np
from scipy import stats

from clickworthy.constants import SEED_BETA_BINOMIAL_PPC


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
