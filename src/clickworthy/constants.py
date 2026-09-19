# Single source of truth for fixed seeds used anywhere in this project
# (CLAUDE.md "Everywhere" convention: seeds live in one constants module,
# not scattered literals). Add new SEED_* constants here as new stochastic
# procedures are introduced; never inline a raw seed elsewhere.

SEED_WINNERS_CURSE = 20250301  # Stage 1 Task 4: split-half winner's-curse simulation
SEED_PEEKING = 20250302  # Stage 1 Task 5: A/A peeking simulation
SEED_BETA_BINOMIAL_PPC = 20250401  # Stage 2 Task 1: prior/posterior predictive draws
SEED_HIERARCHICAL_SUBSAMPLE = 20250402  # Stage 2 Task 2: ~500-test stratified (by impressions) subsample
SEED_HIERARCHICAL_PRIOR_PRED = 20250403  # Stage 2 Task 2: hierarchical prior predictive draws
SEED_HIERARCHICAL_NUTS = 20250404  # Stage 2 Task 2: PyMC NUTS sampling
SEED_HIERARCHICAL_PPC = 20250405  # Stage 2 Task 2: posterior predictive draws
SEED_FULL_SAMPLE_NUTS = 20250406  # Stage 2 Task 4: PyMC NUTS sampling, full exploratory sample
SEED_FULL_SAMPLE_NUTS_RETRY = 20250407  # Stage 2 Task 4: retry w/ tune=4000, target_accept=0.95
SEED_FULL_SAMPLE_NUTPIE = 20250408  # Stage 2 Task 4: nutpie backend retry, same model spec (unused -- nutpie incompatible w/ py3.11, see CLAUDE.md pitfalls)
SEED_FULL_SAMPLE_NUTS_DRAWS4000 = 20250409  # Stage 2 Task 4: same tune=4000/target_accept=0.95, draws=4000 decisive check
