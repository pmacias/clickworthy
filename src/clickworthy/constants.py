# Single source of truth for fixed seeds used anywhere in this project
# (CLAUDE.md "Everywhere" convention: seeds live in one constants module,
# not scattered literals). Add new SEED_* constants here as new stochastic
# procedures are introduced; never inline a raw seed elsewhere.

SEED_WINNERS_CURSE = 20250301  # Stage 1 Task 4: split-half winner's-curse simulation
SEED_PEEKING = 20250302  # Stage 1 Task 5: A/A peeking simulation
