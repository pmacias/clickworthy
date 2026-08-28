# CLAUDE.md

This file guides AI coding assistants working in this repo; the architecture and statistical-convention notes below are equally useful background for any human contributor.

## Project: clickworthy

Statistical analysis of the Upworthy Research Archive — the publicly-available exploratory sample of 4,873 real headline A/B tests (22,666 packages/arms) run by Upworthy from Jan 2013 to Apr 2015. (Note: the archive's full scale, including a separately-gated confirmatory sample, is 32,487 tests total — see Data section for why we work from the exploratory sample only.) The project demonstrates, in order of priority: (1) classical experiment analysis, (2) hierarchical Bayesian meta-analysis in PyMC, (3) causal inference on observational data benchmarked against randomized ground truth, (4) text mining of headlines.

See `PLAN.md` for the staged roadmap. **Do not start work on a later stage until the current stage's acceptance criteria in PLAN.md are met and confirmed by Phil.**

## Working style (read this first)

- Work in small, verifiable steps. After any change to `src/`, run the relevant test or a quick sanity check before moving on. Do not batch large multi-file changes without checkpoints.
- If you are uncertain about a statistical choice, a column meaning, or an API, **stop and ask** rather than guessing. A wrong statistical convention silently propagated through notebooks is the most expensive failure mode in this repo.
- Never invent column names or dataset facts. Inspect the actual schema (`duckdb` query or `pd.read_csv(..., nrows=5)`) before writing code against it.
- Prefer established library implementations over hand-rolled statistics: `statsmodels`/`scipy` for classical tests and intervals, `PyMC` for Bayesian models, `scikit-learn` for propensity models. Do not implement MCMC, test statistics, or variance formulas from scratch unless the notebook's explicit purpose is pedagogical derivation.
- Commits are done manually by Phil in his own terminal, except for complex multi-file changes where he explicitly delegates.
- Paste code changes as targeted cells or diffs, not full-file regeneration, when working conversationally.

## Repo layout

```
clickworthy/
├── CLAUDE.md
├── PLAN.md
├── README.md
├── pyproject.toml          # package name: clickworthy
├── src/clickworthy/
│   ├── __init__.py
│   ├── data_access.py      # DuckDB layer: load CSVs, filtered views
│   ├── classical.py        # Stage 1: tests, CIs, power (thin wrappers + conventions)
│   ├── models.py           # Stage 2: PyMC model builders
│   ├── causal.py           # Stage 3: confounding construction + estimators
│   └── evaluation.py       # shared metrics, diagnostics checks
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_classical_ab.ipynb
│   ├── 03_hierarchical_bayes.ipynb
│   ├── 04_mixture_effects.ipynb
│   ├── 05_causal_benchmark.ipynb
│   └── 06_results_summary.ipynb
└── data/ -> NOT in repo; see below
```

Notebooks are numbered in dependency order. Shared logic lives in `src/clickworthy/`; notebooks import from the package rather than duplicating functions. Each notebook should run top-to-bottom cleanly from a fresh kernel.

## Data

- Raw archive CSV lives at `~/ml_datasets/clickworthy/` (outside Dropbox and Time Machine). Never commit data; `.gitignore` covers `*.csv`, `*.duckdb`, `data/`.
- Source: Upworthy Research Archive, OSF `https://osf.io/jd64p/`. Actual file in use: `upworthy-archive-exploratory-packages-03.12.2020.csv` (~13.4MB, a March 2020 snapshot — see version note below).
- **The confirmatory dataset is NOT freely downloadable.** Per the archive's own documentation, it is shared only with researchers whose analysis plans have been peer reviewed — it is not a file sitting alongside the exploratory data on OSF. PLAN.md's Stage 4c (confirmatory validation) needs to be revisited in light of this: either pursue the archive's access-request process, or reframe Stage 4c as validation against a held-out split constructed from the exploratory data itself. **Do not assume a confirmatory file exists or attempt to locate/download one — flag Phil instead.**
- The exploratory dataset (per the archive's own docs) contains **22,666 packages from 4,873 tests**. Verify this against the actual row/test counts on ingest — a mismatch means something is wrong with the file, not the documentation.

### Schema facts (verified against the archive's own documentation — see `about-the-archive.md` in the `natematias/upworthy-archive` GitHub repo)

Actual columns in the file in hand: `created_at`, `updated_at`, `clickability_test_id`, `excerpt`, `headline`, `lede`, `slug`, `eyecatcher_id`, `impressions`, `clicks`, `significance`, `first_place`, `winner`, `share_text`, `square`, `test_week`.

- Each row = one **package** (an arm: a headline/image bundle) of one **test**. Packages in the same test share `clickability_test_id`; viewers were randomly assigned to packages within a test.
- `impressions`: viewers assigned to that package. `clicks`: how many of those clicked. CTR = clicks / impressions. Baseline CTRs are low (order 1%), so treat proportions carefully — normal approximations can be poor for small arms.
- `significance` and `first_place` are **not** statistical significance in our sense — per the archive maintainers, the exact method Upworthy used to generate `significance` was never recovered from former staff, and both columns were shown to editors only to guide which package to select as `winner`. **Do not use these as a proxy for statistical significance in Stage 1 — compute it ourselves from `impressions`/`clicks`.**
- `winner`: whether editors selected this package for the live site after the test concluded. Note this introduces the same selection-effect logic as → winner's curse (below) if used carelessly as a "ground truth" for "did this package perform best."
- **No `problem`/randomization-flag column exists in this file version.** The current archive (per a June 2024 update, not reflected in our March 2020 snapshot) added a `problem` column, set to 1 for every package in a test where a randomization defect was found, concentrated entirely in tests run **June 25, 2013 – January 10, 2014**. Since we don't have that column, apply the equivalent **date-based exclusion** directly on `created_at` (see Standard filters below). This is a documented, deliberate substitution, not a guess — the docs state the column and the date range are defined identically for the affected tests.

### Standard filters (apply via the DuckDB views, not ad hoc per notebook)

1. Drop packages/tests with `created_at` between 2013-06-25 and 2014-01-10, inclusive — stands in for the `problem` column absent from this file version (see schema note above).
2. Drop degenerate tests: fewer than 2 packages, or any package with 0 impressions.

Every notebook states which view it uses. Exploratory dips outside these filters are fine only in `02_early_exploration`-style contexts and must be labeled as such.

## Statistical conventions (non-negotiable defaults)

These are fixed so results are consistent across notebooks. Deviations require a stated reason in the notebook.

**Classical (Stage 1)**
- Unit of analysis: the experiment (`clickability_test_id`). Within a test, arms are compared pairwise or jointly (chi-square across arms).
- CIs for single proportions: Wilson. CIs for differences/ratios of proportions: Newcombe / delta-method via `statsmodels`; state which.
- Multiple comparisons across arms within a test: Benjamini–Hochberg. Across the whole archive: report raw and BH-adjusted side by side — the contrast is part of the story.
- Power analysis uses the two-proportion formulation at the observed baseline CTR; state alpha, power, and MDE explicitly in every power calc.

**Bayesian (Stage 2)**
- Likelihood: `clicks_arm ~ Binomial(impressions_arm, p_arm)` — model raw counts, never pre-computed CTRs.
- Link: `logit(p_arm) = mu_test + delta_arm`, with `delta` partially pooled across tests. Use the **non-centered parameterization** for all hierarchical scale parameters.
- Weakly informative priors, stated in the notebook. Do not use flat priors.
- Required diagnostics after every fit, checked programmatically via `evaluation.py`, not eyeballed: zero divergences, `r_hat < 1.01` on all parameters, bulk and tail ESS > 400. A fit failing any of these is not a result — it's a bug to fix before interpretation.
- Prior predictive checks before fitting; posterior predictive checks after. Both get a plot.
- Develop on a subsample (~500 tests) first; scale to the full exploratory sample only once the subsample model is clean. If full-scale NUTS is too slow, discuss options with Phil (nutpie, ADVI as a check, minibatching) — do not silently switch inference methods.

**Causal (Stage 3)**
- The confounding in `05_causal_benchmark` is **constructed by us** from randomized data, so ground truth is known. The construction (selection mechanism, which covariates drive it) must be implemented in `causal.py` with a fixed seed and documented in prose — it is the scientific core of the stage, not plumbing.
- Report every estimator (naive difference, regression adjustment, IPW, matching) against the same ground-truth estimand, with bias and coverage. Never report an observational estimate without its experimental benchmark next to it.

**Everywhere**
- Fixed seeds for anything stochastic; seeds live in one constants module, not scattered literals.
- Plots are produced by Phil's own runs; generate plotting code, don't describe imagined figures. Label axes with units. No default matplotlib titles like "Figure 1".

## Environment

- Single env named `clickworthy` (native arm64), created via **Homebrew's Miniforge/mamba**, not the machine's existing Anaconda install. Root prefix: `/opt/homebrew/Caskroom/miniforge/base`. Env lives at `/opt/homebrew/Caskroom/miniforge/base/envs/clickworthy`.
- Core deps: python ≥3.11, pandas, duckdb, statsmodels, scipy, pymc, arviz, scikit-learn, matplotlib, jupyter. No GPU/MPS requirement — PyMC runs CPU here.
- macOS. No Rosetta env needed for this project (unlike flashpoint).

### Why Miniforge/mamba instead of the machine's existing Anaconda

This machine's Anaconda install is conda **4.11.0** — old enough to lack the `solver` config key and the modern libmamba solver entirely. Solving a PyMC-stack environment (pymc/numba/llvmlite have a heavy conda-forge dependency graph) with that installation's classic solver hung indefinitely (30+ minutes, unresponsive to Ctrl-C, required `kill -9`) on more than one attempt. Homebrew's Miniforge ships `mamba` — a fast, modern, C++-based solver — out of the box, which resolves the same environment in low minutes. **Do not attempt to recreate or modify this env using the machine's `anaconda3` conda; use `mamba` exclusively for this project.**

### Setup (manual — outside any agentic layer)

Environment creation is a **manual prerequisite performed by the user before invoking Claude Code**, not something automated by Stage 0 or any other agent workflow. Do this once:

```bash
brew install miniforge          # one-time, if not already installed
mamba shell init --shell zsh --root-prefix=/opt/homebrew/Caskroom/miniforge/base
# open a new terminal after this
mamba create -n clickworthy -c conda-forge python=3.11 pandas duckdb statsmodels scipy pymc arviz scikit-learn matplotlib jupyter
mamba activate clickworthy
pip install -e . --no-deps
```

`--no-deps` on the pip install is required — without it, pip will try to re-resolve pymc/numba/llvmlite itself and may attempt to compile llvmlite from source (which fails on this machine without a system LLVM/CMake setup). All heavy/compiled dependencies come from conda-forge via mamba; pip installs only the `clickworthy` package itself.

### Before running Claude Code (or any bash tooling) in this repo

**Activate the env in your terminal *before* launching `claude`:**
```bash
mamba activate clickworthy
claude
```
Claude Code's bash subshells inherit the parent shell's environment, so activating first means every tool call already has the right `python`/`pip` on `PATH` — no per-command activation needed inside the session.

**Known gotcha — `conda activate` is unreliable in this shell, use `mamba activate`:** the machine's `.zshrc` has an old Anaconda `conda init` block that takes precedence, so a bare `conda activate clickworthy` may silently activate nothing or resolve against the wrong installation. Always use `mamba activate clickworthy` for this project. If a command inside a Claude Code session ever needs to check the active environment, `which python` should resolve to a path under `/opt/homebrew/Caskroom/miniforge/base/envs/clickworthy/`, not `~/opt/anaconda3/...`. If it doesn't, stop and flag it rather than proceeding.

**Do not run `conda create`, `conda install`, or any env-modifying command against this project inside a Claude Code session.** If a new dependency is needed, tell Phil so he can add it via `mamba install -n clickworthy -c conda-forge <package>` manually.

## Known pitfalls in this dataset (learned or anticipated — keep updated)

- **Environment setup, not dataset-specific:** initial env creation via the machine's stock Anaconda (conda 4.11.0) hung indefinitely resolving the pymc/numba/llvmlite dependency graph — see Environment section above for the full story and the Miniforge/mamba fix. If a fresh machine ever needs this repo, use Miniforge/mamba from the start; don't attempt Anaconda's classic solver on this dependency stack.
- **Schema correction:** an earlier draft of this file invented a column name (`randomization_imbalance_risk`) that does not exist in the actual archive. The real column, added in a June 2024 update we don't have, is `problem`. Our file predates that update; see the Data section's date-based workaround. Lesson: always verify column names against the archive's own docs or the file's real header before writing filter logic — see "never invent column names" in Working style above.
- **Confirmatory dataset is access-gated, not freely downloadable** — discovered while sourcing data for Stage 0. PLAN.md's Stage 4c needs revisiting; see the Data section above.
- **Winner's curse / selection on significance:** the largest observed lifts are inflated. This is a feature of the analysis (Stage 2 shrinkage demonstrates it), but never quote a raw top-N lift as an effect estimate.
- **Impressions vary wildly across arms and tests.** Weight or model accordingly; never average CTRs across tests unweighted.
- **Some tests changed mid-flight or have near-duplicate arms.** Treat suspicious tests (identical headlines, absurd CTRs) as data-quality candidates and log them in the ingestion notebook.
- **Peeking/sequential stopping:** Upworthy's own stopping rules were informal. Test durations are not fixed-horizon; this matters for Stage 1 interpretation and is discussed explicitly in `02_classical_ab.ipynb`.

(Add new entries here as they are discovered — this section is the project's institutional memory.)
