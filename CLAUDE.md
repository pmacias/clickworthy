# CLAUDE.md

This file guides AI coding assistants ("CC", for Claude Code, in the notes below) working in this repo; the architecture and statistical-convention notes below are equally useful background for any human contributor.

## Project: clickworthy

Statistical analysis of the Upworthy Research Archive — the full archive of 27,616 real headline A/B tests (128,217 packages/arms total, across a 4,873-test exploratory sample and a 22,743-test confirmatory sample) run by Upworthy from Jan 2013 to Apr 2015. All development, model iteration, and Stage 1–3 work uses the exploratory sample only; the confirmatory sample is touched exactly once, at the very end, in Stage 4c — see Data section for the full preregistration-style rationale. The project demonstrates, in order of priority: (1) classical experiment analysis, (2) hierarchical Bayesian meta-analysis in PyMC, (3) causal inference on observational data benchmarked against randomized ground truth, (4) text mining of headlines.

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
│   ├── evaluation.py       # shared metrics, diagnostics checks
│   └── constants.py        # every fixed seed used in the project
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_classical_ab.ipynb
│   ├── 03_hierarchical_bayes.ipynb
│   ├── 04_mixture_effects.ipynb
│   ├── 05_causal_benchmark.ipynb
│   └── 06_results_summary.ipynb
├── clickworthy.duckdb      # built by notebook 01; gitignored
└── idata/                  # saved InferenceData from every MCMC fit; gitignored

(raw CSVs live outside the repo entirely -- see Data below)
```

Notebooks are numbered in dependency order. Shared logic lives in `src/clickworthy/`; notebooks import from the package rather than duplicating functions. Each notebook should run top-to-bottom cleanly from a fresh kernel.

## Data

- Raw archive CSVs live at `~/ml_datasets/clickworthy/` (outside Dropbox and Time Machine). Never commit data; `.gitignore` covers `*.csv`, `*.duckdb`, `data/`.
- Source: Upworthy Research Archive, OSF `https://osf.io/jd64p/`. Two files in use, both March 2020 snapshots:
  - **Exploratory:** `upworthy-archive-exploratory-packages-03.12.2020.csv` (~13.4MB) — 22,666 packages / 4,873 tests. This is the file Stages 0–3 and all model development use.
  - **Confirmatory:** `upworthy-archive-confirmatory-packages-03.12.2020.csv` (~63MB) — 105,551 packages / 22,743 tests, verified zero test-ID overlap with the exploratory file. **Downloadable directly from the same OSF project (correcting an earlier note in this file that called it access-gated — that appears to have been outdated or wrong; verify current access if this ever changes).**
- **The confirmatory file exists on disk but is under a hard access rule, per the archive's own preregistration design (see `about-the-archive.md`, `natematias/upworthy-archive` on GitHub): all development, exploration, and model-building for Stages 0–3 uses the exploratory file ONLY. The confirmatory file is read exactly once, in Stage 4c (`06_results_summary.ipynb`), after specific claims have been written down in that notebook based only on exploratory-sample results.** This is not a technical limitation — the file is fully readable — it is a scientific-integrity rule. **CC must never open, query, or reference the confirmatory file outside of Stage 4c, even for a schema check or "just to look" — doing so would invalidate the whole point of holding it out. If uncertain whether a task is Stage 4c, stop and ask Phil rather than reading the file.**
- The exploratory dataset (per the archive's own docs) contains **22,666 packages from 4,873 tests**. Verify this against the actual row/test counts on ingest — a mismatch means something is wrong with the file, not the documentation.

### Schema facts (verified against the archive's own documentation — see `about-the-archive.md` in the `natematias/upworthy-archive` GitHub repo)

Actual columns in the file in hand: `created_at`, `updated_at`, `clickability_test_id`, `excerpt`, `headline`, `lede`, `slug`, `eyecatcher_id`, `impressions`, `clicks`, `significance`, `first_place`, `winner`, `share_text`, `square`, `test_week`.

- Each row = one **package** (an arm: a headline/image bundle) of one **test**. Packages in the same test share `clickability_test_id`; viewers were randomly assigned to packages within a test.
- `impressions`: viewers assigned to that package. `clicks`: how many of those clicked. CTR = clicks / impressions. Baseline CTRs are low (order 1%), so treat proportions carefully — normal approximations can be poor for small arms.
- `significance` and `first_place` are **not** statistical significance in our sense — per the archive maintainers, the exact method Upworthy used to generate `significance` was never recovered from former staff, and both columns were shown to editors only to guide which package to select as `winner`. **Do not use these as a proxy for statistical significance in Stage 1 — compute it ourselves from `impressions`/`clicks`.**
- `winner`: whether editors selected this package for the live site after the test concluded. Note this introduces the same selection-effect logic as the winner's curse (see Known pitfalls below) if used carelessly as a "ground truth" for "did this package perform best."
- **No `problem`/randomization-flag column exists in this file version.** The current archive (per a June 2024 update, not reflected in our March 2020 snapshot) added a `problem` column, set to 1 for every package in a test where a randomization defect was found, concentrated entirely in tests run **June 25, 2013 – January 10, 2014**. Since we don't have that column, apply the equivalent **date-based exclusion** directly on `created_at` (see Standard filters below). This is a documented, deliberate substitution, not a guess — the docs state the column and the date range are defined identically for the affected tests.

### Standard filters (apply via the DuckDB views, not ad hoc per notebook)

1. Drop packages/tests with `created_at` between 2013-06-25 and 2014-01-10, inclusive — stands in for the `problem` column absent from this file version (see schema note above).
2. Drop degenerate tests: fewer than 2 packages, or any package with 0 impressions.

Every notebook states which view it uses. Exploratory dips outside these filters (e.g. the schema-verification and filter-accounting sections of `01_data_ingestion.ipynb`, which deliberately look at the raw table) must be labeled as such.

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
- **`mu_test` is deliberately left UNPOOLED** (independent, weakly-informative prior per test, no shared population structure across tests). Reasoning: article-level baseline clickability is expected to vary heterogeneously by topic (a viral scandal vs. a mundane policy piece have no principled shared "normal" baseline), so there's no population-level baseline worth shrinking toward. `delta_arm`, by contrast, represents the genuinely comparable, poolable quantity across tests — "how much did this headline move the needle relative to its own test's baseline" — which is why only `delta_arm` gets the hierarchical treatment. State this asymmetry explicitly in the Stage 2 notebook; don't leave it implicit.
- **`sigma` (the population spread of `delta_arm`) is a single, GLOBAL value shared across the entire archive** — every arm in every test is assumed drawn from one shared population of headline-effect sizes, not a per-test or per-topic-cluster sigma. This is itself a modeling assumption (headline-effect variability is roughly constant archive-wide), not a derived fact — state it as a choice in the notebook. A more complex model could let sigma vary by grouping (e.g. by topic or era) if this assumption seemed to fail posterior predictive checks, but that's out of scope for this project.
- Weakly informative priors, stated in the notebook. Do not use flat priors.
- Required diagnostics after every fit, checked programmatically via `evaluation.py`, not eyeballed: zero divergences, `r_hat < 1.01` on all parameters, bulk and tail ESS > 400. A fit failing any of these is not a result — it's a bug to fix before interpretation.
- **Persist every fit's `InferenceData` to disk immediately after sampling** (`idata.to_netcdf("idata/<descriptive_name>.nc")`, `idata/` gitignored). Notebook kernels are ephemeral once a notebook is re-executed (e.g. via `nbconvert`), so an in-memory-only `idata` is unrecoverable afterward — sampler-level diagnostics not already captured as printed cell output (tree depth, step size, per-chain sample stats) become impossible to inspect without a full re-run. Learned the hard way in Stage 2 Task 4's full-sample diagnostics investigation, where a tree-depth check had to wait on a same-config re-run purely to capture what should have been saved the first time.
- **This machine has 16GB RAM — never hold multiple large `InferenceData` objects in memory at once.** Any comparison across saved fits (e.g. checking whether a parameter's diagnostics improved between two configurations) must use chunked/lazy reads directly off each fit's netCDF file (`xarray.open_dataset(path, group="posterior", engine="h5netcdf")`, sliced by parameter index) rather than loading two or more full objects and comparing in memory — see `evaluation.py`'s `check_diagnostics_from_netcdf` and `compare_params_from_netcdf` for the pattern. Learned by triggering an OOM kill mid-notebook-execution in Stage 2 Task 4, holding a 1.8GB and a 7.2GB posterior simultaneously for a naive in-memory comparison. This will recur at Stage 3/4 scale if not followed by default, not just patched once for that one cell.
- **Load-if-cached, don't blindly resample.** Any fit expensive enough to persist via `idata.to_netcdf()` is expensive enough to check for its cached file before calling `pm.sample()` again — `if path.exists(): idata = az.from_netcdf(path)` else fit-and-save. This is the default pattern for every fitting cell going forward, not a one-off for Task 4: it keeps a fresh clone fully reproducible (no cache → it actually fits) while making a full top-to-bottom notebook re-execution fast once the artifacts already exist and are verified correct.
- Prior predictive checks before fitting; posterior predictive checks after. Both get a plot.
- Develop on a subsample (~500 tests) first; scale to the full exploratory sample only once the subsample model is clean. If full-scale NUTS is too slow, discuss options with Phil (nutpie, ADVI as a check, minibatching) — do not silently switch inference methods.

**Causal (Stage 3)**
- The confounding in `05_causal_benchmark` is **constructed by us** from randomized data, so ground truth is known. The construction (selection mechanism, which covariates drive it) must be implemented in `causal.py` with a fixed seed and documented in prose — it is the scientific core of the stage, not plumbing.
- Report every estimator (naive difference, regression adjustment, IPW, matching) against the same ground-truth estimand, with bias and coverage. Never report an observational estimate without its experimental benchmark next to it.

**Everywhere**
- Fixed seeds for anything stochastic; seeds live in one constants module, not scattered literals.
- Plots are produced by Phil's own runs; generate plotting code, don't describe imagined figures. Label axes with units. No default matplotlib titles like "Figure 1".
- PyMC `sample()` calls (and other PyMC sampling calls with a `progressbar` option, e.g. `sample_posterior_predictive`) always pass `progressbar=False`; more generally, avoid widget-based output (tqdm's default notebook widget, ipywidgets, etc.) in any notebook cell whose output is meant to be viewed outside a live Jupyter session. Reason: this is a portfolio project — notebooks are viewed via GitHub or a static renderer — and a saved `application/vnd.jupyter.widget-view+json` output renders as raw JSON/text there instead of a progress bar, since it depends on a live widget frontend that isn't present.

## Environment

- Single env named `clickworthy` (native arm64), created via **Homebrew's Miniforge/mamba**, not the machine's existing Anaconda install. Root prefix: `/opt/homebrew/Caskroom/miniforge/base`. Env lives at `/opt/homebrew/Caskroom/miniforge/base/envs/clickworthy`.
- Core deps: python ≥3.11, pandas, duckdb, statsmodels, scipy, pymc, arviz, scikit-learn, matplotlib, jupyter. `xarray` and `h5netcdf` (used by `evaluation.py`'s disk-backed diagnostics) arrive as transitive dependencies of arviz and are not installed separately. No GPU/MPS requirement — PyMC runs CPU here.
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

## Pre-approved packages

Install and pin without further checks:

```
# Core / data
numpy, pandas, scipy, duckdb, pyarrow, pyyaml

# Classical stats / A/B testing
statsmodels, scikit-learn, pingouin

# Plotting
matplotlib, seaborn, plotly

# Bayesian / hierarchical
pymc, arviz, pytensor, bambi

# Causal inference
dowhy, econml, causalml

# Text mining (extension)
nltk, spacy, gensim, sentence-transformers

# Notebook / tooling
jupyter, jupyterlab, ipywidgets, tqdm
```

## Package installation policy

For any package NOT on the pre-approved list above:
1. State the exact name and version you intend to install.
2. Look up the package's registry page (pypi.org) or search for it.
   Report: approximate weekly/monthly downloads, and first-release date.
3. Flag it explicitly if any of these are true:
   - Fewer than ~50,000 weekly downloads
   - Published or last updated in the last 30 days
   - The name is a close lookalike of something on the approved list
     (differs by one letter, hyphen, or plural)
4. Always pin the exact version (`==x.y.z`), never a range, unless told
   otherwise.
5. If a package fails these checks, don't install it — report findings
   and wait for a decision.

This applies even mid-task: if a new package is needed while working,
pause and run this check before installing.

When a new package is approved, ask whether it should be added to the
pre-approved list above for future sessions.

## Known pitfalls in this dataset (learned or anticipated — keep updated)

- **Environment setup, not dataset-specific:** initial env creation via the machine's stock Anaconda (conda 4.11.0) hung indefinitely resolving the pymc/numba/llvmlite dependency graph — see Environment section above for the full story and the Miniforge/mamba fix. If a fresh machine ever needs this repo, use Miniforge/mamba from the start; don't attempt Anaconda's classic solver on this dependency stack.
- **Schema correction:** an earlier draft of this file invented a column name (`randomization_imbalance_risk`) that does not exist in the actual archive. The real column, added in a June 2024 update we don't have, is `problem`. Our file predates that update; see the Data section's date-based workaround. Lesson: always verify column names against the archive's own docs or the file's real header before writing filter logic — see "never invent column names" in Working style above.
- **Confirmatory dataset correction:** originally documented here as access-gated (not freely downloadable) based on the archive's stated peer-review process. That turned out to be outdated or inaccurate for this project — the confirmatory CSV was downloaded directly from the same OSF project, verified against documented counts (105,551 packages / 22,743 tests) and confirmed to have zero test-ID overlap with the exploratory file. It now lives on disk under a strict access rule (see Data section) rather than being unavailable. Lesson: an early "this isn't possible" finding should be treated as provisional and re-checked, not baked permanently into project constraints.
- **Winner's curse / selection on significance:** the largest observed lifts are inflated. This is a feature of the analysis (Stage 2 shrinkage demonstrates it), but never quote a raw top-N lift as an effect estimate.
- **Impressions vary wildly across arms and tests.** Weight or model accordingly; never average CTRs across tests unweighted.
- **Some tests changed mid-flight or have near-duplicate arms.** Treat suspicious tests (identical headlines, absurd CTRs) as data-quality candidates and log them in the ingestion notebook.
- **Peeking/sequential stopping:** Upworthy's own stopping rules were informal. Test durations are not fixed-horizon; this matters for Stage 1 interpretation and is discussed explicitly in `02_classical_ab.ipynb`.
- **"Unpooled" prior ≠ unregularized (Stage 2, Task 3):** `mu_test`'s prior
  (`Normal(-4.6, 1.0)`) has no cross-test information sharing — its parameters are
  fixed constants, not learned from other tests — but it is still a proper, non-flat
  prior, and it still pulls `mu_test`'s posterior toward the archive baseline whenever a
  specific test's own data disagrees with it. This bit Stage 2 Task 3's shrinkage plot:
  a hand-derived "raw, no-pooling" comparison baseline was built assuming `mu_test` had
  no regularization at all (an implicitly flat prior), while the model's actual
  `delta_arm` posterior is measured relative to the real, regularized `mu_test`. The
  mismatch produced sign-flipped and sometimes-growing "shrinkage" numbers that looked
  like a modeling bug but were actually a comparison-definition bug. Fixed by
  referencing both sides of any raw-vs-pooled comparison against the model's own fitted
  value for the unpooled-but-regularized component, not a separately-derived proxy.
  Lesson: "unpooled" in this project means *no cross-test sharing*, not *no
  regularization* — any parameter with its own proper prior is still being pulled
  somewhere, even in isolation, and a "raw" comparison that ignores this will be
  silently wrong in exactly the cases (small tests, baselines far from the archive
  average) that matter most for the comparison's whole point.
  **Relevant to Stage 3 too:** any raw-vs-adjusted comparison built for the causal
  estimator ladder should be checked against this same question — is a given model
  component actually unregularized, or just uncorrelated across whatever grouping the
  adjustment method assumes — before trusting a naive baseline for it.
- **nutpie is not usable in this env (checked mid-2026):** as of nutpie 0.16.11, the package requires Python >=3.12, incompatible with this project's pinned `python=3.11`. Not worth a full Python upgrade (and the version-pinning risk to the already-fragile pymc/pytensor/numba stack, see Environment section) just for a secondary NUTS cross-check. If a future stage needs a second inference backend to cross-check a diagnostics failure, use ADVI instead — no new dependency required.
- **Full-sample r_hat borderline failures (Stage 2 Task 4) — investigated, resolved as estimation noise, not a real defect.** The full-exploratory-sample fit at `tune=1000, target_accept=0.8` (and again at `tune=4000, target_accept=0.95, draws=1000`) showed a small, shifting set of `mu_test`/`z_arm` parameters with r_hat marginally above 1.01 (max 1.010–1.028), always with zero divergences and passing ESS. The failing tests were disproportionately high-impression (57th–97th archive percentile), and several had true near-duplicate arms (identical headline **and** `eyecatcher_id`, not just matching text — ~2.4x the ~18% base rate among comparably high-impression tests, though not statistically robust at n=7). Two mechanistic hypotheses were checked directly and ruled out: NUTS tree-depth (0.0% max-treedepth-hit rate everywhere, including specifically at these parameters' most extreme draws — no cut-short-tree/hard-geometry signature) and test duration as a confound (6 of 7 flagged tests sit at unremarkable 31st–59th archive percentile duration). A chain-scatter check (do chains split into consistent groups, suggesting a shallow-ridge incomplete-mixing failure mode) was inconclusive by design — a genuine ridge produces the same "random scatter" signature as noise, since each chain settles independently along a continuous under-explored direction rather than clustering. **The decisive check: re-running the identical `tune=4000, target_accept=0.95` configuration at `draws=4000` instead of 1000** (new seed) collapsed r_hat on all 9 previously-flagged parameters to 1.0001–1.0023, with ESS scaling ~4x in lockstep with the 4x increase in draws, and the archive-wide diagnostics (all ~22,212 parameters) passed cleanly (max r_hat 1.0028, zero divergences). That is the actual signature pure estimation noise predicts — r_hat trending toward 1.0 and ESS scaling linearly as draws increase — and not what real, unresolved mixing would produce. **Settled full-sample configuration going forward: `tune=4000, target_accept=0.95, draws=4000`, 4 chains.** The duplicate-arm/well-powered-test correlation remains real but mechanistically unexplained; worth another look only if it recurs with a larger flagged set at a future scale.

(Add new entries here as they are discovered — this section is the project's institutional memory.)
