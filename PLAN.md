# PLAN.md — clickworthy roadmap

Staged in priority order. Each stage is independently shippable. (Current stage status is tracked in `README.md`; "CC" below means Claude Code, the AI coding assistant used on this project.) **Do not begin a stage until the previous stage's acceptance criteria are met and Phil has signed off.** Within a stage, work through tasks in order unless Phil says otherwise.

Skill targets this project exists to demonstrate, mapped to stages:

| Skill gap | Stage |
|---|---|
| Classical CIs, hypothesis testing, power | 1 |
| Experimentation / A-B testing | 1, 4 |
| Bayesian modeling & inference (PyMC) | 2 |
| Hierarchical modeling | 2 |
| Mixture modeling | 2 |
| Observational → causal inference → experiments | 3 |
| Text mining | 4 |
| Data cleaning/wrangling, visualization | all |

## Notebook convention: pedagogical flow (applies to every notebook, every stage)

Every notebook should read as a teaching artifact, not just a results dump. Concretely, each notebook includes:
- **Exploratory plots before modeling** — look at the data (distributions, raw rates, sample sizes) before fitting anything, and say in prose what you see and why it matters for the method that follows.
- **Diagnostic plots after modeling** — for classical tests: CI/estimate plots showing uncertainty, not just point values. For Bayesian fits: prior predictive, posterior predictive, and diagnostic plots (trace/rank plots, not just a diagnostics table) — see the Statistical conventions section for the required checks. For causal estimators: balance/overlap plots before treating an estimate as trustworthy.
- **A plot is not optional decoration** — if a stage's acceptance criteria in this file mention a specific figure (e.g., the Stage 2 shrinkage plot), it must exist and be discussed in prose, not just generated and left uncaptioned.
- Plots are generated as code by CC; Phil runs them and reviews the output (see CLAUDE.md).

---

## Stage 0 — Scaffold and ingestion (Week 1, first half)

**Goal:** repo skeleton, environment, data on disk, DuckDB layer with standard filtered views.

Tasks:
1. Scaffold repo per CLAUDE.md layout: `pyproject.toml`, `src/clickworthy/`, numbered notebooks, `.gitignore` (data, `*.duckdb`, checkpoints, `.claude/settings.local.json`).
2. Implement `data_access.py`: build `clickworthy.duckdb` from the CSV; create the filtered view applying the standard filters from CLAUDE.md; helper functions to pull a test's packages as a tidy frame.
3. `01_data_ingestion.ipynb`: verify schema against CLAUDE.md's stated facts, quantify what each filter removes (tests/packages/impressions dropped), sanity-check core distributions with exploratory plots (packages per test, impressions per package, CTR range — see Notebook convention above). Also include a light descriptive-stats pass on `significance`, `first_place`, and `winner` (value counts, missingness, rough relationship to observed CTR) — not analysis, just a documented first look, since CLAUDE.md flags these columns as not usable as statistical significance.

(Environment creation and data download are manual prerequisites performed by Phil outside any agentic layer — see CLAUDE.md's Environment and Data sections. Not CC tasks.)

**Acceptance criteria:**
- Fresh clone + documented setup steps → notebook 01 runs top-to-bottom.
- Filter accounting table exists (counts before/after each filter).
- Exploratory plots for the core distributions exist, each with a sentence of interpretation, not just rendered and left uncaptioned.
- The `significance`/`first_place`/`winner` descriptive-stats pass exists and is clearly labeled as descriptive, not inferential.

---

## Stage 1 — Classical experiment analysis (Week 1, second half)

**Goal:** the interview-canonical A/B toolkit, exercised on real experiments. Notebook: `02_classical_ab.ipynb`; shared logic in `classical.py`.

Tasks:
1. Single-test deep dive (pick 2–3 illustrative tests): pairwise two-proportion tests, chi-square across all arms, Wilson CIs per arm, Newcombe CI on the best-vs-control difference. Show absolute and relative lift.
2. Power and MDE: at typical baseline CTR (~1%) and typical per-arm impressions, what lift was each test powered to detect? Plot the distribution of per-test MDE across the archive. This quantifies how underpowered typical tests were — a key finding, state it plainly.
3. Archive-wide testing: fraction of tests "significant" at raw alpha=0.05 vs BH-adjusted; contrast and discuss.
4. Winner's curse demonstration: take each test's best arm by observed CTR, compare its observed lift against a split-half or later-period estimate where feasible; show the inflation.
5. Sequential/peeking discussion section: why Upworthy's informal stopping inflates false positives; simulate a null A/A test with peeking to show the alpha inflation empirically (small simulation, seeded).

**Acceptance criteria:**
- Every reported test/CI names its method and assumptions.
- Power section produces the MDE distribution plot and a one-paragraph plain-language conclusion.
- A/A peeking simulation reproduces the known alpha-inflation effect.
- Phil can explain every number in the notebook without referring back to code (this is interview prep — if a result can't be explained, it isn't done).

---

## Stage 2 — Hierarchical Bayesian meta-analysis (Week 2)

**Goal:** the PyMC centerpiece. Notebooks: `03_hierarchical_bayes.ipynb`, `04_mixture_effects.ipynb`; model builders in `models.py`.

Tasks:
1. Warm-up (pedagogical, small): Beta-Binomial model of a single test's arms. Prior predictive check, fit, posterior predictive check, full diagnostics. This establishes the workflow pattern every later model follows.
2. Hierarchical model on a ~500-test subsample: binomial likelihood on counts, logit link, test-level intercepts, arm effects partially pooled across tests, non-centered parameterization. Verify diagnostics per CLAUDE.md thresholds.
3. Shrinkage analysis: plot raw (no-pooling, Stage 1) effect estimates vs partially-pooled posteriors, ordered by arm sample size. (As built, the "raw" side is defined relative to the model's own fitted `mu_test`, not a separately-derived Stage 1 quantity -- see the "unpooled ≠ unregularized" entry in CLAUDE.md's Known pitfalls for why.) Small noisy tests should shrink hard toward the population; large tests barely move. This plot is the headline figure of the stage — the direct visual of why hierarchical modeling exists.
4. Scale to full exploratory sample. If NUTS runtime is prohibitive, surface options to Phil (nutpie backend, ADVI cross-check) with tradeoffs; decide together. (nutpie later turned out to be incompatible with this env's Python 3.11 -- see CLAUDE.md's Known pitfalls.)
5. Mixture model (`04_mixture_effects.ipynb`): model the population of arm effects as a two-component mixture (near-null spike + wider slab of real effects). Report the posterior fraction of tests with practically meaningful effects, with a stated practical-significance threshold. Compare against the naive "% significant" from Stage 1.
6. Population conclusions: posterior for the effect-size distribution — what does a *typical* headline change actually do to CTR?

**Acceptance criteria:**
- All fits pass diagnostics programmatically (zero divergences, r_hat, ESS via `evaluation.py`).
- Shrinkage figure exists and matches theory (shrinkage ∝ inverse sample size).
- Mixture model's spike fraction is reported with uncertainty, not a point estimate.
- A short prose section explains centered vs non-centered parameterization and why it mattered here.

---

## Stage 3 — Causal inference benchmarked against ground truth (Week 3)

**Goal:** walk the observational → causal → experimental spectrum with a within-study comparison. Notebook: `05_causal_benchmark.ipynb`; logic in `causal.py`.

Design: because Upworthy's tests are randomized, true effects are known. We construct a *confounded observational dataset* by non-randomly deleting assignments — e.g., a selection mechanism where exposure to the "treatment" headline depends on observable covariates (time period, topic/section, weekday, test size). Then we try to recover the known truth using only observational methods.

Tasks:
1. Define the estimand precisely (e.g., ATE of treatment headline vs control on click probability, for a chosen subset of clean two-arm tests). Write it down before any estimation.
2. Implement the selection mechanism in `causal.py`: seeded, parameterized by confounding strength, documented in prose. Verify it induces the intended imbalance (covariate balance table before/after).
3. Estimator ladder, each against the same ground truth: naive difference in means → regression adjustment → propensity scores (IPW and matching; overlap/positivity diagnostics required) → optionally doubly-robust (AIPW).
4. Bias/coverage table: for each estimator, bias vs experimental truth and CI coverage across repeated draws of the selection mechanism (Monte Carlo over seeds).
5. Sensitivity: sweep confounding strength; show where each method breaks. Include one *unobserved* confounder scenario to demonstrate honest failure — the point is knowing the limits, not that methods always work.
6. Prose section connecting this to practice: when would a product DS trust observational estimates, and what diagnostics earn that trust.

**Acceptance criteria:**
- Estimand stated before estimators appear.
- Balance and overlap diagnostics shown for every propensity-based estimator.
- Bias/coverage table complete, including the unobserved-confounder failure case.
- No observational estimate appears anywhere without its experimental benchmark alongside.

---

## Stage 4 — Extensions (Week 4, pick per remaining runway)

**4a. Headline text mining.** Featurize headlines (length, question marks, numbers, sentiment, curiosity-gap constructions like "you won't believe"); model CTR lift as a function of text features (regularized regression first; tree-based second). Connects to the published finding that negativity drives clicks — replicate directionally on the exploratory sample.

**4b. Decision-framework writeup.** A prose-forward notebook or `docs/` essay: fixed-horizon vs sequential testing, alpha spending/mSPRT at a high level, FDR management across a testing program, and how Stages 1–3 inform "ship / don't ship" decisions. This targets the Magic Eye "advanced decision making frameworks" language directly.

**4c. Confirmatory validation (`06_results_summary.ipynb`, required regardless of 4a/4b).** The confirmatory file (`upworthy-archive-confirmatory-packages-03.12.2020.csv`) is confirmed on disk, verified against documented counts, and has zero test-ID overlap with the exploratory sample — see CLAUDE.md's Data section for the hard access rule governing it. Preregister (in the notebook, before touching the confirmatory data) the 2–3 headline claims from Stages 1–3, in specific, falsifiable form (e.g., exact predicted shrinkage direction, exact predicted mixture spike fraction range). Only after those claims are written down, load the confirmatory file and check them once. Report agreement or disagreement honestly — a disagreement is a legitimate, reportable finding, not a failure to fix. This is the capstone and the thing that makes the README credible.

---

## Deliverables checklist (project completion)

- [ ] README with project story, headline figures, and honest limitations section
- [ ] All notebooks run top-to-bottom from fresh kernels
- [ ] Confirmatory validation done exactly once, results reported as found
- [ ] Portfolio site card + resume bullet drafted (separate task, outside this repo)
