# clickworthy

Statistical analysis of the Upworthy Research Archive — see `PLAN.md` for the staged roadmap and `CLAUDE.md` for repo conventions.

## Setup

Environment setup is manual, performed once before working in this repo or running any agentic tooling against it (see `CLAUDE.md` for why Miniforge/mamba is used instead of a plain `conda`):

```bash
brew install miniforge          # one-time, if not already installed
mamba shell init --shell zsh --root-prefix=/opt/homebrew/Caskroom/miniforge/base
# open a new terminal after this step
mamba create -n clickworthy -c conda-forge python=3.11 pandas duckdb statsmodels scipy pymc arviz scikit-learn matplotlib jupyter
mamba activate clickworthy
pip install -e . --no-deps
```

Activate with `mamba activate clickworthy` before working in this repo (including before launching Claude Code) — do not use `conda activate` for this project, see `CLAUDE.md`.

Raw archive CSVs go in `~/ml_datasets/clickworthy/` (not committed). Source: Upworthy Research Archive, OSF `https://osf.io/jd64p/`.
