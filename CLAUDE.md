# CLAUDE.md — non-stationary-transformers

Forecasting non-stationary time series (heavy-tailed levy-stable synthetic data and real
OWID COVID case data) with a positional-encoding **Transformer** and a seq2seq **LSTM**.
Research repo, solo author, experiments tracked in **MLflow**.

## Tech stack & environment

- **Python 3** + PyTorch, NumPy, scikit-learn, pandas, SciPy, MLflow, click, matplotlib,
  tqdm, python-dotenv. Install: `pip install -r requirements.txt`.
- **No `pyproject` packaging** — `setup.py` installs the `src` package (`pip install -e .`).
  `pyproject.toml` holds ONLY tool config (black/isort @ 99).
- **Config via env / `.env`** — `src/config.py` reads everything via `os.getenv` with
  sensible defaults, so the project imports without a `.env`. Copy `.env.example` → `.env`
  to override. Never commit `.env`.
- **Runners are scripts**, run as modules from the repo root, e.g.
  `python3 src/models/train_model.py` or `python3 src/training_pipeline.py`.

## Commands

- Format: `black --line-length 99 src && isort --profile black --line-length 99 src`
- Lint: `flake8 src` (config in `tox.ini`, line length **99**)
- Verify (no test suite): `python3 -m compileall -q src/`
- Full sweep: `python3 src/training_pipeline.py` (downloads OWID once → builds each data
  condition once → trains across `SEEDS`).
- Aggregate results: `python3 src/models/aggregate_results.py` (mean±std per condition).

## Package map (`src/`)

- `config.py` — central config (env-overridable defaults, paths, grids, `SEEDS`).
- `seeding.py` — `seed_everything` (python/numpy/torch/cuda/cudnn).
- `data/download_data.py` — idempotent OWID CSV download (cached, `--force`).
- `data/generate_data.py` — levy-stable non-stationary + synthetic-COVID series.
- `data/process_data.py` — smoothing + **causal** normalization; OWID chunking + groups.
- `data/make_dataset.py` — train/val/test split (random for synthetic, group-by-country
  for OWID); writes `.pt` datasets.
- `models/model.py` — `TransformerWithPE` and seq2seq `LSTM` (+ `PositionalEncoding`).
- `models/utils.py` — split/forward dispatch, losses, metrics, train/eval loops.
- `models/train_model.py` — train the Transformer on synthetic data.
- `models/train_model_real.py` — pre-train on synthetic + fine-tune on OWID (`--model-type`).
- `models/predict_model.py` — load a logged model from MLflow and forecast.
- `models/aggregate_results.py` — mean±std of test metrics across seeds.
- `training_pipeline.py` — orchestrates the multi-condition × multi-seed sweep.
- `features/build_features.py` — reusable feature transforms (differencing, rolling, etc.).
- `visualization/visualize.py` — `visualize_prediction` (src/tgt/pred/infer plot).

## CRITICAL invariants — do not break these silently

These have no automated test; a violation still "runs" but invalidates results.

1. **Causal normalization (no look-ahead leakage).** `causal_normalize` fits the scaler on
   each sequence's HISTORY window only (`leave_ratio`), never the full sequence. `log1p`
   only for non-negative counts (COVID), not levy-stable. No global/full-sequence
   `StandardScaler().fit_transform`.
2. **Group-aware, non-leaky splits.** OWID uses `GroupShuffleSplit` by country (a country
   never spans train/val/test); chunks are non-overlapping; `owid_groups[i]` ↔
   `owid_sequences[i]`. 3-way split, `val_relative = VAL_RATIO/(1-TEST_RATIO)`. Test is
   only touched in `evaluate_model`.
3. **Seq2seq + autoregressive contract.** `forward`/`infer` return `(B, horizon, out_dim)`,
   `batch_first=True`. Transformer keeps the causal `tgt_mask` moved to `tgt.device`. The
   LSTM decoder emits FUTURE steps aligned with `tgt_y` (no per-encoder-step outputs). Both
   `infer()`s run under `eval()` + `no_grad()` and restore mode. Output length ==
   `tgt_y.shape[1]`.
4. **Evaluation honesty.** Autoregressive (`model.infer`) metrics are PRIMARY; teacher-forced
   are "optimistic". Early-stopping/`_validate` use the autoregressive loss with the same
   criterion as training. Metrics are normalized-scale; `mape` is masked + de-emphasized.
5. **MLflow key contract.** Metric keys `{split}_ar_*` / `{split}_tf_*`; consumers
   (`aggregate_results.py`) must read keys logged byte-identically by `evaluate_model`. A
   rename silently NaNs a column.
6. **Reproducibility split.** `RANDOM_STATE` fixes data generation AND the split; `SEEDS`
   varies only model training (so reported variance is model variance). Seed via
   `seed_everything`.
7. **Hermetic data.** OWID download lives only in `download_data.py`, never re-added to
   `generate_data.py`.

## Conventions & gotchas

- Unknown enum values raise `ValueError` (not `AttributeError`). Guard every artifact write
  with `os.makedirs(..., exist_ok=True)`.
- Line length **99** everywhere (black/isort/flake8). The auto-format hook applies
  black+isort after edits (best-effort if installed).
- Artifact dirs are git-ignored and must stay out of git: `data/`, `models/`, `mlruns/`,
  `reports/`, `notebooks/`. Never `git add -f` them or any `.npy/.pt/.ipynb/...` file.
- Verify edits with `python3 -m compileall -q src/` — there is no pytest suite and torch
  may not be installed locally, so do NOT try to run training to "verify".

## Git & commits

- Default branch **`main`**; origin `git@github.com:mgrts/non-stationary-transformers.git`.
- Use **Conventional Commits** (`type(scope): subject`). Prefer the `/commit-push` skill.
- **NEVER attribute commits to Claude** — no `Co-Authored-By`, `--author`, `@anthropic.com`,
  or "Generated with Claude". The `guard_git` hook blocks it.
- No `--force` / `--no-verify` / `git reset --hard` (hook-blocked).

## Claude Code setup (`.claude/`)

- **Skills:** `code-review` (silent-bug review + subagent delegation), `commit-push`
  (review → verify → docs → Conventional Commit → push).
- **Subagents:** `ml-tensor-contract-reviewer` (shapes/infer/device),
  `data-leakage-reproducibility-auditor` (normalization/split/seeds/MLflow).
- **Hooks:** `guard_git` + `block_large_secret` (PreToolUse), `auto_format` (PostToolUse),
  `verify_changes` (Stop: compileall + config-import AST check, dependency-free).
