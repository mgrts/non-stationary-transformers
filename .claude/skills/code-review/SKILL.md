---
name: code-review
description: Review pending changes in the non-stationary-transformers repo for correctness and the silent-bug classes this research codebase actually hits — data-leakage & causal normalization, the train/val/test & group-split design, seq2seq tensor-shape/device/autoregressive contracts (Transformer + LSTM), the MLflow key + seed reproducibility contract, config/CLI parameter sync and the .env defaults, and secrets/large-file hygiene. Read-only by default; surfaces findings grouped by severity. Use before every commit, or via /commit-push.
---

# Code review for non-stationary-transformers

Review the changes currently in the working tree (staged + unstaged + untracked)
against the standards that matter for this codebase specifically. There is **no pytest
suite** and the runtime deps (torch, sklearn, pandas, mlflow, dotenv) are often not
installed, so most bugs here are **silent**: the code still "runs" (or compiles) while
leaking the future into the inputs, mis-splitting the data, breaking the autoregressive
forecast contract, or NaNing a logged metric. The job is to catch those by reading.

The review is **read-only by default** — fixes are surfaced as recommendations and
only applied if the user explicitly asks.

## Arguments

`$ARGUMENTS` — optional. Specific files or globs to scope the review (defaults to the
entire diff).

## Flow

### Step 1: Gather changes

```bash
git status --short
git diff --staged --stat
git diff --stat
```

If there is nothing pending, stop: "Nothing to review."

### Step 2: Read the diff

For each changed file, read the actual diff (not just the file list) so the review
reasons about what changed. Note which subsystems are touched — that selects which
checks below apply and which subagent to delegate to.

### Step 3: Delegate deep audits to subagents

When the diff touches a fragile subsystem, dispatch the matching subagent (via the
Agent tool, `subagent_type`) and fold its findings into the report:

- Touches `src/models/model.py`, `src/models/utils.py` (split/forward/loss/eval), or a
  trainer's model/forward/infer usage → **`ml-tensor-contract-reviewer`**.
- Touches `src/data/*.py`, `src/config.py`, `src/seeding.py`, `evaluate_model`,
  `aggregate_results.py`, or `src/training_pipeline.py` → **`data-leakage-reproducibility-auditor`**.

Run independent subagents in parallel. For a small diff that clearly matches none of
these, do the checks inline.

### Step 4: CRITICAL — Data leakage, normalization & splitting

Applies to `src/data/process_data.py`, `src/data/make_dataset.py`.

- **Causal normalization:** `causal_normalize` fits the scaler on the HISTORY window
  only (`int(len(seq) * leave_ratio)`), never the full sequence. `log1p=True` only for
  non-negative counts (synthetic-COVID, OWID), not the possibly-negative levy-stable main
  data. **Reject** any reintroduced full-sequence / global `StandardScaler().fit_transform`.
- **Non-overlapping OWID chunks:** `slice_array_to_chunks` drops the trailing partial and
  never back-extends (overlap leaks between splits). `owid_groups[i]` is the country of
  `owid_sequences[i]`, appended in the same loop, and both arrays are saved.
- **3-way group split:** `make_dataset` random-splits the independent synthetic datasets
  and `GroupShuffleSplit`-by-country for OWID; `val_relative = VAL_RATIO/(1-TEST_RATIO)`;
  the OWID branch asserts `len(groups)==len(sequences)`. Val drives selection; test is only
  used in `evaluate_model`.

### Step 5: CRITICAL — Seq2seq tensor-shape & autoregressive contracts

Applies to `src/models/model.py`, `src/models/utils.py`.

- `forward`/`infer` return `(B, horizon, out_dim)`; `(batch, seq, feature)` order;
  `nn.Transformer`/`nn.LSTM` stay `batch_first=True`.
- Transformer `forward` builds the causal `tgt_mask` AND re-moves it to `tgt.device`
  (dropping the mask leaks the future; dropping `.to(device)` crashes on CUDA).
- LSTM is encoder→decoder seq2seq; the autoregressive path seeds with `src[:,-1:,:]` and
  emits exactly `output_sequence_length` FUTURE steps aligned with `tgt_y` — NOT
  per-timestep outputs over the encoder window (the old magic-`60` bug).
- BOTH `infer()`s save/restore training mode and run under `eval()` + `no_grad()`.
- `split_sequence_with_decoder` derives the split from the actual length;
  `src=[:split]`, `tgt=[split-1:-1]`, `tgt_y=[split:]` are aligned; output length ==
  `tgt_y.shape[1]`.
- `prepare_batch`/`model_forward` dispatch on `isinstance(model, TransformerWithPE)`; a new
  model implements BOTH `forward(src, tgt)` and `infer(src, tgt_len)`.

### Step 6: CRITICAL — MLflow keys, seeds & evaluation honesty

Applies to `src/models/utils.py`, the trainers, `src/models/aggregate_results.py`,
`src/seeding.py`, `src/config.py`.

- Metric keys are `{split}_ar_*` (autoregressive, PRIMARY) and `{split}_tf_*`
  (teacher-forced, optimistic). Every key read by `aggregate_results.DEFAULT_METRICS` is
  logged byte-identically by `evaluate_model`. Logged hyperparams match what
  `aggregate_results` groups on (`CONDITION_PARAMS`). A rename silently NaNs a column.
- Autoregressive is reported as primary; `_validate`/early-stopping use the autoregressive
  loss with the SAME criterion as training. `mape` is masked on z-scored data and
  de-emphasized; metrics are normalized-scale (flag any original-unit claim without an
  inverse transform).
- `seed_everything` covers python/numpy/torch(+cuda)/cudnn. `RANDOM_STATE` fixes
  data+split; `SEEDS` varies only model training. If the diff claims determinism, confirm
  it matches what is actually seeded.

### Step 7: HIGH — Config / pipeline parameter sync & hermetic data

- Hyperparameters/grids live in `config.py`, env-overridable, and logged: `SEEDS`,
  `INITIAL_ALPHAS`, `FINAL_ALPHAS`, `STABILITY_PERIODS`, `SMOOTHING_TYPES`, `VAL_RATIO`,
  `TEST_RATIO`, `LEAVE_RATIO`, `NORMALIZATION_SCALER`, `FEATURE_DIM`/`NUM_HEADS`/`NUM_LAYERS`/`LR`/`PATIENCE`.
- Every `from src.config import (...)` name still exists in `config.py` (the
  verify-changes Stop hook checks this too, but call it out in review).
- The OWID download lives ONLY in `download_data.py` (idempotent), never re-added to
  `generate_data.py`. `training_pipeline` downloads once → builds each condition once →
  trains across `SEEDS`.
- Unknown enum values raise `ValueError` (not `AttributeError`). `os.makedirs(...,
  exist_ok=True)` guards every artifact write.

### Step 8: MEDIUM — Style, hygiene & docs drift

- Line length **99** consistent across `pyproject.toml` (`[tool.black]`/`[tool.isort]`),
  `tox.ini` (`[flake8]`), and `.pre-commit-config.yaml` (black/isort/flake8 args).
- **No secrets / no hardcoded credentials or PII** in the diff. No staged files under
  `data/`, `models/`, `mlruns/`, `reports/`, `notebooks/`, nor any
  `*.npy`/`*.npz`/`*.pt`/`*.pth`/`*.ckpt`/`*.pkl`/`*.parquet`/`*.ipynb`; no `git add -f`
  bypassing `.gitignore`; no file `> 10 MB`. (The block-large-secret hook enforces this,
  but flag it in review too.)
- New public function/class has a docstring + type hints; tensor shapes in
  `forward`/`infer` docstrings stay in sync with the code.
- README / `.env.example` updated if a new `config.py` constant, CLI flag, or pipeline
  step was added; `CLAUDE.md` package map / invariants updated if a module or a CRITICAL
  contract changed.

### Step 9: Run the static verification gate

There is no pytest suite. Deps are managed by Poetry (in-project `.venv`). Run:

```bash
make verify          # poetry run python -m compileall -q src  (syntax of all sources)
make pre-commit       # poetry run pre-commit run --all-files  (black/isort/flake8 @ 99 + scans)
```

If Poetry is unavailable, fall back to `python3 -m compileall -q src/` (dependency-free).
A compile failure is a critical finding. Also re-run the AST config-import check the
Stop hook uses if `src/config.py` or any importer changed (imports must resolve).

### Step 10: Report findings

Group findings by severity:

- **Critical** — look-ahead leakage / non-causal scaler, country-or-overlap leakage,
  broken autoregressive/shape/teacher-forcing contract, missing causal mask, dropout in
  infer, MLflow-key or seed-contract break, secret/large-file leak, compile failure.
- **High** — split-fraction/`val_relative` error, missing group/length assert, config
  import drift, OWID re-download in generate_data, eval reporting teacher-forced as primary.
- **Medium** — style/line-length/hygiene violation, missing `makedirs`,
  ValueError-vs-AttributeError, docs/`.env.example` drift.
- **Low** — comment / naming / docstring / type-hint polish.

For each finding: file path, the symbol or line, and a concrete suggestion. Do not make
changes unless the user asks.

If there are zero findings: report "Review passed — N files reviewed, M lines changed,
compileall <result>, pre-commit <result-or-skipped>."
