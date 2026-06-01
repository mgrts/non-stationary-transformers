---
name: commit-push
description: Run code-review, the static verification gate (compileall + config-import check; pre-commit if installed), update docs if drifted, write a Conventional Commits message, and commit + push to main on GitHub (origin mgrts/non-stationary-transformers). Stops at every gate (failed review, failed verification, failed hooks, conflicting rebase) and requires explicit confirmation before committing and pushing. NEVER attributes commits to Claude.
---

# Commit & push for non-stationary-transformers

Analyze pending changes, review them, run the verification gate + pre-commit hooks,
update docs if needed, write a Conventional Commits message, and push to `main`.

The default branch is **`main`**; origin is
**`git@github.com:mgrts/non-stationary-transformers.git`** (GitHub, owner `mgrts`). This
is a solo research repo, so the default flow pushes directly to `main` after gates pass
and the user confirms.

## Arguments

`$ARGUMENTS` — optional. A free-form commit message (used verbatim as the subject after
type inference) and/or flags: `--no-push` (commit only). There is **no issue tracker** —
never invent ticket references.

## Important

- **Conventional Commits**: `type(scope): subject`. Types: `feat`, `fix`, `refactor`,
  `perf`, `test`, `docs`, `chore`, `build`, `ci`. Scope optional but encouraged
  (e.g. `model`, `data`, `train`, `eval`, `config`, `pipeline`, `viz`, `hooks`).
- **NEVER** list Claude among commit authors. Do not add a `Co-Authored-By` trailer,
  set `--author` to Claude/Anthropic, use an `@anthropic.com` address, or add a
  "Generated with Claude" line — to the commit message OR a PR body. This is a hard
  project rule: the `guard_git` PreToolUse hook **blocks** any `git commit` carrying such
  attribution, so a slip is denied rather than committed.
- Do **NOT** use `--force`, `--no-verify`, `git reset --hard`, or any destructive git
  flag. The `guard_git` hook blocks these anyway. If a step fails, stop and ask the user.
- **Never `git add -f`** artifact dirs (`data/`, `models/`, `mlruns/`, `reports/`,
  `notebooks/`) or `.npy/.pt/.ipynb/...` files — the `block_large_secret` hook blocks them.

## Flow

### Step 1: Gather changes

```bash
git status --short
git diff --staged --stat
git diff --stat
git branch --show-current
```

If there are no changes, stop: "Nothing to commit." If the current branch is not `main`,
note it and ask whether to proceed on this branch or switch.

### Step 2: Run the code-review skill

Invoke the `code-review` skill on the pending diff.

- **Critical / High** findings: stop. Show them and ask whether to proceed anyway, fix
  automatically, or cancel. Do not move on without explicit acknowledgement.
- **Medium / Low** findings: print as a heads-up and continue.

### Step 3: Run the verification gate

This repo has no pytest suite and deps may not be installed. Run the dependency-free
checks:

```bash
python3 -m compileall -q src/
```

A compile failure is a stop-the-line failure: show it, fix obvious causes from the diff
(e.g. an import-path drift after a rename, or a `from src.config import` name that no
longer exists), and re-run. If `src/config.py` or any importer changed, re-run the
AST config-import check (the same one the verify-changes Stop hook does). If it still
fails, stop and ask.

If a tests/ suite exists, also run `python3 -m pytest -q`.

### Step 4: Run pre-commit hooks (if installed)

```bash
pre-commit run --all-files
```

If `pre-commit` is not installed, note "pre-commit not installed — skipped" and rely on
the auto-format PostToolUse hook having already formatted edited files. If hooks run:
black/isort/flake8/end-of-file/trailing-whitespace auto-fix on a re-run — re-run once. If
they still fail after one auto-fix pass, stop and ask. Never bypass with `--no-verify`. If
`check-added-large-files` or `detect-private-key` trips, do NOT force it — surface the
offending file to the user.

### Step 5: Update documentation

Read `README.md`, `.env.example`, and `CLAUDE.md`; update only sections that drifted:

- **New `config.py` constant / changed default** → README + `.env.example` + `CLAUDE.md`.
- **New module under `src/`** → `CLAUDE.md` package map (+ README structure).
- **New pipeline step / CLI flag** → README "Training pipeline" section.
- **A CRITICAL invariant changed** (causal normalization, group split, MLflow keys,
  autoregressive-primary eval, seq2seq contract) → update the relevant `CLAUDE.md` section.

If nothing drifted, skip this step. Do not rewrite docs that are already correct.

### Step 6: Generate the Conventional Commits message

**Subject** (≤ 72 chars): `type(scope): summary`. Infer the type from the diff:

- new capability (model, dataset, metric, pipeline step) → `feat`
- bug fix → `fix`
- behaviour-preserving restructure → `refactor`
- speed/memory → `perf`
- docs/CLAUDE.md only → `docs`
- tooling/deps/hooks/config → `chore` / `build` / `ci`

If `$ARGUMENTS` supplied a message, use it verbatim as the subject (after the type).

**Body** (after a blank line): one line per significant change. If a methodology contract
changed (normalization, split, MLflow keys, eval protocol, seq2seq shapes), explicitly
note the synchronized consumer/doc updates so the contract reads as kept-whole.

**No AI-attribution trailer** (see Important).

### Step 7: Show summary and confirm

Print: code-review result, verification result, pre-commit result (or "skipped"), doc
updates (or "none"), files to be committed (`git status --short`), and the full commit
message. Then ask with `AskUserQuestion`:

```
question: "Commit and push to origin/main?"
header: "Commit & Push"
options:
  - "Yes" — stage all changes, commit, rebase onto origin/main, push.
  - "No"  — cancel, leave the working tree as-is.
```

Do NOT proceed without an explicit "Yes". If `--no-push` was passed, the option is
"Commit only (no push)".

### Step 8: Commit and push

```bash
git add -A
git commit -m "<subject>

<body>"
git fetch origin main
git rebase origin/main
```

If the rebase conflicts, **abort** (`git rebase --abort`) and tell the user to resolve
manually — do not auto-resolve. Then (unless `--no-push`):

```bash
git push origin main
```

If the push fails (branch protection, auth, network), do NOT retry and do NOT force.
Show the error and suggest pushing a feature branch + opening a PR. `gh` may not be
installed — if it is, `git switch -c <branch> && git push -u origin <branch> && gh pr
create`; otherwise push the branch and give the user the GitHub compare URL.

### Step 9: Final report

```
Pushed to origin/main.
Review: passed (or: N findings)   Verify: compileall passed   Pre-commit: passed | skipped
Doc updates: <files or "none">
```

Or, if the push was blocked, show the error and the feature-branch + PR suggestion.
