# Team workflow

## Branches

- `main` is protected in spirit: always runnable, tests passing.
- Work on short-lived branches: `feat/<topic>`, `exp/<experiment-id>`, `fix/<topic>`.
- Open a pull request into `main`; one teammate reviews before merge.

## Commits

- Small, descriptive commits in the imperative: `Add STFT front-end`, `Log E004 results`.
- **Never commit data, checkpoints or blind-test files** (`.gitignore` covers `data/raw/`, `models/`, `*.npy`, `*.zip`, `*.pt`).
- Run `python -m pytest -q` before pushing.

## Experiments

1. Copy a config, change one thing, fix the seed.
2. Evaluate on the official validation split at all 13 requested SNRs (`radarml.metrics.challenge_report`).
3. Add a row to [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) — even for runs that did not help.

## Notebooks

- Name them `NN_initials_topic.ipynb`; clear large outputs before committing.
- Code reused by more than one notebook moves into `src/radarml/` with a test.

## Non-negotiable rules

- No validation pulse in training; no re-splitting; no filename/metadata/SNR as model input.
- No pretrained weights or external data.
- Blind test: not before Day 5 09:00, never sent to external services (including AI assistants).
