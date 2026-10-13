# IEEE Datathon — RadarML Lite · Track B (time-frequency)

Radar modulation recognition on real over-the-air recordings (UCL ARESTOR RFSoC).
Given one recorded radar pulse, predict one of **7 modulation classes** and stay
robust from **+30 dB down to −30 dB** requested SNR.

| | |
| --- | --- |
| Official challenge repo | [UCLRadarGroup/Radar_ML_lite](https://github.com/UCLRadarGroup/Radar_ML_lite) (vendored at commit `490cb87`) |
| Dataset | [UCL RDR — Radar ML Lite](https://rdr.ucl.ac.uk/articles/dataset/Radar_ML_-_Lite/33977674) · paper: White, Hosford, Ritchie — *RadarML* |
| Our track | **Track B — time-frequency** (STFT / spectrograms or other TF representations) |
| Classes | `barker`, `cw`, `fmcw`, `fsk`, `hyp`, `nlfm`, `quad` |
| Metric | Mean over 13 SNRs of the 7-class **balanced accuracy**; tie-break on −20/−25/−30 dB |

> **Start here:** [docs/CHALLENGE_BRIEF.md](docs/CHALLENGE_BRIEF.md) (summary of the data, rules and
> pitfalls) then [CONTRIBUTING.md](CONTRIBUTING.md) (team workflow).

## Quick start

```powershell
# 1. Environment (Python >= 3.10)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .

# 2. Data: put recorded_dataset.zip at the repo root, then extract + verify (~4.4 GB)
.\.venv\Scripts\python.exe scripts\setup_data.py --zip recorded_dataset.zip

# 3. Sanity tests (includes a bit-exact check of the official SNR expansion)
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS/Linux replace `.\.venv\Scripts\python.exe` by `.venv/bin/python`.

## Running an experiment

One experiment = one YAML file in `configs/` (copy the best one, change one thing).

```powershell
# Train (CPU or GPU). Writes runs/<experiment>_<timestamp>/ (git-ignored):
# config.yaml, model.pt, history.csv, validation_report.json, figures/
.\.venv\Scripts\python.exe scripts\train.py --config configs\e002_multires.yaml [--device auto|cpu|cuda]

# Full validation (20 pulses per configuration x 13 SNRs) of a finished run
.\.venv\Scripts\python.exe scripts\evaluate.py --run runs\<experiment>_<timestamp>

# Compare a run with its parent, SNR by SNR (--full: on the full validation written by evaluate.py)
.\.venv\Scripts\python.exe scripts\compare_runs.py --parent runs\<parent> --run runs\<experiment> [--full]

# 2-minute smoke test of a config
.\.venv\Scripts\python.exe scripts\train.py --config configs\e001_baseline.yaml --epochs 1 --train-pulses 2
```

Then add a row to [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

## Repository layout

```text
.
├── configs/                 # one YAML per experiment (E001 baseline, E002 multi-resolution, ...)
├── data/
│   ├── manifest/            # official 80/20 split (tracked, SHA256-verified)
│   └── raw/                 # extracted recordings (git-ignored, ~4.4 GB)
├── docs/
│   ├── CHALLENGE_BRIEF.md   # data, rules, metric, timeline, pitfalls
│   ├── EXPERIMENT_LOG.md    # one line per experiment (mandatory for the report)
│   └── SUBMISSION_CHECKLIST.md
├── notebooks/
│   ├── 01_eda.ipynb         # exploration + quantitative validation of the data
│   ├── 02_baseline.ipynb    # first end-to-end Track B pipeline (reference score)
│   └── colab_train.ipynb    # train + evaluate + compare configs on a free Colab GPU
├── reports/figures/         # figures for the final presentation
├── models/                  # checkpoints (git-ignored, hashed at freeze)
├── runs/                    # one folder per training run (git-ignored)
├── scripts/
│   ├── setup_data.py        # extract + integrity checks
│   ├── train.py             # train an experiment from its config
│   ├── evaluate.py          # full validation of a finished run
│   ├── compare_runs.py      # a run against its parent: per SNR, per class, costs, verdict
│   ├── validate_submission.py
│   └── hash_artifacts.py    # SHA256 + env capture for the Day-4 model freeze
├── src/radarml/             # shared, tested utilities
│   ├── constants.py         # classes, sample rate, the 13 SNRs
│   ├── manifest.py          # split loading, labels from filenames
│   ├── noise.py             # on-the-fly SNR degradation == official expansion
│   ├── features.py          # crop 19-122 us, per-pulse RMS normalisation, STFT front-ends
│   ├── data.py              # PyTorch datasets (train: random SNR; eval: official noise)
│   ├── models.py            # spectrogram CNN trained from scratch
│   ├── training.py          # class weights, train / predict loops
│   ├── metrics.py           # challenge metric == official scorer
│   ├── evaluation.py        # validation report, score per duration, inference cost
│   ├── comparison.py        # run against parent run: per-SNR deltas, class recalls, decision rule
│   └── plots.py             # validation figures written by the scripts
├── tests/
└── third_party/radarml_lite/  # organizers' scripts, unmodified (GPL-3.0)
```

## Shared utilities

```python
from radarml.manifest import load_manifest, select
from radarml.noise import degrade, degrade_official, to_complex
from radarml.metrics import challenge_report

refs = load_manifest("data/raw/recorded_dataset", "data/manifest/split_manifest.csv")
train = select(refs, "train")                          # 52,800 source pulses
val = select(refs, "validation")                       # 13,200 source pulses

# Validation at a requested SNR, identical to the organizers' expanded arrays:
row = np.load(f"data/raw/recorded_dataset/{val[0].file}", mmap_mode="r")[0, val[0].pulse_index]
iq = to_complex(degrade_official(row, val[0].source_file, -10, val[0].pulse_index))  # (16800,) complex64
```

**Why on-the-fly noise?** The full expansion is 57.66 GB. `radarml.noise` calls the
organizers' own `degrade()` with their seed convention, so validation examples are
bit-identical to the official ones (verified by `tests/test_core.py`), while training
can draw fresh noise realisations every epoch.

## Key rules (details in the brief)

- Train **from scratch** on training pulses only; validation for tuning only. No pretrained weights, no external data.
- Never use filenames, IDs, metadata or requested SNR as model inputs.
- Every SNR variant/augmentation inherits its source pulse's split — never re-split.
- **Do not download `blind_test_participants.zip` before Day 5 09:00.**
- Freeze **Day 4 17:00** · blind test **Day 5 09:00** · submission **Day 6 17:00** (Europe/London).
- Disclose AI assistance and cite reused code in the method summary.

## Attribution & licence

Dataset: Ritchie, White and Hosford (2025), [RadarML, v1](https://doi.org/10.5522/04/30752767.v1) — CC BY 4.0.
Code in `third_party/radarml_lite` © UCL Radar Group, GPL-3.0; `radarml.noise` imports it,
so this repository is distributed under **GPL-3.0** ([LICENSE](LICENSE)).
