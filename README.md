# Radar modulation recognition below the noise floor

**IEEE Datathon — RadarML Lite, Track B (time-frequency) · team POPCORN**

Given one radar pulse recorded over the air (UCL ARESTOR RFSoC platform), predict its modulation among
7 classes, at 13 noise levels from +30 dB down to −30 dB requested SNR, where the noise is 1 000 times
stronger than the signal.

| Submitted model (E007) | Validation score | Tie-break (−20/−25/−30 dB) | CPU inference |
| --- | --- | --- | --- |
| Coherent template matching on the complex STFT | **0.9962** | **0.9836** | **6.1 ms** per signal, end to end |

Scores are the official metric on the full validation split (13 200 pulses × 13 SNRs = 171 600 examples).

## The approach in short

1. **A CNN on spectrograms (E001 → E006).** Multi-resolution magnitude STFTs fed to a compact CNN trained
   from scratch. Six controlled experiments took the score from 0.830 to 0.885, but the CNN stayed near chance
   at −25 dB (0.437) and −30 dB (0.175): a magnitude spectrogram discards the phase, so the signal cannot be
   summed coherently once noise dominates every time-frequency cell.
2. **What the data showed.** Every one of the 660 waveform configurations repeats the same waveform with the
   same carrier phase (phase spread about 0.1° across training pulses), and validation and test contain new
   pulses of these same configurations.
3. **The submitted model (E007).** One template per configuration, the mean of its training pulses; a pulse
   is compared with every template on its **complex STFT** (Hann 256, hop 64, phase kept) and takes the class
   of the template with the largest real correlation. This coherent sum over the whole pulse gives about
   +10.7 dB of effective SNR at −30 dB, and lifts every SNR from +30 to −25 dB to 1.000 (0.951 at −30 dB).

| Requested SNR (dB) | +30 … −5 | −10 | −15 | −20 | −25 | −30 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E006 — CNN on spectrograms | 1.000 | 0.999 | 0.989 | 0.909 | 0.437 | 0.175 | 0.8853 |
| **E007 — complex-STFT template matching** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **0.951** | **0.9962** |

Full method, hardware and limitations: [docs/METHOD_SUMMARY.md](docs/METHOD_SUMMARY.md). Every experiment,
with its settings and validation scores: [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

## Challenge

| | |
| --- | --- |
| Data | 660 configurations (modulation × pulse duration × bandwidth × centre frequency), 80 training and 20 validation pulses each; 16 800 complex I/Q samples per pulse at 120 MS/s |
| Classes | `barker`, `cw`, `fmcw`, `fsk`, `hyp`, `nlfm`, `quad` |
| Metric | mean over the 13 requested SNRs of the 7-class balanced accuracy; tie-break on −20/−25/−30 dB |
| Track B | time-frequency representations computed from the I/Q, included in the inference time |
| Sources | official repository [UCLRadarGroup/Radar_ML_lite](https://github.com/UCLRadarGroup/Radar_ML_lite) · dataset [Radar ML — Lite](https://rdr.ucl.ac.uk/articles/dataset/Radar_ML_-_Lite/33977674) |

## Quick start

```powershell
# Environment (Python >= 3.10)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .

# Data: put recorded_dataset.zip at the repository root, then extract and verify (~4.4 GB)
.\.venv\Scripts\python.exe scripts\setup_data.py --zip recorded_dataset.zip

# Tests (include a bit-exact check of the official noise generation)
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS/Linux replace `.\.venv\Scripts\python.exe` by `.venv/bin/python`.

## Reproduce the results

```powershell
# Submitted model: fit the 660 templates on the training pulses (under a minute on CPU) and score validation
.\.venv\Scripts\python.exe scripts\fit_templates.py --config configs\e007_templates_freq.yaml

# Best CNN (GPU recommended; notebooks/colab_train.ipynb runs it on a free Colab GPU)
.\.venv\Scripts\python.exe scripts\train.py --config configs\e006_seed43.yaml --device cuda

# Full validation of any run, then a comparison with its parent, SNR by SNR
.\.venv\Scripts\python.exe scripts\evaluate.py --run runs\<run>
.\.venv\Scripts\python.exe scripts\compare_runs.py --parent runs\<parent> --run runs\<run> --full

# Blind-test predictions (format sample_id,label), see docs/INFERENCE.md
.\.venv\Scripts\python.exe scripts\predict.py --run runs\<run> --test-dir <blind test>\test --output predictions.csv --device cpu
```

Runs are written to `runs/<experiment>_<timestamp>/` (config, model, validation report, figures). The noise
of each training example is drawn on the fly with the organizers' own `degrade()`, and validation uses their
deterministic seeds, so validation examples are bit-identical to the official expansion without storing its
57.66 GB.

## Repository layout

```text
.
├── configs/                 # one YAML per experiment: E001-E006 (CNN), E007 (submitted model)
├── data/manifest/           # official 80/20 split (tracked, SHA256-verified); recordings go in data/raw/
├── docs/
│   ├── METHOD_SUMMARY.md    # submitted method, validation results, hardware, limitations
│   ├── EXPERIMENT_LOG.md    # every experiment, one change at a time, with its scores
│   ├── INFERENCE.md         # blind-test inference and its checks
│   └── CHALLENGE_BRIEF.md   # data, rules and metric
├── notebooks/
│   ├── 01_eda.ipynb         # data exploration and checks
│   ├── 02_baseline.ipynb    # first end-to-end pipeline
│   ├── 03_error_analysis.ipynb  # where the best CNN fails, and why
│   └── colab_train.ipynb    # train and evaluate configs on a Colab GPU
├── reports/figures/         # figures used in the reports
├── scripts/
│   ├── setup_data.py        # extract and verify the dataset
│   ├── train.py             # train a CNN config
│   ├── fit_templates.py     # fit the template matcher (submitted model)
│   ├── evaluate.py          # full validation of a run
│   ├── compare_runs.py      # a run against its parent: per SNR, per class, costs, decision rule
│   ├── predict.py           # blind test -> predictions.csv
│   ├── make_fake_blind_test.py  # official-format test set built from validation only
│   ├── validate_submission.py   # format check of a predictions file
│   ├── build_submission.py  # reproduction package for the organizers
│   └── hash_artifacts.py    # SHA256 of artifacts and environment capture
├── src/radarml/             # tested library
│   ├── features.py          # fixed crop 19-122 µs, per-pulse RMS normalisation, STFT front-ends
│   ├── templates.py         # complex-STFT template matcher (submitted model)
│   ├── models.py            # spectrogram CNN, model factory
│   ├── data.py, noise.py    # datasets with the official noise generation
│   ├── training.py, metrics.py, evaluation.py, comparison.py, blind.py, plots.py
│   └── constants.py, manifest.py
├── tests/
└── third_party/radarml_lite/  # organizers' scripts, unmodified (GPL-3.0)
```

## Rules we worked under

- Models built from the training pulses only; validation used to evaluate and choose, never to fit.
- No pretrained weights, no external data; no filename, ID, metadata or requested SNR as model input.
- The blind test was never used to train, tune or choose, and never left the team's machine.
- One change per experiment, kept only with a gain of at least +0.010 on the full validation; smaller gains
  were rerun on a second machine or with another seed before being kept.

## Attribution and licence

Dataset: Ritchie, White and Hosford (2025), [RadarML, version 1](https://doi.org/10.5522/04/30752767.v1), CC BY 4.0.
`third_party/radarml_lite/` is © UCL Radar Group under GPL-3.0 and is imported by `radarml.noise`, so this
repository is distributed under **GPL-3.0** ([LICENSE](LICENSE)).
