# Method summary — team POPCORN, Track B (time-frequency)

| | |
| --- | --- |
| Group | POPCORN |
| Kaggle team | `POPCORN_freq` (Track B only; no Track A entry) |
| Submitted model | E006-s43: `configs/e006_seed43.yaml`, run `E006_seed43_20261014_204114`, best epoch 19 of 20 |
| `model.pt` SHA256 | `f6239747e2e397bf0fb2e1be031c08995abb7e0594dfa2174be63dbf0a982993` |
| Code revision, prediction file hash | see `MANIFEST.json` and `INFERENCE_RUN.md` in the submission package |

## 1. Methodology

**Input and processing (per pulse, fixed rules, no learned preprocessing).**

1. Interleaved int16 I/Q `(33600,)` → complex `(16800,)` at 120 MS/s.
2. Fixed crop to 19–122 µs (samples 2 280 to 14 640): every pulse of the 660 configurations lies in
   20.1–120.6 µs (measured on the training data). No filename, ID, metadata or requested SNR is used.
3. Normalisation by the pulse's own RMS, so absolute amplitude is never a cue.

**Time-frequency representation (Track B), computed inside the model and counted in inference time.**
Three STFTs of the same complex pulse with Hann windows of 64, 256 and 1 024 samples, hop 128,
two-sided (complex baseband), fftshifted, log power in dB. Each spectrogram is resized to 256 × 95
(area interpolation when shrinking, bilinear when growing), standardised per example and per channel,
then stacked as a 3 × 256 × 95 image. Long windows concentrate the energy of weak long pulses; short
windows keep fast phase transitions.

**Model.** Compact CNN trained from scratch (no pretrained weights, no external data): four blocks
(16, 32, 64, 128 channels; 3 × 3 convolutions + batch norm + ReLU, one convolution in the first block and
two in the others, 2 × 2 max pooling), global average pooling, dropout 0.3, linear layer to the 7 classes.
292 567 parameters, 1.19 MB checkpoint.

**Training.**

| Setting | Value |
| --- | --- |
| Training data | all 80 training pulses of each of the 660 configurations (52 800 pulses), official 80/20 manifest; no validation pulse used for training |
| Noise | requested SNR drawn at every access with the organizers' `degrade()` (fresh noise each epoch), weights 1 for +30…+5 dB, 2 for 0…−10 dB, 4 for −15…−30 dB |
| Loss | cross-entropy with inverse-frequency class weights (balanced-accuracy metric), label smoothing 0.05 |
| Optimiser | AdamW, one-cycle learning rate up to 2e-3, weight decay 1e-4, batch 64, 20 epochs |
| Checkpoint | best epoch on a monitoring subset of the validation split (1 pulse per configuration): epoch 19 |
| Seed | 43 (`torch`, `numpy` and the noise generator) |

**How the method was chosen.** One change at a time, each compared with its parent on the same
validation pulses, and kept only if the mean balanced accuracy rose by at least 0.010 (gains between
0.005 and 0.010 were rerun before being kept). Full log: `docs/EXPERIMENT_LOG.md`.

| Experiment | Change | Validation mean (20 pulses/config) | Tie-break |
| --- | --- | --- | --- |
| E001 | Baseline: single STFT 256, 128 × 95, 20 pulses/config, uniform SNRs, 8 epochs | 0.8298 (4 pulses/config) | 0.3188 |
| E002 | Multi-resolution STFT (64, 256, 1 024) | 0.8412 (rerun) | 0.3370 |
| E003 | Training SNRs weighted towards low SNR (1/2/4) | 0.8528 (rerun) | 0.3791 |
| E004 | Full frequency resolution, 256 × 95 instead of 128 × 95 | 0.8609 (rerun) | 0.4095 |
| E005 | All 80 training pulses per configuration instead of 20 | 0.8770 / 0.8773 (seeds 42 / 43) | 0.4737 / 0.4739 |
| **E006** | **20 epochs instead of 8** | **0.8837 / 0.8853 (seeds 42 / 43)** | **0.5003 / 0.5069** |

E006 beat E005 with both seeds (+0.0066 and +0.0081) while two seeds of one configuration differed by
0.0003 to 0.0017. The seed-43 model, the better of the two on validation, is submitted.

## 2. Validation results of the submitted model

Official validation split, 20 pulses per configuration at the 13 requested SNRs with the organizers'
deterministic noise (171 600 examples), scored as the official metric.

**Mean balanced accuracy 0.8853; tie-break (−20, −25, −30 dB) 0.5069.**

| Requested SNR (dB) | +30 … 0 | −5 | −10 | −15 | −20 | −25 | −30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced accuracy | 1.0000 | 1.0000 | 0.9989 | 0.9893 | 0.9092 | 0.4370 | 0.1746 |

By pulse duration (analysis only, from filenames): 10 µs 0.9093, 50 µs 0.8840, 100 µs 0.8626.
Confusion matrices: `figures/05_confusion.png` (0, −10 and −20 dB) and the full per-SNR matrices in
`model/evaluation_report.json`; error analysis in `notebooks/03_error_analysis.ipynb`.

Main errors: at −20 dB short Barker codes (Barker 5 and 7) are predicted as `cw` and `nlfm` as `fmcw`;
at −25 dB all seven classes remain above chance; at −30 dB only `quad` and `fmcw` stay above chance
(1/7) and the model mostly answers these two classes.

## 3. Hardware, time and size

| | |
| --- | --- |
| Training hardware | NVIDIA Tesla T4 GPU (Google Colab), 28.0 min for 20 epochs |
| Inference hardware | CPU only: Intel Core i5-1135G7 (4 cores, 8 threads), 7.7 GB RAM, Windows 11 |
| Inference time per pulse, CPU | 12 ms for the model including the three STFTs (batch of 256, median of 5 passes); about 22 ms end to end including reading the shards and preprocessing (measured on a validation-built test set in the official format). Blind-test run: see `INFERENCE_RUN.md` |
| Model size | 292 567 parameters, 1.19 MB |
| Inference software | Python 3.12.10, PyTorch 2.14.1 (CPU), NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, PyYAML 6.0.3; full list in `MANIFEST.json` |
| Training software | the Google Colab GPU image of 14 October 2026 with the same `requirements.txt`; exact package versions of that session were not recorded |

## 4. Attribution and AI assistance

- Dataset: Ritchie, White and Hosford (2025), *RadarML*, version 1 (CC BY 4.0), RadarML Lite release.
- Code: `third_party/radarml_lite/` is copied unmodified from UCLRadarGroup/Radar_ML_lite (GPL-3.0);
  its `degrade()` generates the training noise and validation noise, and its scorer is mirrored by our
  metric. The rest of the code is ours.
- AI assistants were used for code and documentation: Claude Code (Anthropic) and Antigravity with
  Gemini (Google). All results were run and checked by the team.

## 5. Limitations and timeline

- Validation measures new pulses of known configurations, not unseen configurations or receivers.
- One model (one seed) is submitted; −25 dB and −30 dB remain close to chance.
- The exact package versions of the Colab training session were not recorded.
- The model was selected on validation on 14 October 2026; the blind test was downloaded on
  15 October 2026 after that choice. No training, tuning or model selection used the blind test.
  The only change after selection is the output header (`sample_id,label`, required by Kaggle).
