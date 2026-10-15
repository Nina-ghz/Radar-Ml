# Method summary — team POPCORN, Track B (time-frequency)

| | |
| --- | --- |
| Group | POPCORN |
| Kaggle team | `POPCORN_freq` (Track B only; no Track A entry) |
| Submitted model | E007: `configs/e007_templates_freq.yaml`, run `E007_templates_freq_20261016_002354` |
| `model.pt` SHA256 | `6b73a11b0f523c375127ca8a12b7b0440da52b71bc51f448a630f837402ccb68` |
| Prediction file and code revision | `INFERENCE_RUN.md` and `MANIFEST.json` in the submission package |

## 1. Methodology

**Observation.** Each of the 660 waveform configurations repeats the same waveform with the same carrier
phase: across the training pulses of a configuration the phase of the correlation with the first pulse
varies by about 0.1° and the normalised correlation is 0.998 to 1.000. Training, validation and blind
test share these 660 configurations (new pulses of known configurations). The class of a pulse can
therefore be read from the configuration it matches, and a coherent correlation over the whole pulse
keeps that match reliable far below the noise floor, where magnitude spectrograms lose it.

**Processing (per pulse, fixed rules, no learned preprocessing).**

1. Interleaved int16 I/Q `(33600,)` → complex `(16800,)` at 120 MS/s.
2. Fixed crop to 19–122 µs (samples 2 280 to 14 640): every pulse of the 660 configurations lies in
   20.1–120.6 µs (measured on the training data). No filename, ID, metadata or requested SNR is used.
3. Normalisation by the pulse's own RMS.

**Time-frequency representation (Track B), computed inside the model and counted in inference time.**
Complex STFT of the cropped pulse: Hann window of 256 samples, hop 64, two-sided, phase kept
(190 frames × 256 bins = 48 640 complex coefficients).

**Model: coherent template matching in the STFT domain.**

- *Fit (training pulses only):* for each configuration, the template is the mean of its 80 preprocessed
  training pulses; its complex STFT is normalised to unit norm. No validation or test pulse is used, and
  no noise is added. 660 templates (65.3 MB stored as time-domain I/Q; their STFT is computed at load).
- *Decision:* score of a pulse against each template = real part of the inner product between the pulse's
  complex STFT and the template's; the predicted class is the class of the best-scoring template.
- There are no trainable weights and no random seed; fitting is deterministic.

With a Hann window at 75 % overlap the STFT-domain correlation is close to a time-domain matched filter.
The same matcher applied directly to the I/Q (E008, not submitted) gave the same validation scores.

**How the method was chosen.** One change at a time, each compared with its parent on the same
validation pulses (`docs/EXPERIMENT_LOG.md`). The first models were CNNs on magnitude spectrograms; the
best (E006) reached 0.8853 and was limited by −25 dB (0.437) and −30 dB (0.175), where the magnitude
representation discards the phase. Keeping the phase and matching coherently (E007) removed that limit.

| Experiment | Method | Validation mean (20 pulses/config) | Tie-break |
| --- | --- | --- | --- |
| E001 | CNN on a single magnitude STFT, 20 pulses/config, uniform SNRs, 8 epochs | 0.8298 (4 pulses/config) | 0.3188 |
| E002–E004 | Multi-resolution STFT, low-SNR weighting, full frequency resolution | 0.8412 → 0.8609 | 0.3370 → 0.4095 |
| E005–E006 | All 80 training pulses, 20 epochs, two seeds each | 0.8770 → 0.8853 | 0.4737 → 0.5069 |
| **E007** | **Coherent template matching on the complex STFT** | **0.9962** | **0.9836** |

## 2. Validation results of the submitted model

Official validation split, 20 pulses per configuration at the 13 requested SNRs with the organizers'
deterministic noise (171 600 examples), scored as the official metric.

**Mean balanced accuracy 0.9962; tie-break (−20, −25, −30 dB) 0.9836.**

| Requested SNR (dB) | +30 … −25 | −30 |
| --- | --- | --- |
| Balanced accuracy | 1.0000 at every SNR | 0.9508 |

By pulse duration (analysis only, from filenames): 10 µs 0.9964, 50 µs 0.9961, 100 µs 0.9961.
Confusion matrices: `figures/05_confusion.png` and all per-SNR matrices in `model/evaluation_report.json`.
The −30 dB errors are spread over all classes; at that SNR 5.4 % of the I/Q values are clipped to the int16
range by the organizers' noise generation.

## 3. Hardware, time and size

| | |
| --- | --- |
| Fitting hardware and time | CPU only: Intel Core i5-1135G7 (4 cores, 8 threads), 7.7 GB RAM, Windows 11; 0.7 min |
| Inference hardware | the same CPU |
| Inference time per pulse, CPU | 1.9 ms for the model including the STFT (batch of 256, median of 5 passes); end to end on the blind test: see `INFERENCE_RUN.md` |
| Model size | 660 templates of 12 360 complex samples, 65.3 MB; no trainable parameters |
| Software | Python 3.12.10, PyTorch 2.14.1 (CPU), NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, PyYAML 6.0.3; full list in `MANIFEST.json` |

## 4. Attribution and AI assistance

- Dataset: Ritchie, White and Hosford (2025), *RadarML*, version 1 (CC BY 4.0), RadarML Lite release.
- Code: `third_party/radarml_lite/` is copied unmodified from UCLRadarGroup/Radar_ML_lite (GPL-3.0); its
  `degrade()` generates the validation noise and its scorer is mirrored by our metric. The rest of the
  code is ours.
- AI assistants were used for code and documentation: Claude Code (Anthropic) and Antigravity with
  Gemini (Google). All results were run and checked by the team.

## 5. Limitations and timeline

- The method relies on the blind test containing new pulses of the 660 known configurations, as stated
  by the organizers; it would not recognise a waveform configuration absent from the training data.
- Validation measures new pulses of known configurations, not unseen configurations or receivers.
- The model freeze was cancelled by the organizers. The method was developed and selected on 16 October
  2026 using the training and validation splits only. No training, tuning or model selection used the
  blind test, which was only read to produce the final predictions.
