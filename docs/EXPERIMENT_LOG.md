# Experiment log

Required by the organizers (data subset, seed, settings, score, runtime). Change **one thing
at a time**, evaluate on validation only, and add one row per run. Scores = validation,
requested SNR, all 13 SNRs unless stated.

| ID | Date | Author | Representation (STFT params…) | Model | Train subset / SNRs | Seed | Mean bal. acc. | Tie-break (−20/−25/−30) | Train time | Infer. ms/pulse | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E001 | 2026-10-12 | équipe | Crop 19–122 µs, RMS norm, STFT Hann 256 / hop 128, two-sided, log, freq pool 2 (128×95) | SpecCNN 16-32-64-128, GAP, 292k params, weighted CE, AdamW one-cycle 2e-3, 8 epochs | 20 pulses/config (13 200), SNR uniform over the 13 | 42 | **0.8298** (val 4 pulses/config × 13 SNR) | 0.3188 | 40 min CPU | 5.1 | Perfect ≥ +5 dB; barker↔cw from 0 dB; chirps confused at −20 dB; 100 µs worst (0.793) |
| E002 | 2026-10-12 | Urfavcurlygirl | Crop 19–122 µs, RMS norm, MultiResSTFT Hann (64, 256, 1024) / hop 128, log, interpolate (area/bilinear) (3×128×95) | SpecCNN in_channels 3, channels 16-32-64-128, GAP, 293k params, weighted CE, AdamW one-cycle 2e-3, 8 epochs | 20 pulses/config (13 200), SNR uniform over the 13 | 42 | **0.8389** (val 4) / **0.8394** (val 20) | 0.3281 (val 4) / 0.3330 (val 20) | 40 min CPU | 9.9 | Retenu (+0,0091 gain > 0,005, 9,9 ms <= 20 ms) : la fenêtre 1024 booste les pulses 100 µs (0,806 vs 0,793) et la fenêtre 64 renforce le rappel barker à −10 dB (0,976 vs 0,788). |
| E002-r | 2026-10-12 | userdeter | Same as E002 (configs/e002_multires.yaml), rerun on a second machine after the code review refactor | Same as E002, 292 567 params | 20 pulses/config (13 200), SNR uniform over the 13 | 42 | **0.8426** (val 4) / **0.8412** (val 20) | 0.3435 (val 4) / 0.3370 (val 20) | 41.7 min CPU | 6.9 | Reproduction of E002: confirmed (+0.013 vs E001 on val 4, same conclusion; run-to-run gap with E002 is 0.004). Gain is at 0 to −20 dB (−20 dB: 0.655 vs 0.591); barker recall at 0/−5/−10 dB 1.00/1.00/0.93 vs 0.93/0.83/0.79; 100 µs 0.810 vs 0.793; −25/−30 dB still near chance. |
