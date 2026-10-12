# Experiment log

Required by the organizers (data subset, seed, settings, score, runtime). Change **one thing
at a time**, evaluate on validation only, and add one row per run. Scores = validation,
requested SNR, all 13 SNRs unless stated.

| ID | Date | Author | Representation (STFT params…) | Model | Train subset / SNRs | Seed | Mean bal. acc. | Tie-break (−20/−25/−30) | Train time | Infer. ms/pulse | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E001 | 2026-10-12 | équipe | Crop 19–122 µs, RMS norm, STFT Hann 256 / hop 128, two-sided, log, freq pool 2 (128×95) | SpecCNN 16-32-64-128, GAP, 292k params, weighted CE, AdamW one-cycle 2e-3, 8 epochs | 20 pulses/config (13 200), SNR uniform over the 13 | 42 | **0.8298** (val 4 pulses/config × 13 SNR) | 0.3188 | 40 min CPU | 5.1 | Perfect ≥ +5 dB; barker↔cw from 0 dB; chirps confused at −20 dB; 100 µs worst (0.793) |
