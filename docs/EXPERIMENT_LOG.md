# Experiment log

Required by the organizers (data subset, seed, settings, score, runtime). Change **one thing
at a time**, evaluate on validation only, and add one row per run. Scores = validation,
requested SNR, all 13 SNRs unless stated.

| ID | Date | Author | Representation (STFT params…) | Model | Train subset / SNRs | Seed | Mean bal. acc. | Tie-break (−20/−25/−30) | Train time | Infer. ms/pulse | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E001 | 2026-10-12 | équipe | Crop 19–122 µs, RMS norm, STFT Hann 256 / hop 128, two-sided, log, freq pool 2 (128×95) | SpecCNN 16-32-64-128, GAP, 292k params, weighted CE, AdamW one-cycle 2e-3, 8 epochs | 20 pulses/config (13 200), SNR uniform over the 13 | 42 | **0.8298** (val 4 pulses/config × 13 SNR) | 0.3188 | 40 min CPU | 5.1 | Perfect ≥ +5 dB; barker↔cw from 0 dB; chirps confused at −20 dB; 100 µs worst (0.793) |
| E002 | 2026-10-12 | Urfavcurlygirl | Crop 19–122 µs, RMS norm, MultiResSTFT Hann (64, 256, 1024) / hop 128, log, interpolate (area/bilinear) (3×128×95) | SpecCNN in_channels 3, channels 16-32-64-128, GAP, 293k params, weighted CE, AdamW one-cycle 2e-3, 8 epochs | 20 pulses/config (13 200), SNR uniform over the 13 | 42 | **0.8389** (val 4) / **0.8394** (val 20) | 0.3281 (val 4) / 0.3330 (val 20) | 40 min CPU | 9.9 | Retenu (+0,0091 gain > 0,005, 9,9 ms <= 20 ms) : la fenêtre 1024 booste les pulses 100 µs (0,806 vs 0,793) et la fenêtre 64 renforce le rappel barker à −10 dB (0,976 vs 0,788). |
| E002-r | 2026-10-12 | userdeter | Same as E002 (configs/e002_multires.yaml), rerun on a second machine after the code review refactor | Same as E002, 292 567 params | 20 pulses/config (13 200), SNR uniform over the 13 | 42 | **0.8426** (val 4) / **0.8412** (val 20) | 0.3435 (val 4) / 0.3370 (val 20) | 41.7 min CPU | 6.9 | Reproduction of E002: confirmed (+0.013 vs E001 on val 4, same conclusion; run-to-run gap with E002 is 0.004). Gain is at 0 to −20 dB (−20 dB: 0.655 vs 0.591); barker recall at 0/−5/−10 dB 1.00/1.00/0.93 vs 0.93/0.83/0.79; 100 µs 0.810 vs 0.793; −25/−30 dB still near chance. |
| E003 | 2026-10-13 | Urfavcurlygirl | MultiResSTFT Hann (64, 256, 1024) / hop 128, log, interpolate (area/bilinear) (3×128×95) | SpecCNN in_channels 3, channels 16-32-64-128, GAP, 293k params, weighted CE, AdamW 2e-3, 8 epochs | 20 pulses/config (13 200), SNR biased: w=1 (+30 to +5 dB), w=2 (0 to -10 dB), w=4 (-15 to -30 dB) | 42 | **0.8533** (val 4 pulses/config × 13 SNR) | **0.3811** (val 4) | 68.6 min CPU | 15.4 (single pass under load, see E003-r) | Retenu (+0,0144 vs E002, confirmé par E003-r) : parfait >= +5 dB (1,0000 >= 0,99). Gains à bas SNR : -15 dB (0,9608 vs 0,9316), -20 dB (0,7023 vs 0,6083), -25 dB (0,2813 vs 0,2325), -30 dB (0,1598 vs 0,1434). Tie-break +0,0530. Le tirage favorise les SNR bas sans perte à fort SNR ; la cause du gain n'a pas été isolée. |
| E003-r | 2026-10-13 | userdeter | Same as E003 (configs/e003_lowsnr.yaml), rerun on a second machine | Same as E003, 292 567 params | 20 pulses/config (13 200); SNR shares drawn at epoch 1: 3.5 % each from +30 to +5 dB, 7 % each from 0 to −10 dB, 14 % each from −15 to −30 dB | 42 | **0.8533** (val 4) / **0.8528** (val 20) | 0.3812 (val 4) / 0.3791 (val 20) | 63.5 min CPU (same compute as E002, machine under load) | 9.4 | Reproduction of E003: confirmed, kept. On val 20 vs E002-r: +0.0116 mean (rule: >= +0.010), tie-break +0.042; −15 dB 0.957 vs 0.936, −20 dB 0.694 vs 0.624, −25 dB 0.286 vs 0.237, −30 dB 0.157 vs 0.151; +30 to +5 dB all >= 0.9999. By duration 0.885 / 0.854 / 0.819 (10 / 50 / 100 µs). barker→cw is not improved at −15 and −20 dB (131 vs 121 and 442 vs 430 pulses of 1 440). Best epoch is the last one and the monitoring score was still rising (0.849 → 0.854). |
| E004 | 2026-10-13 | Urfavcurlygirl | MultiResSTFT Hann (64, 256, 1024) / hop 128, log, out_shape (256×95) (pleine résolution fréq.) | SpecCNN in_channels 3, channels 16-32-64-128, GAP, 293k params, weighted CE, AdamW 2e-3, 8 epochs | 20 pulses/config (13 200), SNR biased: w=1 (+30 to +5 dB), w=2 (0 to -10 dB), w=4 (-15 to -30 dB) | 42 | **0.8593** (val 4 pulses/config × 13 SNR) | **0.4021** (val 4) | 67.4 min CPU (see notes) | 9.1 (single pass, not comparable) | Gain reproduit par E004-r (voir cette ligne). Chiffres d'origine : +0,0060 vs E003 sur val 4, rappel barker à -10 dB 0,9792 vs 0,9514 (barker->cw : 6 vs 14 pulses sur 288), rappel cw à -15 dB 0,9583 vs 0,8750, tie-break 0,4021 vs 0,3811. Temps d'entraînement et d'inférence mesurés en un seul passage sous charge, non comparables : sur une même machine une époque coûte 1,34 fois celle d'E003 (E004-r). |
| E004-r | 2026-10-14 | userdeter | Same as E004 (configs/e004_fullres.yaml), rerun on a second machine | Same as E004, 292 567 params | Same as E003 (20 pulses/config, SNR weights 1/2/4) | 42 | **0.8614** (val 4) / **0.8609** (val 20) | 0.4117 (val 4) / 0.4095 (val 20) | 80.7 min CPU (7.6 min/epoch vs 5.7 for E003-r) | 12.0 (median of 5 passes; E003-r 9.4) | Reproduction of E004: the gain holds. vs E003-r on the same pulses: +0.0081 on val 4 and on val 20 (provisional band of the rule), second run after E004 (+0.0060), so the rerun the rule asks for is done; tie-break +0.030 (val 20). Gain is at −15 dB (0.971 vs 0.957), −20 dB (0.768 vs 0.694) and −25 dB (0.305 vs 0.286); −30 dB unchanged (0.156), +30 to +5 dB all >= 0.9996. At −20 dB every class improves (barker 0.659 → 0.765, hyp 0.750 → 0.855, quad 0.592 → 0.695, fsk 0.833 → 0.906), so the gain is not limited to barker vs cw; barker→cw drops from 131 to 85 (−15 dB) and from 442 to 306 (−20 dB) of 1 440 barker pulses. By duration 0.886 / 0.862 / 0.835 vs 0.885 / 0.854 / 0.819: the 100 µs pulses gain most. Cost: training ×1.34 per epoch, inference 12.0 ms (< 20 ms). Monitoring score still rising at the last epoch (0.856 → 0.858). |

## Decision rule

An experiment is compared with its parent config on the same validation pulses, on the gain in mean
balanced accuracy (`scripts/compare_runs.py`, `--full` for all 20 validation pulses per configuration):

- **kept** at +0.010 or more;
- **provisional** between +0.005 and +0.010: rerun it and score it on val 20 before building on it;
- **rejected** below +0.005.

Reason: two runs of the same config differ by up to 0.004 (E002 vs E002-r), and val 4 has only 288 barker
and 48 cw pulses per SNR, so a difference of a few pulses moves a class recall by several points. A note
states what was measured; a cause is only written when an experiment isolated it.

## Where the remaining margin is

On val 20 the best run (E004-r) scores 0.8609, so 0.139 is missing. The three tie-break SNRs
(−20, −25, −30 dB) account for 0.136 of it (98 %); −30 dB has not moved since E002 (about 0.16, chance is
0.14). Every experiment so far trained on 20 of the 80 training pulses per configuration (25 %) for
8 epochs, and the monitoring score was still rising at the last epoch: the training budget is the next
variable to test. `configs/e005_fulldata.yaml` is E003 with all 80 pulses; if E004 becomes the parent,
the same change applies to E004 (about 4 times the 80 minutes of E004-r on CPU).
