#!/usr/bin/env python3
"""Compare E003 (low SNR weighted) with E002 baseline SNR by SNR."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

# Find latest E002 and E003 runs
e002_dirs = sorted(RUNS.glob("E002_multires_*"))
e003_dirs = sorted(RUNS.glob("E003_lowsnr_*"))

if not e002_dirs:
    raise FileNotFoundError("No E002 run found")
if not e003_dirs:
    raise FileNotFoundError("No E003 run found")

e002_report = json.loads((e002_dirs[-1] / "validation_report.json").read_text(encoding="utf-8"))
e003_report = json.loads((e003_dirs[-1] / "validation_report.json").read_text(encoding="utf-8"))

e002_snr = {s["requested_snr_db"]: s["balanced_accuracy"] for s in e002_report["per_snr"]}
e003_snr = {s["requested_snr_db"]: s["balanced_accuracy"] for s in e003_report["per_snr"]}

print("=" * 65)
print(f"Comparaison SNR par SNR : E002 (parent) vs E003 (low-SNR weights)")
print(f"E002 run : {e002_dirs[-1].name}")
print(f"E003 run : {e003_dirs[-1].name}")
print("=" * 65)
print(f"{'SNR (dB)':>8} | {'E002 bal acc':>12} | {'E003 bal acc':>12} | {'Delta':>8} | {'Vérif':>6}")
print("-" * 65)

high_snrs_ok = True
for snr in sorted(e002_snr.keys(), reverse=True):
    b02 = e002_snr[snr]
    b03 = e003_snr[snr]
    delta = b03 - b02
    check = ""
    if snr >= 5:
        if b03 >= 0.99:
            check = ">= 0.99"
        else:
            check = "FAIL"
            high_snrs_ok = False
    elif snr in (-15, -20, -25):
        check = f"{delta:+.3f}"
    print(f"{snr:+8d} | {b02:12.4f} | {b03:12.4f} | {delta:+8.4f} | {check:>6}")

print("-" * 65)
mean_02 = e002_report["mean_balanced_accuracy"]
mean_03 = e003_report["mean_balanced_accuracy"]
tie_02 = e002_report["tiebreak_low_snr"]
tie_03 = e003_report["tiebreak_low_snr"]

print(f"{'MOYENNE':>8} | {mean_02:12.4f} | {mean_03:12.4f} | {mean_03 - mean_02:+8.4f} |")
print(f"{'TIE-BREAK':>8} | {tie_02:12.4f} | {tie_03:12.4f} | {tie_03 - tie_02:+8.4f} | (-20/-25/-30 dB)")
print("=" * 65)
print(f"Vérification >= +5 dB resté >= 0.99 : {'SUCCÈS' if high_snrs_ok else 'ATTENTION'}")
