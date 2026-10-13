#!/usr/bin/env python3
"""Compare E004 (full resolution 256x95) with E003 parent."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

e003_dirs = sorted(RUNS.glob("E003_lowsnr_*"))
e004_dirs = sorted(RUNS.glob("E004_fullres_*"))

if not e003_dirs:
    raise FileNotFoundError("No E003 run found")
if not e004_dirs:
    raise FileNotFoundError("No E004 run found")

e003_rep = json.loads((e003_dirs[-1] / "validation_report.json").read_text(encoding="utf-8"))
e004_rep = json.loads((e004_dirs[-1] / "validation_report.json").read_text(encoding="utf-8"))

print("=" * 75)
print(f"COMPARAISON E003 (parent 128x95) vs E004 (full-res 256x95)")
print(f"E003 run : {e003_dirs[-1].name}")
print(f"E004 run : {e004_dirs[-1].name}")
print("=" * 75)

# 1. Temps & Coûts
e03_costs = e003_rep["costs"]
e04_costs = e004_rep["costs"]
e03_ep_time = sum(h["minutes"] for h in e003_rep["history"]) / len(e003_rep["history"])
e04_ep_time = sum(h["minutes"] for h in e004_rep["history"]) / len(e004_rep["history"])

print("\n1. RESSOURCES & TEMPS DE CALCUL :")
print(f"  - Temps moyen / époque : E003 = {e03_ep_time:.2f} min  |  E004 = {e04_ep_time:.2f} min (x{e04_ep_time/e03_ep_time:.2f})")
print(f"  - Temps d'entraînement total : E003 = {e03_costs['entraînement (min)']:.1f} min  |  E004 = {e04_costs['entraînement (min)']:.1f} min")
print(f"  - Inférence totale (ms/pulse) : E003 = {e03_costs['inférence (ms/pulse, STFT incluse)']:.2f} ms  |  E004 = {e04_costs['inférence (ms/pulse, STFT incluse)']:.2f} ms")
print(f"  - Dont STFT (ms/pulse)       : E003 = {e03_costs['dont STFT (ms/pulse)']:.2f} ms  |  E004 = {e04_costs['dont STFT (ms/pulse)']:.2f} ms")

# 2. Scores globaux
mean_03 = e003_rep["mean_balanced_accuracy"]
mean_04 = e004_rep["mean_balanced_accuracy"]
tie_03 = e003_rep["tiebreak_low_snr"]
tie_04 = e004_rep["tiebreak_low_snr"]

print("\n2. SCORES GLOBAUX :")
print(f"  - Mean balanced acc : E003 = {mean_03:.4f}  |  E004 = {mean_04:.4f}  |  Delta = {mean_04 - mean_03:+.4f}")
print(f"  - Tie-break (-20/-25/-30) : E003 = {tie_03:.4f}  |  E004 = {tie_04:.4f}  |  Delta = {tie_04 - tie_03:+.4f}")

# 3. Analyse barker vs cw à 0, -5, -10, -15 dB
print("\n3. ANALYSE SPÉCIFIQUE BARKER / CW (0, -5, -10, -15 dB) :")
print(f"{'SNR':>6} | {'Rappel Barker E003':>18} -> {'E004':>6} | {'Rappel CW E003':>14} -> {'E004':>6} | {'Barker->CW E003':>15} -> {'E004':>6}")
print("-" * 75)

e003_by_snr = {s["requested_snr_db"]: s for s in e003_rep["per_snr"]}
e004_by_snr = {s["requested_snr_db"]: s for s in e004_rep["per_snr"]}

for snr in (0, -5, -10, -15):
    c03 = np.array(e003_by_snr[snr]["confusion_matrix"])
    c04 = np.array(e004_by_snr[snr]["confusion_matrix"])
    
    # row 0 = barker, row 1 = cw
    bk_rec_03 = c03[0, 0] / c03[0].sum()
    bk_rec_04 = c04[0, 0] / c04[0].sum()
    
    cw_rec_03 = c03[1, 1] / c03[1].sum()
    cw_rec_04 = c04[1, 1] / c04[1].sum()
    
    bk_cw_03 = c03[0, 1]
    bk_cw_04 = c04[0, 1]
    bk_cw_pct_03 = 100 * bk_cw_03 / c03[0].sum()
    bk_cw_pct_04 = 100 * bk_cw_04 / c04[0].sum()
    
    print(f"{snr:+5d}dB | {bk_rec_03:17.4f} -> {bk_rec_04:6.4f} | {cw_rec_03:13.4f} -> {cw_rec_04:6.4f} | {bk_cw_03:4d} ({bk_cw_pct_03:4.1f}%) -> {bk_cw_04:4d} ({bk_cw_pct_04:4.1f}%)")

print("=" * 75)
