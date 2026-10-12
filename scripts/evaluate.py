#!/usr/bin/env python3
"""Evaluation script for RadarML Track B runs.

Usage:
    python scripts/evaluate.py --run runs/<experiment_dir> [--val-pulses-per-file N] [--device auto|cpu|cuda]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

# Ensure src is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from radarml.constants import REQUESTED_SNRS_DB
from radarml.data import EvalDataset
from radarml.manifest import load_manifest, select
from radarml.metrics import challenge_report
from radarml.models import build_model
from radarml.training import predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RadarML model on the validation split")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory containing config.yaml and model.pt")
    parser.add_argument("--val-pulses-per-file", type=int, default=20, help="Number of validation pulses per config (default: 20)")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Compute device")
    return parser.parse_args()


def resolve_device(device_choice: str) -> tuple[torch.device, bool, str]:
    if device_choice == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        dev_str = "cuda"
    else:
        dev_str = "cpu"
    device = torch.device(dev_str)
    pin_memory = (dev_str == "cuda")
    return device, pin_memory, dev_str


def duration_of(ref) -> int:
    token = Path(ref.file).name.split("_")[0].lower()
    if token.endswith("us"):
        return int(token[:-2])
    return int(token)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_file = run_dir / "config.yaml"
    model_file = run_dir / "model.pt"

    if not config_file.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    if not model_file.exists():
        raise FileNotFoundError(f"model.pt not found in {run_dir}")

    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device, pin_memory, device_str = resolve_device(args.device)
    print(f"Device: {device_str}")

    stft_cfg = config.get("stft", dict(n_fft=256, hop_length=128, freq_pool=2))
    cnn_cfg = config.get("cnn", dict(channels=(16, 32, 64, 128), dropout=0.3))
    if isinstance(cnn_cfg.get("channels"), list):
        cnn_cfg = dict(cnn_cfg)
        cnn_cfg["channels"] = tuple(cnn_cfg["channels"])

    model = build_model(stft_cfg, cnn_cfg).to(device)
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data_dir = ROOT / "data" / "raw" / "recorded_dataset"
    manifest_csv = ROOT / "data" / "manifest" / "split_manifest.csv"
    refs = load_manifest(data_dir, manifest_csv)

    val_refs = select(refs, "validation", args.val_pulses_per_file)
    snrs = config.get("train_snrs", list(REQUESTED_SNRS_DB))
    val_ds = EvalDataset(val_refs, data_dir, snrs=snrs)

    num_workers = int(config.get("num_workers", 3))
    loader_kw = dict(num_workers=num_workers, persistent_workers=False, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=256, **loader_kw)

    print(f"Évaluation sur {len(val_refs):,} pulses × {len(snrs)} SNRs = {len(val_ds):,} exemples")
    t0 = time.perf_counter()
    out = predict(model, val_loader, device=device)
    eval_time = (time.perf_counter() - t0) / 60

    report = challenge_report(out["y_true"], out["y_pred"], out["snr"])

    print(f"\n=======================================================")
    print(f"RÉSULTATS DE L'ÉVALUATION ({run_dir.name})")
    print(f"=======================================================")
    print(f"Score (moyenne balanced accuracy, 13 SNRs) : {report['mean_balanced_accuracy']:.4f}")
    print(f"Tie-break (-20/-25/-30 dB)                 : {report['tiebreak_low_snr']:.4f}")
    print(f"Temps d'évaluation                         : {eval_time:.2f} min")

    durations = np.array([duration_of(val_refs[i]) for i in out["ref_index"]])
    by_duration = {}
    print("\nBalanced accuracy par durée de pulse :")
    for d in (10, 50, 100):
        mask = (durations == d)
        if mask.any():
            rep_d = challenge_report(out["y_true"][mask], out["y_pred"][mask], out["snr"][mask])
            by_duration[str(d)] = rep_d["mean_balanced_accuracy"]
            print(f"  {d:>3} µs : {rep_d['mean_balanced_accuracy']:.4f}")

    per_snr_df = pd.DataFrame(report["per_snr"]).set_index("requested_snr_db")[["balanced_accuracy", "accuracy"]]
    print("\nPar SNR :")
    print(per_snr_df.round(4).to_string())

    # Sauvegarder evaluation_report.json
    eval_report = {
        "run": str(run_dir),
        "val_pulses_per_file": args.val_pulses_per_file,
        "mean_balanced_accuracy": report["mean_balanced_accuracy"],
        "tiebreak_low_snr": report["tiebreak_low_snr"],
        "by_duration": by_duration,
        "per_snr": report["per_snr"],
        "eval_minutes": round(eval_time, 2),
    }
    with (run_dir / "evaluation_report.json").open("w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=1, default=str)
    print(f"\nRapport enregistré dans : {run_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
