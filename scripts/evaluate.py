#!/usr/bin/env python3
"""Re-run the full validation evaluation of a trained run.

Usage:
    python scripts/evaluate.py --run runs/<experiment>_<timestamp>
        [--val-pulses-per-file N] [--device auto|cpu|cuda]

Reloads model.pt with the run's config.yaml and scores the validation split at the
13 requested SNRs (default: all 20 validation pulses per configuration). Writes
evaluation_report.json and evaluation_figures/ in the run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radarml.evaluation import (describe_hardware, evaluate, inference_ms_per_pulse,  # noqa: E402
                                load_checkpoint, resolve_device)
from radarml.manifest import load_manifest, select  # noqa: E402
from radarml.models import count_parameters  # noqa: E402
from radarml.plots import save_validation_figures  # noqa: E402

DATA_DIR = ROOT / "data" / "raw" / "recorded_dataset"
MANIFEST_CSV = ROOT / "data" / "manifest" / "split_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory with config.yaml and model.pt")
    parser.add_argument("--val-pulses-per-file", type=int, default=20,
                        help="validation pulses per configuration (default: all 20)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((args.run / "config.yaml").read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.run / "model.pt", config, device)

    val_refs = select(load_manifest(DATA_DIR, MANIFEST_CSV), "validation", args.val_pulses_per_file)
    print(f"Device : {device} | validation : {len(val_refs):,} pulses × 13 SNR", flush=True)
    result = evaluate(model, val_refs, DATA_DIR, device, num_workers=config["num_workers"])
    report = result["report"]
    by_duration = {d: rep["mean_balanced_accuracy"] for d, rep in result["by_duration"].items()}
    model_ms, stft_ms = inference_ms_per_pulse(model, val_refs, DATA_DIR, device)

    summary = dict(experiment=config["experiment"], run=args.run.name, epoch=checkpoint["epoch"],
                   val_pulses_per_file=args.val_pulses_per_file, samples=report["samples"],
                   mean_balanced_accuracy=report["mean_balanced_accuracy"],
                   tiebreak_low_snr=report["tiebreak_low_snr"], by_duration=by_duration,
                   costs={"paramètres": count_parameters(model),
                          "inférence (ms/pulse, STFT incluse)": round(model_ms, 2),
                          "dont STFT (ms/pulse)": round(stft_ms, 2),
                          "matériel": describe_hardware(device)},
                   per_snr=report["per_snr"])
    (args.run / "evaluation_report.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                     encoding="utf-8")
    save_validation_figures(report, result["by_duration"], args.run / "evaluation_figures")

    per_snr = pd.DataFrame(report["per_snr"]).set_index("requested_snr_db")[["balanced_accuracy", "accuracy"]]
    print(per_snr.round(4).to_string())
    print("Par durée : " + " | ".join(f"{d} µs {score:.4f}" for d, score in by_duration.items()))
    print(f"SCORE VALIDATION (moyenne des 13 SNR) : {report['mean_balanced_accuracy']:.4f} | "
          f"tie-break (-20/-25/-30 dB) : {report['tiebreak_low_snr']:.4f} | {model_ms:.1f} ms/pulse")
    print(f"Rapport : {args.run / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
