#!/usr/bin/env python3
"""Fit the coherent template matcher of a config on the training pulses, then score it on validation.

Usage:
    python scripts/fit_templates.py --config configs/e007_templates_freq.yaml [--device auto|cpu|cuda]

Writes runs/<experiment>_<timestamp>/ like scripts/train.py: config.yaml, model.pt (the templates),
validation_report.json and figures/, so scripts/evaluate.py, compare_runs.py and predict.py apply as is.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radarml.evaluation import resolve_device, write_validation_report  # noqa: E402
from radarml.manifest import load_manifest, select  # noqa: E402
from radarml.templates import TemplateMatcher  # noqa: E402

DATA_DIR = ROOT / "data" / "raw" / "recorded_dataset"
MANIFEST_CSV = ROOT / "data" / "manifest" / "split_manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = resolve_device(args.device)

    refs = load_manifest(DATA_DIR, MANIFEST_CSV)
    train_refs = select(refs, "train", config["train_pulses_per_file"])
    val_refs = select(refs, "validation", config["val_pulses_per_file"])
    assert not {(r.file, r.pulse_index) for r in train_refs} & {(r.file, r.pulse_index) for r in val_refs}
    config["templates"]["n_templates"] = len({r.file for r in train_refs})

    run_dir = ROOT / "runs" / f"{config['experiment']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    started = time.perf_counter()
    model = TemplateMatcher(**config["templates"]).fit(train_refs, DATA_DIR)
    torch.save({"config": config, "state_dict": model.state_dict(), "epoch": 0}, run_dir / "model.pt")
    minutes = (time.perf_counter() - started) / 60
    print(f"{config['templates']['n_templates']} templates fitted on {len(train_refs):,} training pulses "
          f"in {minutes:.1f} min | validation : {len(val_refs):,} pulses × 13 SNR", flush=True)

    summary = write_validation_report(run_dir, config, val_refs, DATA_DIR, device, minutes,
                                      [dict(epoch=0, minutes=minutes)])
    print(f"SCORE VALIDATION (moyenne des 13 SNR) : {summary['mean_balanced_accuracy']:.4f} | "
          f"tie-break (-20/-25/-30 dB) : {summary['tiebreak_low_snr']:.4f} | {summary['model_ms']:.1f} ms/pulse")
    print(f"Run : {run_dir}")


if __name__ == "__main__":
    main()
