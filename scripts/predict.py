#!/usr/bin/env python3
"""Predict the blind test with a trained run and write predictions.csv.

Usage:
    python scripts/predict.py --run runs/<experiment>_<timestamp> --test-dir <blind test directory>
        [--template <submission_template.csv>] [--output predictions.csv]
        [--batch-size 64] [--num-workers 2] [--device auto|cpu|cuda]

Reads the int16 rows located by <test-dir>/index.csv, applies exactly the training
preprocessing (radarml.features.preprocess), runs the model (STFT front-end included)
under torch.inference_mode, and writes one prediction per template sample_id, in
template order. Run it yourself on the real blind test: never hand those files to
an external service or an AI assistant.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radarml.blind import BlindTestDataset, read_template, write_predictions  # noqa: E402
from radarml.constants import CLASSES  # noqa: E402
from radarml.evaluation import load_checkpoint, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory with config.yaml and model.pt")
    parser.add_argument("--test-dir", type=Path, required=True, help="directory with index.csv and the shards")
    parser.add_argument("--template", type=Path,
                        help="submission_template.csv (default: <test-dir>/submission_template.csv)")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    template = args.template or args.test_dir / "submission_template.csv"
    config = yaml.safe_load((args.run / "config.yaml").read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.run / "model.pt", config, device)
    print(f"Modèle : {args.run.name} (époque {checkpoint['epoch']}) | SHA256 model.pt : "
          f"{sha256(args.run / 'model.pt')} | device : {device}", flush=True)

    dataset = BlindTestDataset(args.test_dir)
    template_ids = read_template(template)
    if set(template_ids) != set(dataset.sample_ids):
        raise ValueError(f"{template} and {args.test_dir / 'index.csv'} list different sample_id")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")

    predictions, started = {}, time.perf_counter()
    for done, (x, index) in enumerate(loader, 1):
        labels = model(x.to(device)).argmax(1).cpu().tolist()
        predictions.update((dataset.sample_ids[i], CLASSES[label]) for i, label in zip(index.tolist(), labels))
        if done % 200 == 0:
            print(f"  {len(predictions):,} / {len(dataset):,}", flush=True)
    seconds = time.perf_counter() - started

    write_predictions(args.output, template_ids, predictions)
    counts = Counter(predictions.values())
    print(f"{len(predictions):,} prédictions -> {args.output}")
    print(f"Temps : {seconds / 60:.1f} min, soit {seconds / len(predictions) * 1e3:.2f} ms par pulse "
          f"(lecture + preprocess + STFT + modèle)")
    print("Prédictions par classe : " + " | ".join(f"{c} {counts.get(c, 0):,}" for c in CLASSES))


if __name__ == "__main__":
    main()
