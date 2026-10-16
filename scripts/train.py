#!/usr/bin/env python3
"""Train one experiment described by a YAML config (CPU or GPU).

Usage:
    python scripts/train.py --config configs/e001_baseline.yaml
        [--device auto|cpu|cuda] [--epochs N] [--train-pulses N] [--val-pulses N]

Writes runs/<experiment>_<timestamp>/: config.yaml (as actually used), model.pt (best
epoch on the monitoring subset of the validation split), history.csv,
validation_report.json and figures/.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radarml.data import EvalDataset, TrainDataset  # noqa: E402
from radarml.evaluation import resolve_device, write_validation_report  # noqa: E402
from radarml.manifest import load_manifest, select  # noqa: E402
from radarml.metrics import challenge_report  # noqa: E402
from radarml.models import build_model, count_parameters  # noqa: E402
from radarml.training import class_weights, predict, train_one_epoch  # noqa: E402

DATA_DIR = ROOT / "data" / "raw" / "recorded_dataset"
MANIFEST_CSV = ROOT / "data" / "manifest" / "split_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="YAML experiment config")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, help="override epochs")
    parser.add_argument("--train-pulses", type=int, help="override train_pulses_per_file")
    parser.add_argument("--val-pulses", type=int, help="override val_pulses_per_file")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    overrides = dict(epochs=args.epochs, train_pulses_per_file=args.train_pulses, val_pulses_per_file=args.val_pulses)
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args)
    device = resolve_device(args.device)
    on_cuda = device.type == "cuda"
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = ROOT / "runs" / f"{config['experiment']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    refs = load_manifest(DATA_DIR, MANIFEST_CSV)
    train_refs = select(refs, "train", config["train_pulses_per_file"])
    monitor_refs = select(refs, "validation", config["monitor_pulses_per_file"])
    val_refs = select(refs, "validation", config["val_pulses_per_file"])
    assert not {(r.file, r.pulse_index) for r in train_refs} & {(r.file, r.pulse_index) for r in val_refs}
    print(f"Device : {device} | entraînement : {len(train_refs):,} pulses | "
          f"validation : {len(val_refs):,} pulses × 13 SNR", flush=True)

    train_ds = TrainDataset(train_refs, DATA_DIR, config["train_snrs"],
                            snr_weights=config.get("train_snr_weights"), seed=seed)
    train_ds.set_epoch(1)
    snr_share = pd.Series(train_ds.drawn_snrs()).value_counts(normalize=True)
    print("SNR tirés à l'époque 1 : " + " | ".join(f"{snr:+d} dB {snr_share.get(snr, 0.0):.1%}"
                                                    for snr in config["train_snrs"]), flush=True)
    loader_kw = dict(num_workers=config["num_workers"], persistent_workers=False, pin_memory=on_cuda)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(seed), **loader_kw)
    monitor_loader = DataLoader(EvalDataset(monitor_refs, DATA_DIR), batch_size=256, **loader_kw)

    model = build_model(config["stft"], config["cnn"]).to(device)
    n_params = count_parameters(model)
    print(f"Paramètres entraînables : {n_params:,}", flush=True)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_refs).to(device),
                                          label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"],
                                                    total_steps=config["epochs"] * len(train_loader))

    history, best_score, started = [], -1.0, time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        train_ds.set_epoch(epoch)
        stats = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device=device)
        out = predict(model, monitor_loader, device=device)
        score = challenge_report(out["y_true"], out["y_pred"], out["snr"])["mean_balanced_accuracy"]
        history.append(dict(epoch=epoch, train_loss=stats["loss"], train_acc=stats["accuracy"],
                            val_monitor_bal_acc=score, minutes=stats["seconds"] / 60))
        if score > best_score:
            best_score = score
            torch.save({"config": config, "state_dict": model.state_dict(), "epoch": epoch}, run_dir / "model.pt")
        print(f"Époque {epoch}/{config['epochs']} | loss {stats['loss']:.3f} | acc train {stats['accuracy']:.3f} | "
              f"balanced acc validation {score:.4f} | {stats['seconds'] / 60:.1f} min", flush=True)
    training_minutes = (time.perf_counter() - started) / 60
    history = pd.DataFrame(history)
    history.to_csv(run_dir / "history.csv", index=False)

    best_epoch = int(history.loc[history["val_monitor_bal_acc"].idxmax(), "epoch"])
    print(f"Entraînement : {training_minutes:.1f} min | meilleure époque : {best_epoch}", flush=True)
    summary = write_validation_report(run_dir, config, val_refs, DATA_DIR, device, training_minutes,
                                      history.to_dict("records"))
    print(f"SCORE VALIDATION (moyenne des 13 SNR) : {summary['mean_balanced_accuracy']:.4f} | "
          f"tie-break (-20/-25/-30 dB) : {summary['tiebreak_low_snr']:.4f} | {summary['model_ms']:.1f} ms/pulse")
    print(f"Run : {run_dir}")


if __name__ == "__main__":
    main()
