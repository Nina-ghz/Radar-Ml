#!/usr/bin/env python3
"""Training script for RadarML Track B experiments.

Usage:
    python scripts/train.py --config configs/e001_baseline.yaml [--device auto|cpu|cuda] [--epochs N] [--train-pulses N]
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

# Ensure src is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from radarml.constants import CLASSES, REQUESTED_SNRS_DB
from radarml.data import EvalDataset, TrainDataset
from radarml.features import preprocess
from radarml.manifest import class_counts, load_manifest, select
from radarml.metrics import challenge_report
from radarml.models import build_model, count_parameters
from radarml.training import class_weights, predict, train_one_epoch

# Styling constants matching notebooks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#9cc3ef", BLUE, "#0d3a73"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RadarML Track B model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Compute device")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--train-pulses", type=int, default=None, help="Override train_pulses_per_file")
    parser.add_argument("--val-pulses", type=int, default=None, help="Override val_pulses_per_file")
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
    """Analyse uniquement (durée lue dans le nom de fichier pour l'analyse, jamais entrée du modèle)."""
    token = Path(ref.file).name.split("_")[0].lower()
    if token.endswith("us"):
        return int(token[:-2])
    return int(token)


def plot_figures(report: dict, out_val: dict, val_refs: list, by_duration: dict, figures_dir: Path) -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "lines.linewidth": 2,
    })

    per_snr = pd.DataFrame(report["per_snr"]).set_index("requested_snr_db")[["balanced_accuracy", "accuracy"]]

    # Figure 1: Balanced accuracy vs SNR
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(per_snr.index, per_snr.balanced_accuracy, color=BLUE, marker="o", ms=5)
    ax.axhline(1 / 7, color=INK2, lw=1, ls="--")
    ax.text(29, 1 / 7 + 0.02, "hasard", color=INK2, ha="right")
    ax.axhline(report["mean_balanced_accuracy"], color=BLUE, lw=1, ls=":")
    ax.text(-29, report["mean_balanced_accuracy"] + 0.02, f"moyenne {report['mean_balanced_accuracy']:.3f}", color=BLUE)
    ax.set(xlabel="requested SNR (dB)", ylabel="balanced accuracy", ylim=(0, 1.02), xticks=list(REQUESTED_SNRS_DB),
           title="Balanced accuracy par SNR (validation)")
    fig.savefig(figures_dir / "03_balanced_accuracy_vs_snr.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Figure 2: Rappel classe x SNR
    recall = pd.DataFrame({
        snr: np.diag(np.array(r["confusion_matrix"])) / np.array(r["confusion_matrix"]).sum(1)
        for snr, r in zip(per_snr.index, report["per_snr"])
    }, index=CLASSES)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.imshow(recall.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(recall.shape[0]):
        for j in range(recall.shape[1]):
            v = recall.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(recall.shape[1]), [f"{s:+d}" for s in recall.columns])
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="requested SNR (dB)", title="Rappel par classe et par SNR")
    ax.grid(False)
    fig.savefig(figures_dir / "04_recall_class_snr.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Figure 3: Matrices de confusion à 0, -10 et -20 dB
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, snr in zip(axes, (0, -10, -20)):
        m_raw = next(r for r in report["per_snr"] if r["requested_snr_db"] == snr)["confusion_matrix"]
        m = np.array(m_raw, float)
        m = m / m.sum(1, keepdims=True)
        ax.imshow(m, cmap=SEQ, vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if m[i, j] > 0.6 else INK)
        ax.set_xticks(range(7), CLASSES, rotation=45)
        ax.set_yticks(range(7), CLASSES)
        ax.set(title=f"Confusion à {snr:+d} dB (lignes = vraie classe)", xlabel="prédite")
        ax.grid(False)
    axes[0].set_ylabel("vraie")
    fig.savefig(figures_dir / "05_confusion.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Figure 4: Balanced accuracy par durée
    durations = np.array([duration_of(val_refs[i]) for i in out_val["ref_index"]])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for d, color in zip((10, 50, 100), (BLUE, ORANGE, AQUA)):
        mask = (durations == d)
        if mask.any():
            rep = challenge_report(out_val["y_true"][mask], out_val["y_pred"][mask], out_val["snr"][mask])
            xs = [r["requested_snr_db"] for r in rep["per_snr"]]
            ys = [r["balanced_accuracy"] for r in rep["per_snr"]]
            ax.plot(xs, ys, color=color, marker="o", ms=4, label=f"{d} µs (moy. {rep['mean_balanced_accuracy']:.3f})")
            ax.text(xs[-1] - 0.6, ys[-1], f"{d} µs", color=color, ha="right", va="center", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.set(xlabel="requested SNR (dB)", ylabel="balanced accuracy", ylim=(0, 1.02), xticks=list(REQUESTED_SNRS_DB),
           title="Balanced accuracy par durée de pulse")
    fig.savefig(figures_dir / "06_by_duration.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply command-line overrides
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.train_pulses is not None:
        config["train_pulses_per_file"] = args.train_pulses
    if args.val_pulses is not None:
        config["val_pulses_per_file"] = args.val_pulses

    device, pin_memory, device_str = resolve_device(args.device)
    print(f"Device: {device_str} (pin_memory={pin_memory})")

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get("experiment", "experiment")
    run_dir = ROOT / "runs" / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save copy of the actual config used
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    data_dir = ROOT / "data" / "raw" / "recorded_dataset"
    manifest_csv = ROOT / "data" / "manifest" / "split_manifest.csv"
    refs = load_manifest(data_dir, manifest_csv)

    train_refs = select(refs, "train", config.get("train_pulses_per_file"))
    monitor_refs = select(refs, "validation", config.get("monitor_pulses_per_file", 1))
    val_pulses_per_file = config.get("val_pulses_per_file", 4)
    val_refs = select(refs, "validation", val_pulses_per_file)

    assert not {(r.file, r.pulse_index) for r in train_refs} & {(r.file, r.pulse_index) for r in val_refs}, "Split leak!"

    print(f"Train subset: {len(train_refs):,} pulses")
    print(f"Monitor subset: {len(monitor_refs):,} pulses × {len(config.get('train_snrs', REQUESTED_SNRS_DB))} SNRs")
    print(f"Validation subset: {len(val_refs):,} pulses × {len(config.get('train_snrs', REQUESTED_SNRS_DB))} SNRs")

    snrs = config.get("train_snrs", list(REQUESTED_SNRS_DB))
    train_snr_weights = config.get("train_snr_weights")
    train_ds = TrainDataset(train_refs, data_dir, snrs=snrs, train_snr_weights=train_snr_weights, seed=seed)
    monitor_ds = EvalDataset(monitor_refs, data_dir, snrs=snrs)
    val_ds = EvalDataset(val_refs, data_dir, snrs=snrs)

    num_workers = int(config.get("num_workers", 3))
    loader_kw = dict(num_workers=num_workers, persistent_workers=False, pin_memory=pin_memory)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_kw,
    )
    monitor_loader = DataLoader(monitor_ds, batch_size=256, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=256, **loader_kw)

    stft_cfg = config.get("stft", dict(n_fft=256, hop_length=128, freq_pool=2))
    cnn_cfg = config.get("cnn", dict(channels=(16, 32, 64, 128), dropout=0.3))
    # YAML channels might be list, convert to tuple if needed
    if isinstance(cnn_cfg.get("channels"), list):
        cnn_cfg = dict(cnn_cfg)
        cnn_cfg["channels"] = tuple(cnn_cfg["channels"])

    model = build_model(stft_cfg, cnn_cfg).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    weights = class_weights(train_refs).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=float(config.get("label_smoothing", 0.05)))
    lr = float(config.get("lr", 2e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(config.get("epochs", 8))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=epochs * max(1, len(train_loader)),
    )

    history = []
    best_score = -1.0
    best_epoch = 1
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        stats = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device=device)
        out_monitor = predict(model, monitor_loader, device=device)
        score = challenge_report(out_monitor["y_true"], out_monitor["y_pred"], out_monitor["snr"])["mean_balanced_accuracy"]

        epoch_mins = stats["seconds"] / 60
        history.append(dict(
            epoch=epoch,
            train_loss=stats["loss"],
            train_acc=stats["accuracy"],
            val_monitor_bal_acc=score,
            minutes=epoch_mins,
        ))

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "config": config,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_monitor_bal_acc": score,
            }, run_dir / "model.pt")

        print(
            f"Époque {epoch}/{epochs} | loss {stats['loss']:.3f} | acc train {stats['accuracy']:.3f} | "
            f"balanced acc val (monitor) {score:.4f} | {epoch_mins:.1f} min"
        )

    training_minutes = (time.perf_counter() - started) / 60
    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)
    print(f"Entraînement terminé : {training_minutes:.1f} min | Meilleure époque : {best_epoch} (score monitor: {best_score:.4f})")

    # Recharger le meilleur modèle pour l'évaluation complète de validation
    ckpt = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    eval_start = time.perf_counter()
    out_val = predict(model, val_loader, device=device)
    eval_minutes = (time.perf_counter() - eval_start) / 60
    report = challenge_report(out_val["y_true"], out_val["y_pred"], out_val["snr"])

    print(f"SCORE VALIDATION (moyenne bal. acc.): {report['mean_balanced_accuracy']:.4f}")
    print(f"TIE-BREAK (-20/-25/-30 dB): {report['tiebreak_low_snr']:.4f}")

    # Calcul par durée de pulse
    durations = np.array([duration_of(val_refs[i]) for i in out_val["ref_index"]])
    by_duration = {}
    for d in (10, 50, 100):
        mask = (durations == d)
        if mask.any():
            rep_d = challenge_report(out_val["y_true"][mask], out_val["y_pred"][mask], out_val["snr"][mask])
            by_duration[str(d)] = rep_d["mean_balanced_accuracy"]

    # Mesure précise des coûts (inférence par pulse dont STFT)
    test_batch_size = min(256, len(val_refs))
    test_batch_iq = [preprocess(np.load(data_dir / r.file, mmap_mode="r")[0, r.pulse_index]) for r in val_refs[:test_batch_size]]
    batch_tensor = torch.from_numpy(np.stack(test_batch_iq)).to(device)

    with torch.inference_mode():
        # Warmup
        model(batch_tensor[:min(32, len(batch_tensor))])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(batch_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_ms = (time.perf_counter() - t0) / len(batch_tensor) * 1e3

        t0 = time.perf_counter()
        model.frontend(batch_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        stft_ms = (time.perf_counter() - t0) / len(batch_tensor) * 1e3

    model_size_mb = round((run_dir / "model.pt").stat().st_size / 1e6, 2)
    if device.type == "cuda":
        hardware = f"GPU, {torch.cuda.get_device_name(0)}"
    else:
        hardware = f"CPU, {torch.get_num_threads()} threads, pas de GPU"

    costs = {
        "paramètres": n_params,
        "taille modèle (Mo)": model_size_mb,
        "inférence (ms/pulse, STFT incluse)": round(model_ms, 2),
        "dont STFT (ms/pulse)": round(stft_ms, 2),
        "entraînement (min)": round(training_minutes, 1),
        "matériel": hardware,
    }

    # validation_report.json
    summary = dict(
        experiment=config.get("experiment", experiment_name),
        config=config,
        best_epoch=best_epoch,
        mean_balanced_accuracy=report["mean_balanced_accuracy"],
        tiebreak_low_snr=report["tiebreak_low_snr"],
        by_duration=by_duration,
        costs=costs,
        per_snr=report["per_snr"],
        history=history_df.to_dict("records"),
    )

    with (run_dir / "validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)

    # Tracer les 4 figures demandées
    plot_figures(report, out_val, val_refs, by_duration, figures_dir)
    print(f"Résultats enregistrés dans : {run_dir}")


if __name__ == "__main__":
    main()
