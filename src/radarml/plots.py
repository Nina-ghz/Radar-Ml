"""Validation figures written by the scripts (same style as notebooks/02_baseline.ipynb)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .constants import CLASSES, REQUESTED_SNRS_DB

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#9cc3ef", BLUE, "#0d3a73"])
STYLE = {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
         "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
         "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 2}
CONFUSION_SNRS_DB = (0, -10, -20)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _row_normalised(confusion_matrix) -> np.ndarray:
    matrix = np.array(confusion_matrix, dtype=float)
    return matrix / matrix.sum(axis=1, keepdims=True)


def _annotated_heatmap(ax, values: np.ndarray, fontsize: float, **imshow_kw) -> None:
    ax.imshow(values, cmap=SEQ, vmin=0, vmax=1, **imshow_kw)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=fontsize,
                    color="white" if values[i, j] > 0.6 else INK)
    ax.grid(False)


def save_validation_figures(report: dict, by_duration: dict[int, dict], figures_dir: Path) -> None:
    """Balanced accuracy vs SNR, recall class x SNR, confusion matrices, score per duration."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    snrs = [r["requested_snr_db"] for r in report["per_snr"]]
    mean = report["mean_balanced_accuracy"]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(snrs, [r["balanced_accuracy"] for r in report["per_snr"]], color=BLUE, marker="o", ms=5)
        ax.axhline(1 / 7, color=INK2, lw=1, ls="--")
        ax.text(29, 1 / 7 + 0.02, "hasard", color=INK2, ha="right")
        ax.axhline(mean, color=BLUE, lw=1, ls=":")
        ax.text(-29, mean + 0.02, f"moyenne {mean:.3f}", color=BLUE)
        ax.set(xlabel="requested SNR (dB)", ylabel="balanced accuracy", ylim=(0, 1.02),
               xticks=list(REQUESTED_SNRS_DB), title="Balanced accuracy par SNR (validation)")
        _save(fig, figures_dir / "03_balanced_accuracy_vs_snr.png")

        recall = np.stack([_row_normalised(r["confusion_matrix"]).diagonal() for r in report["per_snr"]], axis=1)
        fig, ax = plt.subplots(figsize=(11, 3.6))
        _annotated_heatmap(ax, recall, fontsize=7.5, aspect="auto")
        ax.set_xticks(range(len(snrs)), [f"{s:+d}" for s in snrs])
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set(xlabel="requested SNR (dB)", title="Rappel par classe et par SNR")
        _save(fig, figures_dir / "04_recall_class_snr.png")

        fig, axes = plt.subplots(1, len(CONFUSION_SNRS_DB), figsize=(15, 4.4))
        for ax, snr in zip(axes, CONFUSION_SNRS_DB):
            row = next(r for r in report["per_snr"] if r["requested_snr_db"] == snr)
            _annotated_heatmap(ax, _row_normalised(row["confusion_matrix"]), fontsize=7)
            ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
            ax.set_yticks(range(len(CLASSES)), CLASSES)
            ax.set(title=f"Confusion à {snr:+d} dB (lignes = vraie classe)", xlabel="prédite")
        axes[0].set_ylabel("vraie")
        _save(fig, figures_dir / "05_confusion.png")

        fig, ax = plt.subplots(figsize=(8, 3.6))
        for (duration, rep), color in zip(by_duration.items(), (BLUE, ORANGE, AQUA)):
            xs = [r["requested_snr_db"] for r in rep["per_snr"]]
            ys = [r["balanced_accuracy"] for r in rep["per_snr"]]
            ax.plot(xs, ys, color=color, marker="o", ms=4,
                    label=f"{duration} µs (moy. {rep['mean_balanced_accuracy']:.3f})")
            ax.text(xs[-1] - 0.6, ys[-1], f"{duration} µs", color=color, ha="right", va="center", fontweight="bold")
        ax.legend(loc="lower right", frameon=False)
        ax.set(xlabel="requested SNR (dB)", ylabel="balanced accuracy", ylim=(0, 1.02),
               xticks=list(REQUESTED_SNRS_DB), title="Balanced accuracy par durée de pulse")
        _save(fig, figures_dir / "06_by_duration.png")
