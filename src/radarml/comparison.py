"""Compare the reports of two runs (an experiment against its parent config).

Works on the JSON written by scripts/train.py (validation_report.json) and
scripts/evaluate.py (evaluation_report.json). Both runs must be scored on the same
validation pulses, otherwise the difference mixes the experiment with the sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CLASS_TO_INDEX

REPORT_FILES = {"validation": "validation_report.json", "evaluation": "evaluation_report.json"}
# Decision rule of docs/EXPERIMENT_LOG.md, on the gain in mean balanced accuracy.
KEEP_GAIN, PROVISIONAL_GAIN = 0.010, 0.005
SATURATED_FLOOR = 0.99      # requested SNRs >= +5 dB must stay at least here


def load_report(run_dir: Path, kind: str = "validation") -> dict:
    """Report of a run: "validation" (written by train.py) or "evaluation" (evaluate.py)."""
    return json.loads((Path(run_dir) / REPORT_FILES[kind]).read_text(encoding="utf-8"))


def _matrix(report: dict, snr: int) -> np.ndarray:
    rows = [r for r in report["per_snr"] if r["requested_snr_db"] == snr]
    if not rows:
        raise KeyError(f"No requested SNR {snr} dB in the report")
    return np.array(rows[0]["confusion_matrix"])


def class_recall(report: dict, snr: int, label: str) -> float:
    matrix, i = _matrix(report, snr), CLASS_TO_INDEX[label]
    return float(matrix[i, i] / matrix[i].sum())


def confusion_count(report: dict, snr: int, true: str, predicted: str) -> int:
    """Number of `true` pulses predicted as `predicted` at one requested SNR."""
    return int(_matrix(report, snr)[CLASS_TO_INDEX[true], CLASS_TO_INDEX[predicted]])


def _samples(report: dict) -> int:
    return int(sum(np.sum(r["confusion_matrix"]) for r in report["per_snr"]))


def per_snr_table(parent: dict, run: dict) -> pd.DataFrame:
    """Balanced accuracy per requested SNR (high to low): parent, run, delta."""
    if _samples(parent) != _samples(run):
        raise ValueError(f"Runs scored on different samples ({_samples(parent)} vs {_samples(run)})")
    columns = {name: {r["requested_snr_db"]: r["balanced_accuracy"] for r in report["per_snr"]}
               for name, report in (("parent", parent), ("run", run))}
    table = pd.DataFrame(columns).sort_index(ascending=False).rename_axis("requested_snr_db")
    return table.assign(delta=table["run"] - table["parent"])


def class_table(parent: dict, run: dict, labels, snrs) -> pd.DataFrame:
    """Recall of each class in `labels` at each SNR in `snrs`, parent and run side by side."""
    rows = {snr: {f"{label} {name}": class_recall(report, snr, label)
                  for label in labels for name, report in (("parent", parent), ("run", run))}
            for snr in snrs}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("requested_snr_db")


def saturated_snrs_hold(report: dict, floor: float = SATURATED_FLOOR, min_snr: int = 5) -> bool:
    return all(r["balanced_accuracy"] >= floor for r in report["per_snr"] if r["requested_snr_db"] >= min_snr)


def verdict(gain: float) -> str:
    """Decision on a gain in mean balanced accuracy (see the rule in docs/EXPERIMENT_LOG.md)."""
    if gain >= KEEP_GAIN:
        return "retenu"
    return "provisoire (à répliquer)" if gain >= PROVISIONAL_GAIN else "rejeté"
