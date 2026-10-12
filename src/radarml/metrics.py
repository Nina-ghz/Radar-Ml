"""Challenge metric, mirroring third_party/radarml_lite/score_submission.py.

Per requested SNR: balanced accuracy = mean per-class recall over the 7 classes.
Main score = mean over the 13 SNRs. Tie-break = mean over -20, -25, -30 dB.
Confusion matrices: rows = true labels, columns = predictions (CLASSES order).
"""
from __future__ import annotations

import numpy as np

from .constants import CLASSES, TIEBREAK_SNRS_DB


def confusion(y_true, y_pred, n_classes: int = len(CLASSES)) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return matrix


def balanced_accuracy(matrix: np.ndarray) -> float:
    support = matrix.sum(axis=1)
    if np.any(support == 0):
        raise ValueError("Every class must be present to compute the challenge score")
    return float(np.mean(matrix.diagonal() / support))


def challenge_report(y_true, y_pred, snrs) -> dict:
    y_true, y_pred, snrs = map(np.asarray, (y_true, y_pred, snrs))
    per_snr = []
    for snr in sorted(set(snrs.tolist()), reverse=True):
        mask = snrs == snr
        matrix = confusion(y_true[mask], y_pred[mask])
        per_snr.append(dict(requested_snr_db=int(snr),
                            balanced_accuracy=balanced_accuracy(matrix),
                            accuracy=float(matrix.trace() / matrix.sum()),
                            confusion_matrix=matrix.tolist()))
    tiebreak = [r["balanced_accuracy"] for r in per_snr if r["requested_snr_db"] in TIEBREAK_SNRS_DB]
    return dict(classes=list(CLASSES), samples=int(y_true.size),
                mean_balanced_accuracy=float(np.mean([r["balanced_accuracy"] for r in per_snr])),
                tiebreak_low_snr=float(np.mean(tiebreak)) if tiebreak else None,
                per_snr=per_snr)
