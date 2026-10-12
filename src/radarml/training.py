"""Training / inference loops shared by notebooks and scripts."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .constants import CLASSES
from .manifest import PulseRef, class_counts


def class_weights(refs: list[PulseRef]) -> torch.Tensor:
    """Inverse-frequency weights so every class weighs the same in the loss,
    matching the balanced-accuracy metric (fsk has 16x more pulses than cw)."""
    counts = np.array([class_counts(refs)[c] for c in CLASSES], dtype=np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training subset misses classes: {class_counts(refs)}")
    return torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, scheduler=None, device="cpu") -> dict:
    model.train()
    total_loss, correct, seen, start = 0.0, 0, 0, time.perf_counter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    return dict(loss=total_loss / seen, accuracy=correct / seen, seconds=time.perf_counter() - start)


@torch.inference_mode()
def predict(model, loader, device="cpu") -> dict:
    """Predictions on an EvalDataset loader; model time includes the STFT front-end."""
    model.eval()
    out = {k: [] for k in ("y_true", "y_pred", "snr", "ref_index")}
    model_seconds = 0.0
    for x, y, snr, ref_index in loader:
        start = time.perf_counter()
        pred = model(x.to(device)).argmax(1).cpu()
        model_seconds += time.perf_counter() - start
        for key, value in zip(out, (y, pred, snr, ref_index)):
            out[key].append(value)
    out = {k: torch.cat(v).numpy() for k, v in out.items()}
    out["model_seconds_per_example"] = model_seconds / max(1, len(out["y_true"]))
    return out
