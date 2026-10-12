"""Validation evaluation shared by scripts/train.py and scripts/evaluate.py.

Always scored at the 13 requested SNRs with the organizers' deterministic noise,
whatever SNRs the experiment trained on: this is the challenge metric.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import REQUESTED_SNRS_DB
from .data import EvalDataset
from .features import preprocess
from .manifest import PulseRef
from .metrics import challenge_report
from .models import TrackBModel, build_model
from .training import predict

DURATIONS_US = (10, 50, 100)


def resolve_device(choice: str = "auto") -> torch.device:
    """`auto` picks CUDA when available; an explicit `cuda` fails loudly without it."""
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(choice)


def describe_hardware(device: torch.device) -> str:
    if device.type == "cuda":
        return f"GPU, {torch.cuda.get_device_name(0)}"
    return f"CPU, {torch.get_num_threads()} threads, pas de GPU"


def duration_of(ref: PulseRef) -> int:
    """Pulse duration in µs, read from the filename. Analysis only - never a model input."""
    return int(Path(ref.file).name.split("_")[0].removesuffix("us"))


def load_checkpoint(model_file: Path, config: dict, device: torch.device) -> tuple[TrackBModel, dict]:
    """Rebuild the model described by `config` and load the weights of `model_file`."""
    model = build_model(config["stft"], config["cnn"]).to(device)
    checkpoint = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def evaluate(model, refs: list[PulseRef], data_dir: Path, device: torch.device,
             num_workers: int = 0, batch_size: int = 256) -> dict:
    """Challenge report on validation `refs`, plus one report per pulse duration."""
    loader = DataLoader(EvalDataset(refs, data_dir, REQUESTED_SNRS_DB), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    out = predict(model, loader, device=device)
    report = challenge_report(out["y_true"], out["y_pred"], out["snr"])
    durations = np.array([duration_of(refs[i]) for i in out["ref_index"]])
    by_duration = {}
    for duration in DURATIONS_US:
        mask = durations == duration
        if mask.any():
            by_duration[duration] = challenge_report(out["y_true"][mask], out["y_pred"][mask], out["snr"][mask])
    return dict(report=report, by_duration=by_duration)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def inference_ms_per_pulse(model: TrackBModel, refs: list[PulseRef], data_dir: Path,
                           device: torch.device, batch_size: int = 256) -> tuple[float, float]:
    """(whole model, STFT front-end alone) in ms per pulse, on one batch after a warm-up.

    Same protocol as notebooks/02_baseline.ipynb so costs stay comparable across experiments.
    """
    model.eval()
    refs = refs[:batch_size]
    batch = torch.from_numpy(np.stack([
        preprocess(np.load(Path(data_dir) / r.file, mmap_mode="r")[0, r.pulse_index]) for r in refs
    ])).to(device)
    model(batch[:32])
    timings = []
    for module in (model, model.frontend):
        _sync(device)
        start = time.perf_counter()
        module(batch)
        _sync(device)
        timings.append((time.perf_counter() - start) / len(batch) * 1e3)
    return timings[0], timings[1]
