"""Validation evaluation shared by scripts/train.py and scripts/evaluate.py.

Always scored at the 13 requested SNRs with the organizers' deterministic noise,
whatever SNRs the experiment trained on: this is the challenge metric.
"""
from __future__ import annotations

import json
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
from .models import count_parameters, model_from_config
from .plots import save_validation_figures
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


def load_checkpoint(model_file: Path, config: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model described by `config` and load the weights of `model_file`."""
    model = model_from_config(config).to(device)
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
def inference_ms_per_pulse(model: torch.nn.Module, refs: list[PulseRef], data_dir: Path,
                           device: torch.device, batch_size: int = 256, repeats: int = 5) -> tuple[float, float]:
    """(whole model, STFT front-end alone) in ms per pulse, on one batch after a warm-up.

    Same protocol as notebooks/02_baseline.ipynb, with the median of `repeats` passes: a single
    pass varies by a factor of two with the machine load, which hides the cost of an experiment.
    """
    model.eval()
    refs = refs[:batch_size]
    batch = torch.from_numpy(np.stack([
        preprocess(np.load(Path(data_dir) / r.file, mmap_mode="r")[0, r.pulse_index]) for r in refs
    ])).to(device)
    model(batch[:32])
    timings = []
    for module in (model, model.frontend):
        passes = []
        for _ in range(repeats):
            _sync(device)
            start = time.perf_counter()
            module(batch)
            _sync(device)
            passes.append((time.perf_counter() - start) / len(batch) * 1e3)
        timings.append(float(np.median(passes)))
    return timings[0], timings[1]


def write_validation_report(run_dir: Path, config: dict, val_refs: list[PulseRef], data_dir: Path,
                            device: torch.device, fit_minutes: float, history: list[dict]) -> dict:
    """Score the run's model.pt on `val_refs`, then write validation_report.json and figures/."""
    model, checkpoint = load_checkpoint(Path(run_dir) / "model.pt", config, device)
    result = evaluate(model, val_refs, data_dir, device, num_workers=config["num_workers"])
    report = result["report"]
    model_ms, frontend_ms = inference_ms_per_pulse(model, val_refs, data_dir, device)
    costs = {"paramètres": count_parameters(model),
             "taille modèle (Mo)": round((Path(run_dir) / "model.pt").stat().st_size / 1e6, 2),
             "inférence (ms/pulse, STFT incluse)": round(model_ms, 2),
             "dont STFT (ms/pulse)": round(frontend_ms, 2),
             "entraînement (min)": round(fit_minutes, 1),
             "matériel": describe_hardware(device)}
    summary = dict(experiment=config["experiment"], config=config, best_epoch=checkpoint["epoch"],
                   mean_balanced_accuracy=report["mean_balanced_accuracy"],
                   tiebreak_low_snr=report["tiebreak_low_snr"],
                   by_duration={d: rep["mean_balanced_accuracy"] for d, rep in result["by_duration"].items()},
                   costs=costs, per_snr=report["per_snr"], history=history)
    (Path(run_dir) / "validation_report.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                          encoding="utf-8")
    save_validation_figures(report, result["by_duration"], Path(run_dir) / "figures")
    return dict(summary, model_ms=model_ms)
