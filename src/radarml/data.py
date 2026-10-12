"""PyTorch datasets: compact recordings + on-the-fly official SNR degradation.

No 57 GB expansion: noise is generated per access with the organizers' function.
- TrainDataset: random requested SNR and fresh noise at every epoch (augmentation).
- EvalDataset:  every (pulse, SNR) pair with the organizers' deterministic seeds,
                i.e. identical to evaluating on the official expanded arrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import REQUESTED_SNRS_DB
from .features import preprocess
from .manifest import PulseRef
from .noise import degrade, degrade_official


class _Recordings:
    """Lazy per-process memmaps (safe with DataLoader worker processes)."""

    def __init__(self, data_dir: Path):
        self.data_dir, self._arrays = Path(data_dir), {}

    def row(self, ref: PulseRef) -> np.ndarray:
        if ref.file not in self._arrays:
            self._arrays[ref.file] = np.load(self.data_dir / ref.file, mmap_mode="r", allow_pickle=False)
        return np.asarray(self._arrays[ref.file][0, ref.pulse_index])

    def __getstate__(self):
        return {"data_dir": self.data_dir, "_arrays": {}}


class TrainDataset(Dataset):
    """Training pulses only. Each access: requested SNR drawn from `snrs`, fresh noise."""

    def __init__(self, refs: list[PulseRef], data_dir: Path, snrs=REQUESTED_SNRS_DB,
                 snr_weights=None, train_snr_weights=None, seed: int = 0):
        if any(ref.split != "train" for ref in refs):
            raise ValueError("TrainDataset accepts training pulses only")
        self.refs, self.snrs, self.seed, self.epoch = refs, tuple(snrs), seed, 0
        weights = snr_weights if snr_weights is not None else train_snr_weights
        if weights is not None:
            if len(weights) != len(self.snrs):
                raise ValueError(
                    f"train_snr_weights length ({len(weights)}) does not match train_snrs length ({len(self.snrs)})"
                )
            weights_arr = np.asarray(weights, dtype=np.float64)
            if np.any(weights_arr < 0) or weights_arr.sum() <= 0:
                raise ValueError("train_snr_weights must be non-negative with positive sum")
            self.snr_weights = weights_arr / weights_arr.sum()
        else:
            self.snr_weights = None
        self.recordings = _Recordings(data_dir)

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch (with non-persistent workers) to change noise draws."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int):
        ref = self.refs[index]
        rng = np.random.default_rng((self.seed, self.epoch, index))
        if self.snr_weights is not None:
            idx = int(rng.choice(len(self.snrs), p=self.snr_weights))
            snr = float(self.snrs[idx])
        else:
            snr = float(self.snrs[rng.integers(len(self.snrs))])
        row = degrade(self.recordings.row(ref), snr, rng)
        return torch.from_numpy(preprocess(row)), ref.target


class EvalDataset(Dataset):
    """Every (pulse, requested SNR) pair with official deterministic noise.

    Returns (x, target, snr, ref_index); ref_index lets the analysis group results
    (e.g. by pulse duration) without ever feeding metadata to the model.
    """

    def __init__(self, refs: list[PulseRef], data_dir: Path, snrs=REQUESTED_SNRS_DB):
        self.refs, self.snrs = refs, tuple(int(s) for s in snrs)
        self.recordings = _Recordings(data_dir)

    def __len__(self) -> int:
        return len(self.refs) * len(self.snrs)

    def __getitem__(self, index: int):
        ref_index, snr = divmod(index, len(self.snrs))
        ref, snr = self.refs[ref_index], self.snrs[snr]
        row = degrade_official(self.recordings.row(ref), ref.source_file, snr, ref.pulse_index)
        return torch.from_numpy(preprocess(row)), ref.target, snr, ref_index
