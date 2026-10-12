"""On-the-fly SNR degradation, bit-identical to the official expansion.

Instead of materialising the 57.66 GB expanded dataset, we call the organizers'
own `degrade()` (vendored unmodified in third_party/radarml_lite) with the same
seed convention. For a given (source_file, snr, pulse) this reproduces the
official `*_single_channel.npy` values exactly (see tests/test_noise.py).

- Validation: `official_seed(...)` -> identical to the organizer expansion.
- Training:   any RNG -> fresh noise realisations every epoch (augmentation).
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from .constants import OFFICIAL_SEED

_VENDORED = Path(__file__).resolve().parents[2] / "third_party" / "radarml_lite" / "make_dataset_hackathon.py"
_spec = importlib.util.spec_from_file_location("_radarml_lite_official", _VENDORED)
_official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_official)

official_degrade = _official.degrade


def official_seed(source_file: str, snr_db: int, pulse_index: int, seed: int = OFFICIAL_SEED) -> int:
    """Same key as make_dataset_hackathon.process_rows."""
    key = f"{seed}_{Path(source_file).name}_{int(snr_db)}_{pulse_index}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def degrade(row: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Return interleaved int16 I/Q at the requested SNR (official preprocessing)."""
    values, _unattainable, _clipped = official_degrade(row, snr_db, rng)
    return values


def degrade_official(row: np.ndarray, source_file: str, snr_db: int, pulse_index: int) -> np.ndarray:
    """Deterministic variant matching the organizer-expanded arrays exactly."""
    rng = np.random.default_rng(official_seed(source_file, snr_db, pulse_index))
    return degrade(row, snr_db, rng)


def to_complex(row: np.ndarray) -> np.ndarray:
    """Interleaved int16 (33600,) -> complex64 (16800,)."""
    return row[0::2].astype(np.float32) + 1j * row[1::2].astype(np.float32)
