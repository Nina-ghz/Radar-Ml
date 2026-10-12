"""Data processing + Track B feature engineering.

Data processing (per pulse, fixed rules allowed by the event rules):
    1. crop to 19-122 us: every pulse lies in 20.1-120.6 us (measured on all 660
       configurations, notebooks/01_eda.ipynb section 2);
    2. normalise by the pulse's own RMS, so absolute amplitude is never a cue.

Feature engineering (Track B): a fixed, non-learned STFT front-end that turns
[I; Q] into a log-power spectrogram. It lives inside the model, so training and
inference share one code path and its cost is part of the measured inference time.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .constants import SAMPLE_RATE_HZ

CROP_START_US, CROP_END_US = 19.0, 122.0
CROP = slice(round(CROP_START_US * 1e-6 * SAMPLE_RATE_HZ), round(CROP_END_US * 1e-6 * SAMPLE_RATE_HZ))
N_CROPPED = CROP.stop - CROP.start      # 12 360 complex samples


def preprocess(row: np.ndarray) -> np.ndarray:
    """Interleaved int16 (33600,) -> float32 (2, 12360) = cropped [I; Q] / RMS."""
    iq = np.asarray(row, dtype=np.float32).reshape(-1, 2).T[:, CROP]
    rms = np.sqrt(np.mean(iq[0] ** 2 + iq[1] ** 2))
    return np.ascontiguousarray(iq / max(float(rms), 1e-6))


class STFTSpectrogram(nn.Module):
    """(B, 2, N) [I; Q] -> (B, 1, F, T) standardised log-power spectrogram.

    Two-sided (complex baseband has negative frequencies), fftshifted so 0 Hz is
    centred, power in dB, then per-example standardisation (fixed per-pulse rule).
    """

    def __init__(self, n_fft: int = 256, hop_length: int = 128, freq_pool: int = 1, eps: float = 1e-10):
        super().__init__()
        self.n_fft, self.hop_length, self.freq_pool, self.eps = n_fft, hop_length, freq_pool, eps
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        x = torch.complex(iq[:, 0].float(), iq[:, 1].float())
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
                          center=False, onesided=False, return_complex=True)
        spec = torch.fft.fftshift(spec, dim=1)
        img = (10.0 * torch.log10(spec.real ** 2 + spec.imag ** 2 + self.eps)).unsqueeze(1)
        if self.freq_pool > 1:
            img = F.avg_pool2d(img, (self.freq_pool, 1))
        mean = img.mean(dim=(2, 3), keepdim=True)
        std = img.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (img - mean) / std

    def output_shape(self, n_samples: int = N_CROPPED) -> tuple[int, int]:
        return self.n_fft // self.freq_pool, 1 + (n_samples - self.n_fft) // self.hop_length
