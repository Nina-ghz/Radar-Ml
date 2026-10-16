"""Coherent template matching on the complex STFT (Track B).

Every recording of a configuration repeats the same waveform with the same carrier phase
(phase spread about 0.1 degree across training pulses), and train, validation and blind test
share the 660 configurations. The template of a configuration is the mean of its training
pulses after the fixed preprocessing; a pulse is assigned the class of the template whose
complex STFT has the largest real correlation with its own. Keeping the phase and summing
coherently over the whole pulse is what keeps the decision reliable at -25 and -30 dB, where
magnitude spectrograms lose the signal.

Templates are fitted on training pulses only and stored as preprocessed I/Q; their STFT is
computed at load time, inside the model, like the pulse's own.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .constants import CLASSES
from .features import N_CROPPED, preprocess
from .manifest import PulseRef


class ComplexSTFT(nn.Module):
    """(B, 2, N) [I; Q] -> (B, F*T) complex STFT coefficients (Hann, two-sided, phase kept)."""

    def __init__(self, n_fft: int = 256, hop_length: int = 64):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        x = torch.complex(iq[:, 0].float(), iq[:, 1].float())
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
                          center=False, onesided=False, return_complex=True)
        return spec.reshape(spec.shape[0], -1)


class TemplateMatcher(nn.Module):
    """[I; Q] -> per-class scores: the best real STFT correlation with the templates of each class."""

    def __init__(self, n_templates: int, n_fft: int = 256, hop_length: int = 64, n_samples: int = N_CROPPED):
        super().__init__()
        self.frontend = ComplexSTFT(n_fft, hop_length)
        self.register_buffer("templates", torch.zeros(n_templates, 2, n_samples))
        self.register_buffer("template_class", torch.zeros(n_templates, dtype=torch.long))
        self._bank = None

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._bank = None

    def bank(self) -> torch.Tensor:
        """Unit-norm template STFTs, (n_templates, F*T) complex, computed once."""
        if self._bank is None or self._bank.device != self.templates.device:
            with torch.no_grad():
                rep = self.frontend(self.templates)
                self._bank = rep / torch.linalg.vector_norm(rep, dim=1, keepdim=True).clamp_min(1e-12)
        return self._bank

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        scores = (self.frontend(iq) @ self.bank().conj().T).real          # (B, n_templates)
        logits = torch.full((scores.shape[0], len(CLASSES)), -torch.inf, device=scores.device)
        index = self.template_class.expand_as(scores)
        return logits.scatter_reduce(1, index, scores, reduce="amax")

    @torch.no_grad()
    def fit(self, refs: list[PulseRef], data_dir: Path) -> "TemplateMatcher":
        """Template of each configuration = mean of its preprocessed training pulses."""
        if any(ref.split != "train" for ref in refs):
            raise ValueError("Templates are fitted on training pulses only")
        by_file: dict[str, list[PulseRef]] = {}
        for ref in refs:
            by_file.setdefault(ref.file, []).append(ref)
        if len(by_file) != self.templates.shape[0]:
            raise ValueError(f"{len(by_file)} configurations for {self.templates.shape[0]} templates")
        for i, (name, group) in enumerate(sorted(by_file.items())):
            array = np.load(Path(data_dir) / name, mmap_mode="r", allow_pickle=False)
            pulses = np.stack([preprocess(np.asarray(array[0, ref.pulse_index])) for ref in group])
            self.templates[i] = torch.from_numpy(pulses.mean(axis=0))
            self.template_class[i] = group[0].target
        self._bank = None
        return self
