"""Models trained from scratch (pretrained weights are not allowed)."""
from __future__ import annotations

import torch
from torch import nn

from .constants import CLASSES
from .features import MultiResSTFT, STFTSpectrogram
from .templates import TemplateMatcher


def _block(c_in: int, c_out: int, convs: int = 2) -> nn.Sequential:
    layers = []
    for i in range(convs):
        layers += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False),
                   nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.MaxPool2d(2))


class SpecCNN(nn.Module):
    """Compact VGG-style CNN. Global average pooling makes it insensitive to where
    (and how wide) the pattern is on the spectrogram - pulses span 5 to 50 columns.
    The first block has a single convolution: full-resolution layers dominate CPU cost."""

    def __init__(self, channels=(16, 32, 64, 128), dropout: float = 0.3, n_classes: int = len(CLASSES),
                 in_channels: int = 1):
        super().__init__()
        layers, c_in = [], in_channels
        for i, c_out in enumerate(channels):
            layers.append(_block(c_in, c_out, convs=1 if i == 0 else 2))
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(img))


class TrackBModel(nn.Module):
    """[I; Q] -> fixed STFT front-end -> CNN -> logits for the 7 classes."""

    def __init__(self, frontend: STFTSpectrogram | MultiResSTFT, backbone: nn.Module):
        super().__init__()
        self.frontend, self.backbone = frontend, backbone

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.frontend(iq))


FRONTENDS = {"single": STFTSpectrogram, "multires": MultiResSTFT}


def build_model(stft: dict, cnn: dict) -> TrackBModel:
    """`stft["type"]` selects the front-end: "single" (default, E001) or "multires"."""
    stft = dict(stft)
    kind = stft.pop("type", "single")
    if kind not in FRONTENDS:
        raise ValueError(f"Unknown stft.type {kind!r} (expected one of {sorted(FRONTENDS)})")
    frontend = FRONTENDS[kind](**stft)
    return TrackBModel(frontend, SpecCNN(**{"in_channels": frontend.out_channels, **cnn}))


def model_from_config(config: dict) -> nn.Module:
    """CNN (`config["stft"]`, `config["cnn"]`) or, when `config["model"] == "templates"`, the template matcher."""
    if config.get("model", "cnn") == "templates":
        return TemplateMatcher(**config["templates"])
    return build_model(config["stft"], config["cnn"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
