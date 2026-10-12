"""Models trained from scratch (pretrained weights are not allowed)."""
from __future__ import annotations

import torch
from torch import nn

from .constants import CLASSES
from .features import MultiResSTFT, STFTSpectrogram


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

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] | list[int] = (16, 32, 64, 128),
        dropout: float = 0.3,
        n_classes: int = len(CLASSES),
    ):
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

    def __init__(self, frontend: nn.Module, backbone: nn.Module):
        super().__init__()
        self.frontend, self.backbone = frontend, backbone

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.frontend(iq))


def build_model(stft: dict, cnn: dict) -> TrackBModel:
    stft_cfg = dict(stft)
    stft_type = stft_cfg.pop("type", "single")
    if stft_type == "single":
        frontend = STFTSpectrogram(**stft_cfg)
        in_channels = 1
    elif stft_type == "multires":
        frontend = MultiResSTFT(**stft_cfg)
        in_channels = len(frontend.n_ffts)
    else:
        raise ValueError(f"Unknown STFT type: {stft_type!r} (expected 'single' or 'multires')")

    cnn_cfg = dict(cnn)
    if isinstance(cnn_cfg.get("channels"), list):
        cnn_cfg["channels"] = tuple(cnn_cfg["channels"])
    if "in_channels" not in cnn_cfg:
        cnn_cfg["in_channels"] = in_channels
    backbone = SpecCNN(**cnn_cfg)
    return TrackBModel(frontend, backbone)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
