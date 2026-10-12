"""Shape and invariance tests for the Track B pipeline (needs torch)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radarml.features import CROP, N_CROPPED, STFTSpectrogram, preprocess  # noqa: E402
from radarml.models import build_model  # noqa: E402


def test_crop_covers_every_pulse():
    # Pulses span 20.07-120.57 us in all 660 configurations (EDA section 2).
    assert CROP.start / 120e6 * 1e6 < 20.07 and CROP.stop / 120e6 * 1e6 > 120.57
    assert N_CROPPED == 12_360


def test_preprocess_is_scale_invariant():
    rng = np.random.default_rng(0)
    row = rng.integers(-2000, 2000, 33_600).astype(np.int16)
    a, b = preprocess(row), preprocess((row // 2).astype(np.int16) * 2)
    assert a.shape == (2, N_CROPPED) and a.dtype == np.float32
    assert np.sqrt(np.mean(a[0] ** 2 + a[1] ** 2)) == pytest.approx(1.0, rel=1e-4)
    np.testing.assert_allclose(preprocess(row * 0 + 1), preprocess(row * 0 + 3), rtol=1e-5)


def test_frontend_and_model_shapes():
    front = STFTSpectrogram(n_fft=256, hop_length=128, freq_pool=2)
    x = torch.randn(3, 2, N_CROPPED)
    img = front(x)
    assert tuple(img.shape[2:]) == front.output_shape() == (128, 95)
    model = build_model(dict(n_fft=256, hop_length=128, freq_pool=2), dict(channels=(8, 16)))
    assert model(x).shape == (3, 7)
