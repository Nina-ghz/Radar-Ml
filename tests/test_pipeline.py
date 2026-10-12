"""Shape and invariance tests for the Track B pipeline (needs torch)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radarml.features import CROP, N_CROPPED, MultiResSTFT, STFTSpectrogram, preprocess  # noqa: E402
from radarml.models import build_model, count_parameters  # noqa: E402


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


def test_train_snr_weights(tmp_path):
    from radarml.constants import REQUESTED_SNRS_DB
    from radarml.data import TrainDataset
    from radarml.manifest import PulseRef

    arr = np.zeros((1, 5, 33_600), dtype=np.int16)
    arr[0, :, 6000:26000] = 1000
    np.save(tmp_path / "dummy_recorded.npy", arr)
    ref = PulseRef("dummy_recorded.npy", "dummy_recorded.npy", 0, "train", "barker")

    # Mismatched length should raise ValueError
    with pytest.raises(ValueError):
        TrainDataset([ref], tmp_path, train_snr_weights=[1.0, 2.0])

    # Negative / zero sum should raise ValueError
    with pytest.raises(ValueError):
        TrainDataset([ref], tmp_path, train_snr_weights=[-1.0] * len(REQUESTED_SNRS_DB))

    # Weight concentrated on a single SNR
    target_idx = 6  # 0 dB
    weights = [0.0] * len(REQUESTED_SNRS_DB)
    weights[target_idx] = 5.0
    ds = TrainDataset([ref], tmp_path, train_snr_weights=weights)
    assert ds.snr_weights[target_idx] == 1.0

    # Ensure getitem succeeds and produces correct tensor
    x, y = ds[0]
    assert x.shape == (2, N_CROPPED)
    assert y == 0


def test_multires_stft_shapes_and_standardization():
    front = MultiResSTFT(n_ffts=(64, 256, 1024), hop_length=128, out_shape=(128, 95))
    x = torch.randn(3, 2, N_CROPPED)
    out = front(x)
    assert out.shape == (3, 3, 128, 95)
    assert front.output_shape() == (128, 95)

    # Standardisation par canal séparément pour chaque exemple
    for b in range(3):
        for c in range(3):
            assert out[b, c].mean().item() == pytest.approx(0.0, abs=1e-5)
            assert out[b, c].std().item() == pytest.approx(1.0, rel=1e-3)


def test_build_model_multires():
    stft = dict(type="multires", n_ffts=(64, 256, 1024), hop_length=128, out_shape=(128, 95))
    cnn = dict(channels=(8, 16))
    x = torch.randn(2, 2, N_CROPPED)
    model = build_model(stft, cnn)
    assert model(x).shape == (2, 7)


def test_build_model_e001_backward_compatibility():
    stft_e001 = dict(n_fft=256, hop_length=128, freq_pool=2)
    cnn_e001 = dict(channels=(16, 32, 64, 128), dropout=0.3)
    m1 = build_model(stft_e001, cnn_e001)
    assert isinstance(m1.frontend, STFTSpectrogram)
    assert count_parameters(m1) == 292_279

    m2 = build_model(dict(type="single", **stft_e001), cnn_e001)
    assert isinstance(m2.frontend, STFTSpectrogram)
    assert count_parameters(m2) == 292_279
