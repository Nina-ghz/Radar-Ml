"""Shape and invariance tests for the Track B pipeline (needs torch)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radarml.comparison import (class_recall, class_table, confusion_count, per_snr_table,  # noqa: E402
                                saturated_snrs_hold, verdict)
from radarml.constants import CLASSES, REQUESTED_SNRS_DB  # noqa: E402
from radarml.data import TrainDataset  # noqa: E402
from radarml.evaluation import duration_of, resolve_device  # noqa: E402
from radarml.features import CROP, N_CROPPED, MultiResSTFT, STFTSpectrogram, preprocess  # noqa: E402
from radarml.manifest import PulseRef  # noqa: E402
from radarml.metrics import challenge_report  # noqa: E402
from radarml.models import SpecCNN, TrackBModel, build_model, count_parameters  # noqa: E402

E001_STFT = dict(n_fft=256, hop_length=128, freq_pool=2)
E001_CNN = dict(channels=(16, 32, 64, 128), dropout=0.3)


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


@pytest.fixture
def train_ref(tmp_path):
    name = "10us_cw_0Mhz_recorded.npy"
    rng = np.random.default_rng(0)
    np.save(tmp_path / name, rng.integers(-2000, 2000, size=(1, 2, 33_600), dtype=np.int16))
    return PulseRef(name, name, 0, "train", "cw")


@pytest.mark.parametrize("bad", [[1.0, 2.0], [-1.0] * 13, [0.0] * 13])
def test_train_snr_weights_validation(train_ref, tmp_path, bad):
    with pytest.raises(ValueError):
        TrainDataset([train_ref], tmp_path, snr_weights=bad)


def test_train_snr_weights_bias_the_draw(train_ref, tmp_path):
    weights = [1] * 6 + [2] * 3 + [4] * 4       # E003: more low SNRs
    ds = TrainDataset([train_ref], tmp_path, snr_weights=weights)
    draws = np.array([ds._draw_snr(np.random.default_rng(i)) for i in range(4000)])
    assert np.isin(draws, REQUESTED_SNRS_DB[-4:]).mean() == pytest.approx(16 / 28, abs=0.03)

    only_zero_db = TrainDataset([train_ref], tmp_path, snr_weights=[0] * 6 + [5] + [0] * 6)
    assert {only_zero_db._draw_snr(np.random.default_rng(i)) for i in range(50)} == {0.0}
    x, y = only_zero_db[0]
    assert x.shape == (2, N_CROPPED) and y == train_ref.target


def test_drawn_snrs_are_the_training_draws(train_ref, tmp_path):
    weights = [1] * 6 + [2] * 3 + [4] * 4
    ds = TrainDataset([train_ref] * 6000, tmp_path, snr_weights=weights, seed=42)
    ds.set_epoch(1)
    draws = ds.drawn_snrs()
    share = np.array([(draws == snr).mean() for snr in REQUESTED_SNRS_DB])
    np.testing.assert_allclose(share, np.array(weights) / sum(weights), atol=0.015)
    # same generator as __getitem__, so it describes the epoch that is actually trained
    assert draws[7] == ds._draw_snr(np.random.default_rng((42, 1, 7)))
    np.testing.assert_array_equal(draws, ds.drawn_snrs())
    ds.set_epoch(2)
    assert not np.array_equal(draws, ds.drawn_snrs())


def test_train_default_snr_draw_is_unchanged(train_ref, tmp_path):
    """Without weights the draw must stay the E001 one (same generator call)."""
    ds = TrainDataset([train_ref], tmp_path, seed=42)
    for i in range(20):
        expected = REQUESTED_SNRS_DB[np.random.default_rng(i).integers(len(REQUESTED_SNRS_DB))]
        assert ds._draw_snr(np.random.default_rng(i)) == expected


def test_multires_stft_shapes_and_standardisation():
    front = MultiResSTFT(n_ffts=(64, 256, 1024), hop_length=128, out_shape=(128, 95))
    out = front(torch.randn(3, 2, N_CROPPED))
    assert out.shape == (3, 3, 128, 95)
    assert front.output_shape() == (128, 95) and front.out_channels == 3
    assert torch.isfinite(out).all()
    # each channel of each example is standardised on its own
    assert out.mean(dim=(2, 3)).abs().max().item() < 1e-4
    assert out.std(dim=(2, 3)).sub(1).abs().max().item() < 1e-3


def test_multires_256_channel_matches_single_stft():
    """At 256 x 95 the 256 window needs no resize: it is the single STFT without pooling."""
    x = torch.randn(2, 2, N_CROPPED)
    multi = MultiResSTFT(n_ffts=(256,), hop_length=128, out_shape=(256, 95))(x)
    single = STFTSpectrogram(n_fft=256, hop_length=128, freq_pool=1)(x)
    assert torch.allclose(multi, single, atol=1e-5)


def test_build_model_multires():
    stft = dict(type="multires", n_ffts=[64, 256, 1024], hop_length=128, out_shape=[128, 95])
    model = build_model(stft, dict(channels=[8, 16]))
    assert isinstance(model.frontend, MultiResSTFT)
    assert model.backbone.features[0][0].in_channels == 3
    assert model(torch.randn(2, 2, N_CROPPED)).shape == (2, 7)
    with pytest.raises(ValueError):
        build_model(dict(type="wavelet"), dict(channels=[8, 16]))


@pytest.mark.parametrize("stft", [E001_STFT, dict(type="single", **E001_STFT)])
def test_build_model_e001_is_unchanged(stft):
    """Same seed -> exactly the weights of the original STFTSpectrogram + SpecCNN model."""
    torch.manual_seed(42)
    reference = TrackBModel(STFTSpectrogram(**E001_STFT), SpecCNN(**E001_CNN)).eval()
    torch.manual_seed(42)
    model = build_model(stft, E001_CNN).eval()
    assert count_parameters(model) == 292_279
    assert model.state_dict().keys() == reference.state_dict().keys()
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in model.state_dict().items())
    x = torch.randn(2, 2, N_CROPPED)
    assert torch.equal(model(x), reference(x))


def test_configs():
    e001 = yaml.safe_load((ROOT / "configs" / "e001_baseline.yaml").read_text(encoding="utf-8"))
    e002 = yaml.safe_load((ROOT / "configs" / "e002_multires.yaml").read_text(encoding="utf-8"))
    assert e001["stft"] == E001_STFT and e001["cnn"] == {**E001_CNN, "channels": [16, 32, 64, 128]}
    assert e001["train_snrs"] == list(REQUESTED_SNRS_DB)
    # one variable per experiment: E002 only changes the representation
    assert e001.keys() == e002.keys()
    assert {key for key in e001 if e001[key] != e002[key]} == {"experiment", "stft"}
    assert count_parameters(build_model(e002["stft"], e002["cnn"])) == 292_279 + 2 * 16 * 9

    # one variable per experiment: E003 only adds train_snr_weights to E002
    e003 = yaml.safe_load((ROOT / "configs" / "e003_lowsnr.yaml").read_text(encoding="utf-8"))
    assert e003["train_snr_weights"] == [1] * 6 + [2] * 3 + [4] * 4
    assert e003.keys() - e002.keys() == {"train_snr_weights"}
    assert {key for key in e002 if e002[key] != e003[key]} == {"experiment"}

    # one variable per experiment: E004 only changes the image height of E003
    e004 = yaml.safe_load((ROOT / "configs" / "e004_fullres.yaml").read_text(encoding="utf-8"))
    assert e003.keys() == e004.keys()
    assert {key for key in e003 if e003[key] != e004[key]} == {"experiment", "stft"}
    assert {key for key in e003["stft"] if e003["stft"][key] != e004["stft"][key]} == {"out_shape"}
    assert e004["stft"]["out_shape"] == [256, 95]
    # global average pooling: the image size does not change the number of parameters
    assert count_parameters(build_model(e004["stft"], e004["cnn"])) == 292_279 + 2 * 16 * 9

    # one variable per experiment: E005 trains E004 on every training pulse
    e005 = yaml.safe_load((ROOT / "configs" / "e005_fulldata.yaml").read_text(encoding="utf-8"))
    assert {key for key in e004 if e004[key] != e005[key]} == {"experiment", "train_pulses_per_file"}
    assert e005["train_pulses_per_file"] is None

    # one variable per experiment: E006 trains E005 for 20 epochs instead of 8
    e006 = yaml.safe_load((ROOT / "configs" / "e006_long.yaml").read_text(encoding="utf-8"))
    assert {key for key in e005 if e005[key] != e006[key]} == {"experiment", "epochs"}
    assert e006["epochs"] == 20

    # replicates: E005 and E006 with another seed, nothing else changed
    for name, parent in (("e005_seed43", e005), ("e006_seed43", e006)):
        replicate = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert {key for key in parent if parent[key] != replicate[key]} == {"experiment", "seed"}
        assert replicate["seed"] == 43


def _report(barker_as_cw: int) -> dict:
    """7 classes x 10 pulses at +5 and -10 dB; `barker_as_cw` barker pulses go to cw at -10 dB."""
    y_true = np.tile(np.repeat(np.arange(len(CLASSES)), 10), 2)
    snrs = np.repeat([5, -10], 10 * len(CLASSES))
    y_pred = y_true.copy()
    y_pred[10 * len(CLASSES):][:barker_as_cw] = CLASSES.index("cw")
    return challenge_report(y_true, y_pred, snrs)


def test_comparison_helpers():
    parent, run = _report(barker_as_cw=4), _report(barker_as_cw=1)
    assert class_recall(parent, -10, "barker") == pytest.approx(0.6)
    assert confusion_count(parent, -10, "barker", "cw") == 4 and confusion_count(run, -10, "barker", "cw") == 1
    table = per_snr_table(parent, run)
    assert list(table.index) == [5, -10]
    assert table.loc[-10, "delta"] == pytest.approx(0.3 / len(CLASSES)) and table.loc[5, "delta"] == 0
    assert class_table(parent, run, ["barker", "cw"], [-10]).loc[-10].tolist() == pytest.approx([0.6, 0.9, 1, 1])
    assert saturated_snrs_hold(run)
    with pytest.raises(ValueError):       # not the same validation pulses
        per_snr_table(parent, challenge_report([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [5] * 7))
    assert [verdict(g).split()[0] for g in (0.012, 0.006, 0.004, -0.01)] == ["retenu", "provisoire", "rejeté", "rejeté"]


def test_evaluation_helpers():
    assert duration_of(PulseRef("100us_barker5x13_0Mhz_30Mhz_recorded.npy", "", 0, "validation", "barker")) == 100
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")
