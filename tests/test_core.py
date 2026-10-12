"""Fast sanity tests for the shared utilities (no dataset required, except where skipped)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radarml.constants import CLASSES  # noqa: E402
from radarml.manifest import EXPECTED_MANIFEST_SHA256, label_from_filename, sha256  # noqa: E402
from radarml.metrics import challenge_report  # noqa: E402
from radarml.noise import degrade_official, official_seed  # noqa: E402

DATA_DIR = ROOT / "data" / "raw" / "recorded_dataset"


@pytest.mark.parametrize("name, label", [
    ("100us_barker5x13_0Mhz_30Mhz_recorded.npy", "barker"),
    ("10us_fsk#1_40Mhz_20Mhz--20Mhz_recorded.npy", "fsk"),
    ("50us_quad_40Mhz_-40Mhz-0Mhz_recorded.npy", "quad"),
    ("100us_fmcw_10Mhz_25Mhz-35Mhz_recorded.npy", "fmcw"),
])
def test_labels(name, label):
    assert label_from_filename(name) == label


def test_classes_match_scorer_order():
    assert list(CLASSES) == sorted(CLASSES)


def test_manifest_checksum():
    assert sha256(ROOT / "data" / "manifest" / "split_manifest.csv") == EXPECTED_MANIFEST_SHA256


def test_metric_balanced_accuracy():
    y_true = np.repeat(np.arange(7), 10)
    y_pred = y_true.copy()
    y_pred[:5] = 1                          # half of class 0 wrong
    report = challenge_report(y_true, y_pred, np.zeros_like(y_true))
    assert report["mean_balanced_accuracy"] == pytest.approx((0.5 + 6) / 7)


def test_official_seed_is_deterministic():
    assert official_seed("a.npy", -30, 3) == official_seed("dir/a.npy", -30, 3)
    assert official_seed("a.npy", -30, 3) != official_seed("a.npy", -25, 3)


@pytest.mark.skipif(not DATA_DIR.exists(), reason="dataset not extracted")
def test_noise_reproduces_official_expansion(tmp_path):
    """Our on-the-fly noise equals the organizers' compact_dataset.py expansion."""
    import shutil
    import subprocess

    npy = sorted(DATA_DIR.glob("*_recorded.npy"))[0]
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    shutil.copy(npy, pilot)
    shutil.copy(npy.with_suffix(".json"), pilot)
    subprocess.run([sys.executable, str(ROOT / "third_party/radarml_lite/compact_dataset.py"), "expand",
                    "--input", str(pilot), "--output", str(tmp_path / "out"), "--snrs", "10", "-30"],
                   check=True, cwd=ROOT / "third_party/radarml_lite")
    expanded = next((tmp_path / "out").glob("*_single_channel.npy"))
    official = np.load(expanded)
    meta = json.loads(npy.with_suffix(".json").read_text())
    rows = np.load(npy, mmap_mode="r")
    for snr_index, snr in enumerate((10, -30)):
        for pulse in (0, 57, 99):
            ours = degrade_official(np.asarray(rows[0, pulse]), meta["source_file"], snr, pulse)
            np.testing.assert_array_equal(ours, official[0, snr_index, pulse])
