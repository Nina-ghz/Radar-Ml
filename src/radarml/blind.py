"""Blind-test format: read the anonymous shards, write predictions, build a fake blind test.

Official layout (third_party/radarml_lite/participant_loader.py): a directory with
int16 shards of shape (N, 33600) and an index.csv whose columns `file`, `row` and
`sample_id` locate each sample. Only the I/Q rows reach the model; the other
columns of index.csv are never read.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import CLASSES, REQUESTED_SNRS_DB
from .features import preprocess
from .manifest import PulseRef
from .noise import degrade_official

# Kaggle accepts exactly `sample_id,label`; the organizers' offline scorer
# (third_party/radarml_lite/score_submission.py) reads `sample_id,predicted_label`.
KAGGLE_LABEL_COLUMN, SCORER_LABEL_COLUMN = "label", "predicted_label"


class BlindTestDataset(Dataset):
    """Samples of a blind-test directory, in index.csv order: (x, position in index.csv)."""

    def __init__(self, test_dir: Path):
        self.root = Path(test_dir)
        with (self.root / "index.csv").open(newline="", encoding="utf-8") as stream:
            rows = [(r["file"], int(r["row"]), r["sample_id"]) for r in csv.DictReader(stream)]
        self.files, self.rows, self.sample_ids = (list(column) for column in zip(*rows))
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError(f"{self.root / 'index.csv'}: duplicate sample_id")
        self._arrays = {}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _shard(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            self._arrays[name] = np.load(self.root / name, mmap_mode="r", allow_pickle=False)
        return self._arrays[name]

    def __getitem__(self, index: int):
        row = np.asarray(self._shard(self.files[index])[self.rows[index]])
        return torch.from_numpy(preprocess(row)), index

    def __getstate__(self):
        # memmaps are reopened in each DataLoader worker
        return {**self.__dict__, "_arrays": {}}


def read_template(path: Path) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [row["sample_id"] for row in csv.DictReader(stream)]


def write_predictions(path: Path, template_ids: list[str], predictions: dict[str, str],
                      label_column: str = KAGGLE_LABEL_COLUMN) -> None:
    """predictions.csv in template order: header `sample_id,<label_column>`, one row per template ID."""
    missing, extra = set(template_ids) - predictions.keys(), predictions.keys() - set(template_ids)
    if missing or extra:
        raise ValueError(f"Predictions do not match the template: {len(missing)} missing, {len(extra)} unknown IDs")
    if not set(predictions.values()) <= set(CLASSES):
        raise ValueError(f"Unknown labels: {set(predictions.values()) - set(CLASSES)}")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sample_id", label_column))
        writer.writerows((sample_id, predictions[sample_id]) for sample_id in template_ids)


def write_fake_blind_test(refs: list[PulseRef], data_dir: Path, out_dir: Path, shard_size: int = 5000,
                          snrs=REQUESTED_SNRS_DB, seed: int = 0) -> int:
    """Official-format blind test built from `refs` (validation pulses only) with official noise.

    Writes shard_XXX.npy, index.csv, submission_template.csv and the private
    answer_key.csv read by third_party/radarml_lite/score_submission.py.
    """
    if any(ref.split != "validation" for ref in refs):
        raise ValueError("A fake blind test is built from validation pulses only")
    rng = np.random.default_rng(seed)
    samples = [(ref, int(snr)) for ref in refs for snr in snrs]
    order = rng.permutation(len(samples))
    sample_ids = [f"s{value:08x}" for value in rng.choice(16 ** 8, size=len(samples), replace=False)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, index, key = {}, [], []
    for start in range(0, len(order), shard_size):
        chunk = order[start:start + shard_size]
        name = f"shard_{start // shard_size:03d}.npy"
        shard = np.empty((len(chunk), 33_600), dtype=np.int16)
        for row, position in enumerate(chunk):
            ref, snr = samples[position]
            if ref.file not in arrays:
                arrays[ref.file] = np.load(Path(data_dir) / ref.file, mmap_mode="r", allow_pickle=False)
            shard[row] = degrade_official(np.asarray(arrays[ref.file][0, ref.pulse_index]),
                                          ref.source_file, snr, ref.pulse_index)
            sample_id = sample_ids[start + row]
            index.append((name, row, sample_id))
            key.append((sample_id, ref.label, snr, "test"))
        np.save(out_dir / name, shard)
    _write_csv(out_dir / "index.csv", ("file", "row", "sample_id"), index)
    _write_csv(out_dir / "submission_template.csv", ("sample_id",), [(sample_id,) for _, _, sample_id in index])
    _write_csv(out_dir / "answer_key.csv", ("sample_id", "label", "requested_snr_db", "split"), key)
    return len(index)


def _write_csv(path: Path, header, rows) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
