"""Split manifest handling and label extraction.

Filenames are used ONLY to derive training labels and manage examples; they must
never become model inputs (event rules).
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .constants import CLASSES, CLASS_TO_INDEX

EXPECTED_MANIFEST_SHA256 = "4ccc4937f4ba0e9993837bb26a5ba4463f1a880b689a753ad3a0e39b76ebafc7"


def label_from_filename(filename: str) -> str:
    """`100us_barker5x13_0Mhz_30Mhz_recorded.npy` -> `barker`; `fsk#2` -> `fsk`."""
    token = Path(filename).name.split("_")[1].split("#")[0].lower()
    label = "barker" if token.startswith("barker") else token
    if label not in CLASS_TO_INDEX:
        raise ValueError(f"Unknown modulation token {token!r} in {filename}")
    return label


@dataclass(frozen=True)
class PulseRef:
    """One source pulse. All SNR variants of it inherit `split`."""
    file: str            # compact filename, e.g. ..._recorded.npy
    source_file: str     # original filename from the JSON sidecar (noise-seed key)
    pulse_index: int
    split: str           # "train" | "validation"
    label: str

    @property
    def target(self) -> int:
        return CLASS_TO_INDEX[self.label]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(data_dir: Path, manifest_csv: Path | None = None,
                  verify_checksum: bool = True) -> list[PulseRef]:
    """Read the published 80/20 manifest and attach labels and source filenames."""
    data_dir = Path(data_dir)
    manifest_csv = Path(manifest_csv or data_dir / "split_manifest.csv")
    if verify_checksum and sha256(manifest_csv) != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"{manifest_csv}: checksum does not match the published manifest")
    source_names: dict[str, str] = {}
    refs = []
    with manifest_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row["file"]
            if name not in source_names:
                meta = json.loads((data_dir / name).with_suffix(".json").read_text())
                source_names[name] = meta["source_file"]
            refs.append(PulseRef(name, source_names[name], int(row["pulse_index"]),
                                 row["split"], label_from_filename(name)))
    return refs


def select(refs: list[PulseRef], split: str, max_pulses_per_file: int | None = None) -> list[PulseRef]:
    """Deterministic subset: the first N manifest pulses of `split` in each file."""
    chosen, counts = [], {}
    for ref in refs:
        if ref.split != split:
            continue
        if max_pulses_per_file is not None and counts.get(ref.file, 0) >= max_pulses_per_file:
            continue
        counts[ref.file] = counts.get(ref.file, 0) + 1
        chosen.append(ref)
    return chosen


def class_counts(refs: list[PulseRef]) -> dict[str, int]:
    counts = {name: 0 for name in CLASSES}
    for ref in refs:
        counts[ref.label] += 1
    return counts
