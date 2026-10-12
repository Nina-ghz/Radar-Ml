"""Extract the compact dataset, place the split manifest beside it, verify integrity.

    python scripts/setup_data.py --zip recorded_dataset.zip --dest data/raw

Checks: manifest SHA256, 660 recordings, per-file SHA256 (from JSON sidecars),
array layout (1, 100, 33600) int16, complete 80/20 coverage, class counts.
"""
import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from radarml.manifest import load_manifest, sha256, class_counts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", type=Path, help="recorded_dataset.zip (skip if already extracted)")
    p.add_argument("--dest", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--skip-file-hashes", action="store_true", help="Skip per-file SHA256 (faster)")
    args = p.parse_args()

    data_dir = args.dest / "recorded_dataset"
    if args.zip:
        if data_dir.exists():
            print(f"{data_dir} already exists; skipping extraction")
        else:
            print(f"Extracting {args.zip} -> {args.dest} (~4.4 GB)...", flush=True)
            with zipfile.ZipFile(args.zip) as archive:
                archive.extractall(args.dest)
    for name in ("split_manifest.csv", "split_manifest.json"):
        target = data_dir / name
        if not target.exists():
            shutil.copyfile(ROOT / "data" / "manifest" / name, target)

    files = sorted(data_dir.glob("*_recorded.npy"))
    assert len(files) == 660, f"Expected 660 recordings, found {len(files)}"
    for i, path in enumerate(files, 1):
        meta = json.loads(path.with_suffix(".json").read_text())
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        assert array.shape == (1, 100, 33600) and array.dtype == np.int16, path
        if not args.skip_file_hashes:
            assert sha256(path) == meta["output_sha256"], f"Checksum mismatch: {path}"
        if i % 110 == 0:
            print(f"  verified {i}/660", flush=True)

    refs = load_manifest(data_dir)
    per_file = Counter((r.file, r.split) for r in refs)
    assert all(per_file[(f.name, "train")] == 80 and per_file[(f.name, "validation")] == 20 for f in files)
    train = [r for r in refs if r.split == "train"]
    print("OK - 660 recordings, manifest checksum valid, 80/20 split complete")
    print(f"Training pulses: {len(train)} | per class: {class_counts(train)}")


if __name__ == "__main__":
    main()
