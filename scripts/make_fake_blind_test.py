#!/usr/bin/env python3
"""Build a fake blind test from VALIDATION pulses, in the official blind-test format.

Usage:
    python scripts/make_fake_blind_test.py [--pulses-per-file 2] [--out data/fake_blind]

Each selected validation pulse is degraded at the 13 requested SNRs with the official
noise, shuffled and written as anonymous int16 shards (N, 33600) with index.csv,
submission_template.csv and a private answer_key.csv for the official scorer. Used
to test scripts/predict.py end to end without ever touching the real blind test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radarml.blind import write_fake_blind_test  # noqa: E402
from radarml.manifest import load_manifest, select  # noqa: E402

DATA_DIR = ROOT / "data" / "raw" / "recorded_dataset"
MANIFEST_CSV = ROOT / "data" / "manifest" / "split_manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pulses-per-file", type=int, default=2, help="validation pulses per configuration")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "fake_blind")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    refs = select(load_manifest(DATA_DIR, MANIFEST_CSV), "validation", args.pulses_per_file)
    count = write_fake_blind_test(refs, DATA_DIR, args.out, shard_size=args.shard_size, seed=args.seed)
    print(f"Faux blind test : {count:,} échantillons ({len(refs):,} pulses de validation × 13 SNR) dans {args.out}")


if __name__ == "__main__":
    main()
