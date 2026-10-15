#!/usr/bin/env python3
"""Assemble the reproduction package sent to the organizers with the Kaggle predictions.

Usage:
    python scripts/build_submission.py --run runs/E006_seed43_20261014_204114
        --predictions predictions_POPCORN_freq.csv [--inference-log blind_run.log]
        [--out submission/POPCORN_freq]

Package layout:
    README.md            how to reproduce the predictions (command, time, environment)
    METHOD_SUMMARY.md    method, validation results, hardware, attribution
    INFERENCE_RUN.md     the blind-test run that produced the predictions (if --inference-log)
    code/                the inference code at the current commit (git archive)
    model/               model.pt, config.yaml, validation and evaluation reports, history
    figures/             validation figures (balanced accuracy per SNR, confusion matrices)
    predictions_*.csv    the file uploaded to Kaggle
    MANIFEST.json        SHA256 of every file, git commit, Python and package versions
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = ("src", "scripts/predict.py", "scripts/validate_submission.py", "third_party/radarml_lite",
              "configs", "requirements.txt", "pyproject.toml", "docs/INFERENCE.md", "LICENSE")
MODEL_FILES = ("model.pt", "config.yaml", "validation_report.json", "evaluation_report.json", "history.csv")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_code(destination: Path) -> None:
    """Committed files only (git archive), so the package matches the recorded commit."""
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD", *CODE_PATHS], cwd=ROOT,
                             capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")


def readme(run: Path, predictions: Path, commit: str) -> str:
    return f"""# POPCORN — Track B (time-frequency) reproduction package

Kaggle team: `POPCORN_freq`. Model: `{run.name}`. Code commit: `{commit}`.
Method, validation results and hardware: `METHOD_SUMMARY.md`.

## Reproduce `{predictions.name}`

Requirements: Python 3.10 or newer, CPU only is enough.

```
cd code
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\\Scripts\\python.exe
.venv/bin/python scripts/predict.py --run ../model --test-dir <blind_test>/participants/test --output predictions.csv --device cpu
.venv/bin/python scripts/validate_submission.py --predictions predictions.csv --template <blind_test>/participants/submission_template.csv
```

`--test-dir` is the folder that holds `index.csv` and the `signals_*.npy` shards of the organizers'
blind-test archive; `submission_template.csv` is found next to it. The script reads the int16 rows listed
in `index.csv`, applies the fixed per-pulse preprocessing, computes the three STFTs inside the model, and
writes `sample_id,label` in template order. It prints the SHA256 of `model.pt` and of the written file:
compare them with `MANIFEST.json`.

**Expected running time on CPU:** about 22 ms per signal end to end (12 ms for the model with its
STFTs), i.e. about one hour for the 171 600 blind-test signals on a 4-core laptop CPU. See
`INFERENCE_RUN.md` for the measured run.

Prediction is deterministic for a given machine and library versions. On other hardware or versions,
floating-point rounding can in principle change the arg-max of a very small number of near-tied
examples; the versions used are listed in `MANIFEST.json`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inference-log", type=Path, help="log of the blind-test run, summarised in INFERENCE_RUN.md")
    parser.add_argument("--out", type=Path, default=ROOT / "submission" / "POPCORN_freq")
    args = parser.parse_args()

    if git("status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS):
        sys.exit("Commit the inference code first: the package must match a recorded commit.")
    commit = git("rev-parse", "HEAD")
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "model").mkdir(parents=True)

    export_code(args.out / "code")
    for name in MODEL_FILES:
        shutil.copy2(args.run / name, args.out / "model" / name)
    shutil.copytree(args.run / "evaluation_figures", args.out / "figures")
    shutil.copy2(args.predictions, args.out / args.predictions.name)
    shutil.copy2(ROOT / "docs" / "METHOD_SUMMARY.md", args.out / "METHOD_SUMMARY.md")
    (args.out / "README.md").write_text(readme(args.run, args.predictions, commit), encoding="utf-8")
    if args.inference_log:
        lines = [line for line in args.inference_log.read_text(encoding="utf-8").splitlines()
                 if not line.startswith("  ")]
        (args.out / "INFERENCE_RUN.md").write_text(
            "# Blind-test inference run\n\nOutput of `scripts/run_blind.cmd` (progress lines removed).\n\n```\n"
            + "\n".join(lines) + "\n```\n", encoding="utf-8")

    files = sorted(p for p in args.out.rglob("*") if p.is_file())
    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    manifest = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git_commit=commit,
                    python=sys.version.split()[0], packages=packages.splitlines(),
                    files={p.relative_to(args.out).as_posix(): sha256(p) for p in files})
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Package : {args.out} ({len(files)} fichiers, commit {commit[:7]})")
    for name in ("model/model.pt", "model/config.yaml", args.predictions.name):
        print(f"  {manifest['files'][name]}  {name}")


if __name__ == "__main__":
    main()
