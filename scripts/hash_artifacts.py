"""Record SHA256 hashes of frozen artifacts (required at model freeze, Day 4 17:00).

    python scripts/hash_artifacts.py models/final/model.pt configs/final.yaml --output freeze_manifest.json

Also records the git commit and installed package versions.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("freeze_manifest.json"))
    args = p.parse_args()

    manifest = dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=run(["git", "rev-parse", "HEAD"]),
        git_dirty=bool(run(["git", "status", "--porcelain", "--untracked-files=no"])),
        python=sys.version.split()[0],
        packages=(run([sys.executable, "-m", "pip", "freeze"]) or "").splitlines(),
        artifacts={str(f): dict(sha256=sha256(f), bytes=f.stat().st_size) for f in args.files},
    )
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, info in manifest["artifacts"].items():
        print(f"{info['sha256']}  {name}")
    if manifest["git_dirty"]:
        print("WARNING: uncommitted changes - commit before freezing.")


if __name__ == "__main__":
    main()
