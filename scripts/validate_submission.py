"""Check predictions.csv against the blind-test template before submitting.

    python scripts/validate_submission.py --predictions predictions.csv --template submission_template.csv

Same rejection rules as the organizer scorer: exact header, one row per ID,
no missing/duplicate/extra IDs, only the seven permitted labels.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from radarml.constants import CLASSES  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    args = p.parse_args()

    with args.template.open(newline="", encoding="utf-8") as stream:
        expected = [row["sample_id"] for row in csv.DictReader(stream)]
    with args.predictions.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", "predicted_label"]:
            sys.exit(f"Invalid header {reader.fieldnames}; required: sample_id,predicted_label")
        rows = list(reader)

    errors = []
    ids = Counter(r["sample_id"] for r in rows)
    duplicates = [i for i, n in ids.items() if n > 1]
    missing, extra = set(expected) - set(ids), set(ids) - set(expected)
    bad_labels = Counter(r["predicted_label"] for r in rows if r["predicted_label"] not in CLASSES)
    if duplicates:
        errors.append(f"{len(duplicates)} duplicate IDs")
    if missing:
        errors.append(f"{len(missing)} missing IDs")
    if extra:
        errors.append(f"{len(extra)} unknown IDs")
    if bad_labels:
        errors.append(f"invalid labels: {dict(bad_labels)}")
    if errors:
        sys.exit("INVALID: " + "; ".join(errors))
    counts = Counter(r["predicted_label"] for r in rows)
    print(f"VALID: {len(rows)} predictions | label distribution: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
