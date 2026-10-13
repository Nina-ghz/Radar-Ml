#!/usr/bin/env python3
"""Compare a run with its parent run, SNR by SNR.

Usage:
    python scripts/compare_runs.py --parent runs/<parent> --run runs/<experiment>
        [--full] [--classes barker cw] [--snrs 0 -5 -10 -15] [--confusion barker cw]

Default: validation_report.json of both runs (written by train.py). --full reads
evaluation_report.json instead (all 20 validation pulses per configuration, written by
evaluate.py); use it before deciding to keep an experiment. Training and inference costs
always come from validation_report.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radarml.comparison import (SATURATED_FLOOR, class_table, confusion_count, load_report,  # noqa: E402
                                per_snr_table, saturated_snrs_hold, verdict)
from radarml.constants import CLASSES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", type=Path, required=True, help="run directory of the parent config")
    parser.add_argument("--run", type=Path, required=True, help="run directory of the experiment")
    parser.add_argument("--full", action="store_true", help="compare evaluation_report.json (evaluate.py)")
    parser.add_argument("--classes", nargs="+", choices=CLASSES, default=["barker", "cw"],
                        help="classes whose recall is detailed")
    parser.add_argument("--snrs", nargs="+", type=int, default=[0, -5, -10, -15],
                        help="requested SNRs of the per-class detail")
    parser.add_argument("--confusion", nargs=2, choices=CLASSES, default=["barker", "cw"],
                        metavar=("TRUE", "PREDICTED"), help="confusion counted at --snrs")
    return parser.parse_args()


def print_costs(parent: dict, run: dict) -> None:
    for label, key in (("Entraînement (min)", "entraînement (min)"),
                       ("Inférence (ms/pulse, STFT incluse)", "inférence (ms/pulse, STFT incluse)"),
                       ("dont STFT (ms/pulse)", "dont STFT (ms/pulse)")):
        print(f"  {label:<36} {parent['costs'][key]:>8.2f} -> {run['costs'][key]:>8.2f}")
    per_epoch = [sum(h["minutes"] for h in r["history"]) / len(r["history"]) for r in (parent, run)]
    print(f"  {'Temps par époque (min)':<36} {per_epoch[0]:>8.2f} -> {per_epoch[1]:>8.2f} "
          f"(x{per_epoch[1] / per_epoch[0]:.2f})")
    print(f"  Matériel : {parent['costs']['matériel']} | {run['costs']['matériel']}")


def main() -> None:
    args = parse_args()
    kind = "evaluation" if args.full else "validation"
    parent, run = load_report(args.parent, kind), load_report(args.run, kind)
    table = per_snr_table(parent, run)
    gain = run["mean_balanced_accuracy"] - parent["mean_balanced_accuracy"]
    tiebreak_gain = run["tiebreak_low_snr"] - parent["tiebreak_low_snr"]

    print(f"Parent : {args.parent.name}\nRun    : {args.run.name}\nRapport : {kind}\n")
    print(table.round(4).to_string())
    print(f"\nMoyenne des 13 SNR : {parent['mean_balanced_accuracy']:.4f} -> "
          f"{run['mean_balanced_accuracy']:.4f} ({gain:+.4f}) : {verdict(gain)}")
    print(f"Tie-break (-20/-25/-30 dB) : {parent['tiebreak_low_snr']:.4f} -> "
          f"{run['tiebreak_low_snr']:.4f} ({tiebreak_gain:+.4f})")
    print(f"SNR >= +5 dB tous >= {SATURATED_FLOOR} : {'oui' if saturated_snrs_hold(run) else 'NON'}")
    if "by_duration" in parent and "by_duration" in run:
        print("Par durée : " + " | ".join(f"{d} µs {parent['by_duration'][d]:.4f} -> {score:.4f}"
                                          for d, score in run["by_duration"].items()))

    true, predicted = args.confusion
    print(f"\nRappel par classe (parent, run) et confusion {true} -> {predicted} :")
    detail = class_table(parent, run, args.classes, args.snrs)
    for name, report in (("parent", parent), ("run", run)):
        detail[f"{true}->{predicted} {name}"] = [confusion_count(report, snr, true, predicted) for snr in args.snrs]
    print(detail.round(4).to_string())

    print("\nCoûts (parent -> run) :")
    print_costs(load_report(args.parent), load_report(args.run))


if __name__ == "__main__":
    main()
