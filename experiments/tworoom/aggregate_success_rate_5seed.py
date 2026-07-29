#!/usr/bin/env python3
"""Aggregate per-seed TwoRoom success-rate results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY = THIS_DIR / "results" / "tworoom_success_rate_5seed_summary.csv"


def load_rows(summary_path: Path) -> list[dict]:
    with summary_path.open() as f:
        return list(csv.DictReader(f))


def aggregate(summary_path: Path) -> dict:
    rows = load_rows(summary_path)
    by_mode: dict[str, list[float]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(float(row["success_rate"]))

    stats = {}
    for mode, vals in sorted(by_mode.items()):
        arr = np.asarray(vals, dtype=np.float64)
        stats[mode] = {
            "n_seeds": int(len(arr)),
            "mean_success_rate": float(arr.mean()),
            "std_success_rate": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
            "per_seed": vals,
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    stats = aggregate(args.summary)
    out_json = args.out_json or args.summary.with_suffix(".json")

    with out_json.open("w") as f:
        json.dump(stats, f, indent=2)

    print("=== TwoRoom success rate (5 seeds) ===")
    for mode, s in stats.items():
        per = ", ".join(f"{v:.1f}" for v in s["per_seed"])
        print(
            f"{mode:10} mean={s['mean_success_rate']:.2f}% "
            f"± {s['std_success_rate']:.2f}%  per-seed: [{per}]"
        )
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
