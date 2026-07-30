#!/usr/bin/env python3
"""Combine the generic six-method matrix with the human rooms3-50 arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def load_result(path: Path) -> tuple[float, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rate = float(payload["metrics"]["success_rate"])
    if not 0.0 <= rate <= 100.0:
        raise ValueError(f"invalid success-rate scale in {path}: {rate}")
    starts = payload.get("eval_start_indices")
    if not isinstance(starts, list) or len(starts) != 50:
        raise ValueError(f"{path}: expected 50 paired eval_start_indices")
    digest = hashlib.sha256(
        json.dumps([int(value) for value in starts], separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return rate, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/tworoom/matrix"))
    parser.add_argument("--train-seeds", default="0,42,625")
    parser.add_argument("--partition-seeds", default="0,1,2")
    parser.add_argument("--eval-seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    train_seeds = parse_ints(args.train_seeds)
    eval_seeds = parse_ints(args.eval_seeds)

    generic_path = args.root / "matrix_summary.json"
    generic = json.loads(generic_path.read_text(encoding="utf-8"))
    rows = list(generic["rows"])
    official_hashes: dict[int, str] = {}
    for eval_seed in eval_seeds:
        _, digest = load_result(
            args.root / "eval" / "official" / f"eval{eval_seed}" / "results.json"
        )
        official_hashes[eval_seed] = digest

    train_means: list[float] = []
    raw_rows: list[dict[str, object]] = []
    for train_seed in train_seeds:
        values: list[float] = []
        for eval_seed in eval_seeds:
            path = (
                args.root
                / "eval"
                / "human_rooms3"
                / f"train{train_seed}"
                / f"eval{eval_seed}"
                / "results.json"
            )
            value, digest = load_result(path)
            if digest != official_hashes[eval_seed]:
                raise ValueError(f"unpaired evaluation starts in {path}")
            values.append(value)
            raw_rows.append(
                {
                    "method": "Human rooms3-50",
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "success_rate_percent": value,
                }
            )
        train_means.append(statistics.fmean(values))

    rows.append(
        {
            "method": "Human rooms3-50",
            "mean_percent": statistics.fmean(train_means),
            "sd_across_finetuning_seeds_percent": (
                statistics.stdev(train_means) if len(train_means) > 1 else 0.0
            ),
            "num_finetuning_seeds": len(train_seeds),
            "num_partition_seeds": 0,
            "num_eval_seeds": len(eval_seeds),
            "eval_seed_sd_percent": None,
        }
    )

    payload = {
        "dataset": "tworoom",
        "train_seeds": train_seeds,
        "partition_seeds": parse_ints(args.partition_seeds),
        "eval_seeds": eval_seeds,
        "primary_error_bar": (
            "sample SD across fine-tuning seeds after averaging partition and paired evaluation seeds"
        ),
        "paired_start_sha256_by_eval_seed": official_hashes,
        "rows": rows,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "tworoom_main_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.root / "tworoom_main_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.root / "tworoom_human_rooms3_raw.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    for row in rows:
        spread = row.get("sd_across_finetuning_seeds_percent")
        suffix = "" if spread is None else f" ± {float(spread):.2f}%"
        print(f"{row['method']}: {float(row['mean_percent']):.2f}%{suffix}")


if __name__ == "__main__":
    main()
