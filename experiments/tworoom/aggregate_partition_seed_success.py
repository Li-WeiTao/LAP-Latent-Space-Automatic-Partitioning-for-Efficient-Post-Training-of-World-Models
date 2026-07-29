#!/usr/bin/env python3
"""Aggregate success rates across partition seeds and matched eval seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--method-summary", type=Path, action="append", required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument(
        "--reference-summary", type=Path, action="append", required=True
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_summary(path: Path) -> dict[int, dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty summary: {path}")
    by_seed: dict[int, dict] = {}
    for row in rows:
        # New spectral summaries use ``seed`` while the historical K-means++
        # multi-train-seed summaries use ``eval_seed``.  Both identify the same
        # paired evaluation unit and must remain readable for fair comparison.
        seed_value = row.get("seed") or row.get("eval_seed")
        if seed_value is None:
            raise KeyError(f"Summary has neither seed nor eval_seed column: {path}")
        seed = int(seed_value)
        if seed in by_seed:
            raise ValueError(f"Duplicate eval seed {seed} in {path}")
        by_seed[seed] = {
            "success_rate": float(row["success_rate"]),
            "successes": int(row["successes"]),
            "num_eval": int(row["num_eval"]),
        }
    return by_seed


def aggregate(paths: list[Path]) -> dict:
    runs = [load_summary(path) for path in paths]
    eval_seeds = sorted(runs[0])
    if any(sorted(run) != eval_seeds for run in runs[1:]):
        raise ValueError("Partition summaries do not share identical eval seeds")
    num_eval = {run[seed]["num_eval"] for run in runs for seed in eval_seeds}
    if len(num_eval) != 1:
        raise ValueError("Partition summaries do not share one num_eval")

    matrix = [
        [run[seed]["success_rate"] for seed in eval_seeds] for run in runs
    ]
    partition_means = [statistics.fmean(row) for row in matrix]
    partition_eval_sds = [sample_sd(row) for row in matrix]
    per_eval_means = [
        statistics.fmean(matrix[p][e] for p in range(len(matrix)))
        for e in range(len(eval_seeds))
    ]
    return {
        "summary_paths": [str(path.resolve()) for path in paths],
        "num_partition_seeds": len(paths),
        "eval_seeds": eval_seeds,
        "episodes_per_eval_seed": next(iter(num_eval)),
        "success_rate_matrix_percent": matrix,
        "per_partition_mean_percent": partition_means,
        "per_partition_eval_seed_sample_sd_percent": partition_eval_sds,
        "grand_mean_percent": statistics.fmean(partition_means),
        "partition_seed_sample_sd_of_means_percent": sample_sd(partition_means),
        "per_eval_seed_partition_mean_percent": per_eval_means,
        "eval_seed_sample_sd_of_partition_means_percent": sample_sd(per_eval_means),
    }


def main() -> None:
    args = parse_args()
    method = aggregate(args.method_summary)
    reference = aggregate(args.reference_summary)
    if method["eval_seeds"] != reference["eval_seeds"]:
        raise ValueError("Method/reference eval seeds differ")
    if method["episodes_per_eval_seed"] != reference["episodes_per_eval_seed"]:
        raise ValueError("Method/reference eval budgets differ")

    differences = [
        method_value - reference_value
        for method_value, reference_value in zip(
            method["per_eval_seed_partition_mean_percent"],
            reference["per_eval_seed_partition_mean_percent"],
            strict=True,
        )
    ]
    comparison = {
        "method_name": args.method_name,
        "reference_name": args.reference_name,
        "method": method,
        "reference": reference,
        "paired_unit": (
            "eval seed after averaging over the three independent partition seeds"
        ),
        "per_eval_seed_difference_percentage_points": differences,
        "mean_difference_percentage_points": statistics.fmean(differences),
        "difference_sample_sd_percentage_points": sample_sd(differences),
        "interpretation": "descriptive; five eval seeds are not a significance test",
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(comparison, indent=2) + "\n")
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "eval_seed",
                "method_partition_mean_percent",
                "reference_partition_mean_percent",
                "difference_percentage_points",
            ),
        )
        writer.writeheader()
        for seed, method_value, reference_value, difference in zip(
            method["eval_seeds"],
            method["per_eval_seed_partition_mean_percent"],
            reference["per_eval_seed_partition_mean_percent"],
            differences,
            strict=True,
        ):
            writer.writerow(
                {
                    "eval_seed": seed,
                    "method_partition_mean_percent": method_value,
                    "reference_partition_mean_percent": reference_value,
                    "difference_percentage_points": difference,
                }
            )
    print(
        f"{args.method_name}: {method['grand_mean_percent']:.3f}% "
        f"± {method['partition_seed_sample_sd_of_means_percent']:.3f}pp across partitions"
    )
    print(
        f"{args.reference_name}: {reference['grand_mean_percent']:.3f}% "
        f"± {reference['partition_seed_sample_sd_of_means_percent']:.3f}pp across partitions"
    )
    print(
        f"paired per-eval-seed difference: {statistics.fmean(differences):+.3f} "
        f"± {sample_sd(differences):.3f}pp"
    )


if __name__ == "__main__":
    main()
