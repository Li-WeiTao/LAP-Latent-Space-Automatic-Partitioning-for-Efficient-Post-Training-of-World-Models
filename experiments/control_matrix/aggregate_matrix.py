#!/usr/bin/env python3
"""Aggregate one task's LeWM comparison matrix by fine-tuning seed."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


PARTITION_METHODS = ("random_voronoi", "kmeanspp", "spectral")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--train-seeds", default="0,42,625")
    parser.add_argument("--partition-seeds", default="0,1,2")
    parser.add_argument("--eval-seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--skip-joint",
        action="store_true",
        help="Omit Joint-Continue rows when running smoke subsets.",
    )
    parser.add_argument(
        "--skip-regions",
        action="store_true",
        help="Omit partitioned-method rows when running smoke subsets.",
    )
    parser.add_argument(
        "--methods",
        default="random_voronoi,kmeanspp,spectral",
        help="Comma-separated partition methods to aggregate.",
    )
    parser.add_argument(
        "--include-auto-lap",
        action="store_true",
        help="Add Auto-LAP row from eval/auto (deployed Spectral partition seed).",
    )
    parser.add_argument(
        "--skip-official",
        action="store_true",
        help="Omit official-baseline rows (K-ablation matrices that reuse paired starts).",
    )
    parser.add_argument(
        "--deployment-seed",
        type=int,
        default=0,
        help="Spectral partition seed used by Auto-LAP when verifying parity.",
    )
    return parser.parse_args()


def seeds(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def rate(path: Path) -> float:
    result = json.loads(path.read_text(encoding="utf-8"))
    value = float(result["metrics"]["success_rate"])
    if not 0.0 <= value <= 100.0:
        raise ValueError(
            f"success_rate must use the official percentage scale [0, 100]: "
            f"{value} in {path}"
        )
    return value


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    train = seeds(args.train_seeds)
    partition = seeds(args.partition_seeds)
    evaluate = seeds(args.eval_seeds)
    raw_rows: list[dict] = []

    rows: list[dict] = []
    if not args.skip_official:
        official = [
            rate(args.root / "eval" / "official" / f"eval{e}" / "results.json")
            for e in evaluate
        ]
        rows.append(
            {
                "method": "Official baseline",
                "mean_percent": mean(official),
                "sd_across_finetuning_seeds_percent": None,
                "num_finetuning_seeds": 0,
                "num_partition_seeds": 0,
                "num_eval_seeds": len(evaluate),
                "eval_seed_sd_percent": sd(official),
            }
        )
        for e, value in zip(evaluate, official):
            raw_rows.append(
                {
                    "method": "official",
                    "train_seed": "",
                    "partition_seed": "",
                    "eval_seed": e,
                    "success_rate_percent": value,
                }
            )

    for method, label in (("joint", "Joint-Continue 3ep"), ("global", "Global-FT50")):
        if args.skip_joint and method == "joint":
            continue
        train_means: list[float] = []
        all_values: list[float] = []
        for t in train:
            values = [
                rate(
                    args.root
                    / "eval"
                    / method
                    / f"train{t}"
                    / f"eval{e}"
                    / "results.json"
                )
                for e in evaluate
            ]
            train_means.append(mean(values))
            all_values.extend(values)
            for e, value in zip(evaluate, values):
                raw_rows.append(
                    {
                        "method": method,
                        "train_seed": t,
                        "partition_seed": "",
                        "eval_seed": e,
                        "success_rate_percent": value,
                    }
                )
        rows.append(
            {
                "method": label,
                "mean_percent": mean(train_means),
                "sd_across_finetuning_seeds_percent": sd(train_means),
                "num_finetuning_seeds": len(train),
                "num_partition_seeds": 0,
                "num_eval_seeds": len(evaluate),
                "eval_seed_sd_percent": sd(all_values),
            }
        )

    labels = {
        "random_voronoi": "Random-Voronoi",
        "kmeanspp": "K-means++",
        "spectral": "Spectral",
    }
    partition_methods = [
        method.strip()
        for method in args.methods.split(",")
        if method.strip()
    ]
    for method in partition_methods:
        if args.skip_regions:
            continue
        train_means = []
        all_values = []
        for t in train:
            values = []
            for p in partition:
                for e in evaluate:
                    value = rate(
                        args.root
                        / "eval"
                        / method
                        / f"partition{p}_train{t}"
                        / f"eval{e}"
                        / "results.json"
                    )
                    values.append(value)
                    raw_rows.append(
                        {
                            "method": method,
                            "train_seed": t,
                            "partition_seed": p,
                            "eval_seed": e,
                            "success_rate_percent": value,
                        }
                    )
            train_means.append(mean(values))
            all_values.extend(values)
        rows.append(
            {
                "method": labels[method],
                "mean_percent": mean(train_means),
                "sd_across_finetuning_seeds_percent": sd(train_means),
                "num_finetuning_seeds": len(train),
                "num_partition_seeds": len(partition),
                "num_eval_seeds": len(evaluate),
                "eval_seed_sd_percent": sd(all_values),
            }
        )

    if args.include_auto_lap:
        train_means = []
        all_values = []
        for t in train:
            values = [
                rate(
                    args.root
                    / "auto"
                    / "eval"
                    / f"train{t}"
                    / f"eval{e}"
                    / "results.json"
                )
                for e in evaluate
            ]
            train_means.append(mean(values))
            all_values.extend(values)
            for e, value in zip(evaluate, values):
                raw_rows.append(
                    {
                        "method": "auto_lap",
                        "train_seed": t,
                        "partition_seed": args.deployment_seed,
                        "eval_seed": e,
                        "success_rate_percent": value,
                    }
                )
        rows.append(
            {
                "method": "Auto-LAP",
                "mean_percent": mean(train_means),
                "sd_across_finetuning_seeds_percent": sd(train_means),
                "num_finetuning_seeds": len(train),
                "num_partition_seeds": 1,
                "num_eval_seeds": len(evaluate),
                "eval_seed_sd_percent": sd(all_values),
            }
        )

    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "matrix_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with (args.root / "matrix_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(raw_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(raw_rows)
    (args.root / "matrix_summary.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "train_seeds": train,
                "partition_seeds": partition,
                "eval_seeds": evaluate,
                "primary_error_bar": "sample SD across fine-tuning seeds after averaging partition and evaluation seeds",
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for row in rows:
        spread = row["sd_across_finetuning_seeds_percent"]
        suffix = "" if spread is None else f" ± {spread:.2f}%"
        print(f"{row['method']}: {row['mean_percent']:.2f}%{suffix}")


if __name__ == "__main__":
    main()
