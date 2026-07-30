#!/usr/bin/env python3
"""Build an auditable inventory of the migrated TwoRoom experiment runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def classify(name: str) -> str:
    lower = name.lower()
    if lower == "archive" or "bf16" in lower:
        return "Archived precision control"
    if "success_rate" in lower:
        return "Task-success evaluation"
    if "joint_continue" in lower:
        return "Joint-Continue training"
    if "geometry_train_global_ft" in lower:
        return "Global predictor fine-tuning"
    if "geometry_train_region_predictors" in lower:
        return "Manual-partition predictor fine-tuning"
    if lower.startswith(("tworoom_latent_kmeanspp_", "tworoom_latent_random_voronoi_", "tworoom_latent_spectral_")):
        return "Automatic-partition predictor fine-tuning"
    if any(token in lower for token in ("latent_preprocess", "latent_unsup_cluster", "latent_kmeanspp_multirestart", "latent_landmark_spectral", "latent_random_voronoi")):
        return "Automatic partition and stability"
    if "geometry_latent" in lower or "latent_umap" in lower:
        return "Latent-region separability"
    if "trajectory_deviation" in lower or "trajectory_switch" in lower:
        return "Predictor trajectory deviation"
    if "predictor_rule" in lower:
        return "Action-free routing analysis"
    if "encoder" in lower or "resample_check" in lower or "control_v2" in lower:
        return "Encoder/gauge-drift analysis"
    if "speed_benchmark" in lower or "inference_speed" in lower:
        return "Efficiency benchmark"
    if "unique_timestep" in lower or "embedding_loader_validation" in lower:
        return "Lossless cache validation"
    if "smoke" in lower or "validation" in lower:
        return "Smoke/contract validation"
    return "Supporting experiment"


def count_suffix(files: list[Path], suffix: str) -> int:
    return sum(path.suffix.lower() == suffix for path in files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/tworoom/results"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/tworoom/EXPERIMENT_INVENTORY.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, str | int]] = []
    for directory in sorted(path for path in args.results_root.iterdir() if path.is_dir()):
        files = [path for path in directory.rglob("*") if path.is_file()]
        rows.append(
            {
                "experiment_directory": directory.name,
                "category": classify(directory.name),
                "total_files": len(files),
                "json_files": count_suffix(files, ".json"),
                "csv_files": count_suffix(files, ".csv"),
                "npz_files": count_suffix(files, ".npz"),
                "log_files": count_suffix(files, ".log"),
            }
        )

    root_files = [path for path in args.results_root.iterdir() if path.is_file()]
    rows.append(
        {
            "experiment_directory": "__root_orchestration__",
            "category": "Cross-run orchestration and aggregation",
            "total_files": len(root_files),
            "json_files": count_suffix(root_files, ".json"),
            "csv_files": count_suffix(root_files, ".csv"),
            "npz_files": count_suffix(root_files, ".npz"),
            "log_files": count_suffix(root_files, ".log"),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} inventory rows to {args.output}")


if __name__ == "__main__":
    main()
