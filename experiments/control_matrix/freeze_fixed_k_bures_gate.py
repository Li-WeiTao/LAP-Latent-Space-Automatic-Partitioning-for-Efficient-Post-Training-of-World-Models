#!/usr/bin/env python3
"""Freeze and audit the fixed-K Check-1 + Jacobian-Bures gate policy.

The metric and its threshold are developed at one fixed K.  The resulting
threshold is then applied unchanged to other fixed-K experiment matrices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


DEFAULT_METRIC = "jacobian_bures_distance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--threshold-audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-k", type=int, default=4)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--check1-threshold", type=float, default=0.5)
    parser.add_argument("--practical-band-pp", type=float, default=0.5)
    parser.add_argument(
        "--policy",
        choices=("with_check1", "standalone"),
        default="with_check1",
        help="with_check1: Check-1 plus frozen Bures; standalone: frozen Bures only.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected True/False, got {value!r}")


def repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    args = parse_args()
    validations = read_csv(args.validation_csv)
    audits = read_csv(args.threshold_audit_csv)
    candidates = [
        row
        for row in audits
        if int(row["num_clusters"]) == args.development_k
        and row["metric"] == args.metric
        and row["policy"] == "with_check1"
    ]
    if len(candidates) != 1:
        raise SystemExit(
            "expected exactly one development threshold row, "
            f"found {len(candidates)}"
        )
    source = candidates[0]
    if not (
        as_bool(source["empirical_full_separation"])
        and as_bool(source["seed_robust_full_separation"])
        and as_bool(source["threshold_is_deployable_candidate"])
    ):
        raise SystemExit("development threshold does not satisfy the audit checks")

    threshold = float(source["threshold_midpoint_for_audit"])
    metric_column = f"{args.metric}_mean"
    output_rows: list[dict[str, object]] = []
    for row in validations:
        k = int(row["num_clusters"])
        delta = float(row["delta_regional_minus_global_pp"])
        check1_pass = as_bool(row["check1_pass"])
        metric_value = float(row[metric_column])
        if args.policy == "with_check1":
            predicted = (
                "regional"
                if check1_pass and metric_value > threshold
                else "global"
            )
        else:
            predicted = "regional" if metric_value > threshold else "global"
        practical = "regional" if delta > args.practical_band_pp else "global"
        point_estimate = row["point_estimate_winner"]
        role = (
            "metric_and_threshold_development"
            if k == args.development_k
            else "frozen_threshold_validation"
        )
        output_rows.append(
            {
                "task": row["task"],
                "num_clusters": k,
                "experiment_role": role,
                "global_mean_percent": float(row["global_mean_percent"]),
                "regional_mean_percent": float(row["regional_mean_percent"]),
                "delta_regional_minus_global_pp": delta,
                "practical_class_0p5pp_band": practical,
                "point_estimate_winner": point_estimate,
                "check1_retained_safety_fraction": float(
                    row["check1_retained_safety_fraction"]
                ),
                "check1_pass": check1_pass,
                "jacobian_bures_distance_mean": metric_value,
                "frozen_bures_threshold": threshold,
                "predicted_branch": predicted,
                "correct_practical_class": predicted == practical,
                "correct_point_estimate_sign": predicted == point_estimate,
                "training_seeds": row["training_seeds"],
                "partition_seeds": row["partition_seeds"],
                "evaluation_seeds": row["evaluation_seeds"],
            }
        )

    ks = sorted({int(row["num_clusters"]) for row in output_rows})
    summaries: dict[str, dict[str, object]] = {}
    for k in ks:
        subset = [row for row in output_rows if row["num_clusters"] == k]
        summaries[f"K{k}"] = {
            "role": subset[0]["experiment_role"],
            "pairs": len(subset),
            "correct_practical_class": sum(
                bool(row["correct_practical_class"]) for row in subset
            ),
            "correct_point_estimate_sign": sum(
                bool(row["correct_point_estimate_sign"]) for row in subset
            ),
            "predicted_branch_counts": dict(
                Counter(str(row["predicted_branch"]) for row in subset)
            ),
        }

    validation_ks = [k for k in ks if k != args.development_k]
    for k in validation_ks:
        summary = summaries[f"K{k}"]
        if summary["correct_practical_class"] != summary["pairs"]:
            raise SystemExit(f"frozen policy failed practical validation at K={k}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = args.output_dir / "frozen_bures_gate_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    policy = {
        "schema_version": 1,
        "policy_name": (
            "fixed-K Check-1 plus Jacobian-Bures gate"
            if args.policy == "with_check1"
            else "fixed-K Jacobian-Bures gate (standalone)"
        ),
        "status": "frozen",
        "source_repository_commit": repository_commit(),
        "metric": args.metric,
        "metric_direction": "larger_supports_regional",
        "metric_and_threshold_development_k": args.development_k,
        "frozen_bures_threshold": threshold,
        "threshold_operator": ">",
        "check1_threshold": args.check1_threshold,
        "decision_rule": (
            "select Regional iff Check 1 passes and the mean Jacobian-Bures "
            "distance is strictly greater than the frozen threshold; otherwise "
            "select Global"
        ),
        "threshold_source": {
            "policy": source["policy"],
            "positive_tasks": source["positive_tasks"].split(","),
            "nonpositive_tasks_after_check1": source["nonpositive_tasks"].split(","),
            "positive_min_mean": float(source["positive_min_mean"]),
            "nonpositive_max_mean": float(source["nonpositive_max_mean"]),
            "mean_separation_margin": float(source["mean_separation_margin"]),
            "seed_robust_separation_margin": float(
                source["seed_robust_separation_margin"]
            ),
        },
        "validation_protocol": {
            "validation_k": validation_ks,
            "threshold_refit_outside_development_k": False,
            "regional_metric_policy": (
                "average fixed-K results over partition seeds 0,1,2; no deployment "
                "seed selection and no cross-K winner selection"
            ),
            "training_seeds": [0, 42, 625],
            "partition_seeds": [0, 1, 2],
            "evaluation_seeds": [0, 1, 2, 3, 4],
            "practical_positive_definition": (
                f"Regional minus Global > {args.practical_band_pp} percentage points"
            ),
            "inconclusive_policy": "treated as non-positive in sufficiency audit",
        },
        "results": summaries,
        "input_files": {
            str(args.validation_csv): sha256(args.validation_csv),
            str(args.threshold_audit_csv): sha256(args.threshold_audit_csv),
        },
        "output_files": {
            validation_path.name: sha256(validation_path),
        },
    }
    policy_path = args.output_dir / "frozen_bures_gate_policy.json"
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(f"frozen threshold: {threshold:.17g}")
    for key, summary in summaries.items():
        print(
            f"{key}: practical={summary['correct_practical_class']}/"
            f"{summary['pairs']}, point-sign="
            f"{summary['correct_point_estimate_sign']}/{summary['pairs']}"
        )
    print(validation_path)
    print(policy_path)


if __name__ == "__main__":
    main()
