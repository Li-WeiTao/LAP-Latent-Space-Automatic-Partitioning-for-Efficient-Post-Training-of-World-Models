#!/usr/bin/env python3
"""Aggregate the same R_ij for paper methods 11 (mean) and 12 (min)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

LAYER1_TASKS = ("tworoom", "reacher", "cube")
SOURCE = "K4_excluding_pusht_balanced_accuracy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", choices=("lewm", "subjepa"), default="subjepa")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fit(values, labels):
    values = np.asarray(values, float)
    regional = np.asarray(labels) == "regional"
    unique = np.unique(values)
    span = max(float(np.ptp(unique)), 1.0)
    candidates = np.r_[unique[0] - span, (unique[:-1] + unique[1:]) / 2, unique[-1] + span]
    best = None
    for direction in ("higher", "lower"):
        for threshold in candidates:
            pred = values > threshold if direction == "higher" else values < threshold
            balanced = 0.5 * (np.mean(pred[regional]) + np.mean(~pred[~regional]))
            key = (
                balanced,
                np.mean(pred == regional),
                np.min(np.abs(values - threshold)) / span,
                direction == "higher",
                -threshold,
            )
            if best is None or key > best[0]:
                best = (key, direction, float(threshold))
    return best[1], best[2]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    screen = "subjepa_k_geometry_screen" if args.model == "subjepa" else "lewm_k4_geometry_screen"
    by_seed = pd.read_csv(
        repo / f"experiments/control_matrix/assets/{screen}/response_geometry_cross_k_by_seed.csv"
    )
    pair_path = repo / f"experiments/control_matrix/assets/{screen}/jacobian_fixed_k_pair_metrics.csv"
    if pair_path.is_file():
        pairs = pd.read_csv(pair_path)
        recomputed = (
            pairs.groupby(["task", "num_clusters", "partition_seed"], as_index=False)
            .agg(
                mean_pairwise_response=("pairwise_response_contrast", "mean"),
                minimum_pairwise_response=("pairwise_response_contrast", "min"),
            )
        )
        merged = by_seed.merge(
            recomputed,
            on=["task", "num_clusters", "partition_seed"],
            suffixes=("", "_from_pairs"),
            validate="one_to_one",
        )
        mean_gap = float(
            np.max(np.abs(merged.mean_pairwise_response - merged.mean_pairwise_response_from_pairs))
        )
        min_gap = float(
            np.max(
                np.abs(merged.minimum_pairwise_response - merged.minimum_pairwise_response_from_pairs)
            )
        )
        if max(mean_gap, min_gap) > 1e-12:
            raise SystemExit(f"by-seed R_ij does not match pair table: mean {mean_gap} min {min_gap}")
    summary = by_seed[
        [
            "task",
            "num_clusters",
            "partition_seed",
            "mean_pairwise_response",
            "minimum_pairwise_response",
            "pairwise_uniformity_min_over_mean",
        ]
    ].copy()
    summary.to_csv(out / "pairwise_response_summary.csv", index=False)

    cell = by_seed.groupby(["task", "num_clusters"], as_index=False).agg(
        mean_pairwise_response=("mean_pairwise_response", "mean"),
        minimum_pairwise_response=("minimum_pairwise_response", "mean"),
    )
    cell["R_ij"] = "pairwise_response_contrast"
    cell["per_seed_aggregation"] = "arithmetic mean over partition seeds 0,1,2"
    cell["method_11_mean_Rij"] = cell["mean_pairwise_response"]
    cell["method_12_min_Rij"] = cell["minimum_pairwise_response"]
    cell.to_csv(out / "response_pair_aggregation.csv", index=False)
    cell[cell.num_clusters.eq(4)].to_csv(out / "response_pair_aggregation_k4.csv", index=False)

    lewm_k4 = (
        pd.read_csv(
            repo
            / "experiments/control_matrix/assets/lewm_k4_geometry_screen/response_geometry_cross_k_by_seed.csv"
        )
        .query("num_clusters == 4")
        .groupby("task", as_index=False)
        .agg(
            mean_pairwise_response=("mean_pairwise_response", "mean"),
            minimum_pairwise_response=("minimum_pairwise_response", "mean"),
        )
        .set_index("task")
    )
    winners = (
        pd.read_csv(
            repo
            / "experiments/control_matrix/assets/lewm_k4_geometry_screen/frozen_bures_gate_validation.csv"
        )
        .query("num_clusters == 4")
        .set_index("task")
        .point_estimate_winner
    )
    policies = []
    for metric in ("mean_pairwise_response", "minimum_pairwise_response"):
        vals = [float(lewm_k4.loc[task, metric]) for task in LAYER1_TASKS]
        labels = [winners.loc[task] for task in LAYER1_TASKS]
        direction, threshold = fit(vals, labels)
        pred = np.asarray(vals) > threshold if direction == "higher" else np.asarray(vals) < threshold
        hit = pred == (np.asarray(labels) == "regional")
        policies.append(
            dict(
                metric=metric,
                paper_method=11 if metric.startswith("mean") else 12,
                direction=direction,
                threshold=threshold,
                threshold_source=SOURCE,
                layer1_correct=int(hit.sum()),
                layer1_total=len(LAYER1_TASKS),
                layer1_accuracy=float(hit.mean()),
            )
        )
    policies = pd.DataFrame(policies)
    policies.to_csv(out / "layer1_response_pair_policies.csv", index=False)

    targets = pd.read_csv(
        repo / f"experiments/control_matrix/assets/{screen}/jacobian_fixed_k_validation.csv"
        if args.model == "subjepa"
        else repo
        / "experiments/control_matrix/assets/lewm_k4_geometry_screen/frozen_bures_gate_validation.csv"
    )[["task", "num_clusters", "point_estimate_winner", "delta_regional_minus_global_pp"]]
    details = cell.merge(targets, on=["task", "num_clusters"], validate="one_to_one")
    preds = []
    summary_rows = []
    for policy in policies.itertuples(index=False):
        score = details[policy.metric]
        regional = score > policy.threshold if policy.direction == "higher" else score < policy.threshold
        correct = regional.map({True: "regional", False: "global"}).eq(details.point_estimate_winner)
        for row, is_reg, hit in zip(details.itertuples(index=False), regional, correct):
            preds.append(
                dict(
                    metric=policy.metric,
                    paper_method=policy.paper_method,
                    task=row.task,
                    num_clusters=row.num_clusters,
                    score=getattr(row, policy.metric),
                    direction=policy.direction,
                    threshold=policy.threshold,
                    predicted_branch="regional" if is_reg else "global",
                    point_estimate_winner=row.point_estimate_winner,
                    correct=bool(hit),
                    delta_regional_minus_global_pp=row.delta_regional_minus_global_pp,
                )
            )
        n = int(len(details))
        summary_rows.append(
            dict(
                metric=policy.metric,
                paper_method=policy.paper_method,
                correct=int(correct.sum()),
                total=n,
                full_grid_accuracy=float(correct.mean()),
                direction=policy.direction,
                threshold=policy.threshold,
                threshold_source=policy.threshold_source,
            )
        )
    pd.DataFrame(preds).to_csv(out / "layer3_response_pair_predictions.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "layer3_response_pair_accuracy.csv", index=False)

    tworoom_k4 = cell[(cell.task == "tworoom") & (cell.num_clusters == 4)].iloc[0]
    manifest = dict(
        schema_version=1,
        analysis_name=f"{args.model} paper methods 11/12 from the same R_ij",
        repository_commit=subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        model=args.model,
        R_ij="pairwise_response_contrast",
        method_11="mean over region pairs within each partition seed, then mean over seeds",
        method_12="min over region pairs within each partition seed, then mean over seeds",
        layer1_calibration_tasks=list(LAYER1_TASKS),
        layer1_excluded_tasks=["pusht"],
        tworoom_k4_method_11=float(tworoom_k4.mean_pairwise_response),
        tworoom_k4_method_12=float(tworoom_k4.minimum_pairwise_response),
        note="affine_response_contrast_ratio is a different global statistic and must not be used as mean(R_ij).",
        files={},
    )
    for name in (
        "pairwise_response_summary.csv",
        "response_pair_aggregation.csv",
        "response_pair_aggregation_k4.csv",
        "layer1_response_pair_policies.csv",
        "layer3_response_pair_predictions.csv",
        "layer3_response_pair_accuracy.csv",
    ):
        path = out / name
        if path.is_file():
            manifest["files"][name] = sha256_file(path)
    (out / "response_pair_aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
