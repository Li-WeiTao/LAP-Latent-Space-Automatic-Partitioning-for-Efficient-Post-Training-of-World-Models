#!/usr/bin/env python3
"""Evaluate the 22 frozen LeWM criteria for one K-way partition method.

The criterion implementations and frozen policies are reused from
``analyze_layer2_metric_benchmark.py``.  Nineteen criteria are recomputed from
the requested partition labels.  The three task-graph-only criteria
(``eigengap_after_k`` and the two legacy spectral checks) retain their frozen
K=4 task-graph values and are explicitly marked as such in every output.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.control_matrix import analyze_layer2_metric_benchmark as base


TASK_GRAPH_ONLY = {
    "eigengap_after_k",
    "check1_retained_safety_fraction",
    "check2_prominence_ratio",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--partition-method", required=True)
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--tasks", default=",".join(base.TASKS))
    parser.add_argument("--partition-seeds", default="0,1,2")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--chunk", type=int, default=100000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--policies-csv", type=Path)
    parser.add_argument("--outcomes-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--raw-scores-csv", type=Path)
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_root(repo, task, k, method, seed):
    return repo / f"experiments/{task}/matrix_k{k}/partitions/{method}/seed{seed}"


def require_inputs(repo, tasks, k, method, seeds, outcomes):
    missing = []
    for task in tasks:
        for seed in seeds:
            root = method_root(repo, task, k, method, seed)
            for rel in ("manifest.json", "cluster_labels.npz"):
                if not (root / rel).is_file():
                    missing.append(str(root / rel))
    if not outcomes.is_file():
        missing.append(str(outcomes))
    if missing:
        raise SystemExit("missing required inputs:\n" + "\n".join(missing))


def compute_raw(args, repo, tasks, seeds, audit_dir):
    original_root = base.root
    original_meta = base.meta

    def selected_root(_repo, task, k, seed):
        return method_root(_repo, task, k, args.partition_method, seed)

    def selected_meta(root):
        manifest = json.loads((root / "manifest.json").read_text())
        task = manifest["dataset"]
        k = int(manifest["num_clusters"])
        spectral_root = original_root(repo, task, k, 0)
        _, eigengap = original_meta(spectral_root)
        return np.asarray(manifest["cluster_fractions"], dtype=float), eigengap

    base.root = selected_root
    base.meta = selected_meta
    try:
        faiss.omp_set_num_threads(args.cpu_threads)
        rows = []
        worker_args = argparse.Namespace(
            frameskip=args.frameskip,
            ridge=args.ridge,
            chunk=args.chunk,
            audit_dir=audit_dir,
        )
        for task in tasks:
            rows.extend(
                base.task_rows(
                    repo,
                    task,
                    (args.num_clusters,),
                    seeds,
                    worker_args,
                    source_manifest=lambda selected_repo, selected_task: method_root(
                        selected_repo,
                        selected_task,
                        args.num_clusters,
                        args.partition_method,
                        0,
                    )
                    / "manifest.json",
                )
            )
        raw = pd.DataFrame(rows)
    finally:
        base.root = original_root
        base.meta = original_meta
    raw.insert(1, "partition_method", args.partition_method)
    return raw


def score(raw, policies, outcomes, method, k):
    required = set(base.METRICS)
    if set(policies.metric) != required:
        raise SystemExit("frozen policy table does not contain exactly the 22 criteria")
    outcomes = outcomes.copy()
    outcomes["task"] = outcomes["task"].str.lower()
    outcomes["point_estimate_winner"] = outcomes["point_estimate_winner"].str.lower()
    if not outcomes["num_clusters"].eq(k).all():
        raise SystemExit("outcome table contains an unexpected num_clusters value")
    if "method" in outcomes and not outcomes["method"].eq(method).all():
        raise SystemExit("outcome table contains an unexpected partition method")

    means = raw.groupby(["task", "num_clusters"], as_index=False)[list(base.METRICS)].mean(numeric_only=True)
    merged = outcomes.merge(means, on=["task", "num_clusters"], validate="one_to_one")
    prediction_rows = []
    summary_rows = []
    for policy in policies.itertuples(index=False):
        values = merged[policy.metric]
        available = values.notna()
        regional = values.gt(policy.threshold) if policy.direction == "higher" else values.lt(policy.threshold)
        predicted = regional.map({True: "regional", False: "global"}).where(available, "abstain")
        correct = predicted.eq(merged["point_estimate_winner"]) & available
        scope = "task_graph_only" if policy.metric in TASK_GRAPH_ONLY else "partition_specific"
        source = (
            "frozen K=4 task-level spectral graph; not recomputed from partition labels"
            if scope == "task_graph_only"
            else f"recomputed from {method} K={k} partition labels"
        )
        for idx, row in merged.iterrows():
            prediction_rows.append(
                dict(
                    partition_method=method,
                    num_clusters=k,
                    metric=policy.metric,
                    criterion_scope=scope,
                    criterion_source=source,
                    task=row.task,
                    score=row[policy.metric],
                    direction=policy.direction,
                    threshold=policy.threshold,
                    threshold_source=policy.threshold_source,
                    predicted_branch=predicted.iloc[idx],
                    point_estimate_winner=row.point_estimate_winner,
                    correct=bool(correct.iloc[idx]),
                    delta_regional_minus_global_pp=row.delta_regional_minus_global_pp,
                )
            )
        covered = int(available.sum())
        hits = int(correct.sum())
        summary_rows.append(
            dict(
                partition_method=method,
                num_clusters=k,
                metric=policy.metric,
                criterion_scope=scope,
                direction=policy.direction,
                threshold=policy.threshold,
                threshold_source=policy.threshold_source,
                correct=hits,
                covered=covered,
                total=len(merged),
                accuracy=hits / covered if covered else np.nan,
                full_grid_accuracy=hits / len(merged),
            )
        )
    return means, pd.DataFrame(prediction_rows), pd.DataFrame(summary_rows)


def main():
    args = parse_args()
    repo = args.repo.resolve()
    tasks = tuple(filter(None, args.tasks.split(",")))
    seeds = base.ints(args.partition_seeds)
    assets = repo / "experiments/control_matrix/assets"
    audit_dir = (args.audit_dir or assets / "lewm_k4_geometry_screen/audit_cache").resolve()
    policies_path = (
        args.policies_csv or assets / "lewm_layer2_22_criteria/layer1_frozen_policies.csv"
    ).resolve()
    outcomes_path = (
        args.outcomes_csv or assets / f"lewm_k{args.num_clusters}_{args.partition_method}/long_comparison.csv"
    ).resolve()
    output_dir = (
        args.output_dir or assets / f"lewm_k{args.num_clusters}_{args.partition_method}/criteria22"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require_inputs(repo, tasks, args.num_clusters, args.partition_method, seeds, outcomes_path)
    for task in tasks:
        if not (audit_dir / f"{task}.npz").is_file():
            raise SystemExit(f"missing fixed audit cache: {audit_dir / f'{task}.npz'}")
    if not policies_path.is_file():
        raise SystemExit(f"missing frozen policy table: {policies_path}")

    if args.raw_scores_csv:
        raw = base.normalize_raw_scores(base.read_csv(args.raw_scores_csv.resolve()))
    else:
        raw = compute_raw(args, repo, tasks, seeds, audit_dir)
        raw.to_csv(output_dir / "criterion_scores_by_seed.csv", index=False)

    policies = base.read_csv(policies_path)
    outcomes = base.read_csv(outcomes_path)
    means, predictions, summary = score(
        raw, policies, outcomes, args.partition_method, args.num_clusters
    )
    means.to_csv(output_dir / "criterion_scores_by_task.csv", index=False)
    predictions.to_csv(output_dir / "criterion_predictions.csv", index=False)
    summary.to_csv(output_dir / "criterion_selection_accuracy.csv", index=False)

    # Contract check: the recomputed frozen Bures implementation must reproduce
    # the Bures value already used by the completed partition-method experiment.
    comparison = outcomes[["task", "bures_mean"]].merge(
        means[["task", "jacobian_bures_distance"]], on="task", validate="one_to_one"
    )
    comparison["absolute_error"] = (
        comparison["bures_mean"] - comparison["jacobian_bures_distance"]
    ).abs()
    comparison.to_csv(output_dir / "bures_reproduction_check.csv", index=False)
    max_error = float(comparison["absolute_error"].max())
    if max_error > 1e-9:
        raise SystemExit(f"Bures reproduction failed: max absolute error {max_error:.3g}")

    provenance = pd.DataFrame(
        [
            dict(
                metric=metric,
                criterion_scope=("task_graph_only" if metric in TASK_GRAPH_ONLY else "partition_specific"),
                recomputed_from_partition_labels=(metric not in TASK_GRAPH_ONLY),
                source=(
                    "frozen K=4 task-level spectral graph"
                    if metric in TASK_GRAPH_ONLY
                    else f"{args.partition_method} K={args.num_clusters} partition labels"
                ),
            )
            for metric in base.METRICS
        ]
    )
    provenance.to_csv(output_dir / "criterion_provenance.csv", index=False)

    manifest = dict(
        schema_version=1,
        analysis_name="LeWM frozen 22-criterion partition-method transfer",
        repository_commit=subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        partition_method=args.partition_method,
        num_clusters=args.num_clusters,
        tasks=list(tasks),
        partition_seeds=list(seeds),
        criterion_count=len(base.METRICS),
        partition_specific_criterion_count=len(base.METRICS) - len(TASK_GRAPH_ONLY),
        task_graph_only_criteria=sorted(TASK_GRAPH_ONLY),
        threshold_refitted=False,
        policies_csv=str(policies_path),
        policies_sha256=sha256(policies_path),
        outcomes_csv=str(outcomes_path),
        outcomes_sha256=sha256(outcomes_path),
        audit_dir=str(audit_dir),
        ridge=args.ridge,
        bures_reproduction_max_absolute_error=max_error,
        files={},
    )
    for path in sorted(output_dir.glob("*.csv")):
        manifest["files"][path.name] = sha256(path)
    (output_dir / "criteria22_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.sort_values(["full_grid_accuracy", "metric"], ascending=[False, True]).to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
