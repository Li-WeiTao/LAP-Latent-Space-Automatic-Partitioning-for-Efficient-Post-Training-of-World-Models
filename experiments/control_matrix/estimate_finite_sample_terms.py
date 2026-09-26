#!/usr/bin/env python3
"""Estimate epsilon_q and held-out affine-response error.

The split unit is an episode.  For every fixed candidate partition, affine
responses are fitted on training episodes and evaluated on disjoint held-out
episodes.  Score uncertainty is estimated with an episode-block bootstrap.

These are data-driven finite-sample estimates, not a certificate for every
term in epsilon(c).  The held-out response error contains both conditional-
mean approximation error and irreducible conditional noise, so it is reported
as a conservative empirical proxy for the affine-approximation term.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--model", choices=("lewm", "subjepa"), default="subjepa")
    p.add_argument("--tasks", default="tworoom,pusht,reacher,cube")
    p.add_argument("--clusters", default="2,3,4")
    p.add_argument("--partition-seeds", default="0,1,2")
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--transition-stride", type=int, default=1)
    p.add_argument("--ridge", type=float, default=1e-8)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--episode-blocks", type=int, default=20)
    p.add_argument("--bootstrap-replicates", type=int, default=500)
    p.add_argument("--split-seed", type=int, default=20260926)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--frozen-bures-threshold", type=float)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def csv_values(value: str, cast):
    return tuple(cast(item) for item in value.split(",") if item)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_seed(base: int, task: str, k: int = 0, partition_seed: int = 0) -> int:
    digest = hashlib.sha256(f"{task}:{k}:{partition_seed}".encode()).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**32)


def episode_ids_for_rows(data_path: Path, global_ids: np.ndarray) -> np.ndarray:
    with h5py.File(data_path, "r", swmr=True) as h:
        key = "episode_idx" if "episode_idx" in h else "ep_idx"
        return np.asarray(h[key][global_ids], dtype=np.int64)


def assign_episode_blocks(
    episode_ids: np.ndarray, num_blocks: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse = np.unique(episode_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique))
    block_by_unique = np.empty(len(unique), dtype=np.int64)
    block_by_unique[shuffled] = np.arange(len(unique), dtype=np.int64) % num_blocks
    return block_by_unique[inverse], unique, inverse


def sqrt_cov_from_sums(outer: np.ndarray, count: float) -> np.ndarray:
    cov = outer / max(float(count), 1.0)
    vals, vecs = np.linalg.eigh(cov)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T


def fit_region_stats(
    xtx: np.ndarray,
    xty: np.ndarray,
    counts: np.ndarray,
    ridge: float,
) -> list[np.ndarray]:
    p1 = xtx.shape[-1]
    penalty = np.eye(p1, dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefs = []
    for region in range(len(counts)):
        if counts[region] <= p1:
            raise ValueError(f"region {region} has only {counts[region]} rows")
        coefs.append(np.linalg.solve(xtx[region] + penalty, xty[region]))
    return coefs


def collect_block_stats(
    design: np.ndarray,
    response: np.ndarray,
    standardized_actions: np.ndarray,
    labels: np.ndarray,
    blocks: np.ndarray,
    num_blocks: int,
    k: int,
):
    p1, d = design.shape[1], response.shape[1]
    xtx = np.zeros((num_blocks, k, p1, p1), dtype=np.float64)
    xty = np.zeros((num_blocks, k, p1, d), dtype=np.float64)
    counts = np.zeros((num_blocks, k), dtype=np.int64)
    action_outer = np.zeros(
        (num_blocks, standardized_actions.shape[1], standardized_actions.shape[1]),
        dtype=np.float64,
    )
    action_count = np.zeros(num_blocks, dtype=np.int64)
    for block in range(num_blocks):
        bm = blocks == block
        aa = standardized_actions[bm]
        action_outer[block] = aa.T @ aa
        action_count[block] = int(bm.sum())
    group = blocks * k + labels
    for group_id in np.unique(group):
        block, region = divmod(int(group_id), k)
        mask = group == group_id
        xx = design[mask]
        yy = response[mask]
        xtx[block, region] = xx.T @ xx
        xty[block, region] = xx.T @ yy
        counts[block, region] = int(mask.sum())
    return xtx, xty, counts, action_outer, action_count


def q_from_aggregate(
    structural,
    xtx: np.ndarray,
    xty: np.ndarray,
    counts: np.ndarray,
    action_outer: np.ndarray,
    action_count: float,
    ridge: float,
):
    coefs = fit_region_stats(xtx, xty, counts, ridge)
    sqrt_cov = sqrt_cov_from_sums(action_outer, action_count)
    terms, _ = structural.pair_terms(coefs, counts, sqrt_cov)
    return terms, coefs


def cluster_upper_mean(
    row_values: np.ndarray,
    episode_inverse: np.ndarray,
    confidence: float,
) -> tuple[float, float, float]:
    """Cluster-robust normal upper limit for a row-weighted mean."""
    episode_count = np.bincount(episode_inverse).astype(np.float64)
    episode_sum = np.bincount(episode_inverse, weights=row_values).astype(np.float64)
    total_count = float(episode_count.sum())
    mean = float(episode_sum.sum() / total_count)
    active = episode_count > 0
    influence = episode_sum[active] - mean * episode_count[active]
    clusters = int(active.sum())
    variance = float(
        (clusters / max(clusters - 1, 1))
        * np.square(influence).sum()
        / (total_count * total_count)
    )
    # 1.6448536 is the one-sided 95% normal quantile.  The general value is
    # obtained without scipy to keep the audit script dependency-light.
    quantiles = {0.90: 1.2815516, 0.95: 1.6448536, 0.975: 1.9599640, 0.99: 2.3263479}
    z = quantiles.get(round(confidence, 3))
    if z is None:
        raise ValueError("confidence must be one of 0.90, 0.95, 0.975, 0.99")
    se = float(np.sqrt(max(variance, 0.0)))
    return mean, se, max(mean + z * se, 0.0)


def bootstrap_q(
    structural,
    xtx_blocks: np.ndarray,
    xty_blocks: np.ndarray,
    count_blocks: np.ndarray,
    action_outer_blocks: np.ndarray,
    action_count_blocks: np.ndarray,
    ridge: float,
    replicates: int,
    seed: int,
    q_full: float,
    confidence: float,
):
    num_blocks = len(action_count_blocks)
    rng = np.random.default_rng(seed)
    samples = []
    failures = 0
    for _ in range(replicates):
        weights = rng.multinomial(num_blocks, np.full(num_blocks, 1.0 / num_blocks))
        xtx = np.tensordot(weights, xtx_blocks, axes=(0, 0))
        xty = np.tensordot(weights, xty_blocks, axes=(0, 0))
        counts = np.tensordot(weights, count_blocks, axes=(0, 0))
        action_outer = np.tensordot(weights, action_outer_blocks, axes=(0, 0))
        action_count = float(weights @ action_count_blocks)
        try:
            terms, _ = q_from_aggregate(
                structural, xtx, xty, counts, action_outer, action_count, ridge
            )
            samples.append(float(terms["q_min"]))
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
    if len(samples) < max(10, replicates // 2):
        raise RuntimeError(f"only {len(samples)} valid bootstrap replicates")
    values = np.asarray(samples, dtype=np.float64)
    alpha = 1.0 - confidence
    return dict(
        q_bootstrap_mean=float(values.mean()),
        q_ci_low=float(np.quantile(values, alpha / 2.0)),
        q_ci_high=float(np.quantile(values, 1.0 - alpha / 2.0)),
        epsilon_q_95=float(np.quantile(np.abs(values - q_full), confidence)),
        epsilon_score_95=float(2.0 * np.quantile(np.abs(values - q_full), confidence)),
        bootstrap_valid=int(len(values)),
        bootstrap_failures=int(failures),
    )


def main() -> None:
    args = parse_args()
    if args.episode_blocks % args.folds != 0:
        raise ValueError("episode-blocks must be divisible by folds")
    repo = args.repo.resolve()
    geometry = load_module(
        repo / "experiments/control_matrix/analyze_fixed_k_response_geometry.py",
        "lap_fixed_k_geometry",
    )
    structural = load_module(
        repo / "experiments/control_matrix/audit_structural_assumption_terms.py",
        "lap_structural_terms",
    )
    tasks = csv_values(args.tasks, str)
    clusters = csv_values(args.clusters, int)
    partition_seeds = csv_values(args.partition_seeds, int)
    config_rows = []
    region_rows = []

    for task in tasks:
        manifest_path = geometry.gate_manifest(
            repo, args.model, task, 4 if 4 in clusters else clusters[0]
        )
        manifest = json.loads(manifest_path.read_text())
        cache_path = Path(manifest["cache_stats"]["cache"])
        data_path = geometry.resolve_data_file(Path(manifest["data_file"]))
        x, ids = geometry.load_unique(cache_path, args.frameskip)
        left, right, actions = geometry.transition_rows(
            ids, data_path, args.transition_stride
        )
        episode_ids = episode_ids_for_rows(data_path, ids[left])
        blocks, unique_episodes, episode_inverse = assign_episode_blocks(
            episode_ids, args.episode_blocks, task_seed(args.split_seed, task)
        )
        action_mean = actions.mean(axis=0)
        action_scale = actions.std(axis=0) + 1e-12
        standardized_actions = (actions - action_mean) / action_scale
        design = np.column_stack(
            [np.ones(len(actions), dtype=np.float64), standardized_actions]
        )
        response = (
            x[right].astype(np.float64, copy=False)
            - x[left].astype(np.float64, copy=False)
        )

        for k in clusters:
            labels_by_seed = {
                seed: geometry.load_labels(
                    geometry.label_path(repo, args.model, task, k, seed), ids
                )[left]
                for seed in partition_seeds
            }
            for partition_seed, labels in labels_by_seed.items():
                stats = collect_block_stats(
                    design,
                    response,
                    standardized_actions,
                    labels,
                    blocks,
                    args.episode_blocks,
                    k,
                )
                xtx_b, xty_b, count_b, action_outer_b, action_count_b = stats
                full_terms, _ = q_from_aggregate(
                    structural,
                    xtx_b.sum(axis=0),
                    xty_b.sum(axis=0),
                    count_b.sum(axis=0),
                    action_outer_b.sum(axis=0),
                    float(action_count_b.sum()),
                    args.ridge,
                )
                q_boot = bootstrap_q(
                    structural,
                    xtx_b,
                    xty_b,
                    count_b,
                    action_outer_b,
                    action_count_b,
                    args.ridge,
                    args.bootstrap_replicates,
                    task_seed(args.split_seed + 1, task, k, partition_seed),
                    float(full_terms["q_min"]),
                    args.confidence,
                )

                row_squared_error = np.empty(len(left), dtype=np.float64)
                fold_q = []
                for fold in range(args.folds):
                    held_blocks = np.arange(args.episode_blocks) % args.folds == fold
                    train_blocks = ~held_blocks
                    train_terms, coefs = q_from_aggregate(
                        structural,
                        xtx_b[train_blocks].sum(axis=0),
                        xty_b[train_blocks].sum(axis=0),
                        count_b[train_blocks].sum(axis=0),
                        action_outer_b[train_blocks].sum(axis=0),
                        float(action_count_b[train_blocks].sum()),
                        args.ridge,
                    )
                    fold_q.append(float(train_terms["q_min"]))
                    test = held_blocks[blocks]
                    for region in range(k):
                        index = np.flatnonzero(test & (labels == region))
                        if len(index) == 0:
                            raise ValueError(
                                f"empty held-out region: {task} K={k} seed={partition_seed} "
                                f"fold={fold} region={region}"
                            )
                        residual = response[index] - design[index] @ coefs[region]
                        row_squared_error[index] = np.square(residual).sum(axis=1)

                mse, mse_se, mse_upper = cluster_upper_mean(
                    row_squared_error, episode_inverse, args.confidence
                )
                response_scale = float(full_terms["response_scale"])
                config_rows.append(
                    dict(
                        model=args.model,
                        task=task,
                        num_clusters=k,
                        partition_seed=partition_seed,
                        transition_count=len(left),
                        episode_count=len(unique_episodes),
                        q_full=float(full_terms["q_min"]),
                        fold_q_min=float(np.min(fold_q)),
                        fold_q_max=float(np.max(fold_q)),
                        heldout_affine_mse=mse,
                        heldout_affine_mse_cluster_se=mse_se,
                        heldout_affine_mse_upper95=mse_upper,
                        heldout_affine_l2=float(np.sqrt(mse)),
                        heldout_affine_l2_upper95=float(np.sqrt(mse_upper)),
                        response_scale=response_scale,
                        normalized_affine_l2=float(np.sqrt(mse / response_scale)),
                        normalized_affine_l2_upper95=float(
                            np.sqrt(mse_upper / response_scale)
                        ),
                        **q_boot,
                    )
                )
                for region in range(k):
                    mask = labels == region
                    region_episode_ids, region_episode_inverse = np.unique(
                        episode_ids[mask], return_inverse=True
                    )
                    r_mse, r_se, r_upper = cluster_upper_mean(
                        row_squared_error[mask],
                        region_episode_inverse,
                        args.confidence,
                    )
                    region_rows.append(
                        dict(
                            model=args.model,
                            task=task,
                            num_clusters=k,
                            partition_seed=partition_seed,
                            region=region,
                            transition_count=int(mask.sum()),
                            episode_count=len(region_episode_ids),
                            heldout_affine_mse=r_mse,
                            heldout_affine_mse_cluster_se=r_se,
                            heldout_affine_mse_upper95=r_upper,
                            heldout_affine_l2=float(np.sqrt(r_mse)),
                            heldout_affine_l2_upper95=float(np.sqrt(r_upper)),
                        )
                    )
                print(
                    f"{task} K={k} partition_seed={partition_seed}: "
                    f"q={full_terms['q_min']:.6f}, "
                    f"epsilon_q95={q_boot['epsilon_q_95']:.6f}, "
                    f"heldout_norm_upper95={np.sqrt(mse_upper / response_scale):.6f}",
                    flush=True,
                )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = pd.DataFrame(config_rows).sort_values(
        ["task", "num_clusters", "partition_seed"]
    )
    regions = pd.DataFrame(region_rows).sort_values(
        ["task", "num_clusters", "partition_seed", "region"]
    )
    config.to_csv(output / "finite_sample_terms_by_partition.csv", index=False)
    regions.to_csv(output / "heldout_affine_error_by_region.csv", index=False)
    summary = (
        config.groupby(["model", "task", "num_clusters"], as_index=False)
        .agg(
            partition_seed_count=("partition_seed", "nunique"),
            q_mean=("q_full", "mean"),
            epsilon_q95_max=("epsilon_q_95", "max"),
            epsilon_score95_max=("epsilon_score_95", "max"),
            heldout_affine_l2_mean=("heldout_affine_l2", "mean"),
            heldout_affine_l2_upper95_max=("heldout_affine_l2_upper95", "max"),
            normalized_affine_l2_mean=("normalized_affine_l2", "mean"),
            normalized_affine_l2_upper95_max=(
                "normalized_affine_l2_upper95",
                "max",
            ),
            bootstrap_failures=("bootstrap_failures", "sum"),
        )
        .sort_values(["task", "num_clusters"])
    )
    frozen_bures_threshold = args.frozen_bures_threshold
    if frozen_bures_threshold is None:
        policy_path = (
            repo
            / "experiments/control_matrix/assets/lewm_k4_geometry_screen/"
            "frozen_bures_gate_policy.json"
        )
        if policy_path.is_file():
            frozen_bures_threshold = float(
                json.loads(policy_path.read_text())["frozen_bures_threshold"]
            )
    if frozen_bures_threshold is not None:
        q_threshold = frozen_bures_threshold / 2.0
        summary["frozen_bures_threshold"] = frozen_bures_threshold
        summary["q_threshold"] = q_threshold
        summary["absolute_q_margin"] = (summary["q_mean"] - q_threshold).abs()
        summary["q_margin_after_epsilon_q"] = (
            summary["absolute_q_margin"] - summary["epsilon_q95_max"]
        )
        summary["epsilon_q_resolves_threshold_side"] = (
            summary["q_margin_after_epsilon_q"] > 0.0
        )
    summary.to_csv(output / "finite_sample_terms_summary.csv", index=False)
    metadata = {
        "schema_version": 1,
        "status": "empirical_finite_sample_estimates_not_complete_epsilon_c_certificate",
        "repo_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": args.model,
        "tasks": list(tasks),
        "clusters": list(clusters),
        "partition_seeds": list(partition_seeds),
        "ridge": args.ridge,
        "folds": args.folds,
        "episode_blocks": args.episode_blocks,
        "bootstrap_replicates": args.bootstrap_replicates,
        "split_seed": args.split_seed,
        "confidence": args.confidence,
        "frozen_bures_threshold": frozen_bures_threshold,
        "split_unit": "episode",
        "epsilon_q_method": "episode-block nonparametric bootstrap; 95th percentile absolute deviation from full-data q",
        "affine_method": "episode-disjoint cross-fitting; one-sided cluster-robust 95% upper limit",
        "interpretation": {
            "epsilon_q_95": "empirical sampling-error estimate on q scale",
            "epsilon_score_95": "twice epsilon_q_95, on Bures-score scale",
            "heldout_affine_l2_upper95": "conservative empirical upper proxy containing approximation error plus conditional noise",
            "normalized_affine_l2_upper95": "heldout_affine_l2_upper95 divided by sqrt(response_scale)",
        },
        "not_claimed": [
            "deterministic high-probability theorem certificate",
            "separation of conditional noise from conditional-mean approximation error",
            "complete epsilon(c)",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result_lines = [
        "| Task | K | mean q | max epsilon_q (95%) | held-out affine L2 upper (95%) | normalized upper |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        result_lines.append(
            f"| {row.task} | {row.num_clusters} | {row.q_mean:.6f} | "
            f"{row.epsilon_q95_max:.6f} | {row.heldout_affine_l2_upper95_max:.6f} | "
            f"{row.normalized_affine_l2_upper95_max:.6f} |"
        )
    threshold_note = ""
    if frozen_bures_threshold is not None:
        resolved = int(summary["epsilon_q_resolves_threshold_side"].sum())
        unresolved = summary.loc[
            ~summary["epsilon_q_resolves_threshold_side"], ["task", "num_clusters"]
        ]
        unresolved_text = ", ".join(
            f"{row.task} K={row.num_clusters}" for row in unresolved.itertuples()
        ) or "none"
        threshold_note = (
            f"\nUsing the frozen Bures threshold {frozen_bures_threshold:.15f} "
            f"(q threshold {frozen_bures_threshold / 2.0:.15f}), the conservative "
            f"comparison `absolute q margin > max epsilon_q` resolves the threshold "
            f"side in {resolved} of {len(summary)} configurations. Unresolved: "
            f"{unresolved_text}. This isolates score-estimation uncertainty only.\n"
        )
    readme = """# Sub-JEPA finite-sample theory diagnostics

This directory estimates two quantities conditional on each already-fixed
candidate partition.

1. `epsilon_q_95` is the 95th percentile absolute deviation of the weakest-pair
   score `q` under a nonparametric bootstrap of episode blocks.  The associated
   `epsilon_score_95` is on the paper's Bures-score scale and equals twice this
   quantity.
2. `heldout_affine_l2_upper95` is a one-sided, episode-cluster-robust upper
   limit for the response prediction residual of an affine model fitted on
   disjoint episodes.  Region-level values are in
   `heldout_affine_error_by_region.csv`.

The affine residual is conservative for conditional-mean approximation under
the usual zero-mean regression-noise decomposition, because it also contains
conditional noise.  It is not a complete certificate for `epsilon(c)`, and no
unjustified numerical pass/fail threshold is applied.  The partition itself is
treated as fixed: the analysis evaluates score estimation and affine fitting,
not uncertainty from re-estimating the partition.

## Limitation

当前实验制品无法给出完整且可信的 `epsilon(c)` 数值上界。这里报告的结果只覆盖当前数据能够识别的组成项，不能被解释为完整的理论条件证书。

## Aggregate results

Values below average point estimates across the three predeclared partition
seeds; uncertainty columns use the maximum estimate across those seeds.

""" + "\n".join(result_lines) + "\n" + threshold_note
    (output / "README.md").write_text(readme, encoding="utf-8")
    print("\nSummary\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
