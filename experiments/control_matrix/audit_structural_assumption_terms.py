#!/usr/bin/env python3
"""Estimate the directly identifiable structural terms in Appendix G.

The outputs are plug-in estimates computed from the fitted affine response
geometry.  They are not finite-sample confidence bounds and intentionally do
not estimate the neural-model or finite-horizon planning remainders.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", choices=("lewm", "subjepa"), default="subjepa")
    parser.add_argument("--tasks", default="tworoom,pusht,reacher,cube")
    parser.add_argument("--clusters", default="2,3,4")
    parser.add_argument("--partition-seeds", default="0,1,2")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--transition-stride", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--chunk", type=int, default=200_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def csv_values(value: str, cast):
    return tuple(cast(item) for item in value.split(",") if item)


def load_geometry_module(repo: Path):
    source = repo / "experiments/control_matrix/analyze_fixed_k_response_geometry.py"
    spec = importlib.util.spec_from_file_location("lap_fixed_k_geometry", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair_terms(coefs, counts, sqrt_cov):
    total = float(sum(counts))
    probabilities = np.asarray(counts, dtype=np.float64) / total
    factors = [(sqrt_cov @ coef[1:]).T for coef in coefs]
    intercepts = [coef[0] for coef in coefs]
    energies = [float(np.square(factor).sum()) for factor in factors]

    raw_pairs = []
    scale = 0.0
    for i in range(len(coefs)):
        for j in range(i + 1, len(coefs)):
            energy = energies[i] + energies[j]
            product_weight = probabilities[i] * probabilities[j]
            nuclear = float(
                np.linalg.svd(factors[i].T @ factors[j], compute_uv=False).sum()
            )
            bures_squared = max(energy - 2.0 * nuclear, 0.0)
            q_ij = bures_squared / (2.0 * energy)
            intercept_squared = float(np.square(intercepts[i] - intercepts[j]).sum())
            fixed_coordinate_squared = float(np.square(factors[i] - factors[j]).sum())
            orientation_gap = max(fixed_coordinate_squared - bures_squared, 0.0)
            scale += 2.0 * product_weight * energy
            raw_pairs.append(
                (i, j, product_weight, energy, q_ij, intercept_squared, orientation_gap)
            )

    if scale <= 0.0:
        raise ValueError("non-positive response scale")
    q_min = min(item[4] for item in raw_pairs)
    q_max = max(item[4] for item in raw_pairs)
    r_pair = 0.0
    r_b = 0.0
    r_orient = 0.0
    identity_lhs = 0.0
    pair_rows = []
    for i, j, product_weight, energy, q_ij, intercept_squared, orientation_gap in raw_pairs:
        weight = 2.0 * product_weight * energy / scale
        r_pair += weight * (q_ij - q_min)
        r_b += product_weight * intercept_squared / scale
        r_orient += product_weight * orientation_gap / scale
        identity_lhs += (
            product_weight
            * (intercept_squared + 2.0 * energy * q_ij + orientation_gap)
            / scale
        )
        pair_rows.append(
            dict(
                region_left=i,
                region_right=j,
                q_ij=q_ij,
                pair_weight=weight,
                intercept_contribution=product_weight * intercept_squared / scale,
                orientation_contribution=product_weight * orientation_gap / scale,
            )
        )
    return dict(
        response_scale=scale,
        q_min=q_min,
        q_max=q_max,
        r_b=r_b,
        r_orient=r_orient,
        r_pair=r_pair,
        min_region_fraction=float(probabilities.min()),
        max_region_fraction=float(probabilities.max()),
        identity_lhs=identity_lhs,
        identity_rhs=q_min + r_b + r_orient + r_pair,
    ), pair_rows


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    geometry = load_geometry_module(repo)
    tasks = csv_values(args.tasks, str)
    clusters = csv_values(args.clusters, int)
    seeds = csv_values(args.partition_seeds, int)
    rows = []
    pair_rows = []

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
        for k in clusters:
            labels = {
                seed: geometry.load_labels(
                    geometry.label_path(repo, args.model, task, k, seed), ids
                )
                for seed in seeds
            }
            fitted, _, sqrt_cov, _, _ = geometry.sufficient_stats(
                x, left, right, actions, labels, k, args.ridge, args.chunk
            )
            for seed in seeds:
                coefs = [item[0] for item in fitted[seed]]
                counts = [item[1] for item in fitted[seed]]
                terms, pairs = pair_terms(coefs, counts, sqrt_cov)
                rows.append(
                    dict(
                        task=task,
                        model=args.model,
                        num_clusters=k,
                        partition_seed=seed,
                        transition_count=len(left),
                        **terms,
                    )
                )
                for pair in pairs:
                    pair_rows.append(
                        dict(
                            task=task,
                            model=args.model,
                            num_clusters=k,
                            partition_seed=seed,
                            **pair,
                        )
                    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(
        ["task", "num_clusters", "partition_seed"]
    )
    frame.to_csv(output_dir / "structural_terms_by_seed.csv", index=False)
    pd.DataFrame(pair_rows).sort_values(
        ["task", "num_clusters", "partition_seed", "region_left", "region_right"]
    ).to_csv(output_dir / "structural_pair_terms.csv", index=False)
    summary = (
        frame.groupby(["task", "model", "num_clusters"], as_index=False)
        .agg(
            partition_seed_count=("partition_seed", "nunique"),
            q_min_mean=("q_min", "mean"),
            q_min_min=("q_min", "min"),
            q_min_max=("q_min", "max"),
            observed_epsilon_b=("r_b", "max"),
            observed_epsilon_orient=("r_orient", "max"),
            observed_epsilon_pair=("r_pair", "max"),
            min_region_fraction=("min_region_fraction", "min"),
            max_identity_error=("identity_rhs", "size"),
        )
        .sort_values(["task", "num_clusters"])
    )
    errors = (frame["identity_lhs"] - frame["identity_rhs"]).abs()
    error_map = errors.groupby(
        [frame["task"], frame["model"], frame["num_clusters"]]
    ).max()
    summary["max_identity_error"] = [
        error_map.loc[(row.task, row.model, row.num_clusters)]
        for row in summary.itertuples()
    ]
    summary.to_csv(output_dir / "structural_terms_summary.csv", index=False)
    metadata = {
        "schema_version": 1,
        "status": "plug_in_estimates_not_confidence_bounds",
        "repo_commit": __import__("subprocess").check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": args.model,
        "tasks": list(tasks),
        "clusters": list(clusters),
        "partition_seeds": list(seeds),
        "ridge": args.ridge,
        "frameskip": args.frameskip,
        "transition_stride": args.transition_stride,
        "measured_terms": ["r_b", "r_orient", "r_pair"],
        "not_measured": [
            "r_2",
            "kappa_c_minus_kappa",
            "r_extra",
            "epsilon_plan_H",
            "delta_H_over_alpha_H",
            "epsilon_q",
            "epsilon_cal",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
