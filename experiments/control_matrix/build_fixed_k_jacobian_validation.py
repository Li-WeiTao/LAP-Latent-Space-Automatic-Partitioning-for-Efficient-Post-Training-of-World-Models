#!/usr/bin/env python3
"""Join fixed-K matrix summaries, Check-1, and Jacobian geometry into validation CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "tworoom"))

from experiments.control_matrix.analyze_fixed_k_response_geometry import (  # noqa: E402
    gate_manifest,
    matrix_root,
    resolve_data_file,
)
from experiments.control_matrix.estimate_task_spectral_threshold import (  # noqa: E402
    sha256_file,
)
from experiments.control_matrix.fit_partition import episode_ids, load_unique_latents  # noqa: E402
from lap.partition import SpectralDegeneracyGate, SpectralGateConfig  # noqa: E402
from latent_landmark_spectral import exact_cosine_knn_torch  # noqa: E402

PRACTICAL_BAND_PP = 0.5
CHECK1_THRESHOLD = 0.5
SEED_COLUMNS = {
    "training_seeds": "0,42,625",
    "partition_seeds": "0,1,2",
    "evaluation_seeds": "0,1,2,3,4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", choices=("lewm", "subjepa"), default="lewm")
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--clusters", default="2,3,4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args()


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x)


def load_matrix_means(repo: Path, model: str, task: str, k: int) -> tuple[float, float]:
    summary_path = matrix_root(repo, model, task, k) / "manifests/matrix_summary_long.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    by_method = {row["method"]: float(row["mean_percent"]) for row in payload["rows"]}
    return by_method["Global-FT50"], by_method["Spectral"]


def run_check1_gate(
    repo: Path,
    model: str,
    task: str,
    k: int,
    gpu_id: int,
    cpu_threads: int,
    cache_dir: Path,
) -> float:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{model}_{task}_k{k}_gate.json"
    if cache_file.is_file():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return float(payload["gate"]["retained_safety_fraction"])
    manifest = json.loads(gate_manifest(repo, model, task, k).read_text())
    cache_path = Path(manifest["cache_stats"]["cache"])
    data_path = resolve_data_file(Path(manifest["data_file"]))
    frameskip = int(manifest["cache_stats"]["frameskip"])
    raw, sample_ids, _ = load_unique_latents(cache_path, frameskip)
    groups = episode_ids(data_path, sample_ids, "auto")
    config = SpectralGateConfig(
        num_regions=k,
        num_landmarks=min(20_000, len(raw)),
        nominal_knn=30,
        perturb_knn=(27, 33),
        diagnostic_seeds=(0, 1, 2),
        deployment_seed=0,
        cpu_threads=cpu_threads,
    )
    search = lambda values, max_k: exact_cosine_knn_torch(
        values, max_k, gpu_id=gpu_id, query_chunk=2048
    )
    result = SpectralDegeneracyGate(config, neighbor_search=search).evaluate(
        raw, group_ids=groups
    )
    payload = {
        "schema": "lap_empirical_spectral_degeneracy_gate_v1",
        "task_name": task,
        "num_clusters": k,
        "gate": result.to_dict(),
    }
    cache_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return float(result.retained_safety_fraction)


def load_check1(
    repo: Path,
    model: str,
    task: str,
    k: int,
    gpu_id: int,
    cpu_threads: int,
    cache_dir: Path,
) -> tuple[float, bool]:
    manifest_path = gate_manifest(repo, model, task, k)
    manifest = json.loads(manifest_path.read_text())
    gate = manifest.get("method_metadata", {}).get("automatic_gate")
    if gate is None:
        if model != "subjepa":
            raise KeyError(f"missing automatic_gate in {manifest_path}")
        s_val = run_check1_gate(repo, model, task, k, gpu_id, cpu_threads, cache_dir)
    else:
        s_val = float(gate["retained_safety_fraction"])
    return s_val, bool(s_val >= CHECK1_THRESHOLD)


def winner_labels(delta_pp: float) -> tuple[str, str]:
    point = "regional" if delta_pp > 0.0 else "global"
    practical = "regional" if delta_pp > PRACTICAL_BAND_PP else "global"
    if delta_pp > PRACTICAL_BAND_PP:
        band = "regional"
    elif delta_pp < -PRACTICAL_BAND_PP:
        band = "global"
    else:
        band = "inconclusive"
    return point, band if band != "inconclusive" else practical


def aggregate_seed_metrics(df: pd.DataFrame, value_col: str) -> dict[str, float]:
    return {
        "mean": float(df[value_col].mean()),
        "min": float(df[value_col].min()),
        "max": float(df[value_col].max()),
    }


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    geometry = args.geometry_dir.resolve()
    gate_cache = geometry / "gate_cache"
    response = pd.read_csv(geometry / "response_geometry_cross_k_by_seed.csv")
    jacobian = pd.read_csv(geometry / "jacobian_fixed_k_seed_summary.csv")
    rows: list[dict[str, object]] = []
    for k in csv_ints(args.clusters):
        tasks = sorted(set(response["task"]).intersection(set(jacobian["task"])))
        for task in tasks:
            global_mean, regional_mean = load_matrix_means(repo, args.model, task, k)
            delta = regional_mean - global_mean
            point, practical = winner_labels(delta)
            check1_s, check1_pass = load_check1(
                repo, args.model, task, k, args.gpu_id, args.cpu_threads, gate_cache
            )
            resp = response[(response["task"] == task) & (response["num_clusters"] == k)]
            jac = jacobian[(jacobian["task"] == task) & (jacobian["num_clusters"] == k)]
            resp_stats = aggregate_seed_metrics(resp, "minimum_pairwise_response")
            uni_stats = aggregate_seed_metrics(resp, "pairwise_uniformity_min_over_mean")
            chord_stats = aggregate_seed_metrics(jac, "jacobian_subspace_chordal_distance")
            bures_stats = aggregate_seed_metrics(jac, "jacobian_bures_distance")
            rows.append(
                {
                    "task": task,
                    "num_clusters": k,
                    "global_mean_percent": global_mean,
                    "regional_mean_percent": regional_mean,
                    "delta_regional_minus_global_pp": delta,
                    "point_estimate_winner": point,
                    "practical_class_0p5pp_band": practical,
                    "check1_retained_safety_fraction": check1_s,
                    "check1_pass": check1_pass,
                    **SEED_COLUMNS,
                    "minimum_pairwise_response_mean": resp_stats["mean"],
                    "minimum_pairwise_response_seed_min": resp_stats["min"],
                    "minimum_pairwise_response_seed_max": resp_stats["max"],
                    "pairwise_uniformity_mean": uni_stats["mean"],
                    "pairwise_uniformity_seed_min": uni_stats["min"],
                    "pairwise_uniformity_seed_max": uni_stats["max"],
                    "accept_minimum_pairwise_response_rule": bool(
                        resp_stats["mean"] > 0.01
                    ),
                    "accept_pairwise_uniformity_rule": bool(uni_stats["mean"] > 0.7),
                    "jacobian_subspace_chordal_distance_mean": chord_stats["mean"],
                    "jacobian_bures_distance_mean": bures_stats["mean"],
                    "jacobian_subspace_chordal_distance_seed_min": chord_stats["min"],
                    "jacobian_bures_distance_seed_min": bures_stats["min"],
                    "jacobian_subspace_chordal_distance_seed_max": chord_stats["max"],
                    "jacobian_bures_distance_seed_max": bures_stats["max"],
                }
            )
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
