#!/usr/bin/env python3
"""Run LAP's paper-aligned spectral-degeneracy gate without fine-tuning.

This diagnostic is useful for auditing a latent cache.  The production path is
``fit_partition.py --method auto``, which runs the same gate and immediately
selects either the spectral partition or the one-region Global-FT fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TWOROOM_DIR = PROJECT_ROOT / "experiments" / "tworoom"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TWOROOM_DIR))

from experiments.control_matrix.fit_partition import (  # noqa: E402
    comma_separated_ints,
    episode_ids,
    load_unique_latents,
)
from lap.partition import SpectralDegeneracyGate, SpectralGateConfig  # noqa: E402
from latent_landmark_spectral import exact_cosine_knn_torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--source-format",
        choices=("unified_cache", "legacy_region_caches"),
        default="unified_cache",
    )
    parser.add_argument("--latent-cache", type=Path)
    parser.add_argument("--embed-dir", type=Path)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--episode-key", default="auto")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument(
        "--diagnostic-seeds",
        type=comma_separated_ints,
        default=comma_separated_ints("0,1,2"),
    )
    parser.add_argument("--deployment-seed", type=int, default=0)
    parser.add_argument("--num-landmarks", type=int, default=20_000)
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--nominal-knn", type=int, default=30)
    parser.add_argument(
        "--perturb-knn",
        type=comma_separated_ints,
        default=comma_separated_ints("27,33"),
    )
    parser.add_argument("--perturbation-multiplier", type=float, default=2.0)
    parser.add_argument("--retention-threshold", type=float, default=0.5)
    parser.add_argument("--background-gap-count", type=int, default=10)
    parser.add_argument("--background-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    data_path = args.data_file.resolve(strict=True)
    cache_path: Path | None = None
    embed_dir: Path | None = None
    if args.source_format == "unified_cache":
        if args.latent_cache is None:
            raise ValueError("--latent-cache is required for unified_cache")
        cache_path = args.latent_cache.resolve(strict=True)
        raw, sample_ids, cache_stats = load_unique_latents(
            cache_path, args.frameskip
        )
        groups = episode_ids(data_path, sample_ids, args.episode_key)
    else:
        if args.embed_dir is None:
            raise ValueError("--embed-dir is required for legacy_region_caches")
        embed_dir = args.embed_dir.resolve(strict=True)
        from geometry_latent_svm_rooms3 import episode_ids_at_indices
        from latent_cluster_common import load_all_train_dedup_latent_vectors

        raw, sample_ids, cache_stats = load_all_train_dedup_latent_vectors(
            embed_dir, frameskip=args.frameskip
        )
        groups = episode_ids_at_indices(data_path, sample_ids)
    count = min(args.num_landmarks, len(raw))
    config = SpectralGateConfig(
        num_regions=args.num_clusters,
        num_landmarks=count,
        nominal_knn=args.nominal_knn,
        perturb_knn=args.perturb_knn,
        diagnostic_seeds=args.diagnostic_seeds,
        deployment_seed=args.deployment_seed,
        perturbation_multiplier=args.perturbation_multiplier,
        retention_threshold=args.retention_threshold,
        background_gap_count=args.background_gap_count,
        background_mad_multiplier=args.background_mad_multiplier,
        epsilon=args.epsilon,
        cpu_threads=args.cpu_threads,
    )
    search = lambda values, max_k: exact_cosine_knn_torch(
        values,
        max_k,
        gpu_id=args.gpu_id,
        query_chunk=args.query_chunk,
    )
    result = SpectralDegeneracyGate(
        config, neighbor_search=search
    ).evaluate(raw, group_ids=groups)
    output = {
        "schema": "lap_empirical_spectral_degeneracy_gate_v1",
        "task_name": args.task_name,
        "input": {
            "source_format": args.source_format,
            "latent_cache": None if cache_path is None else str(cache_path),
            "latent_cache_sha256": (
                None if cache_path is None else sha256_file(cache_path)
            ),
            "embed_dir": None if embed_dir is None else str(embed_dir),
            "data_file": str(data_path),
            "data_file_sha256": sha256_file(data_path),
            "frameskip": args.frameskip,
            "num_unique_samples": int(len(sample_ids)),
            "cache_stats": cache_stats,
        },
        "gate": result.to_dict(),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"[gate] task={args.task_name} selected={result.selected_method} "
        f"reason={result.reason} S={result.retained_safety_fraction} "
        f"R={result.robust_residual_gap} T_bg={result.background_threshold} "
        f"-> {args.out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
