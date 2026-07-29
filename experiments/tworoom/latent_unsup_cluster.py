#!/usr/bin/env python3
"""Unsupervised clustering of deduplicated train single latent vectors.

Baseline: FAISS spherical K-means (3 clusters). Method is switchable via --method.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from latent_cluster_common import (  # noqa: E402
    CLUSTER_METHODS,
    default_embed_dir,
    load_all_train_dedup_latent_vectors,
    save_cluster_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        default="faiss_spherical_kmeans",
        choices=sorted(CLUSTER_METHODS),
    )
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run multiple clustering seeds (overrides --seed when set)",
    )
    parser.add_argument("--embed-dir", type=Path, default=default_embed_dir())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=THIS_DIR / "results" / "latent_unsup_cluster",
    )
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--niter", type=int, default=100, help="FAISS K-means iterations")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("auto", "cpu", "cuda"),
        help="Cluster backend: cpu=official FAISS subsampling protocol (default)",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--max-points-per-centroid",
        type=int,
        default=256,
        help="FAISS K-means subsample cap per centroid (default 256 → K=3 uses <=768 fit vectors)",
    )
    return parser.parse_args()


def run_one_seed(
    args: argparse.Namespace,
    seed: int,
    X: np.ndarray,
    global_idx: np.ndarray,
    train_stats: dict,
) -> dict:
    print(f"\n==== latent clustering method={args.method} seed={seed} ====", flush=True)
    t_all = time.perf_counter()
    cluster_fn = CLUSTER_METHODS[args.method]
    if args.method in ("faiss_spherical_kmeans", "torch_spherical_kmeans"):
        centroids, labels, timing = cluster_fn(
            X,
            args.num_clusters,
            seed,
            niter=args.niter,
            device=args.device,
            gpu_id=args.gpu_id,
            max_points_per_centroid=args.max_points_per_centroid,
        )
    else:
        centroids, labels, timing = cluster_fn(X, args.num_clusters, seed)

    out_dir = args.out_dir / f"{args.method}_k{args.num_clusters}_seed{seed}"
    save_cluster_artifact(
        out_dir,
        method=args.method,
        seed=seed,
        num_clusters=args.num_clusters,
        centroids=centroids,
        global_idx=global_idx,
        labels=labels,
        train_stats=train_stats,
        timing=timing,
        extra={
            "niter": args.niter,
            "max_points_per_centroid": args.max_points_per_centroid,
            "device": args.device,
        } if args.method == "faiss_spherical_kmeans" else None,
    )
    timing["wall_sec"] = float(time.perf_counter() - t_all)
    summary = {
        "method": args.method,
        "seed": seed,
        "num_clusters": args.num_clusters,
        "out_dir": str(out_dir),
        "num_unique_timesteps": train_stats["num_unique_timesteps"],
        "timing_sec": timing,
        "cluster_counts": {
            f"cluster{k}": int((labels == k).sum()) for k in range(args.num_clusters)
        },
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[load] embedding caches from {args.embed_dir} (shared across {len(seeds)} seeds)",
        flush=True,
    )
    t_load = time.perf_counter()
    X, global_idx, train_stats = load_all_train_dedup_latent_vectors(
        args.embed_dir,
        frameskip=args.frameskip,
    )
    load_sec = time.perf_counter() - t_load
    print(f"[load] done in {load_sec:.2f}s", flush=True)

    summaries = [
        run_one_seed(args, seed, X, global_idx, train_stats) for seed in seeds
    ]
    manifest = {
        "method": args.method,
        "num_clusters": args.num_clusters,
        "seeds": seeds,
        "embed_dir": str(args.embed_dir),
        "device": args.device,
        "gpu_id": args.gpu_id,
        "load_sec": float(load_sec),
        "runs": summaries,
    }
    manifest_path = args.out_dir / f"{args.method}_k{args.num_clusters}_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
