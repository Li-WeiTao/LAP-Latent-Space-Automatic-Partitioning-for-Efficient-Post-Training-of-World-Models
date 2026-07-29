#!/usr/bin/env python3
"""Build deployable random-Voronoi latent partitions for Random-K controls.

The control uses the same frozen latent vectors, Z-score/L2 preprocessing and
nearest-prototype router as the learned K-means/spectral partitions.  The only
difference is that its K routing prototypes are sampled uniformly from the
training latent vectors and are never optimized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from latent_cluster_common import (
    load_all_train_dedup_latent_vectors,
    load_zscore_params,
    sha256_file,
    transform_zscore_l2,
)


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-source-dir",
        type=Path,
        default=Path(
            "experiments/tworoom/results/"
            "tworoom_geometry_train_region_predictors"
        ),
    )
    parser.add_argument(
        "--zscore-params",
        type=Path,
        default=Path(
            "experiments/tworoom/results/"
            "latent_kmeanspp_multirestart_k3/zscore_params.npz"
        ),
        help="Reuse the exact training-set preprocessing fitted for K-means++.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "experiments/tworoom/results/latent_random_voronoi_k3"
        ),
    )
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--chunk-size", type=int, default=131_072)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def atomic_write_json(path: Path, value: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_ready(value), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def assign_random_voronoi(
    X: np.ndarray,
    prototypes: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    labels = np.empty(len(X), dtype=np.int64)
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        labels[start:stop] = (X[start:stop] @ prototypes.T).argmax(axis=1)
    return labels


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def build_one(
    *,
    X: np.ndarray,
    global_idx: np.ndarray,
    data_stats: dict,
    zscore: dict,
    zscore_source: Path,
    out_root: Path,
    num_clusters: int,
    seed: int,
    chunk_size: int,
    overwrite_existing: bool,
) -> Path:
    artifact_dir = out_root / f"random_voronoi_k{num_clusters}_seed{seed}"
    if artifact_dir.exists() and not overwrite_existing:
        raise FileExistsError(
            f"Artifact already exists: {artifact_dir}. Pass --overwrite-existing "
            "only for an intentional replacement."
        )

    rng = np.random.default_rng(seed)
    anchor_rows = rng.choice(len(X), size=num_clusters, replace=False)
    routing_prototypes = np.ascontiguousarray(X[anchor_rows], dtype=np.float32)

    assign_t0 = time.perf_counter()
    labels = assign_random_voronoi(
        X, routing_prototypes, chunk_size=chunk_size
    )
    assign_sec = time.perf_counter() - assign_t0
    counts = np.bincount(labels, minlength=num_clusters)
    if np.any(counts == 0):
        raise RuntimeError(
            f"Random prototypes produced an empty cluster for seed {seed}: {counts}"
        )
    # The labels are defined by the deployed router, so fidelity must be exact.
    check = assign_random_voronoi(
        X, routing_prototypes, chunk_size=chunk_size
    )
    if not np.array_equal(labels, check):
        raise RuntimeError("Random-Voronoi routing is not deterministic")

    diagnostic_centroids = np.stack(
        [
            X[labels == cluster_id].mean(axis=0)
            for cluster_id in range(num_clusters)
        ]
    ).astype(np.float32)
    diagnostic_centroids /= np.maximum(
        np.linalg.norm(diagnostic_centroids, axis=1, keepdims=True),
        np.float32(1e-12),
    )

    out_root.mkdir(parents=True, exist_ok=True)
    write_dir = out_root / f".{artifact_dir.name}.pid{os.getpid()}.tmp"
    if write_dir.exists():
        shutil.rmtree(write_dir)
    write_dir.mkdir()
    try:
        np.save(write_dir / "centroids.npy", diagnostic_centroids)
        np.save(write_dir / "routing_prototypes.npy", routing_prototypes)
        np.save(
            write_dir / "prototype_cluster_ids.npy",
            np.arange(num_clusters, dtype=np.int64),
        )
        np.savez_compressed(
            write_dir / "cluster_labels.npz",
            global_idx=global_idx.astype(np.int64),
            labels=labels,
        )
        np.savez_compressed(
            write_dir / "zscore_params.npz",
            mu=zscore["mu"].astype(np.float32),
            sigma=zscore["sigma"].astype(np.float32),
            eps=np.float32(zscore["eps"]),
        )

        output_files = (
            "centroids.npy",
            "routing_prototypes.npy",
            "prototype_cluster_ids.npy",
            "cluster_labels.npz",
            "zscore_params.npz",
        )
        hashes = {
            name: sha256_file(write_dir / name) for name in output_files
        }
        sizes = {
            name: int((write_dir / name).stat().st_size) for name in output_files
        }
        meta = {
            "version_label": "latent_random_voronoi_v1",
            "assignment_schema_version": 2,
            "method": "zscore_l2_random_voronoi",
            "description": (
                "Random-K capacity control: K train latents are sampled uniformly "
                "without optimization and define a cosine Voronoi router."
            ),
            "seed": seed,
            "num_clusters": num_clusters,
            "cluster_counts": {
                f"cluster{k}": int(counts[k]) for k in range(num_clusters)
            },
            "cluster_fractions": (counts / len(labels)).tolist(),
            "latent_dim": int(X.shape[1]),
            "spherical": True,
            "classification_rule": (
                "zscore and L2-normalize latent; choose maximum-cosine random "
                "training prototype; route to that prototype's cluster id"
            ),
            "prototype_sampling": {
                "algorithm": "uniform_without_replacement",
                "optimized": False,
                "num_restarts": 0,
                "anchor_rows": anchor_rows.tolist(),
                "anchor_global_idx": global_idx[anchor_rows].tolist(),
            },
            "num_routing_prototypes": num_clusters,
            "route_anchor": "transition_start",
            "transition_label_offset_steps": 0,
            "preprocess": "zscore_l2",
            "zscore_eps": float(zscore["eps"]),
            "zscore_source": str(zscore_source.resolve(strict=True)),
            "zscore_source_sha256": sha256_file(zscore_source),
            "full_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "output_file_sha256": hashes,
            "output_file_size_bytes": sizes,
            "train_data_stats": data_stats,
            "timing_sec": {"assign_and_router_check_sec": assign_sec},
            "git_commit": git_commit(),
        }
        atomic_write_json(write_dir / "cluster_meta.json", meta)

        if artifact_dir.exists():
            backup = out_root / f".{artifact_dir.name}.previous.{os.getpid()}"
            os.replace(artifact_dir, backup)
            try:
                os.replace(write_dir, artifact_dir)
            except Exception:
                os.replace(backup, artifact_dir)
                raise
            else:
                shutil.rmtree(backup)
        else:
            os.replace(write_dir, artifact_dir)
    finally:
        if write_dir.exists():
            shutil.rmtree(write_dir)

    print(
        f"[seed {seed}] {artifact_dir} counts={counts.tolist()} "
        f"fractions={np.round(counts / len(labels), 4).tolist()}",
        flush=True,
    )
    return artifact_dir


def main() -> None:
    args = parse_args()
    if args.num_clusters < 2:
        raise SystemExit("--num-clusters must be at least 2")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")

    load_t0 = time.perf_counter()
    raw_X, global_idx, data_stats = load_all_train_dedup_latent_vectors(
        args.embedding_source_dir
    )
    order = np.argsort(global_idx, kind="stable")
    global_idx = np.asarray(global_idx[order], dtype=np.int64)
    raw_X = np.asarray(raw_X[order], dtype=np.float32)
    if len(global_idx) and np.any(global_idx[1:] <= global_idx[:-1]):
        raise RuntimeError("Deduplicated global_idx must be strictly increasing")

    zscore = load_zscore_params(args.zscore_params)
    X = transform_zscore_l2(
        raw_X, zscore["mu"], zscore["sigma"], zscore["eps"]
    )
    data_stats = dict(data_stats)
    data_stats["load_and_preprocess_sec"] = time.perf_counter() - load_t0
    data_stats["preprocess_source"] = str(args.zscore_params.resolve(strict=True))

    for seed in args.seeds:
        build_one(
            X=X,
            global_idx=global_idx,
            data_stats=data_stats,
            zscore=zscore,
            zscore_source=args.zscore_params,
            out_root=args.out_root,
            num_clusters=args.num_clusters,
            seed=seed,
            chunk_size=args.chunk_size,
            overwrite_existing=args.overwrite_existing,
        )


if __name__ == "__main__":
    main()
