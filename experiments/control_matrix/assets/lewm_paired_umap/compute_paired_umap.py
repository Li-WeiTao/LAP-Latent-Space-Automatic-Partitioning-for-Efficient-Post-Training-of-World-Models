#!/usr/bin/env python3
"""Build a label-free UMAP audit for one LeWM spectral-gate task.

The script reuses one frozen latent cache, recreates the nine predeclared
Spectral-K3 candidates (three landmark seeds by three kNN values), aligns their
unordered cluster IDs to the nominal seed-0/kNN-30 candidate with the Hungarian
algorithm, and exports a landmark-disjoint audit sample for ggplot2.

No predictor is trained and no control policy is evaluated.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
TWOROOM_DIR = REPO_ROOT / "experiments" / "tworoom"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TWOROOM_DIR))

from experiments.control_matrix.fit_partition import (  # noqa: E402
    episode_ids,
    load_unique_latents,
)
from lap.partition.landmark import _sample_landmarks  # noqa: E402
from lap.partition.spectral import (  # noqa: E402
    build_self_tuned_graph,
    l2_normalize_rows,
    spectral_labels,
)
from latent_landmark_spectral import exact_cosine_knn_torch  # noqa: E402


EPS = np.float32(1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("tworoom", "pusht"))
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument(
        "--reference-nominal-labels",
        type=Path,
        required=True,
        help="Current seed-0/kNN-30 cluster_labels.npz used for reproduction validation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--episode-key", default="auto")
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--num-landmarks", type=int, default=20_000)
    parser.add_argument("--diagnostic-seeds", default="0,1,2")
    parser.add_argument("--knn-values", default="27,30,33")
    parser.add_argument("--nominal-seed", type=int, default=0)
    parser.add_argument("--nominal-knn", type=int, default=30)
    parser.add_argument("--prototypes-per-cluster", type=int, default=16)
    parser.add_argument("--spectral-n-init", type=int, default=20)
    parser.add_argument("--prototype-n-init", type=int, default=5)
    parser.add_argument("--audit-size", type=int, default=20_000)
    parser.add_argument("--audit-seed", type=int, default=20_260_812)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--umap-neighbors", type=int, default=50)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-seed", type=int, default=20_260_812)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Expected unique comma-separated integers: {text!r}")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_rows(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    transformed = (np.asarray(values, dtype=np.float32) - mean) / scale
    return l2_normalize_rows(transformed).astype(np.float32, copy=False)


def fit_audit_candidate(
    landmarks: np.ndarray,
    neighbors: np.ndarray,
    similarities: np.ndarray,
    audit_transformed: np.ndarray,
    *,
    num_clusters: int,
    knn: int,
    seed: int,
    prototypes_per_cluster: int,
    spectral_n_init: int,
    prototype_n_init: int,
    cpu_threads: int,
) -> tuple[np.ndarray, dict]:
    graph, graph_meta = build_self_tuned_graph(neighbors, similarities, knn)
    landmark_labels, _, eigenvalues, spectral_meta = spectral_labels(
        graph,
        num_clusters=num_clusters,
        seed=seed,
        eig_tol=1e-4,
        eig_maxiter=10_000,
        spectral_n_init=spectral_n_init,
        cpu_threads=cpu_threads,
    )
    prototypes: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    with threadpool_limits(limits=cpu_threads):
        for region_id in range(num_clusters):
            region = landmarks[landmark_labels == region_id]
            if len(region) < prototypes_per_cluster:
                raise RuntimeError(
                    f"seed={seed}, knn={knn}, region={region_id} has only "
                    f"{len(region)} landmarks"
                )
            model = KMeans(
                n_clusters=prototypes_per_cluster,
                init="k-means++",
                n_init=prototype_n_init,
                random_state=seed * 1000 + region_id,
                algorithm="lloyd",
            ).fit(region)
            prototypes.append(l2_normalize_rows(model.cluster_centers_))
            owners.append(
                np.full(prototypes_per_cluster, region_id, dtype=np.int64)
            )
    prototype_matrix = np.concatenate(prototypes).astype(np.float32)
    prototype_owner = np.concatenate(owners)
    nearest = np.argmax(audit_transformed @ prototype_matrix.T, axis=1)
    labels = prototype_owner[nearest].astype(np.int64)
    return labels, {
        "seed": int(seed),
        "knn": int(knn),
        "graph": graph_meta,
        "spectral": spectral_meta,
        "laplacian_eigenvalues": eigenvalues.tolist(),
        "audit_cluster_counts_unaligned": np.bincount(
            labels, minlength=num_clusters
        ).tolist(),
    }


def hungarian_align(
    reference: np.ndarray, labels: np.ndarray, num_clusters: int
) -> tuple[np.ndarray, dict[str, int]]:
    contingency = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    np.add.at(contingency, (reference, labels), 1)
    ref_rows, candidate_cols = linear_sum_assignment(-contingency)
    mapping = np.arange(num_clusters, dtype=np.int64)
    mapping[candidate_cols] = ref_rows
    return mapping[labels], {
        str(int(candidate)): int(reference_id)
        for candidate, reference_id in zip(candidate_cols, ref_rows)
    }


def load_existing_nominal(
    path: Path,
    sample_ids: np.ndarray,
    audit_rows: np.ndarray,
) -> np.ndarray:
    path = path.resolve(strict=True)
    with np.load(path, allow_pickle=False) as data:
        existing_ids = np.asarray(data["global_idx"], dtype=np.int64)
        existing_labels = np.asarray(data["labels"], dtype=np.int64)
    if not np.array_equal(existing_ids, sample_ids):
        raise RuntimeError(f"Existing labels use a different sample order: {path}")
    return existing_labels[audit_rows]


def write_main_csv(
    path: Path,
    task: str,
    sample_ids: np.ndarray,
    coordinates: np.ndarray,
    nominal: np.ndarray,
    stability: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "task",
                "global_idx",
                "umap_1",
                "umap_2",
                "nominal_cluster",
                "stability_fraction",
                "stability_count",
            )
        )
        for index in range(len(sample_ids)):
            writer.writerow(
                (
                    task,
                    int(sample_ids[index]),
                    f"{coordinates[index, 0]:.8f}",
                    f"{coordinates[index, 1]:.8f}",
                    int(nominal[index]) + 1,
                    f"{stability[index]:.8f}",
                    int(round(stability[index] * 9)),
                )
            )


def write_draws_csv(
    path: Path,
    task: str,
    sample_ids: np.ndarray,
    coordinates: np.ndarray,
    aligned_by_draw: dict[tuple[int, int], np.ndarray],
    nominal: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "task",
                "seed",
                "knn",
                "global_idx",
                "umap_1",
                "umap_2",
                "aligned_cluster",
                "agrees_with_nominal",
            )
        )
        for (seed, knn), labels in sorted(aligned_by_draw.items()):
            for index in range(len(sample_ids)):
                writer.writerow(
                    (
                        task,
                        seed,
                        knn,
                        int(sample_ids[index]),
                        f"{coordinates[index, 0]:.8f}",
                        f"{coordinates[index, 1]:.8f}",
                        int(labels[index]) + 1,
                        int(labels[index] == nominal[index]),
                    )
                )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    seeds = int_tuple(args.diagnostic_seeds)
    knn_values = tuple(sorted(int_tuple(args.knn_values)))
    if args.nominal_seed not in seeds or args.nominal_knn not in knn_values:
        raise ValueError("The nominal draw must be included in the diagnostic grid")
    if len(seeds) * len(knn_values) != 9:
        raise ValueError("This audit requires exactly 3 seeds x 3 kNN values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate_manifest_path = args.gate_manifest.resolve(strict=True)
    gate_manifest = json.loads(gate_manifest_path.read_text(encoding="utf-8"))
    gate = gate_manifest["method_metadata"]["automatic_gate"]
    config = gate["configuration"]
    expected = {
        "num_regions": args.num_clusters,
        "num_landmarks": args.num_landmarks,
        "diagnostic_seeds": list(seeds),
        "nominal_knn": args.nominal_knn,
        "perturb_knn": [value for value in knn_values if value != args.nominal_knn],
    }
    for key, value in expected.items():
        if config[key] != value:
            raise RuntimeError(
                f"CLI differs from gate manifest for {key}: {value!r} != "
                f"{config[key]!r}"
            )

    print(f"[{args.task}] loading frozen cache", flush=True)
    raw, sample_ids, cache_stats = load_unique_latents(
        args.latent_cache.resolve(strict=True), args.frameskip
    )
    groups = episode_ids(
        args.data_file.resolve(strict=True), sample_ids, args.episode_key
    )
    print(
        f"[{args.task}] unique latents={len(raw):,}, dim={raw.shape[1]}",
        flush=True,
    )

    mean = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = raw.std(axis=0, dtype=np.float64).astype(np.float32) + EPS
    landmark_rows = {
        seed: _sample_landmarks(len(raw), args.num_landmarks, seed, groups)
        for seed in seeds
    }
    excluded = np.unique(np.concatenate(list(landmark_rows.values())))
    eligible = np.setdiff1d(
        np.arange(len(raw), dtype=np.int64), excluded, assume_unique=True
    )
    if len(eligible) < args.audit_size:
        raise RuntimeError("Not enough landmark-disjoint points for the audit")
    audit_rows = np.sort(
        np.random.default_rng(args.audit_seed).choice(
            eligible, size=args.audit_size, replace=False
        )
    )
    audit_sample_ids = sample_ids[audit_rows].copy()
    audit_raw = raw[audit_rows].copy()
    audit_transformed = transform_rows(audit_raw, mean, scale)
    print(
        f"[{args.task}] audit={len(audit_rows):,}; excluded landmark union="
        f"{len(excluded):,}",
        flush=True,
    )

    raw_labels: dict[tuple[int, int], np.ndarray] = {}
    run_metadata: list[dict] = []
    for seed in seeds:
        print(f"[{args.task}] seed={seed}: exact cosine kNN <= {max(knn_values)}", flush=True)
        landmarks = transform_rows(raw[landmark_rows[seed]], mean, scale)
        neighbors, similarities, neighbor_meta = exact_cosine_knn_torch(
            landmarks,
            max(knn_values),
            gpu_id=args.gpu_id,
            query_chunk=args.query_chunk,
        )
        for knn in knn_values:
            run_started = time.perf_counter()
            labels, metadata = fit_audit_candidate(
                landmarks,
                neighbors,
                similarities,
                audit_transformed,
                num_clusters=args.num_clusters,
                knn=knn,
                seed=seed,
                prototypes_per_cluster=args.prototypes_per_cluster,
                spectral_n_init=args.spectral_n_init,
                prototype_n_init=args.prototype_n_init,
                cpu_threads=args.cpu_threads,
            )
            raw_labels[(seed, knn)] = labels
            metadata["neighbor_search"] = neighbor_meta
            metadata["elapsed_sec"] = time.perf_counter() - run_started
            run_metadata.append(metadata)
            print(
                f"[{args.task}] seed={seed}, kNN={knn}: "
                f"audit counts={np.bincount(labels, minlength=args.num_clusters).tolist()}",
                flush=True,
            )
        del landmarks, neighbors, similarities
        torch.cuda.empty_cache()
        gc.collect()

    nominal = raw_labels[(args.nominal_seed, args.nominal_knn)]
    aligned_by_draw: dict[tuple[int, int], np.ndarray] = {}
    alignment_records: list[dict] = []
    for (seed, knn), labels in sorted(raw_labels.items()):
        aligned, mapping = hungarian_align(nominal, labels, args.num_clusters)
        aligned_by_draw[(seed, knn)] = aligned
        record = {
            "seed": seed,
            "knn": knn,
            "candidate_to_nominal_mapping": mapping,
            "adjusted_rand_index_vs_nominal": float(
                adjusted_rand_score(nominal, labels)
            ),
            "aligned_agreement_vs_nominal": float(np.mean(aligned == nominal)),
        }
        if seed == args.nominal_seed and knn == args.nominal_knn:
            existing = load_existing_nominal(
                args.reference_nominal_labels, sample_ids, audit_rows
            )
            record["adjusted_rand_index_vs_existing_nominal_artifact"] = float(
                adjusted_rand_score(existing, labels)
            )
            if record["adjusted_rand_index_vs_existing_nominal_artifact"] < 0.999999:
                raise RuntimeError(
                    f"Recomputed nominal candidate differs from existing artifact "
                    f"for seed {seed}: {record}"
                )
        alignment_records.append(record)

    aligned_stack = np.stack(
        [aligned_by_draw[key] for key in sorted(aligned_by_draw)], axis=0
    )
    stability = np.mean(aligned_stack == nominal[None, :], axis=0)

    print(f"[{args.task}] label-free StandardScaler -> PCA(50) -> UMAP", flush=True)
    scaler = StandardScaler()
    audit_scaled = scaler.fit_transform(audit_raw)
    pca = PCA(n_components=args.pca_components, random_state=args.umap_seed)
    audit_pca = pca.fit_transform(audit_scaled)
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.umap_seed,
        transform_seed=args.umap_seed,
        n_jobs=1,
        low_memory=True,
        verbose=True,
    )
    coordinates = reducer.fit_transform(audit_pca)

    main_csv = args.output_dir / f"{args.task}_lewm_paired_umap_audit.csv"
    draws_csv = args.output_dir / f"{args.task}_lewm_paired_umap_draws.csv"
    write_main_csv(
        main_csv,
        args.task,
        audit_sample_ids,
        coordinates,
        nominal,
        stability,
    )
    write_draws_csv(
        draws_csv,
        args.task,
        audit_sample_ids,
        coordinates,
        aligned_by_draw,
        nominal,
    )

    metadata = {
        "schema_version": 1,
        "task": args.task,
        "model_family": "lewm",
        "purpose": "accepted-vs-rejected spectral candidate visualization",
        "predictor_training_performed": False,
        "control_evaluation_performed": False,
        "latent_cache": str(args.latent_cache.resolve()),
        "latent_cache_sha256_from_gate_manifest": gate_manifest[
            "latent_cache_sha256"
        ],
        "cache_stats": cache_stats,
        "gate_manifest": str(gate_manifest_path),
        "gate_manifest_sha256": sha256_file(gate_manifest_path),
        "gate_selected_method": gate["selected_method"],
        "gate_reason": gate["reason"],
        "gate_retained_safety_fraction": gate["retained_safety_fraction"],
        "gate_robust_residual_gap": gate["robust_residual_gap"],
        "gate_background_threshold": gate["background_threshold"],
        "spectral_grid": {
            "num_clusters": args.num_clusters,
            "num_landmarks": args.num_landmarks,
            "diagnostic_seeds": list(seeds),
            "knn_values": list(knn_values),
            "nominal_seed": args.nominal_seed,
            "nominal_knn": args.nominal_knn,
            "prototypes_per_cluster": args.prototypes_per_cluster,
        },
        "audit": {
            "sample_unit": "unique global timestep latent",
            "sampling": "uniform without replacement after excluding the union of all diagnostic landmark rows",
            "audit_size": args.audit_size,
            "audit_seed": args.audit_seed,
            "landmark_union_size": int(len(excluded)),
            "landmark_overlap_count": int(np.intersect1d(audit_rows, excluded).size),
        },
        "projection": {
            "label_free": True,
            "fit_scope": "the fixed landmark-disjoint audit latents for this task",
            "preprocessing": "StandardScaler then PCA",
            "pca_components": args.pca_components,
            "pca_explained_variance": float(
                pca.explained_variance_ratio_.sum()
            ),
            "umap_version": umap.__version__,
            "umap_neighbors": args.umap_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "umap_metric": "euclidean",
            "umap_seed": args.umap_seed,
        },
        "stability": {
            "definition": "fraction of 9 Hungarian-aligned candidate labels equal to nominal seed-0/kNN-30 label",
            "mean": float(np.mean(stability)),
            "median": float(np.median(stability)),
            "fraction_unanimous": float(np.mean(stability == 1.0)),
            "fraction_at_least_eight_of_nine": float(
                np.mean(stability >= (8.0 / 9.0))
            ),
            "counts_by_nine": {
                str(value): int(np.sum(np.rint(stability * 9).astype(int) == value))
                for value in range(1, 10)
            },
        },
        "alignments": alignment_records,
        "run_metadata": run_metadata,
        "outputs": {
            "audit_csv": str(main_csv),
            "draws_csv": str(draws_csv),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_id": args.gpu_id,
        },
        "elapsed_sec": time.perf_counter() - started,
    }
    metadata_path = args.output_dir / f"{args.task}_lewm_paired_umap_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[{args.task}] done: mean stability={metadata['stability']['mean']:.4f}, "
        f"unanimous={metadata['stability']['fraction_unanimous']:.4f}",
        flush=True,
    )
    print(f"[{args.task}] wrote {main_csv}", flush=True)
    print(f"[{args.task}] wrote {draws_csv}", flush=True)
    print(f"[{args.task}] wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
