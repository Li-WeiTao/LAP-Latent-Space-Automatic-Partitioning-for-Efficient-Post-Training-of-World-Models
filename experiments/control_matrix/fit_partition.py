#!/usr/bin/env python3
"""Fit one action-free partition from a frozen LeWM latent cache.

This entry point is intentionally limited to the paper comparison matrix.  It
keeps the TwoRoom settings (K=3, Z-score/L2, K-means++ R=50, and landmark
spectral M=20k/k=30/P=16) while accepting any compatible latent cache and
pretrained-model dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TWOROOM_DIR = PROJECT_ROOT / "experiments" / "tworoom"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TWOROOM_DIR))

from lap.partition import (  # noqa: E402
    GatedSpectralPartitioner,
    PartitionArtifact,
    SpectralGateConfig,
)
from lap.partition.landmark import (  # noqa: E402
    LandmarkSpectralConfig,
    LandmarkSpectralPartitioner,
)
from lap.routing import VoronoiRouter  # noqa: E402
from latent_landmark_spectral import exact_cosine_knn_torch  # noqa: E402
from latent_spherical_kmeans_lib import (  # noqa: E402
    cluster_torch_spherical_kmeans_converged,
)


EPS = np.float32(1e-6)


def comma_separated_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected unique comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=(
            "global",
            "random_voronoi",
            "kmeanspp",
            "gmm",
            "controlled_parc",
            "controlled_parc_routable",
            "spectral",
            "auto",
        ),
    )
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Required by spectral sampling to recover episode IDs.",
    )
    parser.add_argument("--episode-key", default="auto")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--kmeans-restarts", type=int, default=50)
    parser.add_argument("--kmeans-max-iter", type=int, default=1000)
    parser.add_argument("--kmeans-rel-tol", type=float, default=1e-7)
    parser.add_argument("--kmeans-patience", type=int, default=10)
    parser.add_argument("--gmm-fit-samples", type=int, default=20_000,
                        help="Predeclared uniform training latent subset; 0 fits all unique latents.")
    parser.add_argument("--gmm-max-iter", type=int, default=100)
    parser.add_argument("--gmm-n-init", type=int, default=1)
    parser.add_argument("--gmm-tol", type=float, default=1e-3)
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    parser.add_argument("--parc-fit-samples", type=int, default=20_000)
    parser.add_argument("--parc-alpha", type=float, default=1e-5)
    parser.add_argument("--parc-sigma", type=float, default=1.0)
    parser.add_argument("--parc-max-iter", type=int, default=15)
    parser.add_argument("--parc-cost-tol", type=float, default=1e-4)
    parser.add_argument("--parc-kmeans-n-init", type=int, default=10)
    parser.add_argument("--parc-min-cluster-size", type=int, default=256)
    parser.add_argument("--num-landmarks", type=int, default=20_000)
    parser.add_argument("--knn", type=int, default=30)
    parser.add_argument("--prototypes-per-cluster", type=int, default=16)
    parser.add_argument("--spectral-n-init", type=int, default=20)
    parser.add_argument("--prototype-n-init", type=int, default=5)
    parser.add_argument(
        "--diagnostic-seeds",
        type=comma_separated_ints,
        default=comma_separated_ints("0,1,2"),
        help="At least three predeclared landmark draws for the auto gate.",
    )
    parser.add_argument("--deployment-seed", type=int, default=0)
    parser.add_argument(
        "--perturb-knn",
        type=comma_separated_ints,
        default=comma_separated_ints("27,33"),
    )
    parser.add_argument("--gate-perturbation-multiplier", type=float, default=2.0)
    parser.add_argument("--gate-retention-threshold", type=float, default=0.5)
    parser.add_argument("--gate-background-gap-count", type=int, default=10)
    parser.add_argument("--gate-background-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--gate-epsilon", type=float, default=1e-8)
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def l2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), np.float32(1e-12)
    )


def load_unique_latents(
    cache_path: Path, frameskip: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(cache_path, allow_pickle=False) as cache:
        emb = np.asarray(cache["emb"], dtype=np.float32)
        starts = np.asarray(cache["region_starts"], dtype=np.int64)
    if emb.ndim != 3 or starts.shape != (len(emb),):
        raise ValueError("latent cache must contain emb[N,T,D] and region_starts[N]")
    global_ids = (
        starts[:, None]
        + np.arange(emb.shape[1], dtype=np.int64)[None, :] * frameskip
    ).reshape(-1)
    flat = emb.reshape(-1, emb.shape[-1])
    # Stable sorting makes the representative for every timestep its first
    # occurrence in the original flattened window order.  Batch-shape-induced
    # FP32 noise must not turn one observation into multiple latent states.
    order = np.argsort(global_ids, kind="stable")
    ordered_ids = global_ids[order]
    ordered = flat[order]
    keep = np.r_[True, ordered_ids[1:] != ordered_ids[:-1]]
    unique_ids = ordered_ids[keep]
    unique = ordered[keep]
    return unique, unique_ids, {
        "cache": str(cache_path.resolve()),
        "num_transitions": int(len(emb)),
        "num_expanded_latents": int(len(flat)),
        "num_unique_timesteps": int(len(unique)),
        "discarded_repeated_window_slots": int(len(flat) - len(unique)),
        "duplicate_policy": "stable_first_occurrence_per_timestep",
        "latent_dim": int(unique.shape[1]),
        "frameskip": int(frameskip),
    }


def episode_ids(path: Path, indices: np.ndarray, requested: str) -> np.ndarray:
    with h5py.File(path, "r", swmr=True) as handle:
        if requested == "auto":
            key = "episode_idx" if "episode_idx" in handle else "ep_idx"
        else:
            key = requested
        if key not in handle:
            raise KeyError(f"episode column {key!r} is missing from {path}")
        return np.asarray(handle[key][indices], dtype=np.int64)


def zscore_l2(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    # The effective denominator is stored so deployment exactly reproduces fit.
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32) + EPS
    return l2((values - mean) / scale), mean, scale


def build_global(
    transformed: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> tuple[PartitionArtifact, np.ndarray, dict]:
    prototype = l2(transformed[:1])
    labels = np.zeros(len(transformed), dtype=np.int64)
    metadata = {
        "algorithm": "global_single_region",
        "num_clusters": 1,
        "routing": "constant_region_0",
    }
    return (
        PartitionArtifact(
            prototypes=prototype,
            prototype_region_ids=np.zeros(1, dtype=np.int64),
            mean=mean,
            scale=scale,
            metadata=metadata,
        ),
        labels,
        metadata,
    )


def build_random(
    transformed: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    num_clusters: int,
    seed: int,
) -> tuple[PartitionArtifact, np.ndarray, dict]:
    rows = np.random.default_rng(seed).choice(
        len(transformed), size=num_clusters, replace=False
    )
    prototypes = transformed[rows].copy()
    metadata = {
        "algorithm": "random_voronoi",
        "num_clusters": num_clusters,
        "seed": seed,
        "prototype_rows": rows.tolist(),
        "optimized": False,
        "routing": "zscore_l2_spherical_voronoi",
    }
    artifact = PartitionArtifact(
        prototypes=prototypes,
        prototype_region_ids=np.arange(num_clusters, dtype=np.int64),
        mean=mean,
        scale=scale,
        metadata=metadata,
    )
    labels = (transformed @ l2(prototypes).T).argmax(axis=1).astype(np.int64)
    return artifact, labels, metadata


def build_kmeans(
    transformed: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    args: argparse.Namespace,
) -> tuple[PartitionArtifact, np.ndarray, dict]:
    device = torch.device(
        f"cuda:{args.gpu_id}" if args.device == "cuda" else args.device
    )
    data = torch.from_numpy(np.ascontiguousarray(transformed)).to(device)
    runs: list[dict] = []
    best = None
    for restart in range(args.kmeans_restarts):
        inner_seed = args.seed * 100_000 + restart
        centroids, labels, info = cluster_torch_spherical_kmeans_converged(
            data,
            num_clusters=args.num_clusters,
            seed=inner_seed,
            max_iter=args.kmeans_max_iter,
            rel_tol=args.kmeans_rel_tol,
            patience=args.kmeans_patience,
            init_mode="kmeanspp",
        )
        record = {
            "restart": restart,
            "seed": inner_seed,
            "objective": float(info["objective_final"]),
            "niter": int(info["niter"]),
            "converged": bool(info["converged"]),
        }
        runs.append(record)
        candidate = (record["objective"], -restart, centroids, labels)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    prototypes = best[2].detach().cpu().numpy().astype(np.float32)
    labels = best[3].detach().cpu().numpy().astype(np.int64)
    metadata = {
        "algorithm": "spherical_kmeanspp",
        "num_clusters": args.num_clusters,
        "seed": args.seed,
        "n_init": args.kmeans_restarts,
        "selected_restart": int(-best[1]),
        "selected_objective": float(best[0]),
        "runs": runs,
        "routing": "zscore_l2_spherical_voronoi",
    }
    return (
        PartitionArtifact(
            prototypes=prototypes,
            prototype_region_ids=np.arange(args.num_clusters, dtype=np.int64),
            mean=mean,
            scale=scale,
            metadata=metadata,
        ),
        labels,
        metadata,
    )


def build_spectral(
    raw: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> tuple[PartitionArtifact, np.ndarray, dict]:
    search = None
    if args.device == "cuda":
        search = lambda values, max_k: exact_cosine_knn_torch(
            values,
            max_k,
            gpu_id=args.gpu_id,
            query_chunk=args.query_chunk,
        )
    partitioner = LandmarkSpectralPartitioner(
        LandmarkSpectralConfig(
            num_regions=args.num_clusters,
            num_landmarks=min(args.num_landmarks, len(raw)),
            knn=args.knn,
            prototypes_per_region=args.prototypes_per_cluster,
            seed=args.seed,
            spectral_n_init=args.spectral_n_init,
            prototype_n_init=args.prototype_n_init,
            cpu_threads=args.cpu_threads,
        ),
        **({} if search is None else {"neighbor_search": search}),
    )
    result = partitioner.fit(
        raw,
        sample_ids=np.arange(len(raw), dtype=np.int64),
        group_ids=groups,
    )
    return result.artifact, result.labels, result.metadata


def build_auto(
    raw: np.ndarray,
    sample_ids: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> tuple[PartitionArtifact, np.ndarray, dict]:
    search = None
    if args.device == "cuda":
        search = lambda values, max_k: exact_cosine_knn_torch(
            values,
            max_k,
            gpu_id=args.gpu_id,
            query_chunk=args.query_chunk,
        )
    gate_config = SpectralGateConfig(
        num_regions=args.num_clusters,
        num_landmarks=min(args.num_landmarks, len(raw)),
        nominal_knn=args.knn,
        perturb_knn=args.perturb_knn,
        diagnostic_seeds=args.diagnostic_seeds,
        deployment_seed=args.deployment_seed,
        perturbation_multiplier=args.gate_perturbation_multiplier,
        retention_threshold=args.gate_retention_threshold,
        background_gap_count=args.gate_background_gap_count,
        background_mad_multiplier=args.gate_background_mad_multiplier,
        epsilon=args.gate_epsilon,
        eig_tol=1e-4,
        eig_maxiter=20_000,
        cpu_threads=args.cpu_threads,
    )
    partition_config = LandmarkSpectralConfig(
        num_regions=args.num_clusters,
        num_landmarks=min(args.num_landmarks, len(raw)),
        knn=args.knn,
        prototypes_per_region=args.prototypes_per_cluster,
        seed=args.deployment_seed,
        spectral_n_init=args.spectral_n_init,
        prototype_n_init=args.prototype_n_init,
        cpu_threads=args.cpu_threads,
    )
    partitioner = GatedSpectralPartitioner(
        gate_config,
        partition_config,
        **({} if search is None else {"neighbor_search": search}),
    )
    result = partitioner.fit(raw, sample_ids=sample_ids, group_ids=groups)
    gate = result.metadata["automatic_gate"]
    method_metadata = {
        "algorithm": "automatic_spectral_degeneracy_gate",
        "selected_method": gate["selected_method"],
        "selected_post_training": result.metadata["selected_post_training"],
        "deployment_seed": gate["deployment_seed"],
        "automatic_gate": gate,
        "deployed_partition": {
            key: value
            for key, value in result.metadata.items()
            if key not in {"automatic_gate", "selected_post_training"}
        },
    }
    return result.artifact, result.labels, method_metadata


def build_gmm(transformed, mean, scale, args):
    import sklearn
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits
    if args.gmm_fit_samples < 0:
        raise ValueError('gmm-fit-samples must be nonnegative')
    count = min(args.gmm_fit_samples or len(transformed), len(transformed))
    rows = np.sort(np.random.default_rng(args.seed).choice(len(transformed), count, replace=False))
    model = GaussianMixture(n_components=args.num_clusters, covariance_type='full',
                            tol=args.gmm_tol, reg_covar=args.gmm_reg_covar,
                            max_iter=args.gmm_max_iter, n_init=args.gmm_n_init,
                            init_params='kmeans', random_state=args.seed, verbose=2, verbose_interval=10)
    print(f'[gmm] fit_samples={count} full_samples={len(transformed)} K={args.num_clusters} seed={args.seed}', flush=True)
    with threadpool_limits(limits=args.cpu_threads):
        model.fit(transformed[rows].astype(np.float64))
    if not model.converged_:
        raise RuntimeError('GMM did not converge; no automatic retry or silent parameter change')
    parameters = dict(means=model.means_.tolist(), weights=model.weights_.tolist(),
                      precisions_cholesky=model.precisions_cholesky_.tolist())
    metadata = dict(algorithm='sklearn_gaussian_mixture', seed=args.seed,
                    num_clusters=args.num_clusters, covariance_type='full',
                    routing='zscore_l2_gmm_full_posterior', gmm_parameters=parameters,
                    fit_samples=count, fit_sample_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
                    sampling='uniform without replacement from unique training latents',
                    init_params='kmeans', n_init=args.gmm_n_init, max_iter=args.gmm_max_iter,
                    tol=args.gmm_tol, reg_covar=args.gmm_reg_covar,
                    sklearn_version=sklearn.__version__, converged=bool(model.converged_),
                    n_iter=int(model.n_iter_), lower_bound=float(model.lower_bound_))
    artifact = PartitionArtifact(prototypes=model.means_.astype(np.float32),
                                 prototype_region_ids=np.arange(args.num_clusters),
                                 mean=mean, scale=scale, metadata=metadata)
    # Route through the serialized deployable parameters, not a nearest-centroid surrogate.
    with threadpool_limits(limits=args.cpu_threads):
        from lap.routing.gaussian_mixture import route_numpy
        labels = route_numpy(transformed, parameters)
        check = model.predict(transformed[:10000].astype(np.float64))
    if not np.array_equal(labels[:10000], check):
        raise RuntimeError('Serialized GMM router differs from sklearn predict')
    return artifact, labels, metadata


def _sample_controlled_windows(cache_path, count, seed, mean, scale):
    with np.load(cache_path, allow_pickle=False) as cache:
        total = int(len(cache["region_starts"]))
    if count <= 0:
        count = total
    count = min(count, total)
    rows = np.sort(
        np.random.default_rng(seed).choice(total, size=count, replace=False)
    )
    # Open the compressed cache separately for the two large arrays so peak
    # memory never contains full emb and full act_emb simultaneously.
    with np.load(cache_path, allow_pickle=False) as cache:
        emb = np.asarray(cache["emb"], dtype=np.float32)
        state_raw = emb[rows, 0].copy()
        response = (emb[rows, 1] - emb[rows, 0]).copy()
    with np.load(cache_path, allow_pickle=False) as cache:
        act = np.asarray(cache["act_emb"], dtype=np.float32)
        action = act[rows, 0].copy()
    state = l2((state_raw - mean) / scale)
    action_mean = action.mean(axis=0, dtype=np.float64).astype(np.float32)
    action_scale = action.std(axis=0, dtype=np.float64).astype(np.float32) + EPS
    response_mean = response.mean(axis=0, dtype=np.float64).astype(np.float32)
    response_scale = response.std(axis=0, dtype=np.float64).astype(np.float32) + EPS
    action = (action - action_mean) / action_scale
    response = (response - response_mean) / response_scale
    stats = {
        "fit_samples": count,
        "total_training_windows": total,
        "fit_sample_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "sampling": "uniform_without_replacement_from_training_windows",
        "state_index": 0,
        "action_embedding_index": 0,
        "response_indices": [0, 1],
        "action_mean_sha256": hashlib.sha256(action_mean.tobytes()).hexdigest(),
        "action_scale_sha256": hashlib.sha256(action_scale.tobytes()).hexdigest(),
        "response_mean_sha256": hashlib.sha256(response_mean.tobytes()).hexdigest(),
        "response_scale_sha256": hashlib.sha256(response_scale.tobytes()).hexdigest(),
    }
    return state, action, response, stats


def build_controlled_parc(transformed, mean, scale, args):
    from lap.partition.controlled_parc import ControlledPARCConfig, fit_controlled_parc

    if args.parc_fit_samples < 0:
        raise ValueError("parc-fit-samples must be nonnegative")
    state, action, response, sampling = _sample_controlled_windows(
        args.latent_cache, args.parc_fit_samples, args.seed, mean, scale
    )
    prototypes, _, metadata = fit_controlled_parc(
        state,
        action,
        response,
        ControlledPARCConfig(
            num_clusters=args.num_clusters,
            seed=args.seed,
            alpha=args.parc_alpha,
            sigma=args.parc_sigma,
            max_iter=args.parc_max_iter,
            cost_tol=args.parc_cost_tol,
            kmeans_n_init=args.parc_kmeans_n_init,
            min_cluster_size=args.parc_min_cluster_size,
            cpu_threads=args.cpu_threads,
            routable_updates=(args.method == "controlled_parc_routable"),
        ),
    )
    if args.method == "controlled_parc_routable" and not metadata["accepted_routable_updates"]:
        raise RuntimeError(
            "Routable Controlled PARC accepted no response-driven update; "
            "refusing to save a renamed K-means partition or launch downstream training"
        )
    metadata.update(
        {
            **sampling,
            "method_scope": "dynamics-informed partition; planning outcomes unused",
            "data_scope": "training latent cache only; evaluation episodes unused",
            "parc_reference": "Bemporad, IEEE TAC 2023, doi:10.1109/TAC.2022.3186812",
        }
    )
    artifact = PartitionArtifact(
        prototypes=prototypes,
        prototype_region_ids=np.asarray(
            metadata["router_prototype_region_ids"], dtype=np.int64
        ),
        mean=mean,
        scale=scale,
        metadata=metadata,
    )
    # ``transformed`` is already z-score/L2 normalized.  Route it directly;
    # VoronoiRouter is reserved for raw latents and would transform twice here.
    labels = artifact.prototype_region_ids[
        (transformed @ l2(prototypes).T).argmax(axis=1)
    ].astype(np.int64)
    return artifact, labels, metadata


def main() -> None:
    args = parse_args()
    if args.frameskip < 1 or (
        args.num_clusters < 2 and args.method != "global"
    ):
        raise ValueError("frameskip must be positive and K must be >=2")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty {args.out_dir}")
    if args.out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(args.out_dir)
    # A prior pre-fit failure can leave an empty directory.  It is safe to
    # reuse, while a non-empty directory still requires --overwrite above.
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    raw, sample_ids, cache_stats = load_unique_latents(
        args.latent_cache.resolve(strict=True), args.frameskip
    )
    transformed, mean, scale = zscore_l2(raw)
    if args.method == "global":
        artifact, labels, method_meta = build_global(transformed, mean, scale)
    elif args.method == "random_voronoi":
        artifact, labels, method_meta = build_random(
            transformed, mean, scale, args.num_clusters, args.seed
        )
    elif args.method == "kmeanspp":
        artifact, labels, method_meta = build_kmeans(
            transformed, mean, scale, args
        )
    elif args.method == "gmm":
        artifact, labels, method_meta = build_gmm(transformed, mean, scale, args)
    elif args.method in ("controlled_parc", "controlled_parc_routable"):
        artifact, labels, method_meta = build_controlled_parc(
            transformed, mean, scale, args
        )
    elif args.method == "spectral":
        if args.data_file is None:
            raise ValueError("--data-file is required for spectral episode sampling")
        groups = episode_ids(
            args.data_file.resolve(strict=True), sample_ids, args.episode_key
        )
        artifact, labels, method_meta = build_spectral(raw, groups, args)
    else:
        if args.data_file is None:
            raise ValueError("--data-file is required for automatic spectral gating")
        groups = episode_ids(
            args.data_file.resolve(strict=True), sample_ids, args.episode_key
        )
        artifact, labels, method_meta = build_auto(raw, sample_ids, groups, args)

    artifact.validate()
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != sample_ids.shape:
        raise RuntimeError("partition produced the wrong number of labels")
    counts = np.bincount(labels, minlength=artifact.num_regions)
    if np.any(counts == 0):
        raise RuntimeError(f"partition contains an empty region: {counts.tolist()}")
    if args.method == 'gmm':
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.cpu_threads):
            deployed_labels = VoronoiRouter(artifact).route(raw).astype(np.int64)
    else:
        deployed_labels = VoronoiRouter(artifact).route(raw).astype(np.int64)
    if not np.array_equal(labels, deployed_labels):
        disagreement = float(np.mean(labels != deployed_labels))
        raise RuntimeError(
            "offline labels disagree with the deployed Voronoi router: "
            f"{disagreement:.6%}"
        )
    artifact.save(args.out_dir / "partition")
    np.savez_compressed(
        args.out_dir / "cluster_labels.npz",
        sample_ids=sample_ids,
        global_idx=sample_ids,
        labels=labels,
    )
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset_name,
        "method": args.method,
        "selected_method": method_meta.get("selected_method", args.method),
        "partition_seed": method_meta.get("deployment_seed", args.seed),
        "num_clusters": artifact.num_regions,
        "cluster_counts": counts.tolist(),
        "cluster_fractions": (counts / len(labels)).tolist(),
        "cache_stats": cache_stats,
        "latent_cache_sha256": sha256_file(args.latent_cache),
        "data_file": str(args.data_file.resolve()) if args.data_file else None,
        "method_metadata": method_meta,
        "elapsed_sec": time.perf_counter() - started,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] {args.method} seed={args.seed} counts={counts.tolist()} "
        f"elapsed={manifest['elapsed_sec']:.2f}s -> {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
