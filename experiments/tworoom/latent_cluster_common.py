"""Shared utilities for latent unsupervised clustering experiments."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from geometry_latent_svm_rooms3 import (  # noqa: E402
    DEFAULT_EMBED_DIR,
    DEFAULT_FRAME_SKIP,
    PRIORITY5_LABELS,
    deduplicate_by_global_idx,
    expand_latent_vectors,
)

ALL_TRAIN_CACHE_REGIONS = PRIORITY5_LABELS
CLUSTER_NAMES = ("cluster0", "cluster1", "cluster2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_all_train_dedup_latent_vectors(
    embed_dir: Path,
    *,
    frameskip: int = DEFAULT_FRAME_SKIP,
    cache_regions: tuple[str, ...] = ALL_TRAIN_CACHE_REGIONS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load deduplicated single latent vectors from P_train region caches."""
    X_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    transition_counts: dict[str, int] = {}

    for region in cache_regions:
        path = embed_dir / f"P_train_{region}_embeddings.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached embeddings: {path}")
        data = np.load(path)
        emb = np.asarray(data["emb"], dtype=np.float32)
        starts = np.asarray(data["region_starts"], dtype=np.int64)
        transition_counts[region] = int(len(starts))
        X_seq, global_idx = expand_latent_vectors(emb, starts, frameskip=frameskip)
        X_parts.append(X_seq)
        idx_parts.append(global_idx)
        print(
            f"  [cache] {region}: {transition_counts[region]} transitions, "
            f"{len(X_seq)} latent vectors from {path.name}",
            flush=True,
        )

    X_all = np.concatenate(X_parts, axis=0)
    global_idx = np.concatenate(idx_parts, axis=0)
    before_dedup = len(global_idx)
    dummy_y = np.zeros(before_dedup, dtype=np.int64)
    X, kept_idx, _, dropped = deduplicate_by_global_idx(X_all, global_idx, dummy_y)
    print(
        f"  [dedup] {before_dedup} expanded vectors -> {len(kept_idx)} unique timesteps "
        f"(dropped_duplicates={dropped})",
        flush=True,
    )
    stats = {
        "embed_dir": str(embed_dir),
        "cache_regions": list(cache_regions),
        "transition_counts": transition_counts,
        "num_expanded_vectors": int(before_dedup),
        "num_dropped_duplicate_timesteps": dropped,
        "num_unique_timesteps": int(len(kept_idx)),
        "frameskip": frameskip,
        "latent_dim": int(X.shape[1]) if len(X) else 0,
    }
    return X, kept_idx, stats


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def resolve_cluster_torch_device(device: str = "auto", gpu_id: int = 0) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda is unavailable")
        return torch.device(f"cuda:{gpu_id}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def resolve_faiss_use_gpu(device: str = "auto", gpu_id: int = 0) -> tuple[bool, int]:
    """Return (use_gpu, gpu_id) for FAISS when faiss-gpu is healthy."""
    if device == "cpu":
        return False, gpu_id
    try:
        import faiss
    except Exception:
        return False, gpu_id
    if device in ("cuda", "gpu") and faiss.get_num_gpus() < 1:
        return False, gpu_id
    if device == "auto" and faiss.get_num_gpus() < 1:
        return False, gpu_id
    try:
        faiss.StandardGpuResources()
        return True, gpu_id
    except Exception:
        return False, gpu_id


def assign_nearest_centroid_faiss(
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    spherical: bool = True,
    use_gpu: bool = False,
    gpu_id: int = 0,
) -> np.ndarray:
    """Batch assign via FAISS search (GPU when available)."""
    import faiss

    X = np.asarray(X, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    d = centroids.shape[1]
    if spherical:
        Xq = l2_normalize_rows(X)
        C = l2_normalize_rows(centroids)
        index = faiss.IndexFlatIP(d)
    else:
        Xq = X
        C = centroids
        index = faiss.IndexFlatL2(d)
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)
    index.add(C)
    _, assign_idx = index.search(Xq, 1)
    return assign_idx.reshape(-1).astype(np.int64)


def assign_nearest_centroid_torch(
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    spherical: bool = True,
    device: torch.device | None = None,
) -> np.ndarray:
    dev = device or resolve_cluster_torch_device("auto")
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(dev)
    Ct = torch.from_numpy(np.asarray(centroids, dtype=np.float32)).to(dev)
    if spherical:
        Xt = F.normalize(Xt, dim=1)
        Ct = F.normalize(Ct, dim=1)
        scores = Xt @ Ct.T
        return scores.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    dists = torch.cdist(Xt, Ct, p=2)
    return dists.argmin(dim=1).detach().cpu().numpy().astype(np.int64)


def assign_nearest_centroid(
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    spherical: bool = True,
    use_gpu: bool = False,
    gpu_id: int = 0,
    torch_device: torch.device | None = None,
) -> np.ndarray:
    """Return cluster id per row via argmax dot product (spherical) or argmin L2."""
    if torch_device is not None and torch_device.type == "cuda":
        return assign_nearest_centroid_torch(
            X, centroids, spherical=spherical, device=torch_device
        )
    if use_gpu:
        try:
            return assign_nearest_centroid_faiss(
                X, centroids, spherical=spherical, use_gpu=True, gpu_id=gpu_id
            )
        except Exception:
            pass
    if spherical:
        Xn = l2_normalize_rows(X.astype(np.float32))
        Cn = l2_normalize_rows(centroids.astype(np.float32))
        scores = Xn @ Cn.T
        return np.argmax(scores, axis=1).astype(np.int64)
    dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dists, axis=1).astype(np.int64)


def build_global_idx_lookup(
    global_idx: np.ndarray,
    labels: np.ndarray,
) -> dict[int, int]:
    if len(global_idx) != len(labels):
        raise ValueError("global_idx and labels length mismatch")
    return {int(g): int(l) for g, l in zip(global_idx, labels)}


def cluster_labels_for_starts(
    starts: np.ndarray,
    global_idx_lookup: dict[int, int],
) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    labels = np.full(len(starts), -1, dtype=np.int64)
    missing = 0
    for i, g in enumerate(starts):
        key = int(g)
        if key not in global_idx_lookup:
            missing += 1
            continue
        labels[i] = global_idx_lookup[key]
    if missing:
        raise RuntimeError(
            f"{missing} transition starts lack cluster labels "
            f"(of {len(starts)} total)"
        )
    return labels


def save_cluster_artifact(
    out_dir: Path,
    *,
    method: str,
    seed: int,
    num_clusters: int,
    centroids: np.ndarray,
    global_idx: np.ndarray,
    labels: np.ndarray,
    train_stats: dict,
    timing: dict,
    extra: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "centroids.npy", centroids.astype(np.float32))
    np.savez_compressed(
        out_dir / "cluster_labels.npz",
        global_idx=np.asarray(global_idx, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
    )
    counts = {
        f"cluster{k}": int((labels == k).sum()) for k in range(num_clusters)
    }
    meta = {
        "method": method,
        "seed": seed,
        "num_clusters": num_clusters,
        "cluster_counts": counts,
        "train_data_stats": train_stats,
        "timing_sec": timing,
        "classification_rule": (
            "spherical: argmax cosine similarity to L2-normalized centroids"
            if method.endswith("spherical_kmeans") or "spherical" in method
            else "argmin L2 distance to centroids"
        ),
        "spherical": "spherical" in method,
    }
    if extra:
        meta.update(extra)
    meta_path = out_dir / "cluster_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [save] cluster artifact -> {out_dir}", flush=True)
    return out_dir


def load_cluster_artifact(artifact_dir: Path) -> dict:
    artifact_dir = Path(artifact_dir)
    diagnostic_centroids = np.asarray(
        np.load(artifact_dir / "centroids.npy"), dtype=np.float32
    )
    data = np.load(artifact_dir / "cluster_labels.npz")
    global_idx = np.asarray(data["global_idx"], dtype=np.int64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    meta_path = artifact_dir / "cluster_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    recorded_hashes = meta.get("output_file_sha256", {})
    recorded_sizes = meta.get("output_file_size_bytes", {})
    if recorded_hashes:
        for filename, expected_hash in recorded_hashes.items():
            if Path(filename).name != filename:
                raise ValueError(f"Unsafe output filename in {meta_path}: {filename!r}")
            output_path = artifact_dir / filename
            if not output_path.is_file():
                raise FileNotFoundError(f"Artifact output listed in metadata is missing: {output_path}")
            expected_size = recorded_sizes.get(filename)
            if expected_size is not None and output_path.stat().st_size != int(expected_size):
                raise ValueError(f"Artifact output size mismatch: {output_path}")
            if sha256_file(output_path) != expected_hash:
                raise ValueError(f"Artifact output SHA-256 mismatch: {output_path}")

    routing_path = artifact_dir / "routing_prototypes.npy"
    owner_path = artifact_dir / "prototype_cluster_ids.npy"
    if routing_path.exists() != owner_path.exists():
        raise ValueError(
            "routing_prototypes.npy and prototype_cluster_ids.npy must either "
            f"both exist or both be absent in {artifact_dir}"
        )
    if routing_path.exists():
        routing_vectors = np.asarray(np.load(routing_path), dtype=np.float32)
        prototype_cluster_ids = np.asarray(np.load(owner_path), dtype=np.int64)
    else:
        # Backward-compatible one-centroid-per-cluster routing.
        routing_vectors = diagnostic_centroids
        prototype_cluster_ids = np.arange(len(routing_vectors), dtype=np.int64)

    if (
        diagnostic_centroids.ndim != 2
        or diagnostic_centroids.shape[1] != routing_vectors.shape[1]
    ):
        raise ValueError(
            f"centroids and routing vectors have incompatible dimensions in {artifact_dir}"
        )

    num_clusters = int(meta.get("num_clusters", len(diagnostic_centroids)))
    _validate_routing_artifact(
        artifact_dir=artifact_dir,
        global_idx=global_idx,
        labels=labels,
        routing_vectors=routing_vectors,
        prototype_cluster_ids=prototype_cluster_ids,
        num_clusters=num_clusters,
    )
    meta.setdefault("num_clusters", num_clusters)
    meta.setdefault("num_routing_prototypes", int(len(routing_vectors)))
    meta.setdefault(
        "classification_rule",
        "argmax cosine similarity to centroid" if len(routing_vectors) == num_clusters
        else "argmax cosine similarity to routing prototype, then use prototype owner",
    )
    lookup = build_global_idx_lookup(global_idx, labels)

    zscore_path = artifact_dir / "zscore_params.npz"
    requires_zscore = (
        str(meta.get("preprocess", "")).lower() == "zscore_l2"
        or "zscore" in str(meta.get("method", "")).lower()
    )
    if requires_zscore and not zscore_path.exists():
        raise FileNotFoundError(
            f"Artifact routing vectors are in Z-score space but {zscore_path} is missing"
        )
    zscore = load_zscore_params(zscore_path) if zscore_path.exists() else None
    if zscore is not None and zscore["mu"].shape != (routing_vectors.shape[1],):
        raise ValueError(
            f"Z-score dimension does not match routing vectors in {artifact_dir}"
        )
    return {
        # `centroids` remains the public routing-vector key used by evaluation.
        "centroids": routing_vectors,
        "routing_vectors": routing_vectors,
        "prototype_cluster_ids": prototype_cluster_ids,
        "diagnostic_centroids": diagnostic_centroids,
        "global_idx": global_idx,
        "labels": labels,
        "lookup": lookup,
        "meta": meta,
        "artifact_dir": artifact_dir,
        "zscore": zscore,
    }


def _validate_routing_artifact(
    *,
    artifact_dir: Path,
    global_idx: np.ndarray,
    labels: np.ndarray,
    routing_vectors: np.ndarray,
    prototype_cluster_ids: np.ndarray,
    num_clusters: int,
) -> None:
    """Validate the common cluster-artifact contract before training or routing."""
    if global_idx.ndim != 1 or labels.ndim != 1 or len(global_idx) != len(labels):
        raise ValueError(f"Invalid global_idx/labels shapes in {artifact_dir}")
    if len(global_idx) and np.any(global_idx[1:] <= global_idx[:-1]):
        raise ValueError(f"global_idx must be strictly increasing in {artifact_dir}")
    if routing_vectors.ndim != 2 or not len(routing_vectors):
        raise ValueError(f"routing vectors must have shape (R,D) in {artifact_dir}")
    if prototype_cluster_ids.shape != (len(routing_vectors),):
        raise ValueError(
            "prototype_cluster_ids must contain one owner for every routing vector "
            f"in {artifact_dir}"
        )
    if num_clusters < 1:
        raise ValueError(f"num_clusters must be positive in {artifact_dir}")
    if labels.size and (labels.min() < 0 or labels.max() >= num_clusters):
        raise ValueError(f"cluster labels out of range [0,{num_clusters}) in {artifact_dir}")
    if prototype_cluster_ids.size and (
        prototype_cluster_ids.min() < 0
        or prototype_cluster_ids.max() >= num_clusters
    ):
        raise ValueError(
            f"prototype owners out of range [0,{num_clusters}) in {artifact_dir}"
        )
    if set(prototype_cluster_ids.tolist()) != set(range(num_clusters)):
        raise ValueError(
            f"every cluster must own at least one routing vector in {artifact_dir}"
        )
    if not np.isfinite(routing_vectors).all():
        raise ValueError(f"routing vectors contain NaN/Inf in {artifact_dir}")


def load_zscore_params(zscore_params_npz: Path) -> dict:
    data = np.load(zscore_params_npz)
    mu = np.asarray(data["mu"], dtype=np.float32)
    sigma = np.asarray(data["sigma"], dtype=np.float32)
    eps = float(np.asarray(data["eps"]).item())
    if mu.ndim != 1 or sigma.shape != mu.shape:
        raise ValueError(f"Invalid mu/sigma shapes in {zscore_params_npz}")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError(f"Z-score parameters contain NaN/Inf in {zscore_params_npz}")
    if np.any(sigma < 0) or not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"Invalid sigma/eps in {zscore_params_npz}")
    return {
        "mu": mu,
        "sigma": sigma,
        "eps": eps,
        "path": str(zscore_params_npz),
    }


def transform_zscore_l2(
    Z: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: float,
) -> np.ndarray:
    Zs = (Z.astype(np.float32) - mu) / (sigma + eps)
    return l2_normalize_rows(Zs)


def load_kmeanspp_label_artifact(
    label_npz: Path,
    *,
    zscore_params_npz: Path | None = None,
    require_zscore: bool = False,
) -> dict:
    """Load an R-budget K-means++ partition.

    ``require_zscore`` is intended for inference.  K-means++ labels can still be
    consumed without preprocessing parameters when they are used only to split
    cached training transitions, but routing a new latent through centroids that
    were fitted in Z-score space must fail closed when those parameters are
    unavailable.
    """
    label_npz = Path(label_npz)
    if require_zscore and zscore_params_npz is None:
        raise FileNotFoundError(
            "K-means++ inference routing requires the fitted zscore_params.npz; "
            "pass zscore_params_npz explicitly"
        )
    data = np.load(label_npz)
    global_idx = np.asarray(data["global_idx"], dtype=np.int64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    centroids = np.asarray(data["centroids"], dtype=np.float32)
    lookup = build_global_idx_lookup(global_idx, labels)

    stem = label_npz.stem
    outer_seed = None
    inner_budget = None
    if "_outer" in stem:
        outer_seed = int(stem.rsplit("outer", 1)[-1])
    if stem.startswith("kmeanspp_R"):
        inner_budget = int(stem.split("_outer", 1)[0].removeprefix("kmeanspp_R"))

    meta = {
        "method": "zscore_l2_spherical_kmeanspp",
        "num_clusters": int(centroids.shape[0]),
        "seed": outer_seed,
        "inner_restart_budget": inner_budget,
        "classification_rule": (
            "zscore L2-normalize latent, then argmax cosine similarity to centroids"
        ),
        "spherical": True,
        "label_npz": str(label_npz),
    }
    if "objective_final" in data.files:
        meta["objective_final"] = float(np.asarray(data["objective_final"]).item())
    if "selected_inner_restart" in data.files:
        meta["selected_inner_restart"] = int(np.asarray(data["selected_inner_restart"]).item())

    zscore = None
    if zscore_params_npz is not None:
        zscore = load_zscore_params(zscore_params_npz)
        if zscore["mu"].shape != (centroids.shape[1],):
            raise ValueError(
                "Z-score dimension does not match K-means++ centroids in "
                f"{label_npz}"
            )
        meta["zscore_params_npz"] = str(zscore_params_npz)

    return {
        "centroids": centroids,
        "routing_vectors": centroids,
        "prototype_cluster_ids": np.arange(centroids.shape[0], dtype=np.int64),
        "diagnostic_centroids": centroids,
        "global_idx": global_idx,
        "labels": labels,
        "lookup": lookup,
        "meta": meta,
        "artifact_dir": label_npz.parent,
        "zscore": zscore,
    }


def resolve_cluster_source(
    *,
    cluster_artifact_dir: Path | None = None,
    kmeanspp_label_npz: Path | None = None,
    zscore_params_npz: Path | None = None,
    require_kmeanspp_zscore: bool = False,
) -> dict:
    if (cluster_artifact_dir is None) == (kmeanspp_label_npz is None):
        raise ValueError(
            "Exactly one cluster source is required: --cluster-artifact-dir XOR "
            "--kmeanspp-label-npz"
        )
    if kmeanspp_label_npz is not None:
        return load_kmeanspp_label_artifact(
            kmeanspp_label_npz,
            zscore_params_npz=zscore_params_npz,
            require_zscore=require_kmeanspp_zscore,
        )
    return load_cluster_artifact(cluster_artifact_dir)


def benchmark_assign_time(
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    spherical: bool,
    repeats: int = 5,
    use_gpu: bool = False,
    gpu_id: int = 0,
    torch_device: torch.device | None = None,
) -> dict:
    times: list[float] = []
    n = len(X)
    backend = "numpy"
    if torch_device is not None and torch_device.type == "cuda":
        backend = "torch_cuda"
    elif use_gpu:
        backend = "faiss_gpu"
    for _ in range(repeats):
        t0 = time.perf_counter()
        assign_nearest_centroid(
            X,
            centroids,
            spherical=spherical,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            torch_device=torch_device,
        )
        times.append(time.perf_counter() - t0)
    per_vector_us = (np.mean(times) / max(n, 1)) * 1e6
    return {
        "assign_all_vectors_sec_mean": float(np.mean(times)),
        "assign_per_vector_us_mean": float(per_vector_us),
        "num_vectors": int(n),
        "repeats": repeats,
        "assign_backend": backend,
    }


def cluster_torch_spherical_kmeans(
    X: np.ndarray,
    num_clusters: int,
    seed: int,
    *,
    niter: int = 100,
    device: str = "auto",
    gpu_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Spherical K-means on GPU via PyTorch (fallback when faiss-gpu unavailable)."""
    torch_device = resolve_cluster_torch_device(device, gpu_id)
    g = torch.Generator(device=torch_device)
    g.manual_seed(seed)

    Xn = l2_normalize_rows(X.astype(np.float32))
    data = torch.from_numpy(Xn).to(torch_device)
    n = data.shape[0]
    perm = torch.randperm(n, generator=g, device=torch_device)
    centroids = F.normalize(data[perm[:num_clusters]].clone(), dim=1)

    t0 = time.perf_counter()
    for _ in range(niter):
        scores = data @ centroids.T
        labels_t = scores.argmax(dim=1)
        new_centroids = []
        for k in range(num_clusters):
            mask = labels_t == k
            if int(mask.sum()) == 0:
                idx = int(torch.randint(0, n, (1,), generator=g, device=torch_device).item())
                new_centroids.append(data[idx])
            else:
                new_centroids.append(F.normalize(data[mask].mean(dim=0), dim=0))
        centroids = torch.stack(new_centroids, dim=0)
    train_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    labels = assign_nearest_centroid_torch(
        Xn, centroids.detach().cpu().numpy(), spherical=True, device=torch_device
    )
    assign_sec = time.perf_counter() - t1
    centroids_np = centroids.detach().cpu().numpy().astype(np.float32)

    timing = {
        "fit_sec": float(train_sec),
        "assign_train_sec": float(assign_sec),
        "total_sec": float(train_sec + assign_sec),
        "niter": niter,
        "device": str(torch_device),
        "backend": "torch_spherical_kmeans",
    }
    timing.update(
        benchmark_assign_time(
            Xn,
            centroids_np,
            spherical=True,
            repeats=3,
            torch_device=torch_device,
        )
    )
    return centroids_np, labels, timing


ClusterFn = Callable[..., tuple[np.ndarray, np.ndarray, dict]]


DEFAULT_FAISS_MAX_POINTS_PER_CENTROID = 256


def cluster_faiss_spherical_kmeans(
    X: np.ndarray,
    num_clusters: int,
    seed: int,
    *,
    niter: int = 100,
    device: str = "cpu",
    gpu_id: int = 0,
    max_points_per_centroid: int = DEFAULT_FAISS_MAX_POINTS_PER_CENTROID,
) -> tuple[np.ndarray, np.ndarray, dict]:
    use_faiss_gpu, gpu_id = resolve_faiss_use_gpu(device, gpu_id)
    if device in ("cuda", "gpu") and not use_faiss_gpu:
        torch_dev = resolve_cluster_torch_device(device, gpu_id)
        if torch_dev.type == "cuda":
            print(
                f"  [cluster] faiss-gpu unavailable, using torch on {torch_dev} "
                "(full-data fit; not the official CPU subsampling protocol)",
                flush=True,
            )
            return cluster_torch_spherical_kmeans(
                X, num_clusters, seed, niter=niter, device=device, gpu_id=gpu_id
            )

    import faiss

    d = X.shape[1]
    n = len(X)
    Xn = l2_normalize_rows(X.astype(np.float32))
    fit_sample_cap = num_clusters * max_points_per_centroid
    t0 = time.perf_counter()
    kmeans_kwargs = {
        "niter": niter,
        "verbose": False,
        "spherical": True,
        "seed": seed,
        "gpu": use_faiss_gpu,
        "max_points_per_centroid": max_points_per_centroid,
    }
    if use_faiss_gpu:
        kmeans_kwargs["gpu_id"] = gpu_id
    kmeans = faiss.Kmeans(d, num_clusters, **kmeans_kwargs)
    kmeans.train(Xn)
    train_sec = time.perf_counter() - t0

    centroids = np.asarray(kmeans.centroids, dtype=np.float32)
    t1 = time.perf_counter()
    labels = assign_nearest_centroid_faiss(
        Xn, centroids, spherical=True, use_gpu=use_faiss_gpu, gpu_id=gpu_id
    )
    assign_sec = time.perf_counter() - t1

    timing = {
        "fit_sec": float(train_sec),
        "assign_train_sec": float(assign_sec),
        "total_sec": float(train_sec + assign_sec),
        "niter": niter,
        "device": "cuda" if use_faiss_gpu else "cpu",
        "backend": "faiss_spherical_kmeans",
        "gpu_id": int(gpu_id) if use_faiss_gpu else None,
        "max_points_per_centroid": int(max_points_per_centroid),
        "fit_sample_cap": int(fit_sample_cap),
        "num_fit_vectors": int(min(n, fit_sample_cap)),
        "num_assign_vectors": int(n),
        "fit_protocol": (
            f"FAISS subsampled centroid fit (<= {fit_sample_cap} vectors) "
            f"+ assign all {n} vectors"
        ),
    }
    timing.update(
        benchmark_assign_time(
            Xn,
            centroids,
            spherical=True,
            repeats=3,
            use_gpu=use_faiss_gpu,
            gpu_id=gpu_id,
        )
    )
    return centroids, labels, timing


CLUSTER_METHODS: dict[str, ClusterFn] = {
    "faiss_spherical_kmeans": cluster_faiss_spherical_kmeans,
    "torch_spherical_kmeans": cluster_torch_spherical_kmeans,
}


def default_embed_dir() -> Path:
    return DEFAULT_EMBED_DIR
