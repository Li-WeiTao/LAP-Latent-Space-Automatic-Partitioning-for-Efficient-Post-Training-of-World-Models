#!/usr/bin/env python3
"""Lightweight landmark spectral partition with prototype out-of-sample routing.

The expensive graph and eigendecomposition are built only on an episode-covered
landmark subset.  Spectral pseudo-labels are compressed into a small set of
spherical prototypes.  Both full-data predictor assignment and online routing
then use the exact same rule: nearest prototype -> prototype owner cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Keep CPU-side sparse algebra polite on the shared server.  The wrapper exports
# the same values before Python starts; these defaults protect direct invocations.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import scipy
import sklearn
import torch
import threadpoolctl
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import DATASETS  # noqa: E402
from geometry_latent_svm_rooms3 import episode_ids_at_indices  # noqa: E402
from latent_cluster_common import (  # noqa: E402
    ALL_TRAIN_CACHE_REGIONS,
    default_embed_dir,
    l2_normalize_rows,
    load_all_train_dedup_latent_vectors,
)
from latent_spherical_kmeans_lib import (  # noqa: E402
    cluster_torch_spherical_kmeans_converged,
)

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embed-dir", type=Path, default=default_embed_dir())
    p.add_argument(
        "--data-root", type=Path, default=Path("/data/sicong/weitao/datasets/lewm")
    )
    p.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Explicit dataset file; overrides --data-root/DATASET_DEFAULT_FILE.",
    )
    # v1 reuses the TwoRoom priority5 embedding caches.  The algorithm is
    # dataset-agnostic, but other datasets need their own cache-manifest adapter.
    p.add_argument("--dataset", default="tworoom", choices=("tworoom",))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=THIS_DIR / "results" / "latent_landmark_spectral_k3",
    )
    p.add_argument("--seeds", default="0", help="Comma-separated landmark seeds")
    p.add_argument("--num-clusters", type=int, default=3)
    p.add_argument(
        "--auto-k",
        action="store_true",
        help=(
            "Select K once per landmark graph by the largest normalized-Laplacian "
            "eigengap in the predeclared [--auto-k-min,--auto-k-max] range. "
            "Disabled by default so the established fixed-K=3 protocol is unchanged."
        ),
    )
    p.add_argument("--auto-k-min", type=int, default=2)
    p.add_argument("--auto-k-max", type=int, default=6)
    p.add_argument("--num-landmarks", type=int, default=20_000)
    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--knn-fallback", type=int, default=50)
    p.add_argument(
        "--knn-backend",
        choices=("auto", "torch_exact", "faiss_hnsw"),
        default="auto",
    )
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--query-chunk", type=int, default=2048)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--eig-tol", type=float, default=1e-4)
    p.add_argument("--eig-maxiter", type=int, default=10_000)
    p.add_argument("--spectral-n-init", type=int, default=20)
    p.add_argument("--prototypes-per-cluster", type=int, default=16)
    p.add_argument("--max-prototypes-per-cluster", type=int, default=32)
    p.add_argument("--prototype-n-init", type=int, default=5)
    p.add_argument("--prototype-max-iter", type=int, default=100)
    p.add_argument("--prototype-holdout-fraction", type=float, default=0.20)
    p.add_argument("--min-prototype-macro-f1", type=float, default=0.97)
    p.add_argument("--min-prototype-class-recall", type=float, default=0.95)
    p.add_argument("--min-full-cluster-fraction", type=float, default=0.02)
    p.add_argument("--assignment-chunk", type=int, default=65_536)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument(
        "--route-anchor",
        choices=("history_end", "transition_start"),
        default="transition_start",
        help=(
            "Training-label anchor for expert routing. TwoRoom online MPC receives "
            "one current observation (time axis length 1), so transition_start is "
            "the deployment-matched default."
        ),
    )
    p.add_argument(
        "--target-wall-sec",
        type=float,
        default=120.0,
        help="Reporting threshold only; the script never auto-kills a run.",
    )
    p.add_argument(
        "--input-cache-hash-mode",
        choices=("sha256", "metadata"),
        default="sha256",
        help=(
            "sha256 hashes every input embedding cache; metadata hashes only "
            "path/size/mtime and is faster but not content-addressed."
        ),
    )
    p.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help="Request PyTorch deterministic algorithms (off by default for v1 parity).",
    )
    p.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Explicitly allow replacing an artifact with the same config fingerprint.",
    )
    return p.parse_args()


def parse_seeds(text: str) -> list[int]:
    seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicates")
    return seeds


def json_ready(value: Any) -> Any:
    """Convert argparse/provenance values to canonical JSON-compatible objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(json_ready(value), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def configuration_fingerprint(value: Any, length: int = 12) -> str:
    if length < 8:
        raise ValueError("configuration fingerprint length must be at least 8")
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(canonical_json(list(contiguous.shape)).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def file_provenance(path: Path, *, hash_mode: str = "sha256") -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    row = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": None,
        "hash_mode": hash_mode,
    }
    if hash_mode == "sha256":
        row["sha256"] = sha256_file(path)
    row["record_fingerprint_sha256"] = hashlib.sha256(
        canonical_json(row).encode("utf-8")
    ).hexdigest()
    return row


def provenance_identity(row: dict) -> dict:
    """Stable identity fields; keep paths/mtimes only in the audit record."""
    if row.get("sha256"):
        return {
            "size_bytes": int(row["size_bytes"]),
            "sha256": row["sha256"],
        }
    return {
        "path": row.get("path"),
        "size_bytes": row.get("size_bytes"),
        "mtime_ns": row.get("mtime_ns"),
        "hash_mode": row.get("hash_mode"),
        "missing": row.get("missing", False),
    }


def collect_code_provenance() -> dict:
    candidates = {
        "generator": Path(__file__),
        "launcher": THIS_DIR / "scripts" / "run_latent_landmark_spectral.sh",
        "cluster_common": THIS_DIR / "latent_cluster_common.py",
        "spherical_kmeans": THIS_DIR / "latent_spherical_kmeans_lib.py",
        "geometry_indexing": THIS_DIR / "geometry_latent_svm_rooms3.py",
        "dataset_spec": THIS_DIR / "gauge_drift.py",
    }
    files: dict[str, dict] = {}
    for name, path in candidates.items():
        files[name] = (
            file_provenance(path, hash_mode="sha256")
            if path.exists()
            else {"path": str(path.resolve()), "missing": True, "sha256": None}
        )
    return {
        "files": files,
        "aggregate_sha256": hashlib.sha256(
            canonical_json(
                {name: provenance_identity(row) for name, row in files.items()}
            ).encode("utf-8")
        ).hexdigest(),
    }


def collect_input_cache_provenance(
    embed_dir: Path,
    cache_regions: list[str],
    *,
    hash_mode: str,
) -> dict:
    files: dict[str, dict] = {}
    for region in cache_regions:
        path = Path(embed_dir) / f"P_train_{region}_embeddings.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing input embedding cache for hashing: {path}")
        files[region] = file_provenance(path, hash_mode=hash_mode)
    return {
        "hash_mode": hash_mode,
        "files": files,
        "aggregate_sha256": hashlib.sha256(
            canonical_json(
                {name: provenance_identity(row) for name, row in files.items()}
            ).encode("utf-8")
        ).hexdigest(),
    }


def configure_determinism(requested: bool) -> None:
    if not requested:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _optional_package_version(
    *, import_names: tuple[str, ...], distribution_names: tuple[str, ...]
) -> str | None:
    for import_name in import_names:
        try:
            module = __import__(import_name)
            value = getattr(module, "__version__", None)
            if value is not None:
                return str(value)
        except Exception:
            pass
    for distribution_name in distribution_names:
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def environment_metadata() -> dict:
    cuda_available = bool(torch.cuda.is_available())
    gpu_names = []
    if cuda_available:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    cudnn = getattr(torch.backends, "cudnn", None)
    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "threadpoolctl_version": str(
            getattr(
                threadpoolctl,
                "__version__",
                _optional_package_version(
                    import_names=("threadpoolctl",),
                    distribution_names=("threadpoolctl",),
                ),
            )
        ),
        "faiss_version": _optional_package_version(
            import_names=("faiss",), distribution_names=("faiss-cpu", "faiss-gpu")
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": gpu_names,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": bool(getattr(cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(cudnn, "benchmark", False)),
        "cuda_matmul_allow_tf32": (
            bool(getattr(matmul_backend, "allow_tf32"))
            if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32")
            else None
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "seeded_components": {
            "landmark_numpy_rng": True,
            "eigsh_v0": True,
            "spectral_sklearn_kmeans": True,
            "prototype_torch_generators": True,
        },
        "backend_determinism_notes": {
            "torch_exact": (
                "seed-independent exact top-k; GPU reductions may not be bitwise "
                "portable unless deterministic algorithms are requested"
            ),
            "faiss_hnsw": (
                "approximate CPU fallback; exact cross-machine bitwise identity "
                "is not guaranteed"
            ),
        },
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=THIS_DIR, text=True
        ).strip()
    except Exception:
        return None


def peak_rss_mb() -> float:
    # Linux reports KiB; this experiment runs only on the Linux research server.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def zscore_l2_inplace(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute training-set Z-score parameters and transform one float32 copy."""
    mu = Z.mean(axis=0, dtype=np.float64).astype(np.float32)
    sigma = Z.std(axis=0, dtype=np.float64).astype(np.float32)
    Z -= mu
    Z /= sigma + np.float32(EPS)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z /= np.maximum(norms, np.float32(1e-12))
    return Z, mu, sigma


def episode_covered_sample(
    episode_ids: np.ndarray,
    num_landmarks: int,
    seed: int,
) -> np.ndarray:
    """Round-robin sample episodes so long episodes cannot dominate landmarks."""
    n = len(episode_ids)
    if num_landmarks < 1 or num_landmarks > n:
        raise ValueError(f"num_landmarks must be in [1,{n}], got {num_landmarks}")
    rng = np.random.default_rng(seed)
    unique_eps, inverse = np.unique(episode_ids, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.flatnonzero(np.r_[True, inverse[order][1:] != inverse[order][:-1], True])
    groups = [order[bounds[i] : bounds[i + 1]].copy() for i in range(len(bounds) - 1)]
    for group in groups:
        rng.shuffle(group)

    if num_landmarks < len(groups):
        chosen_groups = rng.choice(len(groups), size=num_landmarks, replace=False)
        selected = [groups[g][0] for g in chosen_groups]
        return np.sort(np.asarray(selected, dtype=np.int64))

    selected: list[int] = []
    cursor = np.zeros(len(groups), dtype=np.int64)
    while len(selected) < num_landmarks:
        active = np.flatnonzero(cursor < np.asarray([len(g) for g in groups]))
        if not len(active):
            break
        rng.shuffle(active)
        take = min(len(active), num_landmarks - len(selected))
        for group_id in active[:take]:
            selected.append(int(groups[group_id][cursor[group_id]]))
            cursor[group_id] += 1
    if len(selected) != num_landmarks:
        raise RuntimeError("episode-covered landmark sampler exhausted unexpectedly")
    return np.sort(np.asarray(selected, dtype=np.int64))


@torch.inference_mode()
def exact_cosine_knn_torch(
    X: np.ndarray,
    max_k: int,
    *,
    gpu_id: int,
    query_chunk: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("torch_exact kNN requires CUDA; use --knn-backend faiss_hnsw")
    device = torch.device(f"cuda:{gpu_id}")
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device.index)
    data = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    n = len(X)
    neighbors = np.empty((n, max_k), dtype=np.int64)
    similarities = np.empty((n, max_k), dtype=np.float32)
    t0 = time.perf_counter()
    for start in range(0, n, query_chunk):
        stop = min(start + query_chunk, n)
        scores = data[start:stop] @ data.T
        local = torch.arange(stop - start, device=device)
        scores[local, start + local] = -torch.inf
        vals, idx = torch.topk(scores, k=max_k, dim=1, largest=True, sorted=True)
        neighbors[start:stop] = idx.cpu().numpy()
        similarities[start:stop] = vals.cpu().numpy()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated(device.index) / (1024.0**2)
    del data
    torch.cuda.empty_cache()
    return neighbors, similarities, {
        "backend": "torch_cuda_exact_cosine",
        "device": str(device),
        "query_chunk": query_chunk,
        "sec": elapsed,
        "peak_cuda_allocated_mb": peak_mb,
    }


def cosine_knn_faiss_hnsw(
    X: np.ndarray,
    max_k: int,
    *,
    cpu_threads: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import faiss

    faiss.omp_set_num_threads(cpu_threads)
    X = np.asarray(X, dtype=np.float32)
    index = faiss.IndexHNSWFlat(X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = max(80, 2 * max_k)
    index.hnsw.efSearch = max(80, 2 * max_k)
    t0 = time.perf_counter()
    index.add(X)
    scores, raw_neighbors = index.search(X, max_k + 1)
    neighbors = np.empty((len(X), max_k), dtype=np.int64)
    similarities = np.empty((len(X), max_k), dtype=np.float32)
    for i in range(len(X)):
        keep = raw_neighbors[i] != i
        row_idx = raw_neighbors[i][keep][:max_k]
        row_scores = scores[i][keep][:max_k]
        if len(row_idx) != max_k:
            raise RuntimeError(f"HNSW returned fewer than {max_k} non-self neighbors")
        neighbors[i] = row_idx
        similarities[i] = row_scores
    return neighbors, similarities, {
        "backend": "faiss_cpu_hnsw_cosine",
        "device": "cpu",
        "hnsw_m": 32,
        "ef_construction": int(index.hnsw.efConstruction),
        "ef_search": int(index.hnsw.efSearch),
        "sec": time.perf_counter() - t0,
    }


def auto_select_num_clusters(
    W: sparse.csr_matrix,
    *,
    k_min: int,
    k_max: int,
    seed: int,
    eig_tol: float,
    eig_maxiter: int,
) -> tuple[int, dict]:
    """Lightweight auto-K diagnostic; the selected K is then fit normally."""
    degree = np.asarray(W.sum(axis=1)).reshape(-1)
    inv_sqrt = 1.0 / np.sqrt(degree)
    S = sparse.diags(inv_sqrt) @ W @ sparse.diags(inv_sqrt)
    num_nodes = W.shape[0]
    num_eigs = min(k_max + 2, num_nodes - 1)
    if num_eigs <= k_max:
        raise ValueError("Not enough landmarks for the requested auto-K range")
    v0 = np.random.default_rng(seed).standard_normal(num_nodes)
    t0 = time.perf_counter()
    values = eigsh(
        S,
        k=num_eigs,
        which="LA",
        tol=eig_tol,
        maxiter=eig_maxiter,
        v0=v0,
        return_eigenvectors=False,
    )
    values = np.sort(values)[::-1]
    laplacian = 1.0 - values
    selected, meta = select_k_from_laplacian_eigenvalues(
        laplacian, k_min=k_min, k_max=k_max
    )
    meta.update(
        {
            "diagnostic_eigensolver_sec": time.perf_counter() - t0,
            "normalized_adjacency_eigenvalues_desc": values.tolist(),
            "laplacian_eigenvalues_asc": laplacian.tolist(),
            "diagnostic_num_eigenpairs": int(num_eigs),
        }
    )
    return selected, meta


def episode_group_holdout(
    labels: np.ndarray,
    episode_ids: np.ndarray,
    *,
    num_clusters: int,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    for attempt in range(50):
        split_seed = seed * 1000 + 7919 + attempt
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_fraction, random_state=split_seed
        )
        train_idx, test_idx = next(
            splitter.split(np.zeros(len(labels)), labels, groups=episode_ids)
        )
        if (
            len(np.unique(labels[train_idx])) == num_clusters
            and len(np.unique(labels[test_idx])) == num_clusters
        ):
            return train_idx, test_idx, split_seed
    raise RuntimeError("Could not create episode-disjoint prototype holdout with all clusters")


def finite_float_or_none(value: Any) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def prototype_restart_record(restart: int, restart_seed: int, info: dict) -> dict:
    """Keep enough optimizer state to audit convergence and selected restarts."""
    return {
        "restart": int(restart),
        "seed": int(restart_seed),
        "objective_final": finite_float_or_none(info["objective_final"]),
        "converged": bool(info.get("converged", False)),
        "niter": int(info.get("niter", 0)),
        "max_iter": int(info.get("max_iter", 0)),
        "final_rel_obj_change": finite_float_or_none(
            info.get("final_rel_obj_change", float("nan"))
        ),
        "final_label_change_frac": finite_float_or_none(
            info.get("final_label_change_frac", float("nan"))
        ),
        "fit_sec": finite_float_or_none(info.get("fit_sec", float("nan"))),
    }


def fit_spherical_prototypes(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    num_clusters: int,
    prototypes_per_cluster: int,
    n_init: int,
    max_iter: int,
    seed: int,
    cpu_threads: int,
    gpu_id: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    prototype_parts: list[np.ndarray] = []
    owner_parts: list[np.ndarray] = []
    cluster_fit_records: list[dict] = []
    for cluster_id in range(num_clusters):
        cluster_X = X[labels == cluster_id]
        if len(cluster_X) < prototypes_per_cluster:
            raise RuntimeError(
                f"cluster{cluster_id} has {len(cluster_X)} landmarks, fewer than "
                f"P={prototypes_per_cluster}"
            )
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        cluster_tensor = torch.from_numpy(
            np.ascontiguousarray(cluster_X, dtype=np.float32)
        ).to(device)
        best_objective = -float("inf")
        best_centers: np.ndarray | None = None
        selected_restart: dict | None = None
        restart_records: list[dict] = []
        for restart in range(n_init):
            restart_seed = seed * 100_000 + cluster_id * 1_000 + restart
            centers_t, _labels_t, info = cluster_torch_spherical_kmeans_converged(
                cluster_tensor,
                num_clusters=prototypes_per_cluster,
                seed=restart_seed,
                max_iter=max_iter,
                rel_tol=1e-6,
                patience=5,
                init_mode="kmeanspp",
            )
            restart_record = prototype_restart_record(restart, restart_seed, info)
            restart_records.append(restart_record)
            if info["objective_final"] > best_objective:
                best_objective = float(info["objective_final"])
                best_centers = centers_t.detach().cpu().numpy().astype(np.float32)
                selected_restart = dict(restart_record)
        if best_centers is None:
            raise RuntimeError(f"No prototype fit completed for cluster{cluster_id}")
        if selected_restart is None:
            raise RuntimeError(f"No prototype restart selected for cluster{cluster_id}")
        centers = l2_normalize_rows(best_centers)
        prototype_parts.append(centers)
        owner_parts.append(
            np.full(prototypes_per_cluster, cluster_id, dtype=np.int64)
        )
        cluster_fit_records.append(
            {
                "cluster_id": int(cluster_id),
                "num_landmarks": int(len(cluster_X)),
                "selected_restart": selected_restart,
                "all_restarts": restart_records,
            }
        )
    fit_meta = {
        "num_clusters": int(num_clusters),
        "prototypes_per_cluster": int(prototypes_per_cluster),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "cluster_fits": cluster_fit_records,
        "all_selected_restarts_converged": bool(
            all(row["selected_restart"]["converged"] for row in cluster_fit_records)
        ),
    }
    return (
        np.concatenate(prototype_parts),
        np.concatenate(owner_parts),
        fit_meta,
    )


def assign_prototype_owners_numpy(
    X: np.ndarray,
    routing_prototypes: np.ndarray,
    prototype_cluster_ids: np.ndarray,
    *,
    chunk_size: int = 65_536,
) -> np.ndarray:
    prototypes = l2_normalize_rows(
        np.asarray(routing_prototypes, dtype=np.float32)
    )
    owners = np.asarray(prototype_cluster_ids, dtype=np.int64)
    out = np.empty(len(X), dtype=np.int64)
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        prototype_ids = (X[start:stop] @ prototypes.T).argmax(axis=1)
        out[start:stop] = owners[prototype_ids]
    return out


@torch.inference_mode()
def assign_prototype_owners(
    X: np.ndarray,
    routing_prototypes: np.ndarray,
    prototype_cluster_ids: np.ndarray,
    *,
    gpu_id: int,
    chunk_size: int,
) -> tuple[np.ndarray, dict]:
    if not torch.cuda.is_available():
        t0 = time.perf_counter()
        labels = assign_prototype_owners_numpy(
            X, routing_prototypes, prototype_cluster_ids, chunk_size=chunk_size
        )
        return labels, {"backend": "numpy_cpu", "sec": time.perf_counter() - t0}
    device = torch.device(f"cuda:{gpu_id}")
    prototypes = torch.from_numpy(
        l2_normalize_rows(np.asarray(routing_prototypes, dtype=np.float32))
    ).to(device)
    owners = torch.from_numpy(
        np.asarray(prototype_cluster_ids, dtype=np.int64)
    ).to(device)
    labels = np.empty(len(X), dtype=np.int64)
    t0 = time.perf_counter()
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        batch = torch.from_numpy(np.asarray(X[start:stop], dtype=np.float32)).to(device)
        prototype_ids = (batch @ prototypes.T).argmax(dim=1)
        labels[start:stop] = owners[prototype_ids].cpu().numpy()
    torch.cuda.synchronize(device)
    return labels, {
        "backend": "torch_cuda",
        "device": str(device),
        "sec": time.perf_counter() - t0,
    }


def fidelity_metrics(target: np.ndarray, predicted: np.ndarray, k: int) -> dict:
    recalls = recall_score(
        target,
        predicted,
        labels=np.arange(k),
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(target, predicted, average="macro")),
        "per_class_recall": recalls.tolist(),
        "min_class_recall": float(recalls.min()),
        "accuracy": float((target == predicted).mean()),
    }


def normalized_cut(W: sparse.csr_matrix, labels: np.ndarray, k: int) -> float:
    degree = np.asarray(W.sum(axis=1)).reshape(-1)
    value = 0.0
    for cluster_id in range(k):
        mask = labels == cluster_id
        volume = float(degree[mask].sum())
        internal = float(W[mask][:, mask].sum())
        value += (volume - internal) / max(volume, 1e-12)
    return float(value)


def benchmark_router(
    X: np.ndarray,
    prototypes: np.ndarray,
    owners: np.ndarray,
) -> dict:
    rows: dict[str, dict] = {}
    for batch_size in (1, 300, 15_000):
        n = min(batch_size, len(X))
        repeats = 100 if n == 1 else 10 if n == 300 else 3
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            assign_prototype_owners_numpy(
                X[:n], prototypes, owners, chunk_size=max(n, 1)
            )
            times.append(time.perf_counter() - t0)
        rows[str(batch_size)] = {
            "actual_batch_size": n,
            "repeats": repeats,
            "mean_ms": float(np.mean(times) * 1000.0),
            "mean_us_per_vector": float(np.mean(times) / n * 1e6),
        }
    return rows


# All runs below resolve these globals at execution time.  Point them at the
# architecture-neutral LAP implementation so this TwoRoom program is an
# adapter/configuration layer, not a second spectral algorithm implementation.
from lap.partition.spectral import (  # noqa: E402
    build_self_tuned_graph as _lap_build_self_tuned_graph,
    l2_normalize_rows as _lap_l2_normalize_rows,
    select_k_from_laplacian_eigenvalues as _lap_select_k,
    spectral_labels as _lap_spectral_labels,
)

l2_normalize_rows = _lap_l2_normalize_rows
build_self_tuned_graph = _lap_build_self_tuned_graph
select_k_from_laplacian_eigenvalues = _lap_select_k
spectral_labels = _lap_spectral_labels


def result_affecting_args(args: argparse.Namespace) -> dict:
    """Exclude reporting/output controls while retaining all numerical controls."""
    row = json_ready(vars(args))
    for key in (
        "out_dir",
        "embed_dir",
        "data_root",
        "seeds",
        "target_wall_sec",
        "overwrite_existing",
        "input_cache_hash_mode",
    ):
        row.pop(key, None)
    return row


def artifact_directory_name(
    *,
    fingerprint: str,
    num_clusters: int,
    num_landmarks: int,
    effective_knn: int,
    prototypes_per_cluster: int,
    seed: int,
) -> str:
    return (
        f"spectral_cfg{fingerprint}_K{num_clusters}_M{num_landmarks}_"
        f"k{effective_knn}_P{prototypes_per_cluster}_seed{seed}"
    )


def _run_seed_impl(
    X: np.ndarray,
    global_idx: np.ndarray,
    episode_ids: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    data_stats: dict,
    shared_load_sec: float,
    provenance: dict,
    args_record: dict,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Path, np.ndarray, dict]:
    seed_t0 = time.perf_counter()
    timing: dict[str, float | dict] = {}
    landmark_rows = episode_covered_sample(episode_ids, args.num_landmarks, seed)
    X_landmark = np.ascontiguousarray(X[landmark_rows], dtype=np.float32)
    landmark_eps = episode_ids[landmark_rows]
    landmark_global_idx = global_idx[landmark_rows]
    timing["landmark_sampling_sec"] = time.perf_counter() - seed_t0

    max_k = max(args.knn, args.knn_fallback)
    if max_k >= len(X_landmark):
        raise ValueError("kNN fallback must be smaller than number of landmarks")
    backend = args.knn_backend
    if backend == "auto":
        backend = "torch_exact" if torch.cuda.is_available() else "faiss_hnsw"
    if backend == "torch_exact":
        neighbors, similarities, knn_meta = exact_cosine_knn_torch(
            X_landmark,
            max_k,
            gpu_id=args.gpu_id,
            query_chunk=args.query_chunk,
        )
    else:
        neighbors, similarities, knn_meta = cosine_knn_faiss_hnsw(
            X_landmark, max_k, cpu_threads=args.cpu_threads
        )
    timing["knn"] = knn_meta

    graph_t0 = time.perf_counter()
    W, graph_meta = build_self_tuned_graph(neighbors, similarities, args.knn)
    max_supported_clusters = args.auto_k_max if args.auto_k else args.num_clusters
    if (
        graph_meta["num_connected_components"] > max_supported_clusters
        and args.knn_fallback > args.knn
    ):
        W, graph_meta = build_self_tuned_graph(
            neighbors, similarities, args.knn_fallback
        )
    if graph_meta["num_connected_components"] > max_supported_clusters:
        raise RuntimeError(
            "kNN graph has more connected components than requested clusters: "
            f"{graph_meta['num_connected_components']} > {max_supported_clusters}"
        )
    timing["graph_build_sec"] = time.perf_counter() - graph_t0

    if args.auto_k:
        effective_k_min = max(
            args.auto_k_min, int(graph_meta["num_connected_components"])
        )
        effective_num_clusters, cluster_count_selection = auto_select_num_clusters(
            W,
            k_min=effective_k_min,
            k_max=args.auto_k_max,
            seed=seed,
            eig_tol=args.eig_tol,
            eig_maxiter=args.eig_maxiter,
        )
        cluster_count_selection["requested_k_min"] = int(args.auto_k_min)
        cluster_count_selection["connected_component_lower_bound"] = int(
            graph_meta["num_connected_components"]
        )
    else:
        effective_num_clusters = args.num_clusters
        cluster_count_selection = {
            "mode": "fixed",
            "requested_num_clusters": int(args.num_clusters),
            "selected_num_clusters": int(args.num_clusters),
        }

    spectral, spectral_embedding, lap_eigs, spectral_meta = spectral_labels(
        W,
        num_clusters=effective_num_clusters,
        seed=seed,
        eig_tol=args.eig_tol,
        eig_maxiter=args.eig_maxiter,
        spectral_n_init=args.spectral_n_init,
        cpu_threads=args.cpu_threads,
    )
    spectral_meta["normalized_cut"] = normalized_cut(
        W, spectral, effective_num_clusters
    )
    spectral_meta["landmark_cluster_counts"] = np.bincount(
        spectral, minlength=effective_num_clusters
    ).tolist()
    timing["spectral"] = spectral_meta

    train_rows, holdout_rows, split_seed = episode_group_holdout(
        spectral,
        landmark_eps,
        num_clusters=effective_num_clusters,
        test_fraction=args.prototype_holdout_fraction,
        seed=seed,
    )
    prototype_candidates = [args.prototypes_per_cluster]
    if args.max_prototypes_per_cluster > args.prototypes_per_cluster:
        prototype_candidates.append(args.max_prototypes_per_cluster)
    fidelity_attempts: list[dict] = []
    selected_p: int | None = None
    proto_t0 = time.perf_counter()
    for p in prototype_candidates:
        trial_prototypes, trial_owners, trial_fit_meta = fit_spherical_prototypes(
            X_landmark[train_rows],
            spectral[train_rows],
            num_clusters=effective_num_clusters,
            prototypes_per_cluster=p,
            n_init=args.prototype_n_init,
            max_iter=args.prototype_max_iter,
            seed=seed,
            cpu_threads=args.cpu_threads,
            gpu_id=args.gpu_id,
        )
        holdout_pred = assign_prototype_owners_numpy(
            X_landmark[holdout_rows], trial_prototypes, trial_owners
        )
        fidelity = fidelity_metrics(
            spectral[holdout_rows], holdout_pred, effective_num_clusters
        )
        fidelity["prototypes_per_cluster"] = p
        fidelity["prototype_fit"] = trial_fit_meta
        fidelity_attempts.append(fidelity)
        if (
            fidelity["macro_f1"] >= args.min_prototype_macro_f1
            and fidelity["min_class_recall"] >= args.min_prototype_class_recall
        ):
            selected_p = p
            break
    if selected_p is None:
        raise RuntimeError(
            "Prototype out-of-sample extension failed holdout fidelity thresholds: "
            f"{fidelity_attempts}"
        )
    (
        routing_prototypes,
        prototype_cluster_ids,
        final_prototype_fit,
    ) = fit_spherical_prototypes(
        X_landmark,
        spectral,
        num_clusters=effective_num_clusters,
        prototypes_per_cluster=selected_p,
        n_init=args.prototype_n_init,
        max_iter=args.prototype_max_iter,
        seed=seed,
        cpu_threads=args.cpu_threads,
        gpu_id=args.gpu_id,
    )
    timing["prototype_fit_and_validation_sec"] = time.perf_counter() - proto_t0

    full_labels, assignment_meta = assign_prototype_owners(
        X,
        routing_prototypes,
        prototype_cluster_ids,
        gpu_id=args.gpu_id,
        chunk_size=args.assignment_chunk,
    )
    timing["full_assignment"] = assignment_meta
    full_counts = np.bincount(full_labels, minlength=effective_num_clusters)
    full_fractions = full_counts / len(full_labels)
    if float(full_fractions.min()) < args.min_full_cluster_fraction:
        raise RuntimeError(
            f"Degenerate full partition: cluster fractions={full_fractions.tolist()}"
        )

    landmark_router_labels = assign_prototype_owners_numpy(
        X_landmark, routing_prototypes, prototype_cluster_ids
    )
    all_landmark_fidelity = fidelity_metrics(
        spectral, landmark_router_labels, effective_num_clusters
    )
    diagnostic_centroids = np.stack(
        [
            l2_normalize_rows(X[full_labels == k].mean(axis=0, keepdims=True))[0]
            for k in range(effective_num_clusters)
        ]
    ).astype(np.float32)

    fingerprint_payload = {
        "schema": "latent_landmark_spectral_config_v2",
        "result_affecting_args": result_affecting_args(args),
        "seed": int(seed),
        "effective_num_clusters": int(effective_num_clusters),
        "effective_knn": int(graph_meta["effective_knn"]),
        "selected_prototypes_per_cluster": int(selected_p),
        "actual_knn_backend": knn_meta["backend"],
        "code_aggregate_sha256": provenance["code"]["aggregate_sha256"],
        "input_cache_aggregate_sha256": provenance["input_caches"][
            "aggregate_sha256"
        ],
        "global_idx_sha256": provenance["semantic_inputs"]["global_idx_sha256"],
        "episode_ids_sha256": provenance["semantic_inputs"]["episode_ids_sha256"],
        "preprocessed_latent_sha256": provenance["semantic_inputs"][
            "preprocessed_latent_sha256"
        ],
        "zscore_mu_sha256": provenance["semantic_inputs"]["zscore_mu_sha256"],
        "zscore_sigma_sha256": provenance["semantic_inputs"][
            "zscore_sigma_sha256"
        ],
    }
    config_fingerprint = configuration_fingerprint(fingerprint_payload)
    artifact_dir = args.out_dir / artifact_directory_name(
        fingerprint=config_fingerprint,
        num_clusters=effective_num_clusters,
        num_landmarks=args.num_landmarks,
        effective_knn=graph_meta["effective_knn"],
        prototypes_per_cluster=selected_p,
        seed=seed,
    )
    # Serialize writers for the same fingerprint and write into a hidden
    # staging directory.  The completed artifact becomes visible via one
    # atomic directory rename, so readers never observe interleaved arrays.
    import fcntl

    lock_path = args.out_dir / f".{artifact_dir.name}.write.lock"
    lock_handle = lock_path.open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(
            f"Another process is writing the same artifact: {artifact_dir}"
        ) from exc
    if artifact_dir.exists() and not args.overwrite_existing:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        raise FileExistsError(
            f"Artifact already exists: {artifact_dir}. Use --overwrite-existing "
            "only for an intentional same-config replacement."
        )
    write_dir = args.out_dir / f".{artifact_dir.name}.pid{os.getpid()}.tmp"
    write_dir.mkdir(parents=False, exist_ok=False)
    save_t0 = time.perf_counter()
    np.save(write_dir / "centroids.npy", diagnostic_centroids)
    np.save(write_dir / "routing_prototypes.npy", routing_prototypes.astype(np.float32))
    np.save(
        write_dir / "prototype_cluster_ids.npy",
        prototype_cluster_ids.astype(np.int64),
    )
    np.savez_compressed(
        write_dir / "cluster_labels.npz",
        global_idx=global_idx.astype(np.int64),
        labels=full_labels.astype(np.int64),
    )
    np.savez_compressed(
        write_dir / "zscore_params.npz",
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
        eps=np.float32(EPS),
    )
    np.savez_compressed(
        write_dir / "landmark_diagnostics.npz",
        landmark_rows=landmark_rows,
        landmark_global_idx=landmark_global_idx,
        landmark_episode_ids=landmark_eps,
        landmark_spectral_labels=spectral,
        landmark_router_labels=landmark_router_labels,
        spectral_embedding=spectral_embedding,
        laplacian_eigenvalues=lap_eigs,
    )
    timing["artifact_arrays_save_sec"] = time.perf_counter() - save_t0
    output_hash_t0 = time.perf_counter()
    output_file_sha256: dict[str, str] = {}
    output_file_size_bytes: dict[str, int] = {}
    for filename in (
        "centroids.npy",
        "routing_prototypes.npy",
        "prototype_cluster_ids.npy",
        "cluster_labels.npz",
        "zscore_params.npz",
        "landmark_diagnostics.npz",
    ):
        output_path = write_dir / filename
        output_file_sha256[filename] = sha256_file(output_path)
        output_file_size_bytes[filename] = int(output_path.stat().st_size)
    timing["artifact_output_hash_sec"] = time.perf_counter() - output_hash_t0

    transition_label_offset = (
        (args.history_size - 1) * args.frameskip
        if args.route_anchor == "history_end"
        else 0
    )
    router_benchmark = benchmark_router(
        X, routing_prototypes, prototype_cluster_ids
    )
    partition_sec = time.perf_counter() - seed_t0
    end_to_end_sec = shared_load_sec + partition_sec
    timing["partition_seed_sec"] = partition_sec
    timing["shared_load_preprocess_sec"] = shared_load_sec
    timing["end_to_end_sec"] = end_to_end_sec
    labels_hash = hashlib.sha256(full_labels.tobytes()).hexdigest()
    meta = {
        "version_label": "latent_landmark_spectral_v2",
        "assignment_schema_version": 2,
        "method": "zscore_l2_landmark_spectral_prototype",
        "description": "landmark spectral partition with prototype out-of-sample extension",
        "seed": seed,
        "num_clusters": effective_num_clusters,
        "cluster_count_selection": cluster_count_selection,
        "num_landmarks": args.num_landmarks,
        "latent_dim": int(X.shape[1]),
        "spherical": True,
        "classification_rule": (
            "zscore and L2-normalize latent; choose maximum-cosine routing "
            "prototype; route to that prototype's owner cluster"
        ),
        "num_routing_prototypes": int(len(routing_prototypes)),
        "prototypes_per_cluster": selected_p,
        "prototype_fit": {
            "algorithm": "full-landmark spherical K-means++",
            "n_init": args.prototype_n_init,
            "max_iter": args.prototype_max_iter,
            "rel_tol": 1e-6,
            "patience": 5,
            "final_fit": final_prototype_fit,
        },
        "prototype_holdout": {
            "episode_disjoint": True,
            "fraction": args.prototype_holdout_fraction,
            "split_seed": split_seed,
            "num_train": int(len(train_rows)),
            "num_holdout": int(len(holdout_rows)),
            "threshold_macro_f1": args.min_prototype_macro_f1,
            "threshold_min_class_recall": args.min_prototype_class_recall,
            "attempts": fidelity_attempts,
        },
        "all_landmark_router_fidelity": all_landmark_fidelity,
        "graph": graph_meta,
        "knn": knn_meta,
        "spectral": spectral_meta,
        "full_cluster_counts": full_counts.tolist(),
        "full_cluster_fractions": full_fractions.tolist(),
        "min_full_cluster_fraction_threshold": args.min_full_cluster_fraction,
        "route_anchor": args.route_anchor,
        "history_size": args.history_size,
        "frameskip": args.frameskip,
        "transition_label_offset_steps": transition_label_offset,
        "preprocess": "zscore_l2",
        "zscore_eps": EPS,
        "full_labels_sha256": labels_hash,
        "output_file_sha256": output_file_sha256,
        "output_file_size_bytes": output_file_size_bytes,
        "config_fingerprint": config_fingerprint,
        "config_fingerprint_payload": fingerprint_payload,
        "args": args_record,
        "provenance": provenance,
        "train_data_stats": data_stats,
        "timing_sec": timing,
        "router_microbenchmark_cpu": router_benchmark,
        "target_wall_sec": args.target_wall_sec,
        "within_target_wall_sec": end_to_end_sec <= args.target_wall_sec,
        "peak_process_rss_mb": peak_rss_mb(),
        "git_commit": git_commit(),
    }
    atomic_write_json(write_dir / "cluster_meta.json", meta)
    if artifact_dir.exists():
        backup_dir = args.out_dir / f".{artifact_dir.name}.previous.{os.getpid()}"
        os.replace(artifact_dir, backup_dir)
        try:
            os.replace(write_dir, artifact_dir)
        except Exception:
            os.replace(backup_dir, artifact_dir)
            raise
        else:
            shutil.rmtree(backup_dir)
    else:
        os.replace(write_dir, artifact_dir)
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()
    print(
        f"[seed {seed}] saved {artifact_dir} | e2e={end_to_end_sec:.2f}s "
        f"(shared={shared_load_sec:.2f}s, partition={partition_sec:.2f}s) | "
        f"P={selected_p} | fractions={np.round(full_fractions, 4).tolist()} | "
        f"holdout Macro-F1={fidelity_attempts[-1]['macro_f1']:.4f}",
        flush=True,
    )
    return artifact_dir, full_labels, meta


def _cleanup_failed_artifact_transaction(out_dir: Path) -> None:
    """Best-effort cleanup after a failed per-seed artifact transaction.

    The implementation frame owns the advisory-lock file handle.  The wrapper
    clears that finished frame before calling this helper, which closes the
    handle and releases the OS lock even when this module is used as a library
    and the caller catches the exception instead of exiting the process.
    """

    pid = os.getpid()
    for write_dir in out_dir.glob(f".*.pid{pid}.tmp"):
        shutil.rmtree(write_dir, ignore_errors=True)
    for backup_dir in out_dir.glob(f".*.previous.{pid}"):
        marker = f".previous.{pid}"
        artifact_name = backup_dir.name[1 : -len(marker)]
        artifact_dir = out_dir / artifact_name
        try:
            if artifact_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            else:
                os.replace(backup_dir, artifact_dir)
        except OSError:
            # Preserve the original exception; the backup remains recoverable.
            pass


def run_seed(
    X: np.ndarray,
    global_idx: np.ndarray,
    episode_ids: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    data_stats: dict,
    shared_load_sec: float,
    provenance: dict,
    args_record: dict,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Path, np.ndarray, dict]:
    """Run one seed and leave no partial artifact if persistence fails."""

    try:
        return _run_seed_impl(
            X,
            global_idx,
            episode_ids,
            mu,
            sigma,
            data_stats,
            shared_load_sec,
            provenance,
            args_record,
            args,
            seed,
        )
    except BaseException as exc:
        # Clear locals in completed implementation frames so file handles are
        # closed and flock is released before removing/restoring directories.
        tb = exc.__traceback__
        current_frame = sys._getframe()
        while tb is not None:
            if tb.tb_frame is not current_frame:
                try:
                    tb.tb_frame.clear()
                except RuntimeError:
                    pass
            tb = tb.tb_next
        _cleanup_failed_artifact_transaction(args.out_dir)
        raise


def main() -> None:
    args = parse_args()
    if args.num_clusters < 2:
        raise ValueError("--num-clusters must be at least 2")
    if args.auto_k and (
        args.auto_k_min < 2 or args.auto_k_max < args.auto_k_min
    ):
        raise ValueError("auto-K requires 2 <= --auto-k-min <= --auto-k-max")
    if args.knn < 1 or args.knn_fallback < args.knn:
        raise ValueError("Require 1 <= knn <= knn-fallback")
    if args.max_prototypes_per_cluster < args.prototypes_per_cluster:
        raise ValueError("max prototypes must be >= initial prototypes")
    seeds = parse_seeds(args.seeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Must run before the first CUDA API call.
    configure_determinism(args.deterministic_algorithms)

    args_record = json_ready(vars(args))
    args_record["embed_dir"] = str(args.embed_dir.expanduser().resolve())
    args_record["data_root"] = str(args.data_root.expanduser().resolve())
    args_record["data_file"] = (
        str(args.data_file.expanduser().resolve()) if args.data_file else None
    )

    shared_t0 = time.perf_counter()
    provenance_t0 = time.perf_counter()
    code_provenance = collect_code_provenance()
    input_cache_provenance = collect_input_cache_provenance(
        args.embed_dir,
        list(ALL_TRAIN_CACHE_REGIONS),
        hash_mode=args.input_cache_hash_mode,
    )
    provenance_scan_sec = time.perf_counter() - provenance_t0
    try:
        import faiss

        faiss.omp_set_num_threads(args.cpu_threads)
    except Exception:
        if args.knn_backend == "faiss_hnsw":
            raise

    load_t0 = time.perf_counter()
    Z, global_idx, data_stats = load_all_train_dedup_latent_vectors(
        args.embed_dir, frameskip=args.frameskip
    )
    X, mu, sigma = zscore_l2_inplace(np.asarray(Z, dtype=np.float32))
    spec = DATASETS[args.dataset]
    h5_path = args.data_file or (args.data_root / spec.default_file)
    episode_ids = episode_ids_at_indices(h5_path, global_idx)
    data_load_preprocess_sec = time.perf_counter() - load_t0
    semantic_t0 = time.perf_counter()
    semantic_inputs = {
        "preprocessed_latent_sha256": sha256_array(X),
        "global_idx_sha256": sha256_array(global_idx),
        "episode_ids_sha256": sha256_array(episode_ids),
        "zscore_mu_sha256": sha256_array(mu),
        "zscore_sigma_sha256": sha256_array(sigma),
    }
    semantic_hash_sec = time.perf_counter() - semantic_t0
    provenance = {
        "code": code_provenance,
        "input_caches": input_cache_provenance,
        "dataset_file": file_provenance(h5_path, hash_mode="metadata"),
        "semantic_inputs": semantic_inputs,
        "environment": environment_metadata(),
        "timing_sec": {
            "code_and_input_cache_scan": provenance_scan_sec,
            "data_load_preprocess": data_load_preprocess_sec,
            "semantic_array_hash": semantic_hash_sec,
        },
    }
    shared_load_sec = time.perf_counter() - shared_t0
    data_stats["provenance_timing_sec"] = provenance["timing_sec"]
    print(
        f"[data] N={len(X)} D={X.shape[1]} episodes={len(np.unique(episode_ids))} "
        f"shared setup={shared_load_sec:.2f}s "
        f"(load+preprocess={data_load_preprocess_sec:.2f}s, "
        f"provenance={provenance_scan_sec + semantic_hash_sec:.2f}s)",
        flush=True,
    )

    artifacts: list[Path] = []
    labels_by_seed: dict[int, np.ndarray] = {}
    meta_by_seed: dict[int, dict] = {}
    for seed in seeds:
        artifact_dir, labels, meta = run_seed(
            X,
            global_idx,
            episode_ids,
            mu,
            sigma,
            data_stats,
            shared_load_sec,
            provenance,
            args_record,
            args,
            seed,
        )
        artifacts.append(artifact_dir)
        labels_by_seed[seed] = labels
        meta_by_seed[seed] = meta

    pairwise: list[dict] = []
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            a = labels_by_seed[seed_a]
            b = labels_by_seed[seed_b]
            pairwise.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "ari": float(adjusted_rand_score(a, b)),
                    "nmi": float(normalized_mutual_info_score(a, b)),
                }
            )
    summary_fingerprint = configuration_fingerprint(
        {
            "schema": "latent_landmark_spectral_stability_v2",
            "seeds": seeds,
            "artifact_config_fingerprints": {
                str(seed): meta_by_seed[seed]["config_fingerprint"] for seed in seeds
            },
            "code_aggregate_sha256": code_provenance["aggregate_sha256"],
            "input_cache_aggregate_sha256": input_cache_provenance[
                "aggregate_sha256"
            ],
        }
    )
    summary_path = args.out_dir / f"stability_summary_cfg{summary_fingerprint}.json"
    latest_summary_path = args.out_dir / "stability_summary.json"
    stability = {
        "version_label": "latent_landmark_spectral_v2",
        "summary_fingerprint": summary_fingerprint,
        "fingerprinted_summary_path": str(summary_path.resolve()),
        "latest_summary_alias": str(latest_summary_path.resolve()),
        "seeds": seeds,
        "artifacts": [str(p) for p in artifacts],
        "artifacts_by_seed": {
            str(seed): str(artifacts[i].resolve()) for i, seed in enumerate(seeds)
        },
        "effective_num_clusters_by_seed": {
            str(seed): int(meta_by_seed[seed]["num_clusters"]) for seed in seeds
        },
        "artifact_config_fingerprints": {
            str(seed): meta_by_seed[seed]["config_fingerprint"] for seed in seeds
        },
        "args": args_record,
        "provenance": provenance,
        "shared_load_preprocess_sec": shared_load_sec,
        "pairwise": pairwise,
        "pairwise_ari_mean": float(np.mean([r["ari"] for r in pairwise]))
        if pairwise
        else None,
        "pairwise_nmi_mean": float(np.mean([r["nmi"] for r in pairwise]))
        if pairwise
        else None,
        "kmeanspp_R50_reference_pairwise_ari": 0.483,
        "predeclared_interpretation": {
            "below_0p433": "materially less stable than K-means++ R50",
            "at_least_0p50": "at least modest stability improvement",
            "at_least_0p60": "clear stability improvement",
        },
        "peak_process_rss_mb": peak_rss_mb(),
    }
    atomic_write_json(summary_path, stability)
    # Convenience alias only; fingerprinted summaries are never lost when a
    # different configuration becomes latest.
    atomic_write_json(latest_summary_path, stability)
    print(f"[done] summary -> {summary_path} (latest alias updated)", flush=True)


if __name__ == "__main__":
    main()
