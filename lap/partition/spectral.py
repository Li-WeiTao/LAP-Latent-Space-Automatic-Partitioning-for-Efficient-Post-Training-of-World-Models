"""Architecture-neutral spectral-partition primitives used by LAP.

Dataset loading, frozen encoding, and predictor training belong to backends and
experiment adapters. This module operates only on a precomputed latent graph.
The implementation is kept numerically aligned with the completed TwoRoom runs.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def l2_normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(eps))


def build_self_tuned_graph(
    neighbors: np.ndarray,
    similarities: np.ndarray,
    k: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Construct the symmetric self-tuned affinity graph used by LAP."""

    if k < 1 or k > neighbors.shape[1]:
        raise ValueError("k must be supported by the supplied neighbor arrays")
    if neighbors.shape != similarities.shape:
        raise ValueError("neighbors and similarities must have identical shapes")
    nbr = np.asarray(neighbors[:, :k], dtype=np.int64)
    sim = np.clip(np.asarray(similarities[:, :k]), -1.0, 1.0)
    dist2 = np.maximum(2.0 - 2.0 * sim, 0.0)
    sigma = np.sqrt(dist2[:, -1])
    positive = sigma[sigma > 0]
    if not len(positive):
        raise RuntimeError("all landmark local scales are zero")
    sigma_floor = max(float(np.quantile(positive, 0.01)) * 0.1, 1e-4)
    sigma = np.maximum(sigma, sigma_floor)
    denominator = sigma[:, None] * sigma[nbr]
    exponent = np.clip(dist2 / denominator, 0.0, 50.0)
    weights = np.exp(-exponent).astype(np.float64)
    rows = np.repeat(np.arange(len(nbr), dtype=np.int64), k)
    graph = sparse.coo_matrix(
        (weights.reshape(-1), (rows, nbr.reshape(-1))),
        shape=(len(nbr), len(nbr)),
        dtype=np.float64,
    ).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    num_components, component_labels = connected_components(
        graph, directed=False, return_labels=True
    )
    component_counts = np.bincount(component_labels, minlength=num_components)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    if np.any(degree <= 0):
        raise RuntimeError("kNN graph contains zero-degree landmarks")
    return graph, {
        "effective_knn": int(k),
        "num_undirected_nonzeros": int(graph.nnz),
        "num_connected_components": int(num_components),
        "component_fractions": (component_counts / len(component_labels)).tolist(),
        "sigma_floor": sigma_floor,
        "degree_min": float(degree.min()),
        "degree_mean": float(degree.mean()),
        "degree_max": float(degree.max()),
    }


def select_k_from_laplacian_eigenvalues(
    laplacian_eigenvalues: np.ndarray,
    *,
    k_min: int,
    k_max: int,
) -> tuple[int, dict[str, Any]]:
    """Select K by the predeclared largest-eigengap rule."""

    eigenvalues = np.asarray(laplacian_eigenvalues, dtype=np.float64)
    if k_min < 2 or k_max < k_min:
        raise ValueError("auto-K requires 2 <= k_min <= k_max")
    if len(eigenvalues) <= k_max:
        raise ValueError(f"at least {k_max + 1} eigenvalues are required")
    candidates = list(range(k_min, k_max + 1))
    gaps = {k: float(eigenvalues[k] - eigenvalues[k - 1]) for k in candidates}
    selected = max(candidates, key=lambda value: (gaps[value], -value))
    return selected, {
        "mode": "auto_largest_eigengap",
        "rule": (
            "argmax_{K in [k_min,k_max]} (lambda_{K+1}-lambda_K) "
            "on the symmetric normalized graph Laplacian; ties choose smaller K"
        ),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "candidate_eigengaps": {str(k): gaps[k] for k in candidates},
        "selected_num_clusters": int(selected),
    }


def spectral_labels(
    graph: sparse.csr_matrix,
    *,
    num_clusters: int,
    seed: int,
    eig_tol: float = 1e-6,
    eig_maxiter: int = 10_000,
    spectral_n_init: int = 20,
    cpu_threads: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute normalized spectral coordinates and discrete region labels."""

    if num_clusters < 2:
        raise ValueError("num_clusters must be at least two")
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    if np.any(degree <= 0):
        raise ValueError("graph contains zero-degree nodes")
    inv_sqrt = 1.0 / np.sqrt(degree)
    normalized_adjacency = (
        sparse.diags(inv_sqrt) @ graph @ sparse.diags(inv_sqrt)
    )
    num_nodes = graph.shape[0]
    num_eigs = min(num_clusters + 2, num_nodes - 1)
    if num_eigs < num_clusters:
        raise ValueError("graph is too small for the requested number of clusters")
    initial = np.random.default_rng(seed).standard_normal(num_nodes)
    eig_start = time.perf_counter()
    values, vectors = eigsh(
        normalized_adjacency,
        k=num_eigs,
        which="LA",
        tol=eig_tol,
        maxiter=eig_maxiter,
        v0=initial,
    )
    eig_seconds = time.perf_counter() - eig_start
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    embedding = l2_normalize_rows(vectors[:, :num_clusters].astype(np.float32))
    cluster_start = time.perf_counter()
    with threadpool_limits(limits=cpu_threads):
        model = KMeans(
            n_clusters=num_clusters,
            init="k-means++",
            n_init=spectral_n_init,
            max_iter=300,
            random_state=seed,
            algorithm="lloyd",
        ).fit(embedding)
    laplacian_eigenvalues = 1.0 - values
    eigengap = None
    if len(laplacian_eigenvalues) > num_clusters:
        eigengap = float(
            laplacian_eigenvalues[num_clusters]
            - laplacian_eigenvalues[num_clusters - 1]
        )
    return (
        model.labels_.astype(np.int64),
        embedding,
        laplacian_eigenvalues,
        {
            "eigensolver_sec": eig_seconds,
            "spectral_kmeans_sec": time.perf_counter() - cluster_start,
            "normalized_adjacency_eigenvalues_desc": values.tolist(),
            "laplacian_eigenvalues_asc": laplacian_eigenvalues.tolist(),
            "eigengap_after_k": eigengap,
            "spectral_kmeans_inertia": float(model.inertia_),
            "spectral_kmeans_n_iter": int(model.n_iter_),
        },
    )
