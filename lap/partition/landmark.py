"""Lightweight, model-agnostic landmark spectral partitioner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from lap.routing.voronoi import VoronoiRouter

from .artifact import PartitionArtifact
from .base import PartitionResult
from .spectral import (
    build_self_tuned_graph,
    l2_normalize_rows,
    spectral_labels,
)

NeighborSearch = Callable[
    [np.ndarray, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]
]


def _zscore_l2(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.array(values, dtype=np.float32, copy=True)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    values -= mean
    values /= scale + np.float32(1e-6)
    return l2_normalize_rows(values), mean, scale


def _sample_landmarks(
    n: int,
    count: int,
    seed: int,
    group_ids: np.ndarray | None,
) -> np.ndarray:
    if count >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if group_ids is None:
        return np.sort(rng.choice(n, size=count, replace=False))
    group_ids = np.asarray(group_ids)
    unique, inverse = np.unique(group_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    for group in groups:
        rng.shuffle(group)
    selected: list[int] = []
    cursors = np.zeros(len(groups), dtype=np.int64)
    while len(selected) < count:
        active = np.flatnonzero(
            cursors < np.asarray([len(group) for group in groups])
        )
        rng.shuffle(active)
        for group_index in active[: count - len(selected)]:
            selected.append(int(groups[group_index][cursors[group_index]]))
            cursors[group_index] += 1
    return np.sort(np.asarray(selected, dtype=np.int64))


def exact_cosine_neighbors_cpu(
    values: np.ndarray, max_k: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Portable default; large experiments may inject a GPU implementation."""

    model = NearestNeighbors(
        n_neighbors=max_k + 1, metric="cosine", algorithm="brute"
    ).fit(values)
    distances, indices = model.kneighbors(values, return_distance=True)
    neighbors = np.empty((len(values), max_k), dtype=np.int64)
    similarities = np.empty((len(values), max_k), dtype=np.float32)
    for row in range(len(values)):
        keep = indices[row] != row
        neighbors[row] = indices[row][keep][:max_k]
        similarities[row] = 1.0 - distances[row][keep][:max_k]
    return neighbors, similarities, {"backend": "sklearn_exact_cosine_cpu"}


@dataclass(frozen=True)
class LandmarkSpectralConfig:
    num_regions: int = 3
    num_landmarks: int = 20_000
    knn: int = 30
    prototypes_per_region: int = 16
    seed: int = 0
    spectral_n_init: int = 20
    prototype_n_init: int = 5
    cpu_threads: int = 4
    eig_tol: float = 1e-4
    eig_maxiter: int = 10_000

    def validate(self, num_samples: int) -> None:
        if self.num_regions < 2:
            raise ValueError("num_regions must be at least two")
        if not 1 <= self.num_landmarks <= num_samples:
            raise ValueError("num_landmarks must be supported by the latent cache")
        if not 1 <= self.knn < self.num_landmarks:
            raise ValueError("knn must be smaller than num_landmarks")
        if self.prototypes_per_region < 1:
            raise ValueError("prototypes_per_region must be positive")


class LandmarkSpectralPartitioner:
    """Fit LAP's action-free spectral partition and deployable router."""

    def __init__(
        self,
        config: LandmarkSpectralConfig | None = None,
        *,
        neighbor_search: NeighborSearch | None = None,
    ) -> None:
        self.config = config or LandmarkSpectralConfig()
        self.neighbor_search = neighbor_search or exact_cosine_neighbors_cpu

    def fit(
        self,
        latents: np.ndarray,
        *,
        sample_ids: np.ndarray,
        group_ids: np.ndarray | None = None,
    ) -> PartitionResult:
        del sample_ids
        values = np.asarray(latents, dtype=np.float32)
        self.config.validate(len(values))
        transformed, mean, scale = _zscore_l2(values)
        landmark_indices = _sample_landmarks(
            len(values),
            self.config.num_landmarks,
            self.config.seed,
            group_ids,
        )
        landmarks = transformed[landmark_indices]
        neighbors, similarities, neighbor_meta = self.neighbor_search(
            landmarks, self.config.knn
        )
        graph, graph_meta = build_self_tuned_graph(
            neighbors, similarities, self.config.knn
        )
        landmark_labels, _, eigenvalues, spectral_meta = spectral_labels(
            graph,
            num_clusters=self.config.num_regions,
            seed=self.config.seed,
            eig_tol=self.config.eig_tol,
            eig_maxiter=self.config.eig_maxiter,
            spectral_n_init=self.config.spectral_n_init,
            cpu_threads=self.config.cpu_threads,
        )
        prototypes: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        with threadpool_limits(limits=self.config.cpu_threads):
            for region_id in range(self.config.num_regions):
                region = landmarks[landmark_labels == region_id]
                if len(region) < self.config.prototypes_per_region:
                    raise RuntimeError(
                        f"region {region_id} has too few landmarks for prototypes"
                    )
                model = KMeans(
                    n_clusters=self.config.prototypes_per_region,
                    init="k-means++",
                    n_init=self.config.prototype_n_init,
                    random_state=self.config.seed * 1000 + region_id,
                    algorithm="lloyd",
                ).fit(region)
                prototypes.append(l2_normalize_rows(model.cluster_centers_))
                owners.append(
                    np.full(
                        self.config.prototypes_per_region,
                        region_id,
                        dtype=np.int64,
                    )
                )
        metadata = {
            "algorithm": "landmark_spectral_voronoi",
            "num_clusters": self.config.num_regions,
            "seed": self.config.seed,
            "num_landmarks": len(landmark_indices),
            "knn": self.config.knn,
            "neighbor_search": neighbor_meta,
            "graph": graph_meta,
            "spectral": spectral_meta,
            "laplacian_eigenvalues": eigenvalues.tolist(),
            "routing": "zscore_l2_spherical_voronoi",
        }
        artifact = PartitionArtifact(
            prototypes=np.concatenate(prototypes).astype(np.float32),
            prototype_region_ids=np.concatenate(owners),
            mean=mean,
            scale=scale,
            metadata=metadata,
        )
        labels = VoronoiRouter(artifact).route(values).astype(np.int64)
        result = PartitionResult(artifact=artifact, labels=labels, metadata=metadata)
        result.validate(len(values))
        return result
