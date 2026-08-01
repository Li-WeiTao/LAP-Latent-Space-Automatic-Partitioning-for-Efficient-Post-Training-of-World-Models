"""Empirical spectral-degeneracy gate and automatic LAP partitioner.

The gate implements Equations (18)--(22) and Appendix A.12 of the LAP
paper.  It uses only frozen current-state latents and optional trajectory
group identifiers; actions, rewards, goals, and task labels are never read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .artifact import PartitionArtifact
from .base import PartitionResult
from .landmark import (
    LandmarkSpectralConfig,
    LandmarkSpectralPartitioner,
    NeighborSearch,
    _sample_landmarks,
    _zscore_l2,
    exact_cosine_neighbors_cpu,
)
from .spectral import build_self_tuned_graph


@dataclass(frozen=True)
class SpectralGateConfig:
    """Predeclared label-free diagnostic suite for one candidate K."""

    num_regions: int = 3
    num_landmarks: int = 20_000
    nominal_knn: int = 30
    perturb_knn: tuple[int, ...] = (27, 33)
    diagnostic_seeds: tuple[int, ...] = (0, 1, 2)
    deployment_seed: int = 0
    perturbation_multiplier: float = 2.0
    retention_threshold: float = 0.5
    background_gap_count: int = 10
    background_mad_multiplier: float = 3.0
    epsilon: float = 1e-8
    eig_tol: float = 1e-4
    eig_maxiter: int = 20_000
    cpu_threads: int = 4

    def validate(self, num_samples: int | None = None) -> None:
        if self.num_regions < 2:
            raise ValueError("num_regions must be at least two")
        if len(self.diagnostic_seeds) < 3:
            raise ValueError("at least three predeclared diagnostic seeds are required")
        if len(set(self.diagnostic_seeds)) != len(self.diagnostic_seeds):
            raise ValueError("diagnostic seeds must be unique")
        if self.deployment_seed not in self.diagnostic_seeds:
            raise ValueError("deployment_seed must be one of the diagnostic seeds")
        graph_ks = (self.nominal_knn, *self.perturb_knn)
        if len(set(graph_ks)) != len(graph_ks):
            raise ValueError("nominal and perturbed kNN values must be unique")
        if min(graph_ks) < 1 or max(graph_ks) >= self.num_landmarks:
            raise ValueError("all kNN values must lie in [1, num_landmarks)")
        if num_samples is not None and not 1 <= self.num_landmarks <= num_samples:
            raise ValueError("num_landmarks must be supported by the latent cache")
        if self.perturbation_multiplier <= 0:
            raise ValueError("perturbation_multiplier must be positive")
        if not 0 < self.retention_threshold < 1:
            raise ValueError("retention_threshold must lie in (0, 1)")
        if self.background_gap_count < 1:
            raise ValueError("background_gap_count must be positive")
        if self.background_mad_multiplier < 0:
            raise ValueError("background_mad_multiplier must be nonnegative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")

    @property
    def graph_knn_values(self) -> tuple[int, ...]:
        return tuple(sorted((self.nominal_knn, *self.perturb_knn)))

    @property
    def required_eigenvalues(self) -> int:
        return self.num_regions + self.background_gap_count + 1


@dataclass(frozen=True)
class SpectralGateResult:
    """Serializable result of the task-level worst-case diagnostic."""

    use_partition: bool
    selected_method: str
    reason: str
    deployment_seed: int
    candidate_gap_min: float
    perturbation_threshold_max: float
    retained_safety_fraction: float | None
    robust_residual_gap: float | None
    background_threshold: float
    draw_results: tuple[dict[str, Any], ...]
    configuration: dict[str, Any]
    elapsed_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def relative_eigengap(
    eigenvalues: np.ndarray, index: int, epsilon: float
) -> float:
    """Return (lambda[index+1]-lambda[index])/(lambda[index+1]+eps).

    ``index`` is one-based mathematical K: index=K compares lambda_K and
    lambda_{K+1}; NumPy positions are therefore index-1 and index.
    """

    values = np.sort(np.asarray(eigenvalues, dtype=np.float64))
    if index < 1 or len(values) <= index:
        raise ValueError(f"at least {index + 1} eigenvalues are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be finite")
    denominator = float(values[index]) + epsilon
    if denominator <= 0:
        raise ValueError("relative eigengap denominator must be positive")
    return float((values[index] - values[index - 1]) / denominator)


def scaled_mad(values: np.ndarray) -> float:
    """Robust 1.4826-scaled median absolute deviation."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("scaled_mad requires finite nonempty values")
    median = float(np.median(array))
    return float(1.4826 * np.median(np.abs(array - median)))


def evaluate_spectral_gate_spectra(
    spectra_by_draw: Mapping[int, Mapping[int, np.ndarray]],
    config: SpectralGateConfig,
) -> SpectralGateResult:
    """Apply the paper gate to precomputed low-frequency spectra."""

    config.validate()
    expected_draws = set(config.diagnostic_seeds)
    if set(spectra_by_draw) != expected_draws:
        raise ValueError(
            "spectra must contain exactly the predeclared diagnostic seeds"
        )
    expected_knn = set(config.graph_knn_values)
    draw_results: list[dict[str, Any]] = []
    background_candidates: list[float] = []

    for seed in config.diagnostic_seeds:
        spectra = spectra_by_draw[seed]
        if set(spectra) != expected_knn:
            raise ValueError(
                f"draw {seed} must contain exactly kNN values {sorted(expected_knn)}"
            )
        relative_by_knn: dict[str, float] = {}
        background_by_knn: dict[str, dict[str, Any]] = {}
        for knn in config.graph_knn_values:
            values = np.sort(np.asarray(spectra[knn], dtype=np.float64))
            if len(values) < config.required_eigenvalues:
                raise ValueError(
                    f"draw {seed}, kNN {knn} needs at least "
                    f"{config.required_eigenvalues} eigenvalues"
                )
            candidate = relative_eigengap(
                values, config.num_regions, config.epsilon
            )
            relative_by_knn[str(knn)] = candidate
            background = np.asarray(
                [
                    relative_eigengap(values, index, config.epsilon)
                    for index in range(
                        config.num_regions + 1,
                        config.num_regions + config.background_gap_count + 1,
                    )
                ],
                dtype=np.float64,
            )
            threshold = float(
                np.median(background)
                + config.background_mad_multiplier * scaled_mad(background)
            )
            background_candidates.append(threshold)
            background_by_knn[str(knn)] = {
                "relative_gaps": background.tolist(),
                "median": float(np.median(background)),
                "scaled_mad": scaled_mad(background),
                "threshold": threshold,
            }

        nominal = relative_by_knn[str(config.nominal_knn)]
        delta = max(
            abs(relative_by_knn[str(knn)] - nominal)
            for knn in config.graph_knn_values
        )
        draw_results.append(
            {
                "seed": int(seed),
                "candidate_relative_gap_by_knn": relative_by_knn,
                "nominal_candidate_relative_gap": nominal,
                "max_candidate_gap_perturbation": delta,
                "background_by_knn": background_by_knn,
            }
        )

    candidate_min = min(
        row["nominal_candidate_relative_gap"] for row in draw_results
    )
    perturbation_max = config.perturbation_multiplier * max(
        row["max_candidate_gap_perturbation"] for row in draw_results
    )
    background_threshold = max(background_candidates)

    retained: float | None = None
    residual: float | None = None
    if candidate_min <= config.epsilon:
        use_partition = False
        reason = "candidate_gap_not_above_epsilon"
    else:
        retained = 1.0 - perturbation_max / candidate_min
        residual = candidate_min - perturbation_max
        safety_ok = retained >= config.retention_threshold
        background_ok = residual > background_threshold
        use_partition = bool(safety_ok and background_ok)
        if use_partition:
            reason = "spectrally_nondegenerate"
        elif not safety_ok and not background_ok:
            reason = "safety_and_background_checks_failed"
        elif not safety_ok:
            reason = "retained_safety_below_threshold"
        else:
            reason = "residual_gap_not_above_background"

    return SpectralGateResult(
        use_partition=use_partition,
        selected_method="spectral" if use_partition else "global",
        reason=reason,
        deployment_seed=config.deployment_seed,
        candidate_gap_min=float(candidate_min),
        perturbation_threshold_max=float(perturbation_max),
        retained_safety_fraction=None if retained is None else float(retained),
        robust_residual_gap=None if residual is None else float(residual),
        background_threshold=float(background_threshold),
        draw_results=tuple(draw_results),
        configuration=asdict(config),
    )


def _smallest_normalized_laplacian_eigenvalues(
    graph: sparse.csr_matrix,
    *,
    count: int,
    seed: int,
    tol: float,
    maxiter: int,
) -> np.ndarray:
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    if np.any(degree <= 0):
        raise RuntimeError("graph contains a zero-degree landmark")
    if count >= graph.shape[0]:
        raise ValueError("the landmark graph is too small for the requested spectrum")
    inv_sqrt = 1.0 / np.sqrt(degree)
    normalized_adjacency = (
        sparse.diags(inv_sqrt) @ graph @ sparse.diags(inv_sqrt)
    )
    initial = np.random.default_rng(seed).standard_normal(graph.shape[0])
    values = eigsh(
        normalized_adjacency,
        k=count,
        which="LA",
        tol=tol,
        maxiter=maxiter,
        v0=initial,
        return_eigenvectors=False,
    )
    return np.sort(1.0 - values)


class SpectralDegeneracyGate:
    """Evaluate the label-free gate directly from a frozen latent cache."""

    def __init__(
        self,
        config: SpectralGateConfig | None = None,
        *,
        neighbor_search: NeighborSearch | None = None,
    ) -> None:
        self.config = config or SpectralGateConfig()
        self.neighbor_search = neighbor_search or exact_cosine_neighbors_cpu

    def evaluate(
        self, latents: np.ndarray, *, group_ids: np.ndarray | None = None
    ) -> SpectralGateResult:
        started = time.perf_counter()
        values = np.asarray(latents, dtype=np.float32)
        self.config.validate(len(values))
        transformed, _, _ = _zscore_l2(values)
        spectra_by_draw: dict[int, dict[int, np.ndarray]] = {}

        for seed in self.config.diagnostic_seeds:
            landmark_indices = _sample_landmarks(
                len(transformed),
                self.config.num_landmarks,
                seed,
                group_ids,
            )
            landmarks = np.ascontiguousarray(
                transformed[landmark_indices], dtype=np.float32
            )
            neighbors, similarities, _ = self.neighbor_search(
                landmarks, max(self.config.graph_knn_values)
            )
            spectra_by_draw[seed] = {}
            for knn in self.config.graph_knn_values:
                graph, _ = build_self_tuned_graph(neighbors, similarities, knn)
                spectra_by_draw[seed][knn] = (
                    _smallest_normalized_laplacian_eigenvalues(
                        graph,
                        count=self.config.required_eigenvalues,
                        seed=seed * 1000 + knn,
                        tol=self.config.eig_tol,
                        maxiter=self.config.eig_maxiter,
                    )
                )

        result = evaluate_spectral_gate_spectra(spectra_by_draw, self.config)
        return SpectralGateResult(
            **{
                **result.to_dict(),
                "elapsed_sec": time.perf_counter() - started,
            }
        )


class GlobalPartitioner:
    """One-region fallback that makes LAP equivalent to Global-FT."""

    def fit(
        self,
        latents: np.ndarray,
        *,
        sample_ids: np.ndarray,
        group_ids: np.ndarray | None = None,
    ) -> PartitionResult:
        del sample_ids, group_ids
        values = np.asarray(latents, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("latents must be a nonempty [N,D] matrix")
        _, mean, scale = _zscore_l2(values)
        artifact = PartitionArtifact(
            prototypes=np.zeros((1, values.shape[1]), dtype=np.float32),
            prototype_region_ids=np.zeros(1, dtype=np.int64),
            mean=mean,
            scale=scale,
            metadata={
                "algorithm": "global_single_region",
                "num_clusters": 1,
                "routing": "constant_region_0",
            },
        )
        return PartitionResult(
            artifact=artifact,
            labels=np.zeros(len(values), dtype=np.int64),
            metadata=dict(artifact.metadata),
        )


class GatedSpectralPartitioner:
    """Choose Spectral LAP or Global-FT before any predictor is trained."""

    def __init__(
        self,
        gate_config: SpectralGateConfig | None = None,
        partition_config: LandmarkSpectralConfig | None = None,
        *,
        neighbor_search: NeighborSearch | None = None,
    ) -> None:
        self.gate_config = gate_config or SpectralGateConfig()
        self.partition_config = partition_config or LandmarkSpectralConfig(
            num_regions=self.gate_config.num_regions,
            num_landmarks=self.gate_config.num_landmarks,
            knn=self.gate_config.nominal_knn,
            seed=self.gate_config.deployment_seed,
            cpu_threads=self.gate_config.cpu_threads,
        )
        if (
            self.partition_config.num_regions != self.gate_config.num_regions
            or self.partition_config.num_landmarks != self.gate_config.num_landmarks
            or self.partition_config.knn != self.gate_config.nominal_knn
            or self.partition_config.seed != self.gate_config.deployment_seed
        ):
            raise ValueError(
                "deployed spectral configuration must match the predeclared gate "
                "K, landmark count, nominal kNN, and deployment seed"
            )
        self.neighbor_search = neighbor_search or exact_cosine_neighbors_cpu

    def fit(
        self,
        latents: np.ndarray,
        *,
        sample_ids: np.ndarray,
        group_ids: np.ndarray | None = None,
    ) -> PartitionResult:
        gate = SpectralDegeneracyGate(
            self.gate_config, neighbor_search=self.neighbor_search
        ).evaluate(latents, group_ids=group_ids)
        if gate.use_partition:
            result = LandmarkSpectralPartitioner(
                self.partition_config, neighbor_search=self.neighbor_search
            ).fit(latents, sample_ids=sample_ids, group_ids=group_ids)
        else:
            result = GlobalPartitioner().fit(
                latents, sample_ids=sample_ids, group_ids=group_ids
            )
        result.metadata = {
            **result.metadata,
            "automatic_gate": gate.to_dict(),
            "selected_post_training": (
                "regional_predictors" if gate.use_partition else "global_predictor"
            ),
        }
        result.artifact.metadata.update(
            {
                "automatic_gate": gate.to_dict(),
                "selected_post_training": result.metadata[
                    "selected_post_training"
                ],
            }
        )
        return result
