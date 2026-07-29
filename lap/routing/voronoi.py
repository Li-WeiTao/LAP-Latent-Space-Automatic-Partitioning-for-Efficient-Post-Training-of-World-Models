"""Lightweight out-of-sample routing for LAP partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lap.partition.artifact import PartitionArtifact


@dataclass(frozen=True)
class ZScoreL2Transform:
    mean: np.ndarray
    scale: np.ndarray
    eps: float = 1e-12

    def __call__(self, latents: np.ndarray) -> np.ndarray:
        values = np.asarray(latents, dtype=np.float32)
        standardized = (values - self.mean) / self.scale
        norms = np.linalg.norm(standardized, axis=-1, keepdims=True)
        return standardized / np.maximum(norms, self.eps)


class VoronoiRouter:
    """Route normalized latents by nearest prototype and prototype ownership."""

    def __init__(self, artifact: PartitionArtifact):
        artifact.validate()
        self.artifact = artifact
        self.transform = ZScoreL2Transform(artifact.mean, artifact.scale)
        prototypes = np.asarray(artifact.prototypes, dtype=np.float32)
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
        self.prototypes = prototypes / np.maximum(norms, 1e-12)

    def route(self, latents: np.ndarray) -> np.ndarray:
        transformed = self.transform(latents)
        original_shape = transformed.shape[:-1]
        flat = transformed.reshape(-1, transformed.shape[-1])
        nearest = np.argmax(flat @ self.prototypes.T, axis=1)
        regions = self.artifact.prototype_region_ids[nearest]
        return regions.reshape(original_shape)
