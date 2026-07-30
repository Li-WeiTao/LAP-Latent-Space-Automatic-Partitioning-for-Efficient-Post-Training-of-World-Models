"""Adapters for replaying a previously fitted LAP partition exactly."""

from __future__ import annotations

import numpy as np

from lap.routing.voronoi import VoronoiRouter

from .artifact import PartitionArtifact
from .base import PartitionResult


class ArtifactPartitioner:
    """Use a committed deployable artifact as a deterministic partitioner.

    This is the reproduction path for a locked partition.  It does not consume
    actions or labels: labels are recomputed from the stored Voronoi router.
    """

    def __init__(self, artifact: PartitionArtifact):
        artifact.validate()
        self.artifact = artifact

    def fit(
        self,
        latents: np.ndarray,
        *,
        sample_ids: np.ndarray,
        group_ids: np.ndarray | None = None,
    ) -> PartitionResult:
        del sample_ids, group_ids
        labels = VoronoiRouter(self.artifact).route(latents).astype(np.int64)
        result = PartitionResult(
            artifact=self.artifact,
            labels=labels,
            metadata={"mode": "locked_artifact_replay"},
        )
        result.validate(len(latents))
        return result


class IndexedPartitioner:
    """Replay exact training assignments keyed by stable sample IDs.

    Spectral pseudo-labels on the landmarks are compressed into a deployable
    Voronoi router.  The historical TwoRoom trainer intentionally used the
    saved full-data assignment array.  This adapter preserves that exact
    training contract while still shipping the Voronoi artifact for inference.
    """

    def __init__(
        self,
        artifact: PartitionArtifact,
        assignment_sample_ids: np.ndarray,
        assignment_labels: np.ndarray,
    ):
        artifact.validate()
        ids = np.asarray(assignment_sample_ids, dtype=np.int64)
        labels = np.asarray(assignment_labels, dtype=np.int64)
        if ids.shape != labels.shape or ids.ndim != 1:
            raise ValueError("assignment IDs and labels must be equal-length vectors")
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        labels = labels[order]
        if len(ids) and np.any(ids[1:] == ids[:-1]):
            raise ValueError("assignment sample IDs must be unique")
        if labels.size and (labels.min() < 0 or labels.max() >= artifact.num_regions):
            raise ValueError("assignment labels refer to a missing region")
        self.artifact = artifact
        self._ids = ids
        self._labels = labels

    def fit(
        self,
        latents: np.ndarray,
        *,
        sample_ids: np.ndarray,
        group_ids: np.ndarray | None = None,
    ) -> PartitionResult:
        del group_ids
        requested = np.asarray(sample_ids, dtype=np.int64)
        positions = np.searchsorted(self._ids, requested)
        valid = positions < len(self._ids)
        valid[valid] &= self._ids[positions[valid]] == requested[valid]
        if not valid.all():
            missing = requested[~valid][:5].tolist()
            raise KeyError(f"partition artifact does not cover sample IDs {missing}")
        labels = self._labels[positions]
        result = PartitionResult(
            artifact=self.artifact,
            labels=labels,
            metadata={"mode": "indexed_assignment_replay"},
        )
        result.validate(len(latents))
        return result

