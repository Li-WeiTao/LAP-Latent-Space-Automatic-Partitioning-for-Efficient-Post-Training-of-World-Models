"""Partitioner contracts shared by all LAP backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .artifact import PartitionArtifact


@dataclass
class PartitionResult:
    artifact: PartitionArtifact
    labels: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, num_samples: int | None = None) -> None:
        self.artifact.validate()
        labels = np.asarray(self.labels)
        if labels.ndim != 1:
            raise ValueError("partition labels must be one-dimensional")
        if num_samples is not None and len(labels) != num_samples:
            raise ValueError(
                f"expected {num_samples} labels, received {len(labels)}"
            )
        if labels.size and (labels.min() < 0 or labels.max() >= self.artifact.num_regions):
            raise ValueError("partition labels refer to a missing region")


@runtime_checkable
class LatentPartitioner(Protocol):
    def fit(
        self,
        latents: np.ndarray,
        *,
        sample_ids: np.ndarray,
        group_ids: np.ndarray | None = None,
    ) -> PartitionResult:
        """Fit an action-free partition using only current-state latents."""

