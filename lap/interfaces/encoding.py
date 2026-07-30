"""Backend-neutral contracts for building frozen-encoder latent caches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass
class EncodingSelection:
    """Ordered transition windows selected from one task dataset.

    ``frame_ids[n, t]`` identifies the raw observation used at latent timestep
    ``t`` of transition ``n``. Repeated IDs are the only assumption required by
    the lossless unique-frame acceleration. The IDs need not be contiguous.
    """

    sample_ids: np.ndarray
    frame_ids: np.ndarray
    source_offset: int = 0
    source_count: int | None = None
    group_ids: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        sample_ids = np.asarray(self.sample_ids)
        frame_ids = np.asarray(self.frame_ids)
        if sample_ids.ndim != 1 or len(sample_ids) == 0:
            raise ValueError("sample_ids must be a non-empty one-dimensional array")
        if frame_ids.ndim != 2 or frame_ids.shape[0] != len(sample_ids):
            raise ValueError("frame_ids must have shape [N, T]")
        if not np.issubdtype(frame_ids.dtype, np.integer):
            raise TypeError("frame_ids must use an integer dtype")
        if self.source_offset < 0:
            raise ValueError("source_offset must be nonnegative")
        source_count = len(sample_ids) if self.source_count is None else self.source_count
        if source_count < self.source_offset + len(sample_ids):
            raise ValueError("source_count does not cover the selected transitions")
        if self.group_ids is not None and np.asarray(self.group_ids).shape != (
            len(sample_ids),
        ):
            raise ValueError("group_ids must contain one ID per transition")


@runtime_checkable
class EncodingDataset(Protocol):
    """Task-dataset adapter used only by the upstream cache builder."""

    def describe(self) -> Mapping[str, Any]:
        """Return serializable dataset provenance without reading all samples."""

    def make_selection(
        self, *, start_offset: int = 0, max_samples: int = 0
    ) -> EncodingSelection:
        """Return ordered transition windows and their reusable frame IDs."""

    def make_frame_dataset(
        self, frame_ids: np.ndarray, *, chunk_aware: bool
    ) -> Any:
        """Return a torch-compatible dataset yielding ``(frame, frame_id)``."""


@runtime_checkable
class LatentEncoderAdapter(Protocol):
    """Model-specific adapter for one frozen world-model encoder."""

    def describe(self) -> Mapping[str, Any]:
        """Return serializable encoder/backend provenance."""

    def load(self, pretrained_model: Any, device: Any) -> Any:
        """Load and freeze the requested pretrained model."""

    def prepare_dataset(self, dataset: EncodingDataset, model: Any) -> None:
        """Resolve model-dependent dataset settings such as action frameskip."""

    def encode_frames(self, model: Any, frames: Any, device: Any) -> np.ndarray:
        """Encode one already batch-shaped frame batch as ``[B, D]``."""

    def encode_auxiliary(
        self,
        model: Any,
        dataset: EncodingDataset,
        selection: EncodingSelection,
        *,
        device: Any,
        batch_size: int,
        exact_batch_shapes: bool,
        log_every: int,
    ) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
        """Encode backend-owned conditioning arrays, e.g. action embeddings."""

    def cache_arrays(
        self,
        latent_windows: np.ndarray,
        selection: EncodingSelection,
        auxiliary: Mapping[str, np.ndarray],
    ) -> Mapping[str, np.ndarray]:
        """Map generic results to the backend's lossless cache schema."""
