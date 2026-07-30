"""Latent-cache contracts for architecture-neutral LAP post-training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class EncodedTransitions:
    """Frozen-encoder transition cache consumed by LAP.

    ``routing_latents`` contains current-state latents used by the action-free
    partitioner.  ``payload`` stores backend-owned latent windows and predictor
    conditioning, such as LeWM's ``emb`` and ``act_emb`` arrays.  Raw images are
    intentionally outside this contract.
    """

    routing_latents: np.ndarray
    sample_ids: np.ndarray
    payload: Any
    group_ids: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        latents = np.asarray(self.routing_latents)
        sample_ids = np.asarray(self.sample_ids)
        if latents.ndim != 2:
            raise ValueError("routing_latents must have shape [N, D]")
        if sample_ids.shape != (len(latents),):
            raise ValueError("sample_ids must contain one ID per transition")
        if self.group_ids is not None and np.asarray(self.group_ids).shape != (
            len(latents),
        ):
            raise ValueError("group_ids must contain one ID per transition")
        if not np.isfinite(latents).all():
            raise ValueError("routing_latents contain non-finite values")

    def subset(self, indices: np.ndarray) -> "EncodedTransitions":
        indices = np.asarray(indices)
        if indices.dtype == bool and indices.shape != (len(self.routing_latents),):
            raise ValueError("boolean subset mask has the wrong length")
        if not hasattr(self.payload, "subset"):
            raise TypeError(
                "backend cache payload must implement subset(indices) for regional training"
            )
        payload = self.payload.subset(indices)
        return EncodedTransitions(
            routing_latents=np.asarray(self.routing_latents)[indices],
            sample_ids=np.asarray(self.sample_ids)[indices],
            group_ids=(
                None
                if self.group_ids is None
                else np.asarray(self.group_ids)[indices]
            ),
            payload=payload,
            metadata=dict(self.metadata),
        )


@runtime_checkable
class LatentCache(Protocol):
    """A prepared or official frozen-encoder cache supplied to LAP."""

    def load(self) -> EncodedTransitions:
        """Load latent transitions without reading or encoding raw observations."""


@dataclass
class InMemoryLatentCache:
    """Minimal adapter for callers that already hold their cache in memory."""

    transitions: EncodedTransitions

    def load(self) -> EncodedTransitions:
        self.transitions.validate()
        return self.transitions
