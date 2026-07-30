"""Lossless LeWM latent-cache adapter for the generic LAP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lap.interfaces import EncodedTransitions


@dataclass
class LeWMCachedPayload:
    emb: torch.Tensor
    act_emb: torch.Tensor

    def subset(self, indices: np.ndarray) -> "LeWMCachedPayload":
        index = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        return LeWMCachedPayload(
            emb=self.emb.index_select(0, index),
            act_emb=self.act_emb.index_select(0, index),
        )


class LeWMLatentCache:
    """Exact latent-transition cache used by the historical TwoRoom runs."""

    def __init__(
        self,
        emb: torch.Tensor,
        act_emb: torch.Tensor,
        sample_ids: np.ndarray,
        *,
        route_index: int = 0,
        group_ids: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if emb.ndim != 3:
            raise ValueError("LeWM emb cache must have shape [N, T, D]")
        if act_emb.ndim != 3 or len(act_emb) != len(emb):
            raise ValueError("LeWM act_emb cache must have shape [N, T-1, D]")
        if not -emb.shape[1] <= route_index < emb.shape[1]:
            raise ValueError("route_index is outside the cached latent sequence")
        self.emb = emb
        self.act_emb = act_emb
        self.sample_ids = np.asarray(sample_ids, dtype=np.int64)
        self.route_index = int(route_index)
        self.group_ids = group_ids
        self.metadata = dict(metadata or {})

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        route_index: int = 0,
        group_ids: np.ndarray | None = None,
    ) -> "LeWMLatentCache":
        path = Path(path)
        with np.load(path) as data:
            return cls(
                torch.from_numpy(np.asarray(data["emb"])),
                torch.from_numpy(np.asarray(data["act_emb"])),
                np.asarray(data["region_starts"], dtype=np.int64),
                route_index=route_index,
                group_ids=group_ids,
                metadata={"source": str(path.resolve())},
            )

    def load(self) -> EncodedTransitions:
        routing = self.emb[:, self.route_index, :].detach().cpu().numpy()
        transitions = EncodedTransitions(
            routing_latents=np.asarray(routing, dtype=np.float32),
            sample_ids=self.sample_ids,
            group_ids=self.group_ids,
            payload=LeWMCachedPayload(self.emb, self.act_emb),
            metadata={
                **self.metadata,
                "encoding": "lossless_precomputed",
                "route_index": self.route_index,
            },
        )
        transitions.validate()
        return transitions

