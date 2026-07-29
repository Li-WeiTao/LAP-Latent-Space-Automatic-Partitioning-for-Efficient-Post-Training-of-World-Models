"""Portable LAP partition-artifact schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PartitionArtifact:
    """Deployable partition produced by an offline geometry algorithm."""

    prototypes: np.ndarray
    prototype_region_ids: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict

    @property
    def num_regions(self) -> int:
        return int(np.max(self.prototype_region_ids)) + 1

    @classmethod
    def load(cls, directory: str | Path) -> "PartitionArtifact":
        directory = Path(directory)
        prototypes = np.load(directory / "routing_prototypes.npy")
        owners = np.load(directory / "prototype_cluster_ids.npy")
        with np.load(directory / "zscore_params.npz") as params:
            mean_key = "mean" if "mean" in params.files else "mu"
            if "std" in params.files:
                scale_key = "std"
            elif "scale" in params.files:
                scale_key = "scale"
            else:
                scale_key = "sigma"
            mean = np.asarray(params[mean_key], dtype=np.float32)
            scale = np.asarray(params[scale_key], dtype=np.float32)
        meta_path = directory / "cluster_meta.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        artifact = cls(
            prototypes=np.asarray(prototypes, dtype=np.float32),
            prototype_region_ids=np.asarray(owners, dtype=np.int64),
            mean=mean,
            scale=scale,
            metadata=metadata,
        )
        artifact.validate()
        return artifact

    def validate(self) -> None:
        if self.prototypes.ndim != 2:
            raise ValueError("prototypes must have shape [P, D]")
        if self.prototype_region_ids.shape != (self.prototypes.shape[0],):
            raise ValueError("one owner ID is required per prototype")
        if self.mean.shape != (self.prototypes.shape[1],):
            raise ValueError("normalization mean dimension mismatch")
        if self.scale.shape != self.mean.shape:
            raise ValueError("normalization scale dimension mismatch")
        if not np.isfinite(self.prototypes).all():
            raise ValueError("prototypes contain non-finite values")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("normalization parameters contain non-finite values")
        if np.any(self.scale <= 0):
            raise ValueError("normalization scale must be strictly positive")
        if np.any(self.prototype_region_ids < 0):
            raise ValueError("region IDs must be non-negative")
