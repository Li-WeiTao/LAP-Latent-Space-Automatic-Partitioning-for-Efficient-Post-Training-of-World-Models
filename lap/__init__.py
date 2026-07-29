"""Latent-space auto-partitioned fine-tuning for world models."""

from .interfaces.world_model import PredictorHandle, WorldModelBackend
from .partition.artifact import PartitionArtifact
from .routing.voronoi import VoronoiRouter, ZScoreL2Transform

__all__ = [
    "PartitionArtifact",
    "PredictorHandle",
    "VoronoiRouter",
    "WorldModelBackend",
    "ZScoreL2Transform",
]

__version__ = "0.1.0"
