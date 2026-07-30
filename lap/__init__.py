"""Latent-space auto-partitioned fine-tuning for world models."""

from .interfaces import (
    EncodedTransitions,
    FrozenWorldModelBackend,
    InMemoryLatentCache,
    LatentCache,
    PredictorHandle,
    PredictorTrainingResult,
    RegionalPredictorTrainer,
    RegionalTrainingConfig,
    WorldModelBackend,
    WorldModelBackendFactory,
)
from .partition import (
    ArtifactPartitioner,
    IndexedPartitioner,
    LandmarkSpectralConfig,
    LandmarkSpectralPartitioner,
    LatentPartitioner,
    PartitionArtifact,
    PartitionResult,
)
from .pipeline import LAP, LAPConfig, LAPFitResult
from .routing.voronoi import VoronoiRouter, ZScoreL2Transform

__all__ = [
    "ArtifactPartitioner",
    "EncodedTransitions",
    "FrozenWorldModelBackend",
    "IndexedPartitioner",
    "InMemoryLatentCache",
    "LAP",
    "LAPConfig",
    "LatentCache",
    "LAPFitResult",
    "LandmarkSpectralConfig",
    "LandmarkSpectralPartitioner",
    "LatentPartitioner",
    "PartitionArtifact",
    "PartitionResult",
    "PredictorHandle",
    "PredictorTrainingResult",
    "RegionalPredictorTrainer",
    "RegionalTrainingConfig",
    "VoronoiRouter",
    "WorldModelBackend",
    "WorldModelBackendFactory",
    "ZScoreL2Transform",
]

__version__ = "0.2.0"
