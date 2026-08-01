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
    GatedSpectralPartitioner,
    GlobalPartitioner,
    IndexedPartitioner,
    LandmarkSpectralConfig,
    LandmarkSpectralPartitioner,
    LatentPartitioner,
    PartitionArtifact,
    PartitionResult,
    SpectralDegeneracyGate,
    SpectralGateConfig,
    SpectralGateResult,
)
from .pipeline import LAP, LAPConfig, LAPFitResult
from .routing.voronoi import VoronoiRouter, ZScoreL2Transform

__all__ = [
    "ArtifactPartitioner",
    "EncodedTransitions",
    "FrozenWorldModelBackend",
    "GatedSpectralPartitioner",
    "GlobalPartitioner",
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
    "SpectralDegeneracyGate",
    "SpectralGateConfig",
    "SpectralGateResult",
    "VoronoiRouter",
    "WorldModelBackend",
    "WorldModelBackendFactory",
    "ZScoreL2Transform",
]

__version__ = "0.3.0"
