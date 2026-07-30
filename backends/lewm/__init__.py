"""LeWM compatibility backend.

The vendored compatibility files are covered by LICENSES/LEWM-LICENSE.
"""

from .adapter import LeWMBackend, LeWMBackendFactory
from .cache import LeWMCachedPayload, LeWMLatentCache
from .encoding import LeWMEncoderAdapter, LeWMHDF5TransitionDataset
from .finetuning import LeWMRegionalPredictorTrainer, LeWMTrainConfig
from .routing import route_voronoi_torch, transform_latent_torch

__all__ = [
    "LeWMBackend",
    "LeWMBackendFactory",
    "LeWMCachedPayload",
    "LeWMEncoderAdapter",
    "LeWMHDF5TransitionDataset",
    "LeWMLatentCache",
    "LeWMRegionalPredictorTrainer",
    "LeWMTrainConfig",
    "route_voronoi_torch",
    "transform_latent_torch",
]
