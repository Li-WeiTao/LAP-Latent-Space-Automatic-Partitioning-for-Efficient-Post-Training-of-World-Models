"""Backend-neutral regional predictor training contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .cache import EncodedTransitions
from .world_model import FrozenWorldModelBackend, PredictorHandle


@dataclass(frozen=True)
class RegionalTrainingConfig:
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    train_seed: int = 42
    min_region_samples: int = 256
    options: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.min_region_samples < 1:
            raise ValueError("min_region_samples must be positive")


@dataclass
class PredictorTrainingResult:
    predictor: PredictorHandle
    metrics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RegionalPredictorTrainer(Protocol):
    def fit_region(
        self,
        backend: FrozenWorldModelBackend,
        region_id: int,
        transitions: EncodedTransitions,
        config: RegionalTrainingConfig,
    ) -> PredictorTrainingResult:
        """Fine-tune one predictor while leaving the backend encoder frozen."""
