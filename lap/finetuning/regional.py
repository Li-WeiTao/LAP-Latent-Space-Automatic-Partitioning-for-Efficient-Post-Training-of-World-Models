"""Backend-neutral regional predictor fine-tuning orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lap.interfaces.cache import EncodedTransitions
from lap.interfaces.training import (
    PredictorTrainingResult,
    RegionalPredictorTrainer,
    RegionalTrainingConfig,
)
from lap.interfaces.world_model import FrozenWorldModelBackend


@dataclass(frozen=True)
class RegionalFineTuningPlan:
    num_regions: int
    epochs: int
    train_seed: int
    output_directory: Path

    def validate(self) -> None:
        if self.num_regions < 1:
            raise ValueError("num_regions must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")


def fit_regional_predictors(
    backend: FrozenWorldModelBackend,
    transitions: EncodedTransitions,
    labels: np.ndarray,
    trainer: RegionalPredictorTrainer,
    config: RegionalTrainingConfig,
    *,
    num_regions: int,
) -> dict[int, PredictorTrainingResult]:
    """Fit one predictor per region using a single backend-neutral loop."""

    transitions.validate()
    config.validate()
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (len(transitions.routing_latents),):
        raise ValueError("labels must contain one region ID per transition")
    if labels.size and (labels.min() < 0 or labels.max() >= num_regions):
        raise ValueError("labels contain an out-of-range region ID")

    results: dict[int, PredictorTrainingResult] = {}
    for region_id in range(num_regions):
        indices = np.flatnonzero(labels == region_id)
        if len(indices) < config.min_region_samples:
            raise RuntimeError(
                f"region {region_id} has {len(indices)} transitions; "
                f"minimum is {config.min_region_samples}"
            )
        results[region_id] = trainer.fit_region(
            backend, region_id, transitions.subset(indices), config
        )
    return results
