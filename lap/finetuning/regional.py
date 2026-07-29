"""Backend-neutral description of a regional predictor fine-tuning run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
