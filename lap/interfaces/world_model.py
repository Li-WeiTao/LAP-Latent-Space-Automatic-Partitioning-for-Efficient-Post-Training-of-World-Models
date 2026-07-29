"""Architecture-neutral interface required by LAP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

PredictorHandle: TypeAlias = Any


@runtime_checkable
class WorldModelBackend(Protocol):
    """Minimal backend contract for regional world-model post-training.

    Implementations may use tensors, arrays, token sequences, or structured
    latent contexts. LAP never requires the encoder to be updated.
    """

    def encode(self, observations: Any) -> Any:
        """Encode observations with a frozen encoder."""

    def predict(
        self,
        latent_context: Any,
        actions: Any,
        predictor: PredictorHandle | None = None,
    ) -> Any:
        """Predict future latent states with the selected predictor."""

    def load_predictor(self, checkpoint: str | Path) -> PredictorHandle:
        """Load a predictor checkpoint."""

    def clone_predictor(self, predictor: PredictorHandle) -> PredictorHandle:
        """Create an independently fine-tunable predictor copy."""

    def save_predictor(
        self, predictor: PredictorHandle, checkpoint: str | Path
    ) -> None:
        """Serialize a regional predictor."""
