"""Adapter from the LeWM model object to LAP's backend protocol."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch


class LeWMBackend:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def freeze_encoder(self) -> None:
        self.model.encoder.requires_grad_(False)
        self.model.projector.requires_grad_(False)
        self.model.encoder.eval()
        self.model.projector.eval()

    def encode(self, observations: Any) -> Any:
        if isinstance(observations, dict):
            return self.model.encode(observations)
        return self.model.encode({"pixels": observations})["emb"]

    def predict(
        self,
        latent_context: torch.Tensor,
        actions: torch.Tensor,
        predictor: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        if predictor is None:
            action_embeddings = self.model.action_encoder(actions)
            return self.model.predict(latent_context, action_embeddings)
        action_embeddings = self.model.action_encoder(actions)
        raw = predictor(latent_context, action_embeddings)
        batch, steps, width = raw.shape
        projected = self.model.pred_proj(raw.reshape(batch * steps, width))
        return projected.reshape(batch, steps, -1)

    def load_predictor(self, checkpoint: str | Path) -> torch.nn.Module:
        try:
            predictor = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only
            predictor = torch.load(checkpoint, map_location="cpu")
        if not isinstance(predictor, torch.nn.Module):
            raise TypeError("LeWM predictor checkpoints must contain a torch module")
        return predictor

    def clone_predictor(self, predictor: torch.nn.Module) -> torch.nn.Module:
        return copy.deepcopy(predictor)

    def save_predictor(
        self, predictor: torch.nn.Module, checkpoint: str | Path
    ) -> None:
        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(predictor, checkpoint)
