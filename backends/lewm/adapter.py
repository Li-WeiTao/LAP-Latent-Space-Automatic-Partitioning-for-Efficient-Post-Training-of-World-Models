"""Adapter from the LeWM model object to LAP's backend protocol."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import torch


class LeWMBackend:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def freeze_encoder(self) -> None:
        self.model.encoder.requires_grad_(False)
        self.model.projector.requires_grad_(False)
        self.model.action_encoder.requires_grad_(False)
        self.model.encoder.eval()
        self.model.projector.eval()
        self.model.action_encoder.eval()

    def encode(self, observations: Any) -> Any:
        if isinstance(observations, dict):
            return self.model.encode(observations)
        return self.model.encode({"pixels": observations})["emb"]

    def routing_latent(self, encoded_context: Any) -> Any:
        value = (
            encoded_context.get("emb")
            if isinstance(encoded_context, dict)
            else encoded_context
        )
        if getattr(value, "ndim", 0) >= 3:
            return value[:, -1, :]
        return value

    def predict(
        self,
        latent_context: torch.Tensor,
        actions: torch.Tensor,
        predictor: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        if predictor is None:
            action_embeddings = self.model.action_encoder(actions)
            return self.model.predict(latent_context, action_embeddings)
        if hasattr(predictor, "predict") and hasattr(predictor, "action_encoder"):
            return predictor.predict(
                latent_context, predictor.action_encoder(actions)
            )
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


class LeWMBackendFactory:
    """Load a LeWM backend without exposing LeWM imports to the LAP core."""

    def __init__(self, loader: Callable[[Any], torch.nn.Module]):
        self.loader = loader

    def load(self, pretrained_model: Any) -> LeWMBackend:
        if isinstance(pretrained_model, torch.nn.Module):
            model = pretrained_model
        else:
            model = self.loader(pretrained_model)
        return LeWMBackend(model)
