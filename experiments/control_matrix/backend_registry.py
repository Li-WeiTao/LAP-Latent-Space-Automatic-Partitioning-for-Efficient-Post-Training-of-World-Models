"""Registry for JEPA object backends shared by LeWM and Sub-JEPA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from backends.lewm import (
    LeWMBackend,
    LeWMBackendFactory,
    LeWMLatentCache,
    LeWMRegionalPredictorTrainer,
)

IMPLEMENTATION_BACKEND = "jepa_object"
SUPPORTED_MODEL_FAMILIES = frozenset({"lewm", "subjepa"})
DEFAULT_MODEL_FAMILY = "lewm"


@dataclass(frozen=True)
class JEPABackendBundle:
    implementation_backend: str
    model_family: str
    backend_factory: LeWMBackendFactory
    latent_cache_loader: Callable[[Any], LeWMLatentCache]
    trainer_factory: Callable[..., LeWMRegionalPredictorTrainer]

    @property
    def is_subjepa(self) -> bool:
        return self.model_family == "subjepa"


def normalize_model_family(value: str | None) -> str:
    family = (value or DEFAULT_MODEL_FAMILY).strip().lower()
    if family not in SUPPORTED_MODEL_FAMILIES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
        raise ValueError(
            f"unsupported model_family={value!r}; expected one of: {supported}"
        )
    return family


def manifest_model_family(manifest: dict[str, Any]) -> str:
    """Legacy manifests without model_family are interpreted as LeWM."""

    return normalize_model_family(manifest.get("model_family"))


def resolve_backend_bundle(
    model_family: str | None,
    *,
    device: torch.device | None = None,
    select_best_by_eval: bool = True,
) -> JEPABackendBundle:
    family = normalize_model_family(model_family)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(path: str | Path) -> torch.nn.Module:
        from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint

        return load_jepa_object_checkpoint(path, model_family=family, map_location="cpu")

    return JEPABackendBundle(
        implementation_backend=IMPLEMENTATION_BACKEND,
        model_family=family,
        backend_factory=LeWMBackendFactory(_load_model),
        latent_cache_loader=LeWMLatentCache.from_npz,
        trainer_factory=lambda: LeWMRegionalPredictorTrainer(
            dev, select_best_by_eval=select_best_by_eval
        ),
    )


def backend_metadata(model_family: str | None) -> dict[str, str]:
    family = normalize_model_family(model_family)
    return {
        "implementation_backend": IMPLEMENTATION_BACKEND,
        "model_family": family,
    }
