"""Checkpoint loading entry points for JEPA object checkpoints."""

from __future__ import annotations

from typing import Any

from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint as _load_jepa_object_checkpoint


def load_jepa_object(
    path: str,
    *,
    map_location: Any = "cpu",
    model_family: str = "lewm",
):
    return _load_jepa_object_checkpoint(
        path,
        model_family=model_family,
        map_location=map_location,
    )
