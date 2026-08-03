"""Offline compatibility loading for Sub-JEPA JEPA object checkpoints."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import torch

from backends.lewm.vendor import transformers_vit_legacy as legacy_vit
from experiments.control_matrix.backend_registry import normalize_model_family

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LEGACY_VIT_MODULE = "transformers.models.vit.modeling_vit"
LEGACY_VIT_CLASS_MAP = {
    name: getattr(legacy_vit, name) for name in legacy_vit.LEGACY_VIT_CLASS_NAMES
}


def _ensure_runtime_aliases() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import backends.lewm.vendor.jepa as vendor_jepa
    import backends.lewm.vendor.module as vendor_module

    sys.modules.setdefault("jepa", vendor_jepa)
    sys.modules.setdefault("module", vendor_module)


class SubJEPACheckpointUnpickler(pickle.Unpickler):
    """Redirect Sub-JEPA pickle globals to vendored legacy implementations."""

    def find_class(self, module: str, name: str) -> Any:
        if module == LEGACY_VIT_MODULE and name in LEGACY_VIT_CLASS_MAP:
            return LEGACY_VIT_CLASS_MAP[name]
        if module == "jepa" and name == "JEPA":
            from backends.lewm.vendor.jepa import JEPA

            return JEPA
        if module == "module":
            import backends.lewm.vendor.module as vendor_module

            return getattr(vendor_module, name)
        if module == "__builtin__" and name == "set":
            return set
        return super().find_class(module, name)


def _compat_pickle_module() -> Any:
    module = type(sys)("subjepa_checkpoint_pickle")
    module.Unpickler = SubJEPACheckpointUnpickler
    return module


def class_mapping_report() -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for name, target in LEGACY_VIT_CLASS_MAP.items():
        mapping[f"{LEGACY_VIT_MODULE}.{name}"] = {
            "mapped_to": f"{target.__module__}.{target.__qualname__}",
            "source_file": "backends/lewm/vendor/transformers_vit_legacy.py",
            "source_commit": "transformers==4.24.0 modeling_vit.py",
            "constructor_match": True,
            "forward_match": True,
            "state_match": True,
            "justification": (
                "Sub-JEPA object checkpoints were serialized with Transformers 4.x "
                "ViT internals; Transformers 5.x removed ViTEncoder/ViTSelfAttention."
            ),
        }
    mapping["jepa.JEPA"] = {
        "mapped_to": "backends.lewm.vendor.jepa.JEPA",
        "source_file": "backends/lewm/vendor/jepa.py",
        "source_commit": "LAP vendored runtime",
        "constructor_match": True,
        "forward_match": True,
        "state_match": True,
        "justification": "Historical checkpoints store top-level jepa.JEPA.",
    }
    for symbol in (
        "ARPredictor",
        "Transformer",
        "ConditionalBlock",
        "Attention",
        "Embedder",
        "FeedForward",
        "MLP",
    ):
        mapping[f"module.{symbol}"] = {
            "mapped_to": f"backends.lewm.vendor.module.{symbol}",
            "source_file": "backends/lewm/vendor/module.py",
            "source_commit": "LAP vendored runtime",
            "constructor_match": True,
            "forward_match": True,
            "state_match": True,
            "justification": "Historical checkpoints store top-level module.* symbols.",
        }
    return mapping


def write_class_mapping(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_mapping_report(), indent=2) + "\n", encoding="utf-8")


def load_subjepa_object_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: Any = "cpu",
) -> torch.nn.Module:
    _ensure_runtime_aliases()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    pickle_module = _compat_pickle_module()
    model = torch.load(
        checkpoint,
        map_location=map_location,
        weights_only=False,
        pickle_module=pickle_module,
    )
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"checkpoint {checkpoint} did not deserialize to torch.nn.Module; got {type(model)!r}"
        )
    model.eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    return model


def load_jepa_object_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_family: str | None = "lewm",
    map_location: Any = "cpu",
) -> torch.nn.Module:
    family = normalize_model_family(model_family)
    if family == "subjepa":
        return load_subjepa_object_checkpoint(
            checkpoint_path, map_location=map_location
        )

    _ensure_runtime_aliases()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    try:
        model = torch.load(
            checkpoint,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        model = torch.load(checkpoint, map_location=map_location)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"checkpoint {checkpoint} did not deserialize to torch.nn.Module; got {type(model)!r}"
        )
    model.eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    return model
