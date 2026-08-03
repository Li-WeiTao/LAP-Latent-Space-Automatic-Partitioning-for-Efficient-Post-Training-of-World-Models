"""Shared checkpoint compatibility probing for JEPA object checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from backends.lewm.encoding import LeWMEncoderAdapter, make_hdf5_transition_dataset
from backends.lewm.finetuning import trainable_predictor_params
from experiments.control_matrix.backend_registry import (
    IMPLEMENTATION_BACKEND,
    backend_metadata,
    normalize_model_family,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TWOROOM_DIR = PROJECT_ROOT / "experiments" / "tworoom"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"checkpoint is empty: {path}")
    head = path.read_bytes()[:512]
    stripped = head.lstrip()
    if stripped.startswith(b"<!DOCTYPE") or stripped.startswith(b"<html"):
        raise ValueError(f"checkpoint appears to be HTML, not a torch object: {path}")
    if b"version https://git-lfs.github.com/spec/v1" in head:
        raise ValueError(f"checkpoint is a Git LFS pointer, not weights: {path}")
    return {"path": str(path.resolve()), "size_bytes": size, "sha256": sha256_file(path)}


def _patch_vit_config(config: object) -> None:
    defaults = {
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
        "torchscript": False,
        "use_bfloat16": False,
        "output_hidden_states": False,
    }
    for attr, value in defaults.items():
        if not hasattr(config, attr):
            setattr(config, attr, value)


from backends.lewm.checkpoint_compat import (
    load_jepa_object_checkpoint as load_jepa_object_checkpoint_compat,
)


def _load_checkpoint_for_probe(
    path: Path, device: torch.device, *, model_family: str
) -> torch.nn.Module:
    model = load_jepa_object_checkpoint_compat(
        path,
        model_family=model_family,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    encoder = getattr(model, "encoder", None)
    config = getattr(encoder, "config", None)
    if config is not None:
        _patch_vit_config(config)
    return model


def validate_jepa_module_contract(model: torch.nn.Module, *, source: Path) -> None:
    required = ("encoder", "predictor", "action_encoder", "projector", "pred_proj")
    missing = [
        name
        for name in required
        if not isinstance(getattr(model, name, None), torch.nn.Module)
    ]
    if missing:
        raise TypeError(
            f"Checkpoint {source} does not implement the JEPA module contract; "
            f"missing components: {missing}"
        )


def _module_summary(module: torch.nn.Module | None) -> dict[str, Any] | None:
    if module is None:
        return None
    return {
        "class": f"{type(module).__module__}.{type(module).__qualname__}",
        "num_parameters": sum(param.numel() for param in module.parameters()),
        "trainable_parameters": sum(
            param.numel() for param in module.parameters() if param.requires_grad
        ),
    }


def _infer_action_input_dim(model: torch.nn.Module) -> int | None:
    action_encoder = getattr(model, "action_encoder", None)
    if action_encoder is None:
        return None
    patch_embed = getattr(action_encoder, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "in_channels"):
        return int(patch_embed.in_channels)
    first = next(action_encoder.parameters(), None)
    if first is not None and first.ndim >= 2:
        return int(first.shape[1])
    return None


def _bootstrap_starts(
    data_file: Path,
    *,
    frameskip: int,
    history_size: int,
    num_preds: int,
    max_samples: int,
    pixel_key: str = "pixels",
) -> np.ndarray:
    sequence_length = history_size + num_preds
    span = frameskip * (sequence_length - 1) + 1
    with h5py.File(data_file, "r", swmr=True) as handle:
        num_frames = int(handle[pixel_key].shape[0])
    max_start = num_frames - span
    if max_start <= 0:
        raise ValueError("dataset too short for requested frameskip/history")
    stride = max(frameskip * sequence_length, 1)
    starts = np.arange(0, max_start, stride, dtype=np.int64)
    if len(starts) == 0:
        starts = np.asarray([0], dtype=np.int64)
    return starts[:max_samples]


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(tensor)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "has_nan": bool(torch.isnan(tensor).any().item()),
        "has_inf": bool(torch.isinf(tensor).any().item()),
        "all_finite": bool(finite.all().item()),
    }


def probe_checkpoint(
    *,
    model_family: str,
    checkpoint: Path,
    dataset: Path,
    frameskip: int = 5,
    history_size: int = 3,
    num_preds: int = 1,
    img_size: int = 224,
    max_samples: int = 16,
    device: str = "cuda",
) -> dict[str, Any]:
    family = normalize_model_family(model_family)
    meta = backend_metadata(family)
    checkpoint_path = checkpoint.expanduser().resolve()
    dataset_path = dataset.expanduser().resolve()
    file_info = validate_checkpoint_file(checkpoint_path)
    dev = torch.device(
        device
        if (device != "cuda" or torch.cuda.is_available())
        else "cpu"
    )

    report: dict[str, Any] = {
        "status": "PASSED",
        "schema_version": 1,
        "model_family": family,
        "implementation_backend": IMPLEMENTATION_BACKEND,
        "checkpoint": file_info,
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "device": str(dev),
        "frameskip": frameskip,
        "history_size": history_size,
        "num_preds": num_preds,
    }

    try:
        model = _load_checkpoint_for_probe(
            checkpoint_path, dev, model_family=family
        )
        validate_jepa_module_contract(model, source=checkpoint_path)
    except Exception as exc:  # noqa: BLE001 - probe must capture and report
        report["status"] = "FAILED"
        report["error"] = repr(exc)
        return report

    report["checkpoint"]["python_class"] = (
        f"{type(model).__module__}.{type(model).__qualname__}"
    )
    report["checkpoint"]["module_import_path"] = type(model).__module__
    report["train_mode"] = bool(model.training)
    report["named_children"] = list(dict(model.named_children()).keys())
    report["named_modules_count"] = len(list(model.named_modules()))
    report["named_parameters_count"] = len(list(model.named_parameters()))
    report["components"] = {
        name: _module_summary(getattr(model, name, None))
        for name in ("encoder", "projector", "action_encoder", "predictor", "pred_proj")
    }
    report["action_encoder_input_dim"] = _infer_action_input_dim(model)
    report["suggested_trainable_allowlist"] = ["predictor", "pred_proj"]
    report["official_prediction_path_parameters"] = [
        name
        for name, param in model.named_parameters()
        if name.startswith("predictor.") or name.startswith("pred_proj.")
    ]

    starts = _bootstrap_starts(
        dataset_path,
        frameskip=frameskip,
        history_size=history_size,
        num_preds=num_preds,
        max_samples=max_samples,
    )
    starts_path = checkpoint_path.parent / ".probe_starts.npy"
    np.save(starts_path, starts)
    transition_dataset = make_hdf5_transition_dataset(
        data_file=str(dataset_path),
        starts=str(starts_path),
        history_size=history_size,
        num_preds=num_preds,
        frameskip=frameskip,
    )
    adapter = LeWMEncoderAdapter(img_size=img_size, frameskip=frameskip)
    adapter.prepare_dataset(transition_dataset, model)
    report["inferred_frame_skip"] = transition_dataset.frameskip

    selection = transition_dataset.make_selection(max_samples=min(max_samples, 1))

    try:
        frame_dataset = transition_dataset.make_frame_dataset(
            selection.frame_ids.reshape(-1), chunk_aware=True
        )
        seq_len = transition_dataset.sequence_length
        frame_stack = np.stack([frame_dataset[i][0] for i in range(seq_len)], axis=0)
        frames_chw = (
            torch.from_numpy(frame_stack).float().permute(0, 3, 1, 2).to(dev)
        )
        pixels = frames_chw.unsqueeze(0).unsqueeze(0)
        aux, _aux_meta = adapter.encode_auxiliary(
            model,
            transition_dataset,
            selection,
            device=dev,
            batch_size=1,
            exact_batch_shapes=True,
            log_every=0,
        )
        act_emb_np = np.asarray(aux["act_emb"][:1], dtype=np.float32)
        act_emb = torch.from_numpy(act_emb_np).to(dev)

        with torch.inference_mode():
            encoded_frames = adapter.encode_frames(
                model, torch.from_numpy(frame_stack).to(dev), dev
            )
            emb = torch.from_numpy(encoded_frames).unsqueeze(0).to(dev)
            if emb.ndim == 2:
                emb = emb.unsqueeze(1)
            pred = model.predict(emb[:, :history_size], act_emb[:, :history_size])
            report["encode"] = {
                "emb": _tensor_stats(emb),
                "act_emb": _tensor_stats(act_emb),
            }
            report["predictor"] = {"output": _tensor_stats(pred)}

            planning_ok = False
            planning_error = None
            if hasattr(model, "get_cost"):
                try:
                    horizon = history_size + num_preds
                    action_dim = int(
                        getattr(
                            getattr(model.action_encoder, "patch_embed", None),
                            "in_channels",
                            transition_dataset.raw_action_dim,
                        )
                    )
                    action_history = torch.zeros(1, 1, horizon, action_dim, device=dev)
                    action_candidates = torch.zeros(1, 2, horizon, action_dim, device=dev)
                    info = {
                        "pixels": pixels[:, :, :horizon],
                        "action": action_history,
                        "goal": pixels[:, :, -1:].expand(
                            1, 1, horizon, *pixels.shape[-3:]
                        ),
                    }
                    cost = model.get_cost(info, action_candidates)
                    report["planning"] = {
                        "get_cost": _tensor_stats(cost),
                        "success": bool(torch.isfinite(cost).all().item()),
                    }
                    planning_ok = report["planning"]["success"]
                except Exception as exc:  # noqa: BLE001
                    planning_error = repr(exc)
                    report["planning"] = {
                        "get_cost": None,
                        "success": False,
                        "error": planning_error,
                    }
            else:
                report["planning"] = {
                    "get_cost": None,
                    "success": False,
                    "error": "missing get_cost",
                }

            nonfinite = any(
                section.get("has_nan") or section.get("has_inf")
                for section in (
                    report["encode"]["emb"],
                    report["encode"]["act_emb"],
                    report["predictor"]["output"],
                )
            )
            if nonfinite or not planning_ok:
                report["status"] = "FAILED"
                if nonfinite:
                    report["error"] = "probe outputs contain NaN/Inf"
                elif planning_error:
                    report["error"] = planning_error
    except Exception as exc:  # noqa: BLE001
        report["status"] = "FAILED"
        report["error"] = repr(exc)
        return report

    trainable = trainable_predictor_params(model)
    report["trainable_parameter_count_probe"] = sum(param.numel() for param in trainable)
    report["suggested_trainable_allowlist"] = ["predictor", "pred_proj"]
    return report


def write_probe_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
