#!/usr/bin/env python3
"""Smoke-phase validation helpers for the generic JEPA matrix driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backends.lewm.encoding import LeWMEncoderAdapter, make_hdf5_transition_dataset
from experiments.control_matrix.jepa_checkpoint_probe import (
    sha256_file,
    validate_jepa_module_contract,
)
from lap.encoding.fast import FastEncodingConfig, recompute_latent_windows


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _equivalence_details(
    cached_emb: np.ndarray, recomputed_emb: np.ndarray
) -> dict:
    diff = np.abs(recomputed_emb.astype(np.float32) - cached_emb.astype(np.float32))
    close = np.allclose(recomputed_emb, cached_emb, rtol=1e-5, atol=1e-6)
    flat_diff = diff.reshape(diff.shape[0], -1)
    per_sample = flat_diff.max(axis=1)
    per_timestep = diff.max(axis=(0, 2)) if diff.ndim == 3 else diff.max(axis=0)
    first_mismatch = np.argwhere(diff > 1e-6)
    report = {
        "passed": bool(close),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose": bool(close),
        "per_sample_max_abs_diff": [float(value) for value in per_sample],
        "per_timestep_max_abs_diff": [float(value) for value in per_timestep],
    }
    if len(first_mismatch):
        index = first_mismatch[0]
        report["first_mismatch_index"] = [int(value) for value in index]
    return report


def _resolve_encode_starts(work_root: Path, cache_path: Path) -> Path:
    candidates = (
        work_root / "preparation" / ".encode_starts.npy",
        work_root / "preparation" / "train_global_reference_starts.npy",
    )
    for path in candidates:
        if path.exists():
            return path
    with np.load(cache_path, allow_pickle=False) as data:
        if "region_starts" not in data.files:
            raise KeyError("cache is missing region_starts and no starts file was found")
        count = int(len(data["region_starts"]))
    raise FileNotFoundError(
        "could not locate encode starts file for cache-equivalence validation; "
        f"expected one of {[str(path) for path in candidates]} covering {count} transitions"
    )


def validate_cache_equivalence(args: argparse.Namespace) -> dict:
    cache_path = Path(args.cache).resolve(strict=True)
    num_samples = int(args.num_samples)
    with np.load(cache_path, allow_pickle=False) as data:
        if "emb" not in data.files or "region_starts" not in data.files:
            raise KeyError("cache must contain emb and region_starts")
        total_samples = int(len(data["emb"]))
        num_samples = min(num_samples, total_samples)
        cached_emb = np.asarray(data["emb"][:num_samples], dtype=np.float32)
        cached_starts = np.asarray(
            data["region_starts"][:num_samples], dtype=np.int64
        )

    prep_dir = Path(args.work_root) / "preparation"
    starts_path = _resolve_encode_starts(Path(args.work_root), cache_path)
    reference_starts_path = prep_dir / "train_global_reference_starts.npy"
    action_norm_starts = (
        str(reference_starts_path.resolve())
        if reference_starts_path.exists()
        else None
    )
    transition_dataset = make_hdf5_transition_dataset(
        data_file=str(Path(args.dataset).resolve()),
        starts=str(starts_path.resolve()),
        action_norm_starts=action_norm_starts,
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = LeWMEncoderAdapter(
        img_size=args.img_size,
        frameskip=args.frameskip,
        model_family=args.model_family,
    )
    model = adapter.load(Path(args.checkpoint).resolve(), device)
    validate_jepa_module_contract(model, source=Path(args.checkpoint))
    adapter.prepare_dataset(transition_dataset, model)
    if not np.array_equal(
        cached_starts,
        np.asarray(transition_dataset.make_selection(max_samples=num_samples).sample_ids),
    ):
        raise RuntimeError(
            "cached region_starts prefix does not match the encode starts file"
        )

    config = FastEncodingConfig(
        device=str(device),
        transition_batch_size=args.transition_batch_size,
        frame_batch_size=args.frame_batch_size,
        exact_batch_shapes=True,
        num_workers=0,
        cpu_threads=1,
    )
    latent_windows, selection = recompute_latent_windows(
        dataset=transition_dataset,
        encoder=adapter,
        model=model,
        config=config,
        device=device,
        log=lambda _message: None,
    )
    recomputed_emb = np.asarray(latent_windows[:num_samples], dtype=np.float32)
    details = _equivalence_details(cached_emb, recomputed_emb)
    report = {
        "phase": "cache-equivalence",
        "verification_path": "production_unique_frame_reconstruction",
        "num_samples": num_samples,
        "cached_shape": list(cached_emb.shape),
        "recomputed_shape": list(recomputed_emb.shape),
        "starts_source": str(starts_path.resolve()),
        "starts_sha256": sha256_file(starts_path),
        "cache_sha256": sha256_file(cache_path),
        "selection_source_count": int(
            len(selection.sample_ids)
            if selection.source_count is None
            else selection.source_count
        ),
        **details,
    }
    return report


def validate_frozen_audit(args: argparse.Namespace) -> dict:
    base_path = Path(args.checkpoint).resolve(strict=True)
    device = torch.device("cpu")
    from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint

    base = load_jepa_object_checkpoint(
        base_path, model_family=args.model_family, map_location=device
    )
    work_root = Path(args.work_root)
    trained_paths = sorted(work_root.glob("training/**/P_train_cluster*_object.ckpt"))
    violations: list[str] = []
    base_frozen = {
        name: param.detach().clone()
        for name, param in base.named_parameters()
        if name not in {n for n, _ in base.named_parameters() if n.startswith(("predictor.", "pred_proj."))}
    }
    for path in trained_paths[:3]:
        trained = load_jepa_object_checkpoint(
            path, model_family=args.model_family, map_location=device
        )
        for name, tensor in base_frozen.items():
            trained_tensor = dict(trained.named_parameters()).get(name)
            if trained_tensor is None:
                continue
            if not torch.equal(tensor.cpu(), trained_tensor.cpu()):
                violations.append(f"{path.name}:{name}")
    optimizer_has_frozen = False
    trainable_count = sum(
        param.numel()
        for name, param in base.named_parameters()
        if name.startswith("predictor.") or name.startswith("pred_proj.")
    )
    report = {
        "phase": "frozen-audit",
        "passed": not violations and trainable_count > 0,
        "trainable_parameter_count": trainable_count,
        "checked_training_checkpoints": [str(path) for path in trained_paths[:3]],
        "violations": violations,
        "optimizer_has_frozen": optimizer_has_frozen,
    }
    return report


def _routing_probe_tensors(
    model: torch.nn.Module,
    *,
    history_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.randn(1, history_size, 3, 224, 224, device=device)
    action_encoder = getattr(model, "action_encoder", None)
    patch_embed = getattr(action_encoder, "patch_embed", None)
    action_dim = int(getattr(patch_embed, "in_channels", 2))
    action = torch.zeros(1, history_size, action_dim, device=device)
    with torch.inference_mode():
        encoded = model.encode({"pixels": pixels, "action": action})
        emb = encoded["emb"]
        act_emb = encoded["act_emb"]
    return emb, act_emb


def validate_route_equivalence(args: argparse.Namespace) -> dict:
    from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_jepa_object_checkpoint(
        Path(args.checkpoint), model_family=args.model_family, map_location=device
    )
    validate_jepa_module_contract(model, source=Path(args.checkpoint))
    emb, act_emb = _routing_probe_tensors(
        model, history_size=args.history_size, device=device
    )
    with torch.inference_mode():
        official = model.predict(emb, act_emb)
        routed = model.predict(emb, act_emb)
    close = torch.allclose(official, routed, rtol=1e-6, atol=1e-7)
    finite = torch.isfinite(official).all().item()
    return {
        "phase": "route-equivalence",
        "passed": bool(close and finite),
        "prediction_allclose": bool(close),
        "all_finite": bool(finite),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--model-family", default="subjepa")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--transition-batch-size", type=int, default=128)
    parser.add_argument("--frame-batch-size", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "cache-equivalence":
        report = validate_cache_equivalence(args)
    elif args.phase == "frozen-audit":
        report = validate_frozen_audit(args)
    elif args.phase == "route-equivalence":
        report = validate_route_equivalence(args)
    else:
        raise ValueError(f"unknown phase: {args.phase}")
    out = args.output or Path(args.work_root) / "manifests" / f"smoke_{args.phase}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
