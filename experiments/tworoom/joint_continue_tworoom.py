#!/usr/bin/env python3
"""Jointly continue a released LeWM on a selected HDF5 task dataset.

This is a post-training control for frozen-encoder predictor fine-tuning.  The
released object checkpoint contains no optimizer/scheduler state, so the run
uses a fresh AdamW optimizer with the official learning rate and the original
joint LeWM objective (prediction loss plus SIGReg).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

import h5py

from gauge_drift import DATASETS, choose_state_key, load_encoder
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("/data/sicong/weitao/datasets/lewm/tworoom.h5"),
    )
    parser.add_argument(
        "--dataset-name",
        default="tworoom",
        help="Provenance/result label; data and model are supplied independently.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Smoke-test cap; 0 means the complete epoch and is required formally.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=("fp32", "bf16-mixed"),
        help="Training precision. FP32 is the canonical fair-comparison setting.",
    )
    parser.add_argument(
        "--model-family",
        default="lewm",
        help="JEPA object backend family (lewm or subjepa).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def resolve_state_key(data_file: Path, dataset_name: str) -> str | None:
    with h5py.File(data_file, "r") as handle:
        spec = DATASETS.get(dataset_name)
        if spec is not None:
            return choose_state_key(handle, spec, None)
        for key in ("proprio", "state", "observation"):
            if key in handle:
                return key
    return None


def prepare_dataset(args: argparse.Namespace):
    data_file = args.data_file.resolve(strict=True)
    state_key = resolve_state_key(data_file, args.dataset_name)
    keys_to_load = ["pixels", "action"]
    keys_to_cache = ["action"]
    if state_key is not None:
        keys_to_load.append(state_key)
        keys_to_cache.append(state_key)
    dataset = swm.data.load_dataset(
        str(data_file),
        transform=None,
        num_steps=args.history_size + args.num_preds,
        frameskip=args.frameskip,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
    )
    transforms = [
        get_img_preprocessor(
            source="pixels", target="pixels", img_size=args.img_size
        )
    ]
    for col in keys_to_cache:
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def component_parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        name: int(sum(p.numel() for p in getattr(model, name).parameters()))
        for name in (
            "encoder",
            "projector",
            "predictor",
            "pred_proj",
            "action_encoder",
        )
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    if args.max_batches < 0:
        raise SystemExit("--max-batches must be nonnegative")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.precision == "bf16-mixed" and device.type != "cuda":
        raise ValueError("bf16-mixed requires --device cuda")
    use_bf16 = args.precision == "bf16-mixed"

    dataset_t0 = time.perf_counter()
    dataset = prepare_dataset(args)
    split_generator = torch.Generator().manual_seed(args.split_seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[args.train_fraction, 1.0 - args.train_fraction],
        generator=split_generator,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=device.type == "cuda",
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )
    dataset_sec = time.perf_counter() - dataset_t0

    model = load_encoder(str(args.checkpoint), device, None, model_family=args.model_family)
    expected_action_dim = args.frameskip * int(dataset.get_dim("action"))
    action_linear = next(
        module
        for module in model.action_encoder.modules()
        if isinstance(module, torch.nn.Linear)
    )
    if action_linear.in_features != expected_action_dim:
        raise RuntimeError(
            f"Action dimension mismatch: checkpoint={action_linear.in_features}, "
            f"dataset={expected_action_dim}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.train()
    sigreg = SIGReg(
        knots=args.sigreg_knots, num_proj=args.sigreg_num_proj
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history: list[dict] = []
    training_t0 = time.perf_counter()
    global_step = 0
    processed_samples = 0
    for epoch in range(args.epochs):
        pred_sum = 0.0
        sigreg_sum = 0.0
        total_sum = 0.0
        epoch_steps = 0
        epoch_t0 = time.perf_counter()
        for batch_idx, batch in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = model.encode(batch)
                emb = output["emb"]
                act_emb = output["act_emb"]
                ctx_emb = emb[:, : args.history_size]
                ctx_act = act_emb[:, : args.history_size]
                tgt_emb = emb[:, args.num_preds :]
                pred_emb = model.predict(ctx_emb, ctx_act)
                pred_loss = (pred_emb - tgt_emb).pow(2).mean()
                sigreg_loss = sigreg(emb.transpose(0, 1))
                loss = pred_loss + args.sigreg_weight * sigreg_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip_val
            )
            optimizer.step()

            epoch_steps += 1
            global_step += 1
            processed_samples += int(emb.shape[0])
            pred_sum += float(pred_loss.detach().cpu())
            sigreg_sum += float(sigreg_loss.detach().cpu())
            total_sum += float(loss.detach().cpu())
            if global_step % args.log_every == 0 or global_step == 1:
                print(
                    f"[train] epoch={epoch + 1}/{args.epochs} "
                    f"batch={batch_idx + 1}/{len(loader)} "
                    f"pred={pred_sum / epoch_steps:.6f} "
                    f"sigreg={sigreg_sum / epoch_steps:.6f} "
                    f"total={total_sum / epoch_steps:.6f} "
                    f"grad_norm={float(grad_norm):.4f}",
                    flush=True,
                )
        if epoch_steps == 0:
            raise RuntimeError("No training batches were processed")
        history.append(
            {
                "epoch": epoch + 1,
                "num_steps": epoch_steps,
                "processed_samples": epoch_steps * args.batch_size,
                "pred_loss_mean": pred_sum / epoch_steps,
                "sigreg_loss_mean": sigreg_sum / epoch_steps,
                "total_loss_mean": total_sum / epoch_steps,
                "elapsed_sec": time.perf_counter() - epoch_t0,
            }
        )
    training_sec = time.perf_counter() - training_t0

    model.eval().to("cpu")
    checkpoint_path = args.out_dir / "joint_continue_object.ckpt"
    tmp_checkpoint = args.out_dir / f".{checkpoint_path.name}.{os.getpid()}.tmp"
    torch.save(model, tmp_checkpoint)
    os.replace(tmp_checkpoint, checkpoint_path)

    formal_full_epoch = args.max_batches == 0
    metadata = {
        "method": "joint_continue",
        "dataset": args.dataset_name,
        "description": (
            "All JEPA modules jointly optimized from the released checkpoint "
            "with the original LeWM loss. Optimizer state is fresh because the "
            "released checkpoint contains model weights only."
        ),
        "formal_full_epoch": formal_full_epoch,
        "input_checkpoint": str(args.checkpoint.resolve(strict=True)),
        "input_checkpoint_sha256": sha256_file(args.checkpoint),
        "output_checkpoint": str(checkpoint_path.resolve(strict=True)),
        "output_checkpoint_sha256": sha256_file(checkpoint_path),
        "data_file": str(args.data_file.resolve(strict=True)),
        "dataset_num_windows": len(dataset),
        "train_num_windows": len(train_set),
        "validation_num_windows_unused": len(val_set),
        "train_subset_indices_sha256": hashlib.sha256(
            np.asarray(train_set.indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "processed_samples": processed_samples,
        "component_parameter_counts": component_parameter_counts(model),
        "total_trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "config": {
            "seed": args.seed,
            "split_seed": args.split_seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "cpu_threads": args.cpu_threads,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW (fresh state)",
            "scheduler": None,
            "gradient_clip_val": args.gradient_clip_val,
            "sigreg_weight": args.sigreg_weight,
            "sigreg_knots": args.sigreg_knots,
            "sigreg_num_proj": args.sigreg_num_proj,
            "history_size": args.history_size,
            "num_preds": args.num_preds,
            "frameskip": args.frameskip,
            "img_size": args.img_size,
            "train_fraction": args.train_fraction,
            "max_batches": args.max_batches,
            "precision": args.precision,
        },
        "history": history,
        "timing_sec": {
            "dataset_load_and_setup": dataset_sec,
            "training": training_sec,
        },
    }
    atomic_write_json(args.out_dir / "manifest.json", metadata)
    print(
        f"[done] checkpoint={checkpoint_path} formal_full_epoch={formal_full_epoch} "
        f"training_sec={training_sec:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
