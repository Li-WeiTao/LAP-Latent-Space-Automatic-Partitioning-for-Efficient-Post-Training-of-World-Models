#!/usr/bin/env python3
"""Validate exact equivalence and benchmark the accelerated embedding loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import DATASETS, load_encoder  # noqa: E402
from predictor_rule_drift import (  # noqa: E402
    infer_frameskip,
    read_action_blocks,
    read_sequence_dataset,
)
from trajectory import (  # noqa: E402
    TrainConfig,
    TransitionSequenceDataset,
    precompute_embeddings,
    read_action_blocks_vectorized,
    seq_len,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="tworoom")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("/data/sicong/weitao/datasets/lewm/tworoom.h5"),
    )
    parser.add_argument(
        "--checkpoint",
        default="/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt",
    )
    parser.add_argument(
        "--starts",
        type=Path,
        default=Path(
            "experiments/tworoom/results/"
            "tworoom_geometry_train_region_predictors/train_global_reference_starts.npy"
        ),
    )
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=20260717)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/tworoom/results/"
            "embedding_loader_validation/validation.json"
        ),
    )
    return parser.parse_args()


def tensor_diff(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> dict:
    delta = (a.float() - b.float()).abs()
    return {
        "shape": list(a.shape),
        "dtype_legacy": str(a.dtype),
        "dtype_dataloader": str(b.dtype),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "exact": bool(torch.equal(a, b)),
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")

    all_starts = np.asarray(np.load(args.starts), dtype=np.int64)
    rng = np.random.default_rng(args.sample_seed)
    sample_size = min(args.sample_size, len(all_starts))
    positions = np.sort(rng.choice(len(all_starts), size=sample_size, replace=False))
    starts = all_starts[positions]

    device = torch.device(args.device)
    spec = DATASETS[args.dataset]
    model = load_encoder(args.checkpoint, device, None).eval()
    cfg = TrainConfig(batch_size=args.batch_size)
    with h5py.File(args.data_file, "r") as h5:
        frameskip = infer_frameskip(model, int(h5["action"].shape[1]), cfg.frameskip)
        legacy_pixels = read_sequence_dataset(
            h5[spec.pixel_key], starts, seq_len(cfg), frameskip
        )
        legacy_actions = read_action_blocks(
            h5["action"], starts, seq_len(cfg) - 1, frameskip
        ).astype(np.float32)
        vectorized_actions = read_action_blocks_vectorized(
            np.asarray(h5["action"][:], dtype=np.float32),
            starts,
            seq_len(cfg) - 1,
            frameskip,
        )

    dataset = TransitionSequenceDataset(
        args.data_file,
        spec.pixel_key,
        starts,
        seq_len(cfg),
        frameskip,
    )
    fast_pixels = []
    fast_actions = []
    fast_starts = []
    for i in range(len(dataset)):
        pixels, actions, start = dataset[i]
        fast_pixels.append(pixels)
        fast_actions.append(actions)
        fast_starts.append(start)
    fast_pixels_np = np.stack(fast_pixels)
    fast_actions_np = np.stack(fast_actions)
    fast_starts_np = np.asarray(fast_starts, dtype=np.int64)

    raw_checks = {
        "starts_exact": bool(np.array_equal(starts, fast_starts_np)),
        "pixels_exact": bool(np.array_equal(legacy_pixels, fast_pixels_np)),
        "actions_exact": bool(np.array_equal(legacy_actions, fast_actions_np)),
        "vectorized_actions_exact": bool(
            np.array_equal(legacy_actions, vectorized_actions)
        ),
    }

    action_dim = legacy_actions.shape[-1]
    action_mean = np.zeros((1, action_dim), dtype=np.float32)
    action_std = np.ones((1, action_dim), dtype=np.float32)

    legacy_start = time.perf_counter()
    legacy_emb, legacy_act = precompute_embeddings(
        model,
        args.data_file,
        spec,
        starts,
        cfg,
        action_mean,
        action_std,
        device,
        log_every=max(1, (sample_size + args.batch_size - 1) // args.batch_size),
        backend="legacy",
    )
    legacy_sec = time.perf_counter() - legacy_start

    fast_start = time.perf_counter()
    fast_emb, fast_act = precompute_embeddings(
        model,
        args.data_file,
        spec,
        starts,
        cfg,
        action_mean,
        action_std,
        device,
        log_every=max(1, (sample_size + args.batch_size - 1) // args.batch_size),
        backend="dataloader",
        num_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
    )
    fast_sec = time.perf_counter() - fast_start

    comparisons = {
        "emb": tensor_diff(legacy_emb, fast_emb, args.atol, args.rtol),
        "act_emb": tensor_diff(legacy_act, fast_act, args.atol, args.rtol),
    }
    passed = all(raw_checks.values()) and all(x["allclose"] for x in comparisons.values())
    result = {
        "status": "passed" if passed else "failed",
        "dataset": args.dataset,
        "data_file": str(args.data_file),
        "checkpoint": args.checkpoint,
        "starts_file": str(args.starts),
        "sample_size": sample_size,
        "sample_seed": args.sample_seed,
        "sample_starts_sha256": hashlib.sha256(starts.tobytes()).hexdigest(),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "prefetch_factor": args.prefetch_factor,
        "raw_checks": raw_checks,
        "comparisons": comparisons,
        "timing_sec": {
            "legacy": legacy_sec,
            "dataloader": fast_sec,
            "speedup": legacy_sec / fast_sec if fast_sec > 0 else None,
        },
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
