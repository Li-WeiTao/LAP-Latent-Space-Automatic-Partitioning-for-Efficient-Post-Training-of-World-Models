#!/usr/bin/env python3
"""Compare multi-step rollout deviation between two predictors on test-set transitions.

Given two predictor checkpoints and a maximum rollout horizon, roll out both
predictors from the same test-set initial histories (with ground-truth actions)
and report per-step average deviation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import DATASETS, choose_state_key, load_encoder  # noqa: E402
from predictor_rule_drift import (  # noqa: E402
    action_block_stats,
    infer_frameskip,
    json_ready,
    load_transition_cache,
    predict_next,
    read_action_blocks,
    read_sequence_dataset,
    sample_starts,
    save_transition_cache,
    valid_transition_starts,
)
from gauge_drift import preprocess_pixels  # noqa: E402


def load_predictor(path: Path, device: torch.device) -> torch.nn.Module:
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(path, map_location="cpu")
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def test_pool_starts(
    h5_path: Path,
    spec,
    state_key: str,
    seq_len: int,
    frameskip: int,
    train_fraction: float,
    split_seed: int,
    test_max_samples: int,
    seed: int,
) -> np.ndarray:
    with h5py.File(h5_path, "r") as h5:
        all_starts = valid_transition_starts(h5, spec, state_key, seq_len, frameskip, 0, seed)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(all_starts)
    train_n = int(round(len(perm) * train_fraction))
    test_pool = np.sort(perm[train_n:])
    return sample_starts(test_pool, test_max_samples, seed + 29)


@torch.no_grad()
def compare_rollout_deviation(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    latents: torch.Tensor,
    actions: torch.Tensor,
    history: int,
    max_steps: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Return per-step mean metrics over samples.

    Metrics at step h (1-indexed):
    - pairwise_mse: MSE between the two predictors' h-step predictions
    - predictor_a_mse_vs_gt / predictor_b_mse_vs_gt: MSE vs encoder ground truth
    """
    if latents.shape[1] < history + max_steps:
        raise ValueError(
            f"Need at least {history + max_steps} latent steps, got {latents.shape[1]}"
        )

    pairwise: list[list[float]] = [[] for _ in range(max_steps)]
    mse_a: list[list[float]] = [[] for _ in range(max_steps)]
    mse_b: list[list[float]] = [[] for _ in range(max_steps)]

    for offset in range(0, latents.shape[0], batch_size):
        y_true = latents[offset : offset + batch_size]
        a_true = actions[offset : offset + batch_size]

        emb_a = y_true[:, :history].clone()
        emb_b = y_true[:, :history].clone()

        for step in range(max_steps):
            ctx_emb_a = emb_a[:, -history:]
            ctx_emb_b = emb_b[:, -history:]
            ctx_action = a_true[:, step : step + history]

            pred_a = predict_next(model_a, ctx_emb_a, ctx_action)
            pred_b = predict_next(model_b, ctx_emb_b, ctx_action)
            target = y_true[:, history + step]

            pairwise[step].append(((pred_a - pred_b).pow(2).mean(dim=1)).detach().cpu().numpy())
            mse_a[step].append(((pred_a - target).pow(2).mean(dim=1)).detach().cpu().numpy())
            mse_b[step].append(((pred_b - target).pow(2).mean(dim=1)).detach().cpu().numpy())

            emb_a = torch.cat([emb_a, pred_a[:, None]], dim=1)
            emb_b = torch.cat([emb_b, pred_b[:, None]], dim=1)

    return {
        "pairwise_mse": np.array([np.concatenate(chunks) for chunks in pairwise], dtype=np.float64),
        "predictor_a_mse_vs_gt": np.array([np.concatenate(chunks) for chunks in mse_a], dtype=np.float64),
        "predictor_b_mse_vs_gt": np.array([np.concatenate(chunks) for chunks in mse_b], dtype=np.float64),
    }


def summarize_step_metrics(per_sample: dict[str, np.ndarray]) -> list[dict]:
    rows: list[dict] = []
    max_steps = per_sample["pairwise_mse"].shape[0]
    for step in range(max_steps):
        h = step + 1
        rows.append(
            {
                "step": h,
                "pairwise_mse_mean": float(per_sample["pairwise_mse"][step].mean()),
                "pairwise_mse_std": float(per_sample["pairwise_mse"][step].std()),
                "predictor_a_mse_vs_gt_mean": float(per_sample["predictor_a_mse_vs_gt"][step].mean()),
                "predictor_a_mse_vs_gt_std": float(per_sample["predictor_a_mse_vs_gt"][step].std()),
                "predictor_b_mse_vs_gt_mean": float(per_sample["predictor_b_mse_vs_gt"][step].mean()),
                "predictor_b_mse_vs_gt_std": float(per_sample["predictor_b_mse_vs_gt"][step].std()),
            }
        )
    return rows


def subsample_starts(starts: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or len(starts) <= max_samples:
        return starts
    return sample_starts(starts, max_samples, seed)


@torch.no_grad()
def encode_test_transitions_with_progress(
    encoder_model: torch.nn.Module,
    h5_path: Path,
    spec,
    test_starts: np.ndarray,
    seq_len: int,
    frameskip: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    batch_size: int,
    img_size: int,
    log_every: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        actions_raw = read_action_blocks(
            h5["action"], test_starts, seq_len - 1, frameskip
        ).astype(np.float32)
    actions_np = (actions_raw - action_mean.astype(np.float32)) / action_std.astype(np.float32)
    pad = np.zeros((actions_np.shape[0], 1, actions_np.shape[2]), dtype=np.float32)
    actions_np = np.concatenate([actions_np, pad], axis=1)

    encoder_model.eval()
    emb_chunks: list[np.ndarray] = []
    num_batches = (len(test_starts) + batch_size - 1) // batch_size
    print(f"[encode] {len(test_starts)} test trajectories, {num_batches} batches", flush=True)
    with h5py.File(h5_path, "r") as h5:
        for batch_idx, offset in enumerate(range(0, len(test_starts), batch_size)):
            batch_starts = test_starts[offset : offset + batch_size]
            pixels_np = read_sequence_dataset(
                h5[spec.pixel_key], batch_starts, seq_len, frameskip
            )
            b, t = pixels_np.shape[:2]
            pixels = pixels_np.reshape(b * t, *pixels_np.shape[2:])
            pixels = preprocess_pixels(pixels, device, img_size)
            pixels = pixels.reshape(b, t, *pixels.shape[1:])
            out = encoder_model.encode({"pixels": pixels})
            emb_chunks.append(out["emb"].detach().cpu().numpy())
            if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == num_batches:
                done = min(offset + batch_size, len(test_starts))
                print(
                    f"  [encode] batch {batch_idx + 1}/{num_batches} ({done}/{len(test_starts)})",
                    flush=True,
                )
    return np.concatenate(emb_chunks, axis=0), actions_np


def load_or_build_test_transitions(
    args: argparse.Namespace,
    spec,
    h5_path: Path,
    encoder_model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if args.load_test_cache is not None:
        latents_np, actions_np, starts, metadata = load_transition_cache(args.load_test_cache)
        if int(metadata.get("seq_len", seq_len)) < seq_len:
            raise ValueError(
                f"Cached seq_len={metadata.get('seq_len')} < required seq_len={seq_len}"
            )
        if int(metadata.get("seq_len", seq_len)) > seq_len:
            latents_np = latents_np[:, :seq_len]
            actions_np = actions_np[:, :seq_len]
        print(f"[cache] loaded test transitions from {args.load_test_cache}", flush=True)
        return latents_np, actions_np, starts, metadata

    with h5py.File(h5_path, "r") as h5:
        state_key = choose_state_key(h5, spec, args.state_key)
        frameskip = infer_frameskip(encoder_model, int(h5["action"].shape[1]), args.frameskip)

    if args.test_starts is not None:
        test_starts = np.load(args.test_starts).astype(np.int64)
        print(f"[data] loaded {len(test_starts)} test starts from {args.test_starts}", flush=True)
    else:
        test_starts = test_pool_starts(
            h5_path,
            spec,
            state_key,
            seq_len,
            frameskip,
            args.train_fraction,
            args.split_seed,
            args.test_max_samples,
            args.seed,
        )
        print(f"[data] sampled {len(test_starts)} test starts from held-out pool", flush=True)

    test_starts = subsample_starts(test_starts, args.test_max_samples, args.seed + 29)
    if args.test_max_samples > 0:
        print(f"[data] using {len(test_starts)} test starts after subsampling", flush=True)

    with h5py.File(h5_path, "r") as h5:
        rng = np.random.default_rng(args.split_seed)
        perm = rng.permutation(
            valid_transition_starts(
                h5, spec, state_key, seq_len, frameskip, 0, args.seed
            )
        )
        train_n = int(round(len(perm) * args.train_fraction))
        train_pool = np.sort(perm[:train_n])
        train_starts = sample_starts(train_pool, args.train_max_samples, args.seed + 17)
        train_actions_raw = read_action_blocks(
            h5["action"], train_starts, seq_len - 1, frameskip
        ).astype(np.float32)
    action_mean, action_std = action_block_stats(train_actions_raw)

    latents_np, actions_np = encode_test_transitions_with_progress(
        encoder_model,
        h5_path,
        spec,
        test_starts,
        seq_len,
        frameskip,
        action_mean,
        action_std,
        device,
        args.batch_size,
        args.img_size,
        log_every=args.encode_log_every,
    )

    metadata = {
        "dataset": args.dataset,
        "data_file": str(h5_path),
        "state_key": state_key,
        "num_starts": int(len(test_starts)),
        "seq_len": seq_len,
        "history_size": args.history_size,
        "frameskip": frameskip,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_fraction": args.train_fraction,
        "test_max_samples": args.test_max_samples,
        "encoder_checkpoint": args.encoder_checkpoint,
        "action_mean": action_mean.astype(np.float32),
        "action_std": action_std.astype(np.float32),
        "source": "test_pool",
    }

    if args.save_test_cache is not None:
        save_transition_cache(
            args.save_test_cache,
            latents_np,
            actions_np,
            test_starts,
            metadata,
        )
        print(f"[cache] saved test transitions to {args.save_test_cache}", flush=True)

    return latents_np, actions_np, test_starts, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-a", type=Path, required=True, help="First predictor checkpoint")
    parser.add_argument("--predictor-b", type=Path, required=True, help="Second predictor checkpoint")
    parser.add_argument("--predictor-a-name", default="predictor_a")
    parser.add_argument("--predictor-b-name", default="predictor_b")
    parser.add_argument(
        "--encoder-checkpoint",
        default="/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt",
        help="Frozen encoder used to build ground-truth test latents",
    )
    parser.add_argument("--checkpoint-cache-dir", default=None)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="tworoom")
    parser.add_argument("--data-root", type=Path, default=Path("/data/sicong/weitao/datasets/lewm"))
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--frameskip", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--train-max-samples", type=int, default=5000)
    parser.add_argument("--test-max-samples", type=int, default=0, help="0 = use all held-out test transitions")
    parser.add_argument("--encode-log-every", type=int, default=20)
    parser.add_argument(
        "--test-starts",
        type=Path,
        default=None,
        help="Optional precomputed test transition starts (.npy). Default: derive full test pool from split.",
    )
    parser.add_argument("--load-test-cache", type=Path, default=None)
    parser.add_argument("--save-test-cache", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/tworoom/results/tworoom_trajectory_deviation"),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test_starts is not None and str(args.test_starts) == "":
        args.test_starts = None

    device = resolve_device(args.device)
    seq_len = args.history_size + args.max_steps
    spec = DATASETS[args.dataset]
    h5_path = args.data_file or (args.data_root / spec.default_file)

    print(f"[load] predictor A: {args.predictor_a}", flush=True)
    print(f"[load] predictor B: {args.predictor_b}", flush=True)
    model_a = load_predictor(args.predictor_a, device)
    model_b = load_predictor(args.predictor_b, device)

    encoder = load_encoder(args.encoder_checkpoint, device, args.checkpoint_cache_dir)
    encoder.eval()

    latents_np, actions_np, test_starts, cache_metadata = load_or_build_test_transitions(
        args, spec, h5_path, encoder, device, seq_len
    )
    latents = torch.as_tensor(latents_np, device=device)
    actions = torch.as_tensor(actions_np, device=device)

    print(
        f"[rollout] {len(test_starts)} test samples, history={args.history_size}, "
        f"max_steps={args.max_steps}",
        flush=True,
    )
    per_sample = compare_rollout_deviation(
        model_a,
        model_b,
        latents,
        actions,
        args.history_size,
        args.max_steps,
        args.batch_size,
    )
    rows = summarize_step_metrics(per_sample)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "trajectory_deviation.csv"
    json_path = args.out_dir / "trajectory_deviation.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "config": json_ready(vars(args)),
        "cache_metadata": json_ready(cache_metadata),
        "num_test_samples": int(len(test_starts)),
        "predictor_a": str(args.predictor_a),
        "predictor_b": str(args.predictor_b),
        "predictor_a_name": args.predictor_a_name,
        "predictor_b_name": args.predictor_b_name,
        "metrics": rows,
    }
    with json_path.open("w") as f:
        json.dump(json_ready(payload), f, indent=2)

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print("step  pairwise_mse  A_vs_gt  B_vs_gt", flush=True)
    for row in rows:
        print(
            f"{row['step']:4d}  "
            f"{row['pairwise_mse_mean']:.6f}  "
            f"{row['predictor_a_mse_vs_gt_mean']:.6f}  "
            f"{row['predictor_b_mse_vs_gt_mean']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
