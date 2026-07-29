#!/usr/bin/env python3
"""Diagnose region-dependent predictor dynamics rule drift on LeWM datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import (  # noqa: E402
    DATASETS,
    finite_rows,
    fit_pca,
    load_encoder,
    preprocess_pixels,
    read_columns,
    split_reference_and_iid,
    standardize_from_reference,
    transform_with_pca,
)


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def choose_state_key(h5: h5py.File, spec, requested: str | None) -> str:
    if requested:
        if requested not in h5:
            raise KeyError(f"Requested state key '{requested}' not found")
        return requested
    for key in spec.state_keys:
        if key in h5:
            return key
    raise KeyError(f"No state proxy found. Tried {spec.state_keys}")


def valid_transition_starts(
    h5: h5py.File,
    spec,
    state_key: str,
    seq_len: int,
    step_stride: int,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    n = h5[state_key].shape[0]
    end = np.arange(n) + (seq_len - 1) * step_stride
    valid = end < n
    if "ep_idx" in h5:
        ep = h5["ep_idx"][:]
        valid &= ep == ep[np.minimum(end, n - 1)]
    if "step_idx" in h5 and "ep_len" in h5:
        step = h5["step_idx"][:]
        ep_len = h5["ep_len"][:]
        valid &= step + (seq_len - 1) * step_stride < ep_len[h5["ep_idx"][:]]
    valid &= finite_rows(h5[state_key][:])
    valid &= finite_rows(h5["action"][:])
    if spec.pixel_key in h5:
        valid &= end < h5[spec.pixel_key].shape[0]

    starts = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)
    if max_samples > 0 and len(starts) > max_samples:
        starts = rng.choice(starts, size=max_samples, replace=False)
    return np.sort(starts.astype(np.int64))


def read_sequence_dataset(dataset, starts: np.ndarray, seq_len: int, step_stride: int) -> np.ndarray:
    pieces = [dataset[start + np.arange(seq_len) * step_stride] for start in starts]
    return np.stack(pieces, axis=0)


def read_action_blocks(dataset, starts: np.ndarray, num_steps: int, frameskip: int) -> np.ndarray:
    rows = []
    for start in starts:
        blocks = []
        for step in range(num_steps):
            lo = start + step * frameskip
            hi = lo + frameskip
            blocks.append(np.asarray(dataset[lo:hi]).reshape(-1))
        rows.append(np.stack(blocks, axis=0))
    return np.stack(rows, axis=0)


def action_block_stats(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = blocks.reshape(-1, blocks.shape[-1]).astype(np.float64)
    flat = flat[np.isfinite(flat).all(axis=1)]
    return flat.mean(axis=0, keepdims=True), flat.std(axis=0, keepdims=True) + 1e-8


@torch.no_grad()
def encode_sequences(
    model,
    h5_path: Path,
    spec,
    starts: np.ndarray,
    seq_len: int,
    step_stride: int,
    device: torch.device,
    batch_size: int,
    img_size: int,
) -> np.ndarray:
    chunks = []
    with h5py.File(h5_path, "r") as h5:
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            pixels_np = read_sequence_dataset(h5[spec.pixel_key], batch_starts, seq_len, step_stride)
            b, t = pixels_np.shape[:2]
            pixels = pixels_np.reshape(b * t, *pixels_np.shape[2:])
            pixels = preprocess_pixels(pixels, device, img_size)
            pixels = pixels.reshape(b, t, *pixels.shape[1:])
            out = model.encode({"pixels": pixels})
            chunks.append(out["emb"].detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def make_splits(
    h5_path: Path,
    spec,
    starts: np.ndarray,
    state_key: str,
    args,
) -> tuple[dict, str, np.ndarray, dict]:
    with h5py.File(h5_path, "r") as h5:
        split_keys = sorted(
            set(spec.state_keys)
            | {"pos_agent", "pos_target", "goal_state", "goal_proprio", state_key}
        )
        cols = read_columns(h5, split_keys, starts)
        state = np.asarray(h5[state_key][starts], dtype=np.float64)
    raw_splits = spec.split_fn(cols)
    raw_splits = {
        name: np.asarray(mask, dtype=bool) & finite_rows(state)
        for name, mask in raw_splits.items()
        if len(mask) == len(starts)
    }
    split_masks, ref_name = split_reference_and_iid(
        splits=raw_splits,
        reference_split=args.reference_split,
        seed=args.seed,
        reference_fraction=args.reference_fraction,
        min_split_size=args.min_split_size,
        disjoint_reference=args.disjoint_reference,
    )
    return split_masks, ref_name, state, raw_splits


def make_raw_splits(h5_path: Path, spec, starts: np.ndarray, state_key: str) -> tuple[dict, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        split_keys = sorted(
            set(spec.state_keys)
            | {"pos_agent", "pos_target", "goal_state", "goal_proprio", state_key}
        )
        cols = read_columns(h5, split_keys, starts)
        state = np.asarray(h5[state_key][starts], dtype=np.float64)
    raw_splits = spec.split_fn(cols)
    raw_splits = {
        name: np.asarray(mask, dtype=bool) & finite_rows(state)
        for name, mask in raw_splits.items()
        if len(mask) == len(starts)
    }
    raw_splits["test_all"] = np.ones(len(starts), dtype=bool)
    return raw_splits, state


def prepare_projection(latents: np.ndarray, ref_mask: np.ndarray, align_dim: int | None):
    flat = latents[:, 0, :]
    latent_z, mean, std = standardize_from_reference(flat, ref_mask)
    max_dim = min(flat.shape[1], int(ref_mask.sum()) - 1)
    dim = max(1, min(int(align_dim or 2), max_dim))
    pca_mean, basis = fit_pca(latent_z[ref_mask], dim)
    projected = transform_with_pca(latent_z, pca_mean, basis)
    return {
        "dim": dim,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "pca_mean": pca_mean.astype(np.float32),
        "basis": basis.astype(np.float32),
        "projected_reference_var": projected[ref_mask].var(axis=0).tolist(),
    }


def predict_next(model, ctx_emb: torch.Tensor, ctx_action: torch.Tensor) -> torch.Tensor:
    act_emb = model.action_encoder(ctx_action)
    return model.predict(ctx_emb, act_emb)[:, -1, :]


def rollout_errors(
    model,
    latents: torch.Tensor,
    actions: torch.Tensor,
    history: int,
    horizons: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    max_h = max(horizons)
    out = {h: [] for h in horizons}
    for offset in range(0, latents.shape[0], batch_size):
        y_true = latents[offset : offset + batch_size]
        a_true = actions[offset : offset + batch_size]
        emb = y_true[:, :history].clone()
        for step in range(max_h):
            ctx_emb = emb[:, -history:]
            ctx_action = a_true[:, step : step + history]
            pred = predict_next(model, ctx_emb, ctx_action)
            emb = torch.cat([emb, pred[:, None]], dim=1)
            h = step + 1
            if h in out:
                target = y_true[:, history + step]
                mse = (pred - target).pow(2).mean(dim=1).detach().cpu().numpy()
                out[h].append(mse)
    return {h: np.concatenate(vals, axis=0) for h, vals in out.items()}


def mean_projected_jacobian(
    model,
    latents: torch.Tensor,
    actions: torch.Tensor,
    indices: np.ndarray,
    history: int,
    proj: dict,
    device: torch.device,
    jacobian_batch_size: int,
) -> tuple[np.ndarray, float]:
    mean = torch.as_tensor(proj["mean"], device=device)
    std = torch.as_tensor(proj["std"], device=device)
    pca_mean = torch.as_tensor(proj["pca_mean"], device=device)
    basis = torch.as_tensor(proj["basis"], device=device)

    if len(indices) == 0:
        width = history * (latents.shape[-1] + actions.shape[-1])
        return np.full((proj["dim"], width), np.nan), float("nan")

    total = None
    batch_means = []
    seen = 0
    for start in range(0, len(indices), jacobian_batch_size):
        batch_idx = indices[start : start + jacobian_batch_size]
        batch_n = len(batch_idx)
        ctx_emb = latents[batch_idx, :history].detach().clone().requires_grad_(True)
        ctx_action = actions[batch_idx, :history].detach().clone().requires_grad_(True)
        pred = predict_next(model, ctx_emb, ctx_action)
        pred_proj = ((pred - mean) / std - pca_mean) @ basis

        jac_rows = []
        for col in range(pred_proj.shape[1]):
            # Gradient of the batch mean gives per-sample gradients scaled by 1/B.
            # Summing over the batch recovers the mean Jacobian row for this split.
            grad_emb, grad_action = torch.autograd.grad(
                pred_proj[:, col].mean(),
                (ctx_emb, ctx_action),
                retain_graph=col + 1 < pred_proj.shape[1],
                allow_unused=False,
            )
            jac_rows.append(
                torch.cat(
                    [
                        grad_emb.reshape(batch_n, -1).sum(dim=0),
                        grad_action.reshape(batch_n, -1).sum(dim=0),
                    ]
                )
                .detach()
                .cpu()
                .numpy()
            )
        batch_mean = np.stack(jac_rows, axis=0)
        total = batch_mean * batch_n if total is None else total + batch_mean * batch_n
        batch_means.append(batch_mean)
        seen += batch_n

    mean_jac = total / max(seen, 1)
    batch_std = float(np.stack(batch_means, axis=0).reshape(len(batch_means), -1).std(axis=0).mean())
    return mean_jac, batch_std


def split_sample(mask: np.ndarray, n: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    if len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    return np.sort(idx)


def iid_bootstrap_rule_drift(
    model,
    latents: torch.Tensor,
    actions: torch.Tensor,
    base_mask: np.ndarray,
    history: int,
    proj: dict,
    device: torch.device,
    jacobian_samples: int,
    jacobian_batch_size: int,
    reference_fraction: float,
    min_split_size: int,
    trials: int,
    seed: int,
) -> list[dict]:
    base_idx = np.flatnonzero(base_mask)
    if trials <= 0 or len(base_idx) < 2 * min_split_size:
        return []

    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(trials):
        perm = rng.permutation(base_idx)
        ref_n = int(round(len(perm) * reference_fraction))
        ref_n = min(max(ref_n, min_split_size), len(perm) - min_split_size)
        ref_pool = perm[:ref_n]
        holdout_pool = perm[ref_n:]

        ref_take = min(jacobian_samples, len(ref_pool))
        holdout_take = min(jacobian_samples, len(holdout_pool))
        ref_idx = np.sort(rng.choice(ref_pool, size=ref_take, replace=False))
        holdout_idx = np.sort(rng.choice(holdout_pool, size=holdout_take, replace=False))

        j_ref, _ = mean_projected_jacobian(
            model,
            latents,
            actions,
            ref_idx,
            history,
            proj,
            device,
            jacobian_batch_size,
        )
        j_holdout, _ = mean_projected_jacobian(
            model,
            latents,
            actions,
            holdout_idx,
            history,
            proj,
            device,
            jacobian_batch_size,
        )
        ref_norm = np.linalg.norm(j_ref, ord="fro") + 1e-12
        rows.append(
            {
                "trial": trial,
                "reference_n": int(len(ref_idx)),
                "holdout_n": int(len(holdout_idx)),
                "iid_rule_drift": float(np.linalg.norm(j_holdout - j_ref, ord="fro") / ref_norm),
                "iid_rule_diff_fro": float(np.linalg.norm(j_holdout - j_ref, ord="fro")),
                "iid_reference_jacobian_fro": float(ref_norm),
                "iid_holdout_jacobian_fro": float(np.linalg.norm(j_holdout, ord="fro")),
            }
        )
    return rows


def pool_to_reference_bootstrap(
    model,
    latents: torch.Tensor,
    actions: torch.Tensor,
    pool_mask: np.ndarray,
    j_ref: np.ndarray,
    ref_norm: float,
    history: int,
    proj: dict,
    device: torch.device,
    jacobian_samples: int,
    jacobian_batch_size: int,
    trials: int,
    seed: int,
) -> list[dict]:
    pool_idx = np.flatnonzero(pool_mask)
    if trials <= 0 or len(pool_idx) == 0:
        return []
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(trials):
        take = min(jacobian_samples, len(pool_idx))
        idx = np.sort(rng.choice(pool_idx, size=take, replace=False))
        j_mean, _ = mean_projected_jacobian(
            model,
            latents,
            actions,
            idx,
            history,
            proj,
            device,
            jacobian_batch_size,
        )
        rows.append(
            {
                "trial": trial,
                "sample_n": int(len(idx)),
                "iid_rule_drift": float(np.linalg.norm(j_mean - j_ref, ord="fro") / ref_norm),
                "iid_rule_diff_fro": float(np.linalg.norm(j_mean - j_ref, ord="fro")),
                "iid_reference_jacobian_fro": float(ref_norm),
                "iid_holdout_jacobian_fro": float(np.linalg.norm(j_mean, ord="fro")),
            }
        )
    return rows


def parse_horizons(text: str) -> list[int]:
    vals = [int(x) for x in text.split(",") if x.strip()]
    if not vals or min(vals) < 1:
        raise ValueError("--rollout-horizons must contain positive integers")
    return sorted(set(vals))


def save_transition_cache(
    path: Path,
    latents: np.ndarray,
    actions: np.ndarray,
    starts: np.ndarray,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        latents=latents.astype(np.float32),
        actions=actions.astype(np.float32),
        starts=starts.astype(np.int64),
        metadata=json.dumps(json_ready(metadata)),
    )


def load_transition_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    return (
        data["latents"],
        data["actions"],
        data["starts"],
        json.loads(str(data["metadata"])),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data/sicong/weitao/datasets/lewm"))
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-cache-dir", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-mode", choices=("within-split", "train-test"), default="within-split")
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--train-max-samples", type=int, default=5000)
    parser.add_argument("--test-max-samples", type=int, default=5000)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--jacobian-samples", type=int, default=64)
    parser.add_argument("--jacobian-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=0)
    parser.add_argument("--rollout-horizons", default="1,5,10")
    parser.add_argument("--align-dim", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-split", default="common")
    parser.add_argument("--reference-fraction", type=float, default=0.5)
    parser.add_argument("--disjoint-reference", action="store_true")
    parser.add_argument("--min-split-size", type=int, default=256)
    parser.add_argument("--iid-bootstrap-trials", type=int, default=0)
    parser.add_argument("--include-reference-source-split", action="store_true")
    parser.add_argument("--save-transition-cache", type=Path, default=None)
    parser.add_argument("--load-transition-cache", type=Path, default=None)
    return parser.parse_args()


def infer_frameskip(model, raw_action_dim: int, requested: int) -> int:
    if requested > 0:
        return requested
    expected = getattr(getattr(model, "action_encoder", None), "patch_embed", None)
    expected_dim = getattr(expected, "in_channels", None)
    if expected_dim is None:
        return 1
    if expected_dim % raw_action_dim != 0:
        raise ValueError(
            f"Cannot infer frameskip: action_encoder expects {expected_dim}, raw action dim is {raw_action_dim}"
        )
    return max(1, int(expected_dim // raw_action_dim))


def sample_starts(pool: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(pool, dtype=np.int64)
    if max_samples > 0 and len(out) > max_samples:
        out = rng.choice(out, size=max_samples, replace=False)
    return np.sort(out)


def encode_and_normalize_actions(
    model,
    h5_path: Path,
    spec,
    starts: np.ndarray,
    seq_len: int,
    frameskip: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    batch_size: int,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        actions_raw = read_action_blocks(h5["action"], starts, seq_len - 1, frameskip).astype(np.float32)
    latents_np = encode_sequences(
        model=model,
        h5_path=h5_path,
        spec=spec,
        starts=starts,
        seq_len=seq_len,
        step_stride=frameskip,
        device=device,
        batch_size=batch_size,
        img_size=img_size,
    )
    actions_np = ((actions_raw - action_mean.astype(np.float32)) / action_std.astype(np.float32))
    pad = np.zeros((actions_np.shape[0], 1, actions_np.shape[2]), dtype=np.float32)
    actions_np = np.concatenate([actions_np, pad], axis=1)
    return latents_np, actions_np


def run_train_test(args, spec, h5_path: Path, horizons: list[int], seq_len: int, device: torch.device, model) -> None:
    if args.save_transition_cache is not None or args.load_transition_cache is not None:
        raise ValueError("Transition cache is only supported for --eval-mode within-split")

    with h5py.File(h5_path, "r") as h5:
        state_key = choose_state_key(h5, spec, args.state_key)
        raw_action_dim = int(h5["action"].shape[1])
        frameskip = infer_frameskip(model, raw_action_dim, args.frameskip)
        all_starts = valid_transition_starts(h5, spec, state_key, seq_len, frameskip, 0, args.seed)

    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(all_starts)
    train_n = int(round(len(perm) * args.train_fraction))
    train_pool = np.sort(perm[:train_n])
    test_pool = np.sort(perm[train_n:])
    train_starts = sample_starts(train_pool, args.train_max_samples, args.seed + 17)
    test_starts = sample_starts(test_pool, args.test_max_samples, args.seed + 29)

    with h5py.File(h5_path, "r") as h5:
        train_actions_raw = read_action_blocks(h5["action"], train_starts, seq_len - 1, frameskip).astype(np.float32)
        action_mean, action_std = action_block_stats(train_actions_raw)

    train_latents_np, train_actions_np = encode_and_normalize_actions(
        model,
        h5_path,
        spec,
        train_starts,
        seq_len,
        frameskip,
        action_mean,
        action_std,
        device,
        args.batch_size,
        args.img_size,
    )
    test_latents_np, test_actions_np = encode_and_normalize_actions(
        model,
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
    )

    train_ref_mask = np.ones(len(train_starts), dtype=bool)
    proj = prepare_projection(train_latents_np, train_ref_mask, args.align_dim)
    train_latents = torch.as_tensor(train_latents_np, device=device)
    train_actions = torch.as_tensor(train_actions_np, device=device)
    test_latents = torch.as_tensor(test_latents_np, device=device)
    test_actions = torch.as_tensor(test_actions_np, device=device)

    ref_j_idx = split_sample(train_ref_mask, args.jacobian_samples, args.seed + 1009)
    j_ref, j_ref_std = mean_projected_jacobian(
        model,
        train_latents,
        train_actions,
        ref_j_idx,
        args.history_size,
        proj,
        device,
        args.jacobian_batch_size,
    )
    ref_norm = np.linalg.norm(j_ref, ord="fro") + 1e-12

    raw_splits, test_state = make_raw_splits(h5_path, spec, test_starts, state_key)
    rollout = rollout_errors(
        model=model,
        latents=test_latents,
        actions=test_actions,
        history=args.history_size,
        horizons=horizons,
        batch_size=args.batch_size,
    )

    iid_bootstrap_rows = pool_to_reference_bootstrap(
        model=model,
        latents=test_latents,
        actions=test_actions,
        pool_mask=raw_splits["test_all"],
        j_ref=j_ref,
        ref_norm=ref_norm,
        history=args.history_size,
        proj=proj,
        device=device,
        jacobian_samples=args.jacobian_samples,
        jacobian_batch_size=args.jacobian_batch_size,
        trials=args.iid_bootstrap_trials,
        seed=args.seed + 12345,
    )
    iid_values = np.asarray([row["iid_rule_drift"] for row in iid_bootstrap_rows], dtype=np.float64)
    iid_mean = float(iid_values.mean()) if len(iid_values) else float("nan")
    iid_std = float(iid_values.std(ddof=1)) if len(iid_values) > 1 else float("nan")

    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jacobian_dir = args.out_dir / "jacobians"
    jacobian_dir.mkdir(parents=True, exist_ok=True)
    np.save(jacobian_dir / "train_global_mean_projected_jacobian.npy", j_ref)

    for name, mask in sorted(raw_splits.items()):
        n = int(mask.sum())
        if n < args.min_split_size:
            continue
        one_step = rollout[1][mask]
        row = {
            "split": name,
            "n": n,
            "one_step_mse": float(np.mean(one_step)),
            "one_step_mse_std": float(np.std(one_step)),
        }
        for horizon in horizons:
            vals = rollout[horizon][mask]
            row[f"rollout_h{horizon}_mse"] = float(np.mean(vals))
            row[f"rollout_h{horizon}_mse_std"] = float(np.std(vals))

        j_idx = split_sample(mask, args.jacobian_samples, args.seed + 7919 + sum(map(ord, name)))
        j_mean, j_within_std = mean_projected_jacobian(
            model,
            test_latents,
            test_actions,
            j_idx,
            args.history_size,
            proj,
            device,
            args.jacobian_batch_size,
        )
        np.save(jacobian_dir / f"{name}_mean_projected_jacobian.npy", j_mean)
        rule_drift = float(np.linalg.norm(j_mean - j_ref, ord="fro") / ref_norm)
        rule_excess = float(rule_drift - iid_mean) if np.isfinite(iid_mean) else float("nan")
        rule_z = (
            float((rule_drift - iid_mean) / (iid_std + 1e-12))
            if np.isfinite(iid_mean) and np.isfinite(iid_std)
            else float("nan")
        )
        row.update(
            {
                "jacobian_samples": int(len(j_idx)),
                "rule_drift_to_train": rule_drift,
                "rule_diff_fro": float(np.linalg.norm(j_mean - j_ref, ord="fro")),
                "train_reference_jacobian_fro": float(ref_norm),
                "mean_jacobian_fro": float(np.linalg.norm(j_mean, ord="fro")),
                "mean_within_jacobian_std": j_within_std,
                "train_reference_within_jacobian_std": j_ref_std,
                "test_iid_bootstrap_mean": iid_mean,
                "test_iid_bootstrap_std": iid_std,
                "rule_excess_vs_test_iid": rule_excess,
                "rule_z_vs_test_iid": rule_z,
            }
        )
        rows.append(row)

    with (args.out_dir / "predictor_rule_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["split"])
        writer.writeheader()
        writer.writerows(rows)
    if iid_bootstrap_rows:
        with (args.out_dir / "iid_bootstrap.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(iid_bootstrap_rows[0].keys()))
            writer.writeheader()
            writer.writerows(iid_bootstrap_rows)
    with (args.out_dir / "predictor_rule_metrics.json").open("w") as f:
        json.dump(
            {
                "config": json_ready(vars(args)),
                "metadata": {
                    "eval_mode": "train-test",
                    "dataset": args.dataset,
                    "data_file": str(h5_path),
                    "state_key": state_key,
                    "num_valid_starts": int(len(all_starts)),
                    "train_pool_size": int(len(train_pool)),
                    "test_pool_size": int(len(test_pool)),
                    "train_num_starts": int(len(train_starts)),
                    "test_num_starts": int(len(test_starts)),
                    "seq_len": int(seq_len),
                    "history_size": int(args.history_size),
                    "frameskip": int(frameskip),
                    "train_fraction": float(args.train_fraction),
                    "split_seed": int(args.split_seed),
                    "rollout_horizons": horizons,
                    "available_test_splits": {k: int(v.sum()) for k, v in raw_splits.items()},
                    "jacobian_batch_size": int(args.jacobian_batch_size),
                    "iid_bootstrap_trials": int(args.iid_bootstrap_trials),
                    "test_iid_bootstrap_mean": iid_mean,
                    "test_iid_bootstrap_std": iid_std,
                    "projection": json_ready(proj),
                    "action_mean": action_mean.squeeze(0).tolist(),
                    "action_std": action_std.squeeze(0).tolist(),
                    "test_state_shape": list(test_state.shape),
                    "train_latent_shape": list(train_latents_np.shape),
                    "test_latent_shape": list(test_latents_np.shape),
                    "cwd": os.getcwd(),
                },
                "metrics": rows,
                "iid_bootstrap": iid_bootstrap_rows,
            },
            f,
            indent=2,
        )
    np.save(args.out_dir / "train_transition_starts.npy", train_starts)
    np.save(args.out_dir / "test_transition_starts.npy", test_starts)
    print(json.dumps({"metrics": rows}, indent=2))
    print(f"Wrote train-test predictor rule drift results to {args.out_dir}")


def main() -> None:
    args = parse_args()
    if args.save_transition_cache is not None and args.load_transition_cache is not None:
        raise ValueError("Use only one of --save-transition-cache or --load-transition-cache")
    spec = DATASETS[args.dataset]
    h5_path = args.data_file or (args.data_root / spec.default_file)
    horizons = parse_horizons(args.rollout_horizons)
    seq_len = args.history_size + max(horizons)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = load_encoder(args.checkpoint, device, args.checkpoint_cache_dir)
    model.eval()

    if args.eval_mode == "train-test":
        run_train_test(args, spec, h5_path, horizons, seq_len, device, model)
        return

    cache_metadata = {}
    if args.load_transition_cache is not None:
        latents_np, actions_np, starts, cache_metadata = load_transition_cache(args.load_transition_cache)
        state_key = cache_metadata.get("state_key", args.state_key)
        if state_key is None:
            raise ValueError("Cached transitions do not record state_key; pass --state-key")
        frameskip = int(cache_metadata.get("frameskip", args.frameskip or 1))
        action_mean = np.asarray(cache_metadata.get("action_mean", [[0.0]]), dtype=np.float32)
        action_std = np.asarray(cache_metadata.get("action_std", [[1.0]]), dtype=np.float32)
        cached_seq_len = int(cache_metadata.get("seq_len", seq_len))
        if cached_seq_len < seq_len:
            raise ValueError(
                f"Cached seq_len={cached_seq_len} is smaller than requested seq_len={seq_len}"
            )
        if cached_seq_len > seq_len:
            latents_np = latents_np[:, :seq_len]
            actions_np = actions_np[:, :seq_len]
    else:
        with h5py.File(h5_path, "r") as h5:
            state_key = choose_state_key(h5, spec, args.state_key)
            raw_action_dim = int(h5["action"].shape[1])
            frameskip = infer_frameskip(model, raw_action_dim, args.frameskip)
            starts = valid_transition_starts(
                h5, spec, state_key, seq_len, frameskip, args.max_samples, args.seed
            )
            actions_raw = read_action_blocks(h5["action"], starts, seq_len - 1, frameskip).astype(np.float32)
            action_mean, action_std = action_block_stats(actions_raw)

        latents_np = encode_sequences(
            model=model,
            h5_path=h5_path,
            spec=spec,
            starts=starts,
            seq_len=seq_len,
            step_stride=frameskip,
            device=device,
            batch_size=args.batch_size,
            img_size=args.img_size,
        )
        actions_np = ((actions_raw - action_mean.astype(np.float32)) / action_std.astype(np.float32))
        pad = np.zeros((actions_np.shape[0], 1, actions_np.shape[2]), dtype=np.float32)
        actions_np = np.concatenate([actions_np, pad], axis=1)
        if args.save_transition_cache is not None:
            save_transition_cache(
                args.save_transition_cache,
                latents=latents_np,
                actions=actions_np,
                starts=starts,
                metadata={
                    "dataset": args.dataset,
                    "data_file": str(h5_path),
                    "state_key": state_key,
                    "num_starts": int(len(starts)),
                    "seq_len": int(seq_len),
                    "history_size": int(args.history_size),
                    "frameskip": int(frameskip),
                    "rollout_horizons": horizons,
                    "seed": int(args.seed),
                    "checkpoint": args.checkpoint,
                    "action_mean": action_mean.tolist(),
                    "action_std": action_std.tolist(),
                },
            )

    split_masks, ref_name, state, raw_splits = make_splits(h5_path, spec, starts, state_key, args)

    proj = prepare_projection(latents_np, split_masks[ref_name], args.align_dim)
    latents = torch.as_tensor(latents_np, device=device)
    actions = torch.as_tensor(actions_np, device=device)

    rollout = rollout_errors(
        model=model,
        latents=latents,
        actions=actions,
        history=args.history_size,
        horizons=horizons,
        batch_size=args.batch_size,
    )

    ref_j_idx = split_sample(
        split_masks[ref_name],
        args.jacobian_samples,
        args.seed + 1009,
    )
    j_ref, j_ref_std = mean_projected_jacobian(
        model,
        latents,
        actions,
        ref_j_idx,
        args.history_size,
        proj,
        device,
        args.jacobian_batch_size,
    )
    ref_norm = np.linalg.norm(j_ref, ord="fro") + 1e-12

    iid_bootstrap_rows = iid_bootstrap_rule_drift(
        model=model,
        latents=latents,
        actions=actions,
        base_mask=raw_splits[args.reference_split],
        history=args.history_size,
        proj=proj,
        device=device,
        jacobian_samples=args.jacobian_samples,
        jacobian_batch_size=args.jacobian_batch_size,
        reference_fraction=args.reference_fraction,
        min_split_size=args.min_split_size,
        trials=args.iid_bootstrap_trials,
        seed=args.seed + 12345,
    )
    iid_values = np.asarray([row["iid_rule_drift"] for row in iid_bootstrap_rows], dtype=np.float64)
    iid_mean = float(iid_values.mean()) if len(iid_values) else float("nan")
    iid_std = float(iid_values.std(ddof=1)) if len(iid_values) > 1 else float("nan")

    rows = []
    jacobian_dir = args.out_dir / "jacobians"
    jacobian_dir.mkdir(parents=True, exist_ok=True)
    np.save(jacobian_dir / f"{ref_name}_mean_projected_jacobian.npy", j_ref)

    for name, mask in sorted(split_masks.items()):
        if (
            name == args.reference_split
            and name != ref_name
            and not args.include_reference_source_split
        ):
            continue
        report_name = "iid_holdout" if name == "iid_nonoverlap" else name
        n = int(mask.sum())
        if n < args.min_split_size:
            continue
        one_step = rollout[1][mask]
        row = {
            "split": report_name,
            "n": n,
            "one_step_mse": float(np.mean(one_step)),
            "one_step_mse_std": float(np.std(one_step)),
        }
        for horizon in horizons:
            vals = rollout[horizon][mask]
            row[f"rollout_h{horizon}_mse"] = float(np.mean(vals))
            row[f"rollout_h{horizon}_mse_std"] = float(np.std(vals))

        if name == ref_name:
            j_idx = ref_j_idx
            j_mean = j_ref
            j_within_std = j_ref_std
        else:
            j_idx = split_sample(mask, args.jacobian_samples, args.seed + 7919 + sum(map(ord, name)))
            j_mean, j_within_std = mean_projected_jacobian(
                model,
                latents,
                actions,
                j_idx,
                args.history_size,
                proj,
                device,
                args.jacobian_batch_size,
            )
        np.save(jacobian_dir / f"{report_name}_mean_projected_jacobian.npy", j_mean)
        rule_drift = float(np.linalg.norm(j_mean - j_ref, ord="fro") / ref_norm)
        rule_excess = float(rule_drift - iid_mean) if np.isfinite(iid_mean) else float("nan")
        rule_z = (
            float((rule_drift - iid_mean) / (iid_std + 1e-12))
            if np.isfinite(iid_mean) and np.isfinite(iid_std)
            else float("nan")
        )
        row.update(
            {
                "jacobian_samples": int(len(j_idx)),
                "rule_drift": rule_drift,
                "rule_diff_fro": float(np.linalg.norm(j_mean - j_ref, ord="fro")),
                "reference_jacobian_fro": float(ref_norm),
                "mean_jacobian_fro": float(np.linalg.norm(j_mean, ord="fro")),
                "mean_within_jacobian_std": j_within_std,
                "reference_within_jacobian_std": j_ref_std,
                "iid_bootstrap_mean": iid_mean,
                "iid_bootstrap_std": iid_std,
                "rule_excess_vs_iid": rule_excess,
                "rule_z_vs_iid": rule_z,
            }
        )
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "predictor_rule_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["split"])
        writer.writeheader()
        writer.writerows(rows)
    if iid_bootstrap_rows:
        with (args.out_dir / "iid_bootstrap.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(iid_bootstrap_rows[0].keys()))
            writer.writeheader()
            writer.writerows(iid_bootstrap_rows)
    with (args.out_dir / "predictor_rule_metrics.json").open("w") as f:
        json.dump(
            {
                "config": json_ready(vars(args)),
                "metadata": {
                    "dataset": args.dataset,
                    "data_file": str(h5_path),
                    "state_key": state_key,
                    "num_starts": int(len(starts)),
                    "seq_len": int(seq_len),
                    "history_size": int(args.history_size),
                    "frameskip": int(frameskip),
                    "rollout_horizons": horizons,
                    "reference_split": ref_name,
                    "available_splits": {k: int(v.sum()) for k, v in split_masks.items()},
                    "raw_splits": {k: int(v.sum()) for k, v in raw_splits.items()},
                    "jacobian_batch_size": int(args.jacobian_batch_size),
                    "iid_bootstrap_trials": int(args.iid_bootstrap_trials),
                    "iid_bootstrap_mean": iid_mean,
                    "iid_bootstrap_std": iid_std,
                    "projection": json_ready(proj),
                    "action_mean": action_mean.squeeze(0).tolist(),
                    "action_std": action_std.squeeze(0).tolist(),
                    "state_shape": list(state.shape),
                    "latent_shape": list(latents_np.shape),
                    "save_transition_cache": str(args.save_transition_cache)
                    if args.save_transition_cache
                    else None,
                    "load_transition_cache": str(args.load_transition_cache)
                    if args.load_transition_cache
                    else None,
                    "cache_metadata": cache_metadata,
                    "cwd": os.getcwd(),
                },
                "metrics": rows,
                "iid_bootstrap": iid_bootstrap_rows,
            },
            f,
            indent=2,
        )
    np.save(args.out_dir / "transition_starts.npy", starts)
    print(json.dumps({"metrics": rows}, indent=2))
    print(f"Wrote predictor rule drift results to {args.out_dir}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    main()
