#!/usr/bin/env python3
"""Encoder alignment drift using train-global reference and test-region reports."""

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

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import (  # noqa: E402
    DATASETS,
    choose_state_key,
    compute_metrics,
    encode_latents,
    finite_rows,
    json_ready,
    load_latent_cache,
    read_columns,
    save_latent_cache,
)


def valid_indices(h5: h5py.File, spec, state_key: str) -> np.ndarray:
    n = h5[state_key].shape[0]
    valid = finite_rows(h5[state_key][:])
    if spec.pixel_key in h5:
        valid &= np.arange(n) < h5[spec.pixel_key].shape[0]
    return np.flatnonzero(valid).astype(np.int64)


def sample_from_pool(pool: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(pool, dtype=np.int64)
    if max_samples > 0 and len(out) > max_samples:
        out = rng.choice(out, size=max_samples, replace=False)
    return np.sort(out)


def make_test_splits(h5_path: Path, spec, indices: np.ndarray, state_key: str) -> tuple[dict, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        split_keys = sorted(
            set(spec.state_keys)
            | {"pos_agent", "pos_target", "goal_state", "goal_proprio", state_key}
        )
        cols = read_columns(h5, split_keys, indices)
        state = np.asarray(h5[state_key][indices], dtype=np.float64)
    raw = spec.split_fn(cols)
    splits = {
        name: np.asarray(mask, dtype=bool) & finite_rows(state)
        for name, mask in raw.items()
        if len(mask) == len(indices)
    }
    splits["test_all"] = np.ones(len(indices), dtype=bool)
    return splits, state


def read_h5_rows(dataset, indices: np.ndarray) -> np.ndarray:
    order = np.argsort(indices)
    sorted_indices = np.asarray(indices, dtype=np.int64)[order]
    data = np.asarray(dataset[sorted_indices])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return data[inverse]


def rows_by_split(rows: list[dict]) -> dict[str, dict]:
    return {row["split"]: row for row in rows}


def iid_bootstrap(
    latent: np.ndarray,
    state: np.ndarray,
    train_n: int,
    test_mask: np.ndarray,
    align_dim: int | None,
    min_split_size: int,
    bootstrap_samples: int,
    r2_mode: str,
    trials: int,
    seed: int,
) -> list[dict]:
    if trials <= 0:
        return []
    test_idx = np.flatnonzero(test_mask)
    if len(test_idx) < min_split_size:
        return []
    rng = np.random.default_rng(seed)
    out = []
    for trial in range(trials):
        requested = bootstrap_samples if bootstrap_samples > 0 else train_n
        take = min(len(test_idx), max(min_split_size, requested))
        sampled = np.sort(rng.choice(test_idx, size=take, replace=False))
        mask = np.zeros(latent.shape[0], dtype=bool)
        mask[:train_n] = True
        test_all = np.zeros(latent.shape[0], dtype=bool)
        test_all[sampled] = True
        splits = {"train_global": mask, "test_iid": test_all}
        rows, _ = compute_metrics(
            latent=latent,
            state=state,
            splits=splits,
            reference_split="train_global",
            align_dim=align_dim,
            min_split_size=min_split_size,
            seed=seed + trial * 17,
            r2_mode=r2_mode,
        )
        row = rows_by_split(rows).get("test_iid")
        if row:
            out.append(
                {
                    "trial": trial,
                    "n": int(row["n"]),
                    "pca_drift": float(row["pca_drift"]),
                    "pca_residual_ratio": float(row["pca_residual_ratio"]),
                    "frame_drift": float(row["frame_drift"]),
                    "frame_residual_ratio": float(row["frame_residual_ratio"]),
                }
            )
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def add_iid_excess(rows: list[dict], boot_rows: list[dict]) -> tuple[float, float, float, float]:
    frame_mean, frame_std = mean_std([float(row["frame_drift"]) for row in boot_rows])
    pca_mean, pca_std = mean_std([float(row["pca_drift"]) for row in boot_rows])
    for row in rows:
        frame = float(row["frame_drift"])
        pca = float(row["pca_drift"])
        row["test_iid_frame_drift_mean"] = frame_mean
        row["test_iid_frame_drift_std"] = frame_std
        row["frame_drift_excess_vs_test_iid"] = frame - frame_mean if math.isfinite(frame_mean) else float("nan")
        row["frame_drift_z_vs_test_iid"] = (
            (frame - frame_mean) / (frame_std + 1e-12)
            if math.isfinite(frame_mean) and math.isfinite(frame_std)
            else float("nan")
        )
        row["test_iid_pca_drift_mean"] = pca_mean
        row["test_iid_pca_drift_std"] = pca_std
        row["pca_drift_excess_vs_test_iid"] = pca - pca_mean if math.isfinite(pca_mean) else float("nan")
        row["pca_drift_z_vs_test_iid"] = (
            (pca - pca_mean) / (pca_std + 1e-12)
            if math.isfinite(pca_mean) and math.isfinite(pca_std)
            else float("nan")
        )
    return frame_mean, frame_std, pca_mean, pca_std


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data/sicong/weitao/datasets/lewm"))
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-cache-dir", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-max-samples", type=int, default=5000)
    parser.add_argument("--test-max-samples", type=int, default=5000)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--align-dim", type=int, default=None)
    parser.add_argument("--min-split-size", type=int, default=256)
    parser.add_argument("--r2-mode", choices=("holdout", "insample"), default="holdout")
    parser.add_argument("--iid-bootstrap-trials", type=int, default=20)
    parser.add_argument("--iid-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--save-latents", type=Path, default=None)
    parser.add_argument("--load-latents", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = DATASETS[args.dataset]
    h5_path = args.data_file or (args.data_root / spec.default_file)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if args.save_latents and args.load_latents:
        raise ValueError("Use only one of --save-latents or --load-latents")

    cache_meta = {}
    if args.load_latents:
        latent, state, all_indices, cache_meta = load_latent_cache(args.load_latents)
        train_n = int(cache_meta["train_num_samples"])
        train_indices = all_indices[:train_n]
        test_indices = all_indices[train_n:]
        state_key = cache_meta.get("state_key", args.state_key)
    else:
        with h5py.File(h5_path, "r") as h5:
            state_key = choose_state_key(h5, spec, args.state_key)
            all_valid = valid_indices(h5, spec, state_key)
        rng = np.random.default_rng(args.split_seed)
        perm = rng.permutation(all_valid)
        split_n = int(round(len(perm) * args.train_fraction))
        train_pool = np.sort(perm[:split_n])
        test_pool = np.sort(perm[split_n:])
        train_indices = sample_from_pool(train_pool, args.train_max_samples, args.seed + 17)
        test_indices = sample_from_pool(test_pool, args.test_max_samples, args.seed + 29)
        all_indices = np.concatenate([train_indices, test_indices])

        latent = encode_latents(
            h5_path=h5_path,
            indices=all_indices,
            spec=spec,
            checkpoint=args.checkpoint,
            cache_dir=args.checkpoint_cache_dir,
            device=device,
            batch_size=args.batch_size,
            img_size=args.img_size,
        )
        with h5py.File(h5_path, "r") as h5:
            state = np.asarray(read_h5_rows(h5[state_key], all_indices), dtype=np.float64)
        train_n = len(train_indices)
        if args.save_latents:
            save_latent_cache(
                args.save_latents,
                latent=latent,
                state=state,
                indices=all_indices,
                metadata={
                    "dataset": args.dataset,
                    "data_file": str(h5_path),
                    "state_key": state_key,
                    "train_num_samples": int(train_n),
                    "test_num_samples": int(len(test_indices)),
                    "train_fraction": float(args.train_fraction),
                    "split_seed": int(args.split_seed),
                    "seed": int(args.seed),
                    "checkpoint": args.checkpoint,
                },
            )

    test_splits, _ = make_test_splits(h5_path, spec, test_indices, state_key)
    full_splits = {"train_global": np.zeros(len(all_indices), dtype=bool)}
    full_splits["train_global"][:train_n] = True
    for name, mask in test_splits.items():
        full_mask = np.zeros(len(all_indices), dtype=bool)
        full_mask[train_n:] = mask
        full_splits[name] = full_mask

    rows, metadata = compute_metrics(
        latent=np.asarray(latent, dtype=np.float64),
        state=np.asarray(state, dtype=np.float64),
        splits=full_splits,
        reference_split="train_global",
        align_dim=args.align_dim,
        min_split_size=args.min_split_size,
        seed=args.seed,
        r2_mode=args.r2_mode,
    )
    rows = [row for row in rows if row["split"] != "train_global"]

    boot_rows = iid_bootstrap(
        latent=np.asarray(latent, dtype=np.float64),
        state=np.asarray(state, dtype=np.float64),
        train_n=train_n,
        test_mask=full_splits["test_all"],
        align_dim=args.align_dim,
        min_split_size=args.min_split_size,
        bootstrap_samples=args.iid_bootstrap_samples,
        r2_mode=args.r2_mode,
        trials=args.iid_bootstrap_trials,
        seed=args.seed + 12345,
    )
    frame_mean, frame_std, pca_mean, pca_std = add_iid_excess(rows, boot_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "metrics.csv", rows)
    if boot_rows:
        write_csv(args.out_dir / "iid_bootstrap.csv", boot_rows)
    config = json_ready(vars(args).copy())
    config.update(
        {
            "eval_mode": "train-global/test-region",
            "data_file": str(h5_path),
            "state_key": state_key,
            "train_num_samples": int(train_n),
            "test_num_samples": int(len(test_indices)),
            "available_test_splits": {k: int(v.sum()) for k, v in test_splits.items()},
            "test_iid_frame_drift_mean": frame_mean,
            "test_iid_frame_drift_std": frame_std,
            "test_iid_pca_drift_mean": pca_mean,
            "test_iid_pca_drift_std": pca_std,
            "cache_metadata": cache_meta,
            "metadata": metadata,
            "cwd": os.getcwd(),
        }
    )
    with (args.out_dir / "metrics.json").open("w") as f:
        json.dump({"config": config, "metrics": rows, "iid_bootstrap": boot_rows}, f, indent=2)
    np.save(args.out_dir / "train_indices.npy", train_indices)
    np.save(args.out_dir / "test_indices.npy", test_indices)
    print(json.dumps({"metrics": rows}, indent=2))
    print(f"Wrote encoder train-global drift results to {args.out_dir}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    main()
