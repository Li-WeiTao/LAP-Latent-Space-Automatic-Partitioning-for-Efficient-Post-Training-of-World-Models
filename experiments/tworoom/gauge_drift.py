#!/usr/bin/env python3
"""Measure encoder latent/state alignment drift on real LeWM datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
import torch.nn.functional as F

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    default_file: str
    pixel_key: str
    state_keys: tuple[str, ...]
    split_fn: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def finite_rows(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 1:
        return np.isfinite(arr)
    flat = arr.reshape(arr.shape[0], -1)
    return np.isfinite(flat).all(axis=1)


def middle_quantile_mask(x: np.ndarray, lo: float = 0.25, hi: float = 0.75) -> np.ndarray:
    lower, upper = np.quantile(x, [lo, hi])
    return (x >= lower) & (x <= upper)


def extreme_quantile_mask(x: np.ndarray, q: float = 0.1) -> np.ndarray:
    lower, upper = np.quantile(x, [q, 1.0 - q])
    return (x <= lower) | (x >= upper)


def get_first_available(cols: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in cols:
            return cols[key]
    return None


def tworoom_thresholds_from_reference(
    pos: np.ndarray,
    target_x: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute TwoRoom natural-region thresholds from a reference pool (train-only)."""
    x = np.asarray(pos[:, 0], dtype=np.float64)
    y = np.asarray(pos[:, 1], dtype=np.float64)
    x33, x67 = np.quantile(x, [0.33, 0.67])
    x45, x55 = np.quantile(x, [0.45, 0.55])
    x_lo_c, x_hi_c = np.quantile(x, [0.25, 0.75])
    y_lo_c, y_hi_c = np.quantile(y, [0.20, 0.80])
    x_lo_w, x_hi_w = np.quantile(x, [0.08, 0.92])
    y_lo_w, y_hi_w = np.quantile(y, [0.08, 0.92])
    thresholds: dict[str, float] = {
        "x33": float(x33),
        "x67": float(x67),
        "x45": float(x45),
        "x55": float(x55),
        "x_lo_common": float(x_lo_c),
        "x_hi_common": float(x_hi_c),
        "y_lo_common": float(y_lo_c),
        "y_hi_common": float(y_hi_c),
        "x_lo_wall": float(x_lo_w),
        "x_hi_wall": float(x_hi_w),
        "y_lo_wall": float(y_lo_w),
        "y_hi_wall": float(y_hi_w),
        "x_mid": float(np.median(x)),
    }
    return thresholds


def tworoom_splits_with_thresholds(
    cols: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, np.ndarray]:
    pos = get_first_available(cols, ("pos_agent", "proprio", "observation"))
    if pos is None or pos.ndim != 2 or pos.shape[1] < 2:
        return {}

    x = pos[:, 0]
    y = pos[:, 1]
    splits = {
        "common": (x >= thresholds["x_lo_common"])
        & (x <= thresholds["x_hi_common"])
        & (y >= thresholds["y_lo_common"])
        & (y <= thresholds["y_hi_common"]),
        "left_room": x <= thresholds["x33"],
        "right_room": x >= thresholds["x67"],
        "doorway_corridor": (x >= thresholds["x45"]) & (x <= thresholds["x55"]),
        "near_wall": (x <= thresholds["x_lo_wall"])
        | (x >= thresholds["x_hi_wall"])
        | (y <= thresholds["y_lo_wall"])
        | (y >= thresholds["y_hi_wall"]),
    }

    if "pos_target" in cols:
        target_x = cols["pos_target"][:, 0]
        mid = thresholds["x_mid"]
        splits["goal_other_side"] = ((x <= mid) & (target_x > mid)) | (
            (x > mid) & (target_x <= mid)
        )

    return splits


def tworoom_splits(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pos = get_first_available(cols, ("pos_agent", "proprio", "observation"))
    if pos is None or pos.ndim != 2 or pos.shape[1] < 2:
        return {}
    thresholds = tworoom_thresholds_from_reference(pos)
    return tworoom_splits_with_thresholds(cols, thresholds)


# Fixed TwoRoom geometry (stable_worldmodel TwoRoomEnv, 224x224 layout).
TWOROOM_GEOMETRY = {
    "img_size": 224.0,
    "border": 14.0,
    "wall_center": 112.0,
    "wall_width": 10.0,
    "near_wall_margin": 15.0,
    "common_interior_margin": 20.0,
}


def tworoom_geometry_thresholds(
    *,
    img_size: float = TWOROOM_GEOMETRY["img_size"],
    border: float = TWOROOM_GEOMETRY["border"],
    wall_center: float = TWOROOM_GEOMETRY["wall_center"],
    wall_width: float = TWOROOM_GEOMETRY["wall_width"],
    near_wall_margin: float = TWOROOM_GEOMETRY["near_wall_margin"],
    common_interior_margin: float = TWOROOM_GEOMETRY["common_interior_margin"],
) -> dict[str, float]:
    """Fixed task-geometry thresholds for TwoRoom (no data-dependent quantiles)."""
    pos_min = border
    pos_max = img_size - border - 1.0
    wall_lo = wall_center - wall_width / 2.0
    wall_hi = wall_center + wall_width / 2.0
    return {
        "img_size": float(img_size),
        "border": float(border),
        "wall_center": float(wall_center),
        "wall_width": float(wall_width),
        "wall_lo": float(wall_lo),
        "wall_hi": float(wall_hi),
        "pos_min": float(pos_min),
        "pos_max": float(pos_max),
        "near_wall_margin": float(near_wall_margin),
        "common_interior_margin": float(common_interior_margin),
        "x_lo_common": float(pos_min + common_interior_margin),
        "x_hi_common_left": float(wall_lo - common_interior_margin),
        "x_lo_common_right": float(wall_hi + common_interior_margin),
        "x_hi_common": float(pos_max - common_interior_margin),
        "y_lo_common": float(pos_min + common_interior_margin),
        "y_hi_common": float(pos_max - common_interior_margin),
        "x_lo_wall": float(pos_min + near_wall_margin),
        "x_hi_wall": float(pos_max - near_wall_margin),
        "y_lo_wall": float(pos_min + near_wall_margin),
        "y_hi_wall": float(pos_max - near_wall_margin),
        "x_mid": float(wall_center),
    }


def tworoom_geometry_splits_with_thresholds(
    cols: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, np.ndarray]:
    pos = get_first_available(cols, ("pos_agent", "proprio", "observation"))
    if pos is None or pos.ndim != 2 or pos.shape[1] < 2:
        return {}

    x = pos[:, 0]
    y = pos[:, 1]
    wall_lo = thresholds["wall_lo"]
    wall_hi = thresholds["wall_hi"]
    left_interior = (x >= thresholds["x_lo_common"]) & (x <= thresholds["x_hi_common_left"])
    right_interior = (x >= thresholds["x_lo_common_right"]) & (x <= thresholds["x_hi_common"])
    y_interior = (y >= thresholds["y_lo_common"]) & (y <= thresholds["y_hi_common"])
    splits = {
        "left_room": x < wall_lo,
        "right_room": x > wall_hi,
        "doorway_corridor": (x >= wall_lo) & (x <= wall_hi),
        "near_wall": (x <= thresholds["x_lo_wall"])
        | (x >= thresholds["x_hi_wall"])
        | (y <= thresholds["y_lo_wall"])
        | (y >= thresholds["y_hi_wall"]),
        "common": (left_interior | right_interior) & y_interior,
    }

    if "pos_target" in cols:
        target_x = cols["pos_target"][:, 0]
        mid = thresholds["x_mid"]
        splits["goal_other_side"] = ((x <= mid) & (target_x > mid)) | (
            (x > mid) & (target_x <= mid)
        )

    return splits


def tworoom_geometry_splits(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return tworoom_geometry_splits_with_thresholds(cols, tworoom_geometry_thresholds())


def resolve_tworoom_region_splits(
    cols: dict[str, np.ndarray],
    *,
    split_mode: str = "quantile",
    quantile_reference_pos: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float] | None]:
    """Return region masks and optional saved thresholds for TwoRoom."""
    if split_mode == "geometry":
        thresholds = tworoom_geometry_thresholds()
        return tworoom_geometry_splits_with_thresholds(cols, thresholds), thresholds
    if split_mode == "quantile":
        if quantile_reference_pos is not None:
            thresholds = tworoom_thresholds_from_reference(quantile_reference_pos, None)
            return tworoom_splits_with_thresholds(cols, thresholds), thresholds
        return tworoom_splits(cols), None
    raise ValueError(f"Unknown TwoRoom region split_mode: {split_mode!r}")


def pusht_splits(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state = get_first_available(cols, ("state", "proprio"))
    if state is None or state.ndim != 2:
        return {}

    splits: dict[str, np.ndarray] = {}
    if state.shape[1] >= 4:
        obj_x = state[:, 2]
        obj_y = state[:, 3]
        x_mid = np.median(obj_x)
        y_mid = np.median(obj_y)
        splits["common"] = middle_quantile_mask(obj_x, 0.25, 0.75) & middle_quantile_mask(
            obj_y, 0.25, 0.75
        )
        splits["object_pos_q1"] = (obj_x <= x_mid) & (obj_y <= y_mid)
        splits["object_pos_q2"] = (obj_x <= x_mid) & (obj_y > y_mid)
        splits["object_pos_q3"] = (obj_x > x_mid) & (obj_y <= y_mid)
        splits["object_pos_q4"] = (obj_x > x_mid) & (obj_y > y_mid)

        if state.shape[1] >= 2:
            dist = np.linalg.norm(state[:, :2] - state[:, 2:4], axis=1)
            d20, d80 = np.quantile(dist, [0.20, 0.80])
            splits["contact"] = dist <= d20
            splits["non_contact"] = dist >= d80

    if state.shape[1] >= 5:
        angle = state[:, 4]
        a20, a80 = np.quantile(angle, [0.20, 0.80])
        splits["object_angle_low"] = angle <= a20
        splits["object_angle_high"] = angle >= a80

    if "goal_state" in cols and cols["goal_state"].shape == state.shape:
        dist_goal = np.linalg.norm(state - cols["goal_state"], axis=1)
        g20, g80 = np.quantile(dist_goal, [0.20, 0.80])
        splits["near_goal"] = dist_goal <= g20
        splits["far_goal"] = dist_goal >= g80

    if "common" not in splits:
        splits["common"] = np.ones(state.shape[0], dtype=bool)
    return splits


def reacher_splits(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Matrix protocol parity with PushT: no manually designed partition, only
    # the "common" region needed to seed the global training-start pool.
    state = get_first_available(cols, ("observation", "qpos", "finger_pos"))
    if state is None:
        return {}
    return {"common": np.ones(state.shape[0], dtype=bool)}


def cube_splits(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Matrix protocol parity with PushT/Reacher: no manually designed
    # partition for OGBench Cube, only the "common" region needed to seed the
    # global training-start pool. Regional splits (if any) come from LAP's
    # automatic spectral partitioning on the latent cache, not from this file.
    state = get_first_available(cols, ("observation", "qpos"))
    if state is None:
        return {}
    return {"common": np.ones(state.shape[0], dtype=bool)}


DATASETS = {
    "tworoom": DatasetSpec(
        name="tworoom",
        default_file="tworoom.h5",
        pixel_key="pixels",
        state_keys=("proprio", "pos_agent", "observation"),
        split_fn=tworoom_splits,
    ),
    "pusht": DatasetSpec(
        name="pusht",
        default_file="pusht_expert_train.h5",
        pixel_key="pixels",
        state_keys=("state", "proprio"),
        split_fn=pusht_splits,
    ),
    "reacher": DatasetSpec(
        name="reacher",
        default_file="reacher.h5",
        pixel_key="pixels",
        state_keys=("observation", "qpos", "finger_pos"),
        split_fn=reacher_splits,
    ),
    "cube": DatasetSpec(
        name="cube",
        default_file="cube_single_expert.h5",
        pixel_key="pixels",
        state_keys=("observation", "qpos"),
        split_fn=cube_splits,
    ),
}


def read_columns(h5: h5py.File, keys: list[str], indices: np.ndarray) -> dict[str, np.ndarray]:
    cols = {}
    for key in keys:
        if key in h5:
            cols[key] = h5[key][indices]
    return cols


def choose_state_key(h5: h5py.File, spec: DatasetSpec, requested: str | None) -> str:
    if requested:
        if requested not in h5:
            raise KeyError(f"Requested state key '{requested}' not found in {list(h5.keys())}")
        return requested
    for key in spec.state_keys:
        if key in h5:
            return key
    raise KeyError(f"No state proxy found. Tried: {spec.state_keys}")


def sample_indices(h5: h5py.File, spec: DatasetSpec, state_key: str, max_samples: int, seed: int) -> np.ndarray:
    n = h5[state_key].shape[0]
    valid = finite_rows(h5[state_key][:])
    if spec.pixel_key in h5:
        valid &= np.arange(n) < h5[spec.pixel_key].shape[0]
    valid_idx = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)
    if len(valid_idx) > max_samples:
        valid_idx = rng.choice(valid_idx, size=max_samples, replace=False)
    return np.sort(valid_idx)


def load_encoder(
    checkpoint: str,
    device: torch.device,
    cache_dir: str | None,
    *,
    model_family: str = "lewm",
):
    if not checkpoint:
        raise ValueError("--checkpoint is required when --latent-source encoder")

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists() and checkpoint_path.suffix == ".ckpt":
        from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint

        model = load_jepa_object_checkpoint(
            checkpoint_path,
            model_family=model_family,
            map_location="cpu",
        )
        model = model.to(device).eval()
        model.requires_grad_(False)
        return model

    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(checkpoint, cache_dir=cache_dir)
    model = model.to(device).eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    return model


def preprocess_pixels(pixels: np.ndarray, device: torch.device, img_size: int) -> torch.Tensor:
    x = torch.from_numpy(pixels)
    if x.ndim != 4:
        raise ValueError(f"Expected pixels as (B,H,W,C), got {tuple(x.shape)}")
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-2:] != (img_size, img_size):
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


def read_h5_rows(dataset, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices)
    if len(indices) == 0:
        return np.asarray(dataset[indices])
    order = np.argsort(indices)
    sorted_indices = indices[order]
    rows = np.asarray(dataset[sorted_indices])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return rows[inverse]


@torch.inference_mode()
def encode_latents(
    h5_path: Path,
    indices: np.ndarray,
    spec: DatasetSpec,
    checkpoint: str,
    cache_dir: str | None,
    device: torch.device,
    batch_size: int,
    img_size: int,
) -> np.ndarray:
    model = load_encoder(checkpoint, device, cache_dir)
    chunks = []
    with h5py.File(h5_path, "r") as h5:
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            pixels = preprocess_pixels(read_h5_rows(h5[spec.pixel_key], idx), device, img_size)
            batch = {"pixels": pixels.unsqueeze(1)}
            out = model.encode(batch)
            emb = out["emb"][:, 0].detach().cpu().numpy()
            chunks.append(emb)
    return np.concatenate(chunks, axis=0)


def privileged_control_latents(state: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = np.asarray(state, dtype=np.float64)
    state = (state - state.mean(axis=0, keepdims=True)) / (state.std(axis=0, keepdims=True) + 1e-8)
    random_matrix = rng.normal(size=(state.shape[1], state.shape[1]))
    u, _, vt = np.linalg.svd(random_matrix, full_matrices=False)
    q = u @ vt
    return state @ q


def fit_pca(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:dim].T
    return mean, basis


def transform_with_pca(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - mean) @ basis


def procrustes(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    matrix = source.T @ target
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    q = u @ vt
    residual = np.mean(np.sum((source @ q - target) ** 2, axis=1))
    return q, float(residual)


def linear_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([source, np.ones((source.shape[0], 1), dtype=source.dtype)], axis=1)
    coef, *_ = np.linalg.lstsq(x, target, rcond=None)
    weight = coef[:-1]
    bias = coef[-1]
    return weight, bias


def linear_predict(source: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return source @ weight + bias


def r2_score(target: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((target - pred) ** 2))
    centered = target - target.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum(centered**2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def linear_r2(
    source: np.ndarray,
    target: np.ndarray,
    seed: int,
    mode: str,
) -> tuple[float, np.ndarray]:
    if source.shape[0] < 4 or target.shape[0] < 4:
        return float("nan"), np.full((source.shape[1], target.shape[1]), np.nan)

    if mode == "insample":
        train_idx = test_idx = np.arange(source.shape[0])
    elif mode == "holdout":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(source.shape[0])
        cut = max(2, source.shape[0] // 2)
        train_idx = perm[:cut]
        test_idx = perm[cut:]
        if len(test_idx) < 2:
            return float("nan"), np.full((source.shape[1], target.shape[1]), np.nan)
    else:
        raise ValueError(f"Unknown R2 mode: {mode}")

    weight, bias = linear_fit(source[train_idx], target[train_idx])
    pred = linear_predict(source[test_idx], weight, bias)
    return r2_score(target[test_idx], pred), weight


def standardize_from_reference(x: np.ndarray, ref_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[ref_mask].mean(axis=0, keepdims=True)
    std = x[ref_mask].std(axis=0, keepdims=True) + 1e-8
    return (x - mean) / std, mean, std


def split_reference_and_iid(
    splits: dict[str, np.ndarray],
    reference_split: str,
    seed: int,
    reference_fraction: float,
    min_split_size: int,
    disjoint_reference: bool,
) -> tuple[dict[str, np.ndarray], str]:
    if reference_split not in splits:
        raise KeyError(f"Reference split '{reference_split}' missing. Available: {sorted(splits)}")

    base_idx = np.flatnonzero(splits[reference_split])
    if len(base_idx) < 2 * min_split_size:
        raise ValueError(
            f"Reference base split '{reference_split}' too small for disjoint reference/IID: {len(base_idx)}"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(base_idx)
    ref_n = int(round(len(base_idx) * reference_fraction))
    ref_n = min(max(ref_n, min_split_size), len(base_idx) - min_split_size)

    ref_mask = np.zeros_like(splits[reference_split], dtype=bool)
    iid_mask = np.zeros_like(splits[reference_split], dtype=bool)
    ref_mask[perm[:ref_n]] = True
    iid_mask[perm[ref_n:]] = True

    out = {name: mask.copy() for name, mask in splits.items()}
    out["reference"] = ref_mask
    out["iid_nonoverlap"] = iid_mask

    if disjoint_reference:
        for name in list(out):
            if name == "reference":
                continue
            out[name] = out[name] & ~ref_mask

    return out, "reference"


def compute_metrics(
    latent: np.ndarray,
    state: np.ndarray,
    splits: dict[str, np.ndarray],
    reference_split: str,
    align_dim: int | None,
    min_split_size: int,
    seed: int,
    r2_mode: str,
) -> tuple[list[dict], dict]:
    if reference_split not in splits:
        raise KeyError(f"Reference split '{reference_split}' missing. Available: {sorted(splits)}")

    ref_mask = splits[reference_split]
    if ref_mask.sum() < min_split_size:
        raise ValueError(f"Reference split '{reference_split}' too small: {int(ref_mask.sum())}")

    state_ref_std_raw = state[ref_mask].std(axis=0)
    state_keep = state_ref_std_raw > 1e-6
    if not np.any(state_keep):
        raise ValueError("All state dimensions are near-constant on the reference split")
    state = state[:, state_keep]

    max_dim = min(latent.shape[1], state.shape[1], int(ref_mask.sum()) - 1)
    dim = int(align_dim or max_dim)
    dim = max(1, min(dim, max_dim))

    state_z, state_mean, state_std = standardize_from_reference(state, ref_mask)
    latent_z, latent_mean, latent_std = standardize_from_reference(latent, ref_mask)

    latent_pca_mean, latent_basis = fit_pca(latent_z[ref_mask], dim)
    latent_proj = transform_with_pca(latent_z, latent_pca_mean, latent_basis)

    if state_z.shape[1] > dim:
        state_pca_mean, state_basis = fit_pca(state_z[ref_mask], dim)
        state_proj = transform_with_pca(state_z, state_pca_mean, state_basis)
    elif state_z.shape[1] == dim:
        state_proj = state_z
        state_pca_mean = np.zeros((1, state_z.shape[1]))
        state_basis = np.eye(state_z.shape[1])
    else:
        raise ValueError(
            f"State dim {state_z.shape[1]} is smaller than requested alignment dim {dim}"
        )

    q_ref_pca, residual_ref_pca = procrustes(state_proj[ref_mask], latent_proj[ref_mask])
    q_ref_frame, residual_ref_frame = procrustes(state_z[ref_mask], latent_z[ref_mask])
    rows = []
    for name, mask in sorted(splits.items()):
        n = int(mask.sum())
        if n < min_split_size:
            continue
        q_pca, residual_pca = procrustes(state_proj[mask], latent_proj[mask])
        pca_drift = float(np.linalg.norm(q_pca - q_ref_pca, ord="fro") / math.sqrt(dim))
        pca_residual_ratio = float(residual_pca / (residual_ref_pca + 1e-12))

        q_frame, residual_frame = procrustes(state_z[mask], latent_z[mask])
        frame_dim = min(state_z.shape[1], latent_z.shape[1])
        frame_drift = float(np.linalg.norm(q_frame - q_ref_frame, ord="fro") / math.sqrt(frame_dim))
        frame_residual_ratio = float(residual_frame / (residual_ref_frame + 1e-12))

        split_seed = seed + sum(ord(ch) for ch in name)
        state_to_latent_r2, state_to_latent_weight = linear_r2(
            state_z[mask], latent_z[mask], split_seed, r2_mode
        )
        latent_to_state_r2, _ = linear_r2(latent_z[mask], state_z[mask], split_seed + 17, r2_mode)

        if state_z.shape[1] == latent_z.shape[1]:
            orthogonality_error = float(
                np.linalg.norm(state_to_latent_weight.T @ state_to_latent_weight - np.eye(state_z.shape[1]), ord="fro")
                / math.sqrt(state_z.shape[1])
            )
            orthogonal_recovery_error = residual_frame
        else:
            orthogonality_error = float("nan")
            orthogonal_recovery_error = float("nan")

        overlap_n = int(np.logical_and(mask, ref_mask).sum())
        overlap_ratio = float(overlap_n / n) if n else float("nan")
        rows.append(
            {
                "split": name,
                "n": n,
                "dim": dim,
                "pca_residual": residual_pca,
                "pca_drift": pca_drift,
                "pca_residual_ratio": pca_residual_ratio,
                "pca_drift_to_residual_ratio": float(pca_drift / (pca_residual_ratio + 1e-12)),
                "frame_residual": residual_frame,
                "frame_drift": frame_drift,
                "frame_residual_ratio": frame_residual_ratio,
                "frame_drift_to_residual_ratio": float(frame_drift / (frame_residual_ratio + 1e-12)),
                "state_to_latent_r2": state_to_latent_r2,
                "latent_to_state_r2": latent_to_state_r2,
                "orthogonality_error": orthogonality_error,
                "orthogonal_recovery_error": orthogonal_recovery_error,
                "reference_overlap_n": overlap_n,
                "reference_overlap_ratio": overlap_ratio,
            }
        )

    metadata = {
        "reference_split": reference_split,
        "reference_pca_residual": residual_ref_pca,
        "reference_frame_residual": residual_ref_frame,
        "align_dim": dim,
        "latent_dim": int(latent.shape[1]),
        "state_dim": int(state.shape[1]),
        "state_dims_kept": np.flatnonzero(state_keep).tolist(),
        "state_mean": state_mean.squeeze(0).tolist(),
        "state_std": state_std.squeeze(0).tolist(),
        "latent_mean_shape": list(latent_mean.shape),
        "latent_std_shape": list(latent_std.shape),
        "latent_pca_basis_shape": list(latent_basis.shape),
        "state_pca_basis_shape": list(state_basis.shape),
        "r2_mode": r2_mode,
    }
    return rows, metadata


def write_outputs(out_dir: Path, rows: list[dict], metadata: dict, config: dict, indices: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["split"])
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump({"metadata": metadata, "metrics": rows}, f, indent=2)
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
    np.save(out_dir / "sample_indices.npy", indices)


def save_latent_cache(path: Path, latent: np.ndarray, state: np.ndarray, indices: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        latent=latent.astype(np.float32),
        state=state.astype(np.float32),
        indices=indices.astype(np.int64),
        metadata=json.dumps(json_ready(metadata)),
    )


def load_latent_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    return data["latent"], data["state"], data["indices"], metadata


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data/sicong/weitao/datasets/lewm"))
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--latent-source", choices=("encoder", "privileged-control"), default="encoder")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-cache-dir", default=None)
    parser.add_argument("--save-latents", type=Path, default=None)
    parser.add_argument("--load-latents", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-split", default="common")
    parser.add_argument("--reference-fraction", type=float, default=0.5)
    parser.add_argument("--disjoint-reference", action="store_true")
    parser.add_argument("--align-dim", type=int, default=None)
    parser.add_argument("--min-split-size", type=int, default=256)
    parser.add_argument("--r2-mode", choices=("holdout", "insample"), default="holdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = DATASETS[args.dataset]
    h5_path = args.data_file or (args.data_root / spec.default_file)
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {h5_path}")
    if args.save_latents is not None and args.load_latents is not None:
        raise ValueError("Use only one of --save-latents or --load-latents")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cache_metadata = {}
    if args.load_latents is not None:
        latent, state, indices, cache_metadata = load_latent_cache(args.load_latents)
        state_key = cache_metadata.get("state_key", args.state_key or "cached_state")
        with h5py.File(h5_path, "r") as h5:
            split_keys = sorted(
                set(spec.state_keys)
                | {
                    "pos_agent",
                    "pos_target",
                    "goal_state",
                    "goal_proprio",
                    state_key,
                }
            )
            cols = read_columns(h5, split_keys, indices)
    else:
        with h5py.File(h5_path, "r") as h5:
            state_key = choose_state_key(h5, spec, args.state_key)
            indices = sample_indices(h5, spec, state_key, args.max_samples, args.seed)
            split_keys = sorted(
                set(spec.state_keys)
                | {
                    "pos_agent",
                    "pos_target",
                    "goal_state",
                    "goal_proprio",
                    state_key,
                }
            )
            cols = read_columns(h5, split_keys, indices)
            state = np.asarray(h5[state_key][indices], dtype=np.float64)

    split_masks = spec.split_fn(cols)
    split_masks = {
        name: np.asarray(mask, dtype=bool) & finite_rows(state)
        for name, mask in split_masks.items()
        if len(mask) == len(indices)
    }
    if not split_masks:
        raise RuntimeError(f"No valid splits generated for dataset '{args.dataset}'")
    split_masks, metric_reference_split = split_reference_and_iid(
        splits=split_masks,
        reference_split=args.reference_split,
        seed=args.seed,
        reference_fraction=args.reference_fraction,
        min_split_size=args.min_split_size,
        disjoint_reference=args.disjoint_reference,
    )

    if args.load_latents is not None:
        latent = np.asarray(latent, dtype=np.float64)
        state = np.asarray(state, dtype=np.float64)
    elif args.latent_source == "encoder":
        latent = encode_latents(
            h5_path=h5_path,
            indices=indices,
            spec=spec,
            checkpoint=args.checkpoint,
            cache_dir=args.checkpoint_cache_dir,
            device=device,
            batch_size=args.batch_size,
            img_size=args.img_size,
        )
    else:
        latent = privileged_control_latents(state, args.seed)

    if args.save_latents is not None:
        save_latent_cache(
            args.save_latents,
            latent=np.asarray(latent),
            state=np.asarray(state),
            indices=indices,
            metadata={
                "dataset": args.dataset,
                "data_file": str(h5_path),
                "state_key": state_key,
                "latent_source": args.latent_source,
                "checkpoint": args.checkpoint,
                "num_sampled": int(len(indices)),
                "seed": args.seed,
            },
        )

    rows, metadata = compute_metrics(
        latent=np.asarray(latent, dtype=np.float64),
        state=state,
        splits=split_masks,
        reference_split=metric_reference_split,
        align_dim=args.align_dim,
        min_split_size=args.min_split_size,
        seed=args.seed,
        r2_mode=args.r2_mode,
    )

    config = json_ready(vars(args).copy())
    config.update(
        {
            "data_file": str(h5_path),
            "out_dir": str(args.out_dir),
            "state_key": state_key,
            "metric_reference_split": metric_reference_split,
            "num_sampled": int(len(indices)),
            "available_splits": {k: int(v.sum()) for k, v in split_masks.items()},
            "save_latents": str(args.save_latents) if args.save_latents else None,
            "load_latents": str(args.load_latents) if args.load_latents else None,
            "cache_metadata": cache_metadata,
            "cwd": os.getcwd(),
        }
    )
    write_outputs(args.out_dir, rows, metadata, config, indices)

    print(json.dumps({"metadata": metadata, "metrics": rows}, indent=2))
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    main()
