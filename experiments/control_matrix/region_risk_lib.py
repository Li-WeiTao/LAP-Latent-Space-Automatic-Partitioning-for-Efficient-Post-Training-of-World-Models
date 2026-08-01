"""Core utilities for region-conditioned prediction risk evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from backends.lewm.cache import LeWMLatentCache
from backends.lewm.routing import route_voronoi_torch
from lap.partition import PartitionArtifact

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

# Formal region-risk runs hold out episodes from LAP partition/post-training only.
# The shared official LeWM checkpoint may have seen all dataset episodes at pretraining.
FORMAL_CLAIM_SCOPE = "held_out_from_LAP_partition_and_posttraining"
FORMAL_BASE_PRETRAINING_EPISODE_DISJOINT = False


@dataclass(frozen=True)
class CacheContract:
    history_size: int
    num_preds: int
    frameskip: int
    route_index: int
    latent_dim: int


@dataclass(frozen=True)
class ExpertBundle:
    train_seed: int
    global_model: torch.nn.Module
    regional_models: dict[int, torch.nn.Module]
    checkpoint_hashes: dict[str, str]
    manifest: dict[str, Any]
    global_manifest: dict[str, Any] | None = None


@dataclass(frozen=True)
class RolloutLosses:
    terminal_mse: np.ndarray
    mean_trajectory_mse: np.ndarray


@dataclass(frozen=True)
class MultiHorizonRolloutLosses:
    by_horizon: dict[int, RolloutLosses]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return output.strip() or None


def runtime_provenance(*, python_executable: str | None = None) -> dict[str, Any]:
    executable = python_executable or sys.executable
    cuda_available = bool(torch.cuda.is_available())
    cuda_version = torch.version.cuda
    stable_worldmodel_version: str | None = None
    try:
        import stable_worldmodel as stable_wm

        stable_worldmodel_version = getattr(stable_wm, "__version__", stable_wm.__file__)
    except ImportError:
        stable_worldmodel_version = None
    return {
        "python_executable": str(executable),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "stable_worldmodel_version": stable_worldmodel_version,
        "git_commit": git_commit(),
    }


def load_model(path: str | Path) -> torch.nn.Module:
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(path, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"checkpoint is not a torch.nn.Module: {path}")
    model.eval()
    model.requires_grad_(False)
    return model


def episode_exclusive_ends(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r", swmr=False) as handle:
        if "ep_offset" in handle and "ep_len" in handle:
            offset = np.asarray(handle["ep_offset"], dtype=np.int64)
            length = np.asarray(handle["ep_len"], dtype=np.int64)
            return offset + length
        if "episode_ends" in handle:
            return np.asarray(handle["episode_ends"], dtype=np.int64)
        raise KeyError(f"{h5_path} lacks ep_offset/ep_len or episode_ends")


def episode_ids_at_starts(h5_path: Path, starts: np.ndarray) -> np.ndarray:
    ends = episode_exclusive_ends(h5_path)
    return np.searchsorted(ends, np.asarray(starts, dtype=np.int64), side="right")


def load_cache_contract(
    cache: LeWMLatentCache,
    *,
    history_size: int,
    num_preds: int,
    frameskip: int,
) -> CacheContract:
    seq_len = cache.emb.shape[1]
    if seq_len != history_size + num_preds:
        raise ValueError(
            f"cache sequence length {seq_len} != history_size({history_size}) + "
            f"num_preds({num_preds})"
        )
    latent_dim = int(cache.emb.shape[-1])
    if cache.act_emb.shape[-1] != latent_dim:
        raise ValueError("act_emb latent dim must match emb latent dim")
    if cache.act_emb.shape[1] != seq_len:
        raise ValueError(
            "act_emb time dimension must match emb for LeWM predictor contract"
        )
    return CacheContract(
        history_size=history_size,
        num_preds=num_preds,
        frameskip=frameskip,
        route_index=cache.route_index,
        latent_dim=latent_dim,
    )


def resolve_action_norm_starts(
    train_latent_cache: Path,
    *,
    override: Path | None = None,
) -> Path:
    if override is not None:
        return override.expanduser().resolve(strict=True)
    report_path = Path(f"{train_latent_cache.resolve()}.report.json")
    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        starts_source = payload.get("selection", {}).get("starts_source")
        if starts_source:
            candidate = Path(starts_source).expanduser()
            if candidate.is_file():
                return candidate.resolve(strict=True)
    sidecar = train_latent_cache.with_name(
        f"{train_latent_cache.stem}_action_norm_starts.npy"
    )
    if not sidecar.is_file():
        cache = load_lewm_cache(train_latent_cache)
        np.save(sidecar, cache.sample_ids)
    return sidecar.resolve(strict=True)


def load_lewm_cache(
    path: Path,
    *,
    route_index: int | None = None,
) -> LeWMLatentCache:
    path = path.resolve(strict=True)
    with np.load(path, allow_pickle=False) as data:
        missing = {"emb", "act_emb", "region_starts"} - set(data.files)
        if missing:
            raise KeyError(f"{path} missing required arrays: {sorted(missing)}")
    cache = LeWMLatentCache.from_npz(path, route_index=0 if route_index is None else route_index)
    return cache


def start_index_map(sample_ids: np.ndarray) -> dict[int, int]:
    return {int(value): int(index) for index, value in enumerate(sample_ids)}


def audit_episode_disjointness(
    *,
    data_file: Path,
    train_starts: np.ndarray,
    eval_starts: np.ndarray,
    require_disjoint: bool = True,
) -> dict[str, Any]:
    train_starts = np.asarray(train_starts, dtype=np.int64)
    eval_starts = np.asarray(eval_starts, dtype=np.int64)
    train_episodes = episode_ids_at_starts(data_file, train_starts)
    eval_episodes = episode_ids_at_starts(data_file, eval_starts)
    episode_overlap = sorted(set(map(int, train_episodes)).intersection(map(int, eval_episodes)))
    start_overlap = np.intersect1d(train_starts, eval_starts)
    result = {
        "train_num_starts": int(len(train_starts)),
        "eval_num_starts": int(len(eval_starts)),
        "train_num_episodes": int(len(np.unique(train_episodes))),
        "eval_num_episodes": int(len(np.unique(eval_episodes))),
        "episode_overlap_count": int(len(episode_overlap)),
        "episode_overlap_ids": episode_overlap[:20],
        "region_start_overlap_count": int(len(start_overlap)),
        "episode_disjoint": len(episode_overlap) == 0,
        "region_start_disjoint": len(start_overlap) == 0,
    }
    if require_disjoint and (
        result["episode_overlap_count"] > 0 or result["region_start_overlap_count"] > 0
    ):
        raise RuntimeError(
            "no_episode_disjoint_evaluation_cache: train/eval episode or start overlap "
            f"episode_overlap={result['episode_overlap_count']} "
            f"start_overlap={result['region_start_overlap_count']}"
        )
    return result


def audit_partition_train_contract(
    *,
    data_file: Path,
    train_starts: np.ndarray,
    eval_starts: np.ndarray,
    train_cache_hash: str,
    partition_manifest_path: Path | None,
    nominal_train_episode_ids: set[int] | None = None,
    require_train_only: bool = False,
) -> dict[str, Any]:
    train_starts = np.asarray(train_starts, dtype=np.int64)
    eval_starts = np.asarray(eval_starts, dtype=np.int64)
    train_episodes = set(map(int, np.unique(episode_ids_at_starts(data_file, train_starts))))
    eval_episodes = set(map(int, np.unique(episode_ids_at_starts(data_file, eval_starts))))
    episode_overlap = sorted(train_episodes.intersection(eval_episodes))

    manifest_payload: dict[str, Any] | None = None
    manifest_latent_cache_sha256: str | None = None
    if partition_manifest_path is not None and partition_manifest_path.is_file():
        manifest_payload = json.loads(partition_manifest_path.read_text(encoding="utf-8"))
        manifest_latent_cache_sha256 = manifest_payload.get("latent_cache_sha256")

    partition_latent_cache_hash_match = (
        manifest_latent_cache_sha256 == train_cache_hash
        if manifest_latent_cache_sha256
        else None
    )
    train_cache_within_nominal_train: bool | None = None
    eval_episodes_within_holdout: bool | None = None
    if nominal_train_episode_ids is not None:
        train_cache_within_nominal_train = train_episodes.issubset(nominal_train_episode_ids)
        holdout_episodes = set(nominal_train_episode_ids)
        # nominal_train_episode_ids is the 90% train set; holdout is complement checked via eval.
        eval_episodes_within_holdout = eval_episodes.isdisjoint(nominal_train_episode_ids)

    gate_partition_train_only_valid = (
        len(episode_overlap) == 0
        and partition_latent_cache_hash_match is True
        and (train_cache_within_nominal_train is not False)
        and (eval_episodes_within_holdout is not False)
    )

    result = {
        "partition_manifest": str(partition_manifest_path) if partition_manifest_path else None,
        "partition_manifest_latent_cache_sha256": manifest_latent_cache_sha256,
        "train_cache_sha256": train_cache_hash,
        "partition_latent_cache_hash_match": partition_latent_cache_hash_match,
        "train_cache_num_episodes": len(train_episodes),
        "eval_cache_num_episodes": len(eval_episodes),
        "train_eval_episode_overlap_count": len(episode_overlap),
        "train_eval_episode_overlap_ids": episode_overlap[:20],
        "train_cache_within_nominal_train_split": train_cache_within_nominal_train,
        "eval_episodes_disjoint_from_nominal_train_split": eval_episodes_within_holdout,
        "gate_partition_train_only_valid": gate_partition_train_only_valid,
    }
    if require_train_only and not gate_partition_train_only_valid:
        raise RuntimeError(
            "train_only_partition_contract_failed: gate/partition must use the same "
            f"train-only latent cache as predictors partition_hash_match="
            f"{partition_latent_cache_hash_match} train_within_split="
            f"{train_cache_within_nominal_train} eval_holdout="
            f"{eval_episodes_within_holdout} episode_overlap={len(episode_overlap)}"
        )
    return result


def audit_cache_starts_exact(
    *,
    cache_starts: np.ndarray,
    expected_starts: np.ndarray,
    label: str,
    require_exact: bool = False,
) -> dict[str, Any]:
    cache_starts = np.asarray(cache_starts, dtype=np.int64)
    expected_starts = np.asarray(expected_starts, dtype=np.int64)
    exact_match = bool(np.array_equal(cache_starts, expected_starts))
    result = {
        f"{label}_starts_exact_match": exact_match,
        f"{label}_starts_expected_count": int(len(expected_starts)),
        f"{label}_starts_actual_count": int(len(cache_starts)),
    }
    if require_exact and not exact_match:
        raise RuntimeError(
            f"{label}_cache_starts_mismatch: cache sample_ids do not exactly match split manifest"
        )
    return result


def route_regions_from_cache(
    cache: LeWMLatentCache,
    artifact: PartitionArtifact,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    mean = torch.as_tensor(artifact.mean, device=device, dtype=torch.float32)
    scale = torch.as_tensor(artifact.scale, device=device, dtype=torch.float32)
    prototypes = torch.as_tensor(artifact.prototypes, device=device, dtype=torch.float32)
    owners = torch.as_tensor(
        artifact.prototype_region_ids, device=device, dtype=torch.long
    )
    routing = cache.emb[:, cache.route_index, :]
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(routing), batch_size):
            batch = routing[offset : offset + batch_size].to(device=device, dtype=torch.float32)
            assigned = route_voronoi_torch(
                batch,
                prototypes,
                owners,
                mean=mean,
                scale=scale,
                spherical=True,
            )
            labels.append(assigned.detach().cpu().numpy().astype(np.int64))
    return np.concatenate(labels, axis=0)


def final_token_mse(
    pred_seq: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (pred_seq[:, -1] - target).pow(2).mean(dim=-1)


@torch.inference_mode()
def one_step_losses(
    models: Sequence[torch.nn.Module],
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    *,
    history_size: int,
    num_preds: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    losses = np.empty((len(emb), len(models)), dtype=np.float64)
    for begin in range(0, len(emb), batch_size):
        end = min(begin + batch_size, len(emb))
        ctx_emb = emb[begin:end, :history_size].to(device=device, dtype=torch.float32)
        ctx_act = act_emb[begin:end, :history_size].to(device=device, dtype=torch.float32)
        target = emb[begin:end, -1].to(device=device, dtype=torch.float32)
        for model_index, model in enumerate(models):
            pred = model.predict(ctx_emb, ctx_act)
            mse = final_token_mse(pred, target)
            losses[begin:end, model_index] = mse.detach().cpu().numpy()
    return losses


def wrong_expert_losses(
    loss_matrix: np.ndarray,
    regions: np.ndarray,
    num_regions: int,
) -> tuple[np.ndarray, np.ndarray]:
    wrong_mean = np.empty(len(regions), dtype=np.float64)
    wrong_best = np.empty(len(regions), dtype=np.float64)
    for region in range(num_regions):
        mask = regions == region
        if not np.any(mask):
            continue
        block = loss_matrix[mask]
        for row_index, row in enumerate(block):
            others = [value for expert, value in enumerate(row) if expert != region]
            if len(others) != num_regions - 1:
                raise ValueError("wrong expert set must exclude the correct expert")
            wrong_mean[np.flatnonzero(mask)[row_index]] = float(np.mean(others))
            wrong_best[np.flatnonzero(mask)[row_index]] = float(np.min(others))
    return wrong_mean, wrong_best


def anchor_in_same_episode(
    anchor_start: int,
    horizon: int,
    frameskip: int,
    history_size: int,
    episode_lookup: Mapping[int, int],
    start_map: Mapping[int, int],
) -> bool:
    needed = [anchor_start + step * frameskip for step in range(horizon)]
    needed.append(
        anchor_start + (horizon - 1) * frameskip + history_size * frameskip
    )
    episode_id = episode_lookup.get(anchor_start)
    if episode_id is None:
        return False
    for start in needed:
        if start not in start_map:
            return False
        if episode_lookup.get(start) != episode_id:
            return False
    return True


def collect_rollout_anchors(
    eval_starts: np.ndarray,
    *,
    horizon: int,
    frameskip: int,
    history_size: int,
    start_map: Mapping[int, int],
    episode_lookup: Mapping[int, int],
    max_anchors: int = 0,
) -> np.ndarray:
    anchors: list[int] = []
    for start in np.asarray(eval_starts, dtype=np.int64):
        start = int(start)
        if horizon <= 1:
            if start in start_map:
                anchors.append(start)
        elif anchor_in_same_episode(
            start,
            horizon,
            frameskip,
            history_size,
            episode_lookup,
            start_map,
        ):
            anchors.append(start)
        if max_anchors > 0 and len(anchors) >= max_anchors:
            break
    return np.asarray(anchors, dtype=np.int64)


@torch.inference_mode()
def multi_horizon_open_loop_rollout_losses(
    models: Sequence[torch.nn.Module],
    cache: LeWMLatentCache,
    anchors: np.ndarray,
    *,
    horizons: Sequence[int],
    contract: CacheContract,
    start_map: Mapping[int, int],
    device: torch.device,
    batch_size: int = 512,
) -> MultiHorizonRolloutLosses:
    horizon_values = sorted({int(value) for value in horizons if int(value) > 1})
    if not horizon_values:
        raise ValueError("horizons must include at least one value > 1")
    max_horizon = max(horizon_values)
    history = contract.history_size
    fs = contract.frameskip
    num_models = len(models)
    num_anchors = len(anchors)
    step_losses = np.full((max_horizon, num_anchors, num_models), np.nan, dtype=np.float64)

    for begin in range(0, num_anchors, batch_size):
        end = min(begin + batch_size, num_anchors)
        batch_anchors = np.asarray(anchors[begin:end], dtype=np.int64)
        init_rows = torch.as_tensor(
            [start_map[int(start)] for start in batch_anchors], dtype=torch.long
        )
        ctx_z_init = cache.emb.index_select(0, init_rows)[:, :history].clone()

        for model_index, model in enumerate(models):
            ctx_z = ctx_z_init.clone().to(device=device, dtype=torch.float32)
            for step in range(max_horizon):
                row_starts = batch_anchors + step * fs
                rows = torch.as_tensor(
                    [start_map[int(start)] for start in row_starts], dtype=torch.long
                )
                ctx_a = cache.act_emb.index_select(0, rows)[:, :history].to(
                    device=device, dtype=torch.float32
                )
                target = cache.emb.index_select(0, rows)[:, history].to(
                    device=device, dtype=torch.float32
                )
                pred = model.predict(ctx_z, ctx_a)
                z_hat = pred[:, -1]
                mse = (z_hat - target).pow(2).mean(dim=-1)
                step_losses[step, begin:end, model_index] = mse.detach().cpu().numpy()
                if step + 1 < max_horizon:
                    ctx_z = torch.cat([ctx_z[:, 1:], z_hat.unsqueeze(1)], dim=1)

    by_horizon: dict[int, RolloutLosses] = {}
    for horizon in horizon_values:
        block = step_losses[:horizon]
        by_horizon[horizon] = RolloutLosses(
            terminal_mse=block[-1],
            mean_trajectory_mse=np.mean(block, axis=0),
        )
    return MultiHorizonRolloutLosses(by_horizon=by_horizon)


@torch.inference_mode()
def open_loop_rollout_losses(
    models: Sequence[torch.nn.Module],
    cache: LeWMLatentCache,
    anchors: np.ndarray,
    *,
    horizon: int,
    contract: CacheContract,
    start_map: Mapping[int, int],
    device: torch.device,
    batch_size: int = 512,
) -> RolloutLosses:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if horizon == 1:
        emb = cache.emb
        act_emb = cache.act_emb
        rows = np.asarray([start_map[int(start)] for start in anchors], dtype=np.int64)
        one_step = one_step_losses(
            models,
            emb[rows],
            act_emb[rows],
            history_size=contract.history_size,
            num_preds=contract.num_preds,
            device=device,
            batch_size=max(batch_size, len(rows)),
        )
        return RolloutLosses(terminal_mse=one_step, mean_trajectory_mse=one_step)

    multi = multi_horizon_open_loop_rollout_losses(
        models,
        cache,
        anchors,
        horizons=[horizon],
        contract=contract,
        start_map=start_map,
        device=device,
        batch_size=batch_size,
    )
    return multi.by_horizon[horizon]


def aggregate_region_metrics(
    *,
    task: str,
    train_seed: int,
    horizon: int,
    regions: np.ndarray,
    episode_ids: np.ndarray,
    global_loss: np.ndarray,
    correct_loss: np.ndarray,
    wrong_mean_loss: np.ndarray,
    wrong_best_loss: np.ndarray,
    num_regions: int,
    terminal_global_loss: np.ndarray | None = None,
    terminal_correct_loss: np.ndarray | None = None,
    terminal_wrong_mean_loss: np.ndarray | None = None,
    terminal_wrong_best_loss: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    total = len(regions)
    rows: list[dict[str, Any]] = []
    for region in range(num_regions):
        mask = regions == region
        count = int(mask.sum())
        weight = count / total if total else 0.0
        episodes = np.unique(episode_ids[mask]) if count else np.array([], dtype=np.int64)
        global_mse = float(global_loss[mask].mean()) if count else float("nan")
        correct_mse = float(correct_loss[mask].mean()) if count else float("nan")
        wrong_mean_mse = float(wrong_mean_loss[mask].mean()) if count else float("nan")
        wrong_best_mse = float(wrong_best_loss[mask].mean()) if count else float("nan")
        gain = global_mse - correct_mse
        relative_gain = gain / global_mse if global_mse > 0 else float("nan")
        wrong_penalty = wrong_mean_mse - correct_mse
        row = {
            "task": task,
            "train_seed": train_seed,
            "horizon": horizon,
            "region": region,
            "num_episodes": int(len(episodes)),
            "num_anchors": count,
            "region_weight": weight,
            "global_mse": global_mse,
            "correct_mse": correct_mse,
            "wrong_mean_mse": wrong_mean_mse,
            "wrong_best_mse": wrong_best_mse,
            "global_minus_correct": gain,
            "relative_gain": relative_gain,
            "wrong_penalty": wrong_penalty,
            "mean_trajectory_global_mse": global_mse,
            "mean_trajectory_correct_mse": correct_mse,
            "mean_trajectory_wrong_mean_mse": wrong_mean_mse,
            "mean_trajectory_wrong_best_mse": wrong_best_mse,
            "low_support": len(episodes) < 5 or count < 1000,
        }
        if terminal_global_loss is not None:
            row["terminal_global_mse"] = (
                float(terminal_global_loss[mask].mean()) if count else float("nan")
            )
            row["terminal_correct_mse"] = (
                float(terminal_correct_loss[mask].mean()) if count else float("nan")
            )
            row["terminal_wrong_mean_mse"] = (
                float(terminal_wrong_mean_loss[mask].mean()) if count else float("nan")
            )
            row["terminal_wrong_best_mse"] = (
                float(terminal_wrong_best_loss[mask].mean()) if count else float("nan")
            )
        rows.append(row)
    return rows


def weighted_summary(
    region_rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    train_seed: int,
    horizon: int,
) -> dict[str, Any]:
    weights = np.asarray([row["region_weight"] for row in region_rows], dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("region weights must sum to a positive value")
    weights = weights / weights.sum()

    def weighted(field: str) -> float:
        values = np.asarray([row[field] for row in region_rows], dtype=np.float64)
        return float(np.sum(weights * values))

    global_mse = weighted("global_mse")
    correct_mse = weighted("correct_mse")
    return {
        "task": task,
        "train_seed": train_seed,
        "horizon": horizon,
        "global_mse": global_mse,
        "correct_mse": correct_mse,
        "wrong_mean_mse": weighted("wrong_mean_mse"),
        "wrong_best_mse": weighted("wrong_best_mse"),
        "global_minus_correct": global_mse - correct_mse,
        "relative_gain": (global_mse - correct_mse) / global_mse if global_mse > 0 else float("nan"),
        "wrong_penalty": weighted("wrong_mean_mse") - correct_mse,
    }


def paired_bootstrap_ci(
    samples: Mapping[str, np.ndarray],
    *,
    reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    keys = list(samples)
    arrays = [np.asarray(samples[key], dtype=np.float64) for key in keys]
    if not arrays:
        return []
    length = len(arrays[0])
    if any(len(arr) != length for arr in arrays):
        raise ValueError("bootstrap samples must have equal length")
    estimates = {key: float(arr.mean()) for key, arr in zip(keys, arrays)}
    draws = {key: [] for key in keys}
    indices = np.arange(length)
    for _ in range(reps):
        chosen = rng.choice(indices, size=length, replace=True)
        for key, arr in zip(keys, arrays):
            draws[key].append(float(arr[chosen].mean()))
    rows: list[dict[str, Any]] = []
    pairs = [
        ("correct_minus_global", "correct", "global"),
        ("correct_minus_wrong_mean", "correct", "wrong_mean"),
        ("correct_minus_wrong_best", "correct", "wrong_best"),
    ]
    for label, left, right in pairs:
        delta = np.asarray(draws[left], dtype=np.float64) - np.asarray(
            draws[right], dtype=np.float64
        )
        rows.append(
            {
                "metric": label,
                "estimate": estimates[left] - estimates[right],
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
            }
        )
    return rows


def nested_paired_bootstrap_ci(
    seed_losses: Sequence[Mapping[str, Any]],
    *,
    reps: int,
    seed: int,
    metric_keys: Sequence[str] = ("global", "correct", "wrong_mean", "wrong_best"),
    metric_label: str = "mean_trajectory",
    pairs: Sequence[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    keys = list(metric_keys)
    draws = {key: [] for key in keys}
    if not seed_losses:
        return []
    for _ in range(reps):
        chosen_seeds = rng.choice(len(seed_losses), size=len(seed_losses), replace=True)
        episode_universe = np.unique(
            np.concatenate(
                [np.asarray(seed_losses[int(index)]["episode_ids"], dtype=np.int64) for index in chosen_seeds]
            )
        )
        chosen_episodes = rng.choice(episode_universe, size=len(episode_universe), replace=True)
        per_key_values = {key: [] for key in keys}
        replicate_valid = False
        for seed_index in chosen_seeds:
            block = seed_losses[int(seed_index)]
            episode_ids = np.asarray(block["episode_ids"], dtype=np.int64)
            row_indices: list[int] = []
            for episode in chosen_episodes:
                row_indices.extend(np.flatnonzero(episode_ids == episode).tolist())
            if not row_indices:
                continue
            replicate_valid = True
            for key in keys:
                per_key_values[key].append(float(np.mean(block[key][row_indices])))
        if not replicate_valid:
            continue
        for key in keys:
            if per_key_values[key]:
                draws[key].append(float(np.mean(per_key_values[key])))
    estimates = {
        key: float(np.mean([np.mean(block[key]) for block in seed_losses])) for key in keys
    }
    rows: list[dict[str, Any]] = []
    pair_specs = pairs or [
        ("correct_minus_global", "correct", "global"),
        ("correct_minus_wrong_mean", "correct", "wrong_mean"),
        ("correct_minus_wrong_best", "correct", "wrong_best"),
    ]
    for label, left, right in pair_specs:
        delta = np.asarray(draws[left], dtype=np.float64) - np.asarray(
            draws[right], dtype=np.float64
        )
        rows.append(
            {
                "metric": label,
                "loss_kind": metric_label,
                "estimate": estimates[left] - estimates[right],
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
            }
        )
    return rows


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    train_seed: int,
    history_size: int,
    num_preds: int,
    latent_cache_hash: str,
    pretrained_hash: str,
    partition_hash: str,
    split_manifest_hash: str | None = None,
) -> None:
    manifest_seed = manifest.get("train_seed", manifest.get("seed"))
    if manifest_seed is not None and int(manifest_seed) != train_seed:
        raise ValueError("checkpoint manifest train_seed mismatch")
    training = manifest.get("training_config", manifest.get("immutable_config", {}))
    if isinstance(training, dict) and "training" in training:
        training = training["training"]
    manifest_history = training.get("history_size", manifest.get("history_size"))
    manifest_preds = training.get("num_preds", manifest.get("num_preds"))
    if manifest_history is not None and int(manifest_history) != history_size:
        raise ValueError("checkpoint manifest history_size mismatch")
    if manifest_preds is not None and int(manifest_preds) != num_preds:
        raise ValueError("checkpoint manifest num_preds mismatch")
    if manifest.get("latent_cache_sha256") and manifest["latent_cache_sha256"] != latent_cache_hash:
        raise ValueError("checkpoint manifest latent_cache hash mismatch")
    if manifest.get("pretrained_model_sha256") and manifest["pretrained_model_sha256"] != pretrained_hash:
        raise ValueError("checkpoint manifest pretrained_model hash mismatch")
    if partition_hash and manifest.get("partition_source_manifest_sha256") and (
        manifest["partition_source_manifest_sha256"] != partition_hash
    ):
        raise ValueError("checkpoint manifest partition hash mismatch")
    if split_manifest_hash and manifest.get("split_manifest_sha256") and (
        manifest["split_manifest_sha256"] != split_manifest_hash
    ):
        raise ValueError("checkpoint manifest split_manifest hash mismatch")


def audit_formal_posttraining(
    *,
    episode_audit: Mapping[str, Any],
    partition_contracts: Mapping[str, Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    split_manifest_sha256: str,
    train_cache_hash: str,
    eval_cache_hash: str,
    action_norm_starts_hash: str,
    cache_start_audits: Mapping[str, Mapping[str, Any]],
    checkpoint_manifests: Sequence[Mapping[str, Any]],
    require_valid: bool = True,
) -> dict[str, Any]:
    split_hashes = split_manifest.get("sha256", {})
    action_norm_hash_match = action_norm_starts_hash == split_hashes.get("action_norm_starts")
    checkpoint_provenance_valid = True
    checkpoint_issues: list[str] = []
    for index, manifest in enumerate(checkpoint_manifests):
        if manifest.get("latent_cache_sha256") != train_cache_hash:
            checkpoint_provenance_valid = False
            checkpoint_issues.append(f"manifest[{index}] latent_cache_sha256 mismatch")
        if manifest.get("split_manifest_sha256") != split_manifest_sha256:
            checkpoint_provenance_valid = False
            checkpoint_issues.append(f"manifest[{index}] split_manifest_sha256 mismatch")

    auto_gate_train_only_valid = bool(
        partition_contracts.get("auto", {}).get("gate_partition_train_only_valid")
    )
    forced_spectral_train_only_valid = bool(
        partition_contracts.get("forced_spectral", {}).get("gate_partition_train_only_valid")
    )
    global_partition_train_only_valid = bool(
        partition_contracts.get("global", {}).get("gate_partition_train_only_valid")
    )
    cache_starts_exact_valid = all(
        any(bool(value) for key, value in audit.items() if key.endswith("_starts_exact_match"))
        for audit in cache_start_audits.values()
    )

    posttraining_train_only_valid = (
        bool(episode_audit.get("episode_disjoint"))
        and bool(episode_audit.get("region_start_disjoint"))
        and auto_gate_train_only_valid
        and forced_spectral_train_only_valid
        and global_partition_train_only_valid
        and bool(split_manifest.get("train_eval_episode_disjoint"))
        and action_norm_hash_match
        and cache_starts_exact_valid
        and checkpoint_provenance_valid
    )
    result = {
        "posttraining_train_only_valid": posttraining_train_only_valid,
        "auto_gate_train_only_valid": auto_gate_train_only_valid,
        "forced_spectral_train_only_valid": forced_spectral_train_only_valid,
        "global_partition_train_only_valid": global_partition_train_only_valid,
        "cache_starts_exact_valid": cache_starts_exact_valid,
        "action_norm_starts_hash_match": action_norm_hash_match,
        "checkpoint_provenance_valid": checkpoint_provenance_valid,
        "checkpoint_provenance_issues": checkpoint_issues[:10],
        "eval_cache_sha256": eval_cache_hash,
        "train_cache_sha256": train_cache_hash,
        "split_manifest_sha256": split_manifest_sha256,
        "paper_claim": "held out from LAP partition fitting and post-training",
    }
    result.update(
        {
            f"partition_{name}": contract
            for name, contract in partition_contracts.items()
        }
    )
    result.update(
        {key: value for audit in cache_start_audits.values() for key, value in audit.items()}
    )
    if require_valid and not posttraining_train_only_valid:
        raise RuntimeError(
            "formal_posttraining_audit_failed: "
            f"auto={auto_gate_train_only_valid} "
            f"forced_spectral={forced_spectral_train_only_valid} "
            f"global={global_partition_train_only_valid} "
            f"cache_starts_exact={cache_starts_exact_valid} "
            f"action_norm_hash_match={action_norm_hash_match} "
            f"checkpoint_provenance_valid={checkpoint_provenance_valid}"
        )
    return result


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
