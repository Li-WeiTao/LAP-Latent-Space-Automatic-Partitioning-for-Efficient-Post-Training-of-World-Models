#!/usr/bin/env python3
"""Run Held-out Region-Conditional Prediction-Risk Analysis."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "tworoom"))

from backends.lewm.cache import LeWMLatentCache  # noqa: E402
from backends.lewm.encoding import LeWMEncoderAdapter, make_hdf5_transition_dataset  # noqa: E402
from lap.encoding.fast import FastEncodingConfig, FastLatentCacheEncoder  # noqa: E402
from lap.partition import PartitionArtifact  # noqa: E402

from experiments.control_matrix.episode_split import (  # noqa: E402
    load_split_manifest,
    split_manifest_is_unsubsampled,
    split_paths_from_manifest,
)
from experiments.control_matrix.region_risk_lib import (  # noqa: E402
    ExpertBundle,
    aggregate_region_metrics,
    atomic_savez_compressed,
    atomic_write_json,
    audit_cache_starts_exact,
    audit_episode_disjointness,
    audit_formal_posttraining,
    audit_partition_train_contract,
    collect_rollout_anchors,
    episode_ids_at_starts,
    FORMAL_BASE_PRETRAINING_EPISODE_DISJOINT,
    FORMAL_CLAIM_SCOPE,
    git_commit,
    load_cache_contract,
    load_lewm_cache,
    load_model,
    multi_horizon_open_loop_rollout_losses,
    nested_paired_bootstrap_draws_from_summaries,
    nested_paired_bootstrap_ci,
    one_step_losses,
    precompute_episode_summaries,
    resolve_action_norm_starts,
    route_regions_from_cache,
    runtime_provenance,
    sha256_file,
    stable_json_sha256,
    start_index_map,
    validate_manifest,
    weighted_summary,
    wrong_expert_losses,
    PUBLIC_ANALYSIS_NAME,
    PUBLIC_ANALYSIS_SHORT_NAME,
)
from experiments.tworoom.gauge_drift import DATASETS, choose_state_key  # noqa: E402
from experiments.tworoom.predictor_rule_drift import (  # noqa: E402
    sample_starts,
    valid_transition_starts,
)

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


FORMAL_REGIONAL_RUN_PATTERNS = (
    "train{seed}",
    "tworoom_latent_spectral_spectral_M20000_k30_P16_seed0_trainseed{seed}",
    "partition0_train{seed}",
    "trainseed{seed}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"{PUBLIC_ANALYSIS_NAME}. This is a held-out mechanistic analysis, "
            "not the main planning-performance experiment."
        )
    )
    parser.add_argument("--task", required=True, choices=("tworoom", "pusht"))
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--train-latent-cache", type=Path, required=True)
    parser.add_argument("--eval-latent-cache", type=Path, required=True)
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument("--regional-runs", type=Path, required=True)
    parser.add_argument("--global-runs", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=Path, required=True)
    parser.add_argument("--train-seeds", default="0,42,625")
    parser.add_argument("--horizons", default="1,5,10")
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--route-index", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument("--bootstrap-reps", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    parser.add_argument(
        "--stage",
        choices=("rollout", "bootstrap", "finalize", "all"),
        default="all",
        help="Evaluation stage; default 'all' preserves the legacy one-command workflow.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse fingerprint-matching raw rollout files and bootstrap chunks.",
    )
    parser.add_argument("--bootstrap-chunk-size", type=int, default=1000)
    parser.add_argument("--bootstrap-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--build-eval-cache", action="store_true")
    parser.add_argument("--allow-in-cache", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--max-eval-starts", type=int, default=0)
    parser.add_argument("--global-checkpoint-name", default="auto")
    parser.add_argument("--regional-checkpoint-template", default="P_train_cluster{region}_object.ckpt")
    parser.add_argument(
        "--action-norm-starts",
        type=Path,
        default=None,
        help="Starts used for action mean/std; defaults to the train cache encoding report.",
    )
    parser.add_argument(
        "--encoding-batch-size",
        type=int,
        default=128,
        help="Transition batch size for eval-cache encoding (match train cache report).",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help=(
            "Held-out train/evaluation split manifest. The filename "
            "formal_region_risk_pipeline.py is retained for compatibility."
        ),
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help=(
            "Require held-out partition/post-training audits. 'formal' is an internal "
            "audit/provenance term, not the public analysis name; incompatible with "
            "--allow-in-cache."
        ),
    )
    parser.add_argument(
        "--gate-summary",
        type=Path,
        default=None,
        help="Optional gate summary JSON to copy into the evaluation output directory.",
    )
    parser.add_argument(
        "--global-partition-dir",
        type=Path,
        default=None,
        help="Partition directory used for global predictor provenance checks.",
    )
    parser.add_argument(
        "--audit-partition-dir",
        type=Path,
        default=None,
        help="Auto-gate partition manifest used for gate/partition train-only audit.",
    )
    parser.add_argument(
        "--forced-spectral-partition-dir",
        type=Path,
        default=None,
        help="Forced-spectral partition manifest used for routing audit.",
    )
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--paper-eligible", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def episode_level_test_starts(
    data_file: Path,
    dataset_name: str,
    *,
    history_size: int,
    num_preds: int,
    frameskip: int,
    train_fraction: float,
    split_seed: int,
    seed: int,
) -> np.ndarray:
    train_episodes, _holdout = episode_level_split_episodes(
        data_file,
        dataset_name,
        history_size=history_size,
        num_preds=num_preds,
        frameskip=frameskip,
        train_fraction=train_fraction,
        split_seed=split_seed,
        seed=seed,
    )
    return _starts_for_episodes(
        data_file,
        dataset_name,
        history_size=history_size,
        num_preds=num_preds,
        frameskip=frameskip,
        seed=seed,
        episode_ids=set(_holdout),
    )


def episode_level_split_episodes(
    data_file: Path,
    dataset_name: str,
    *,
    history_size: int,
    num_preds: int,
    frameskip: int,
    train_fraction: float,
    split_seed: int,
    seed: int,
) -> tuple[set[int], set[int]]:
    spec = DATASETS[dataset_name]
    seq_len = history_size + num_preds
    with h5py.File(data_file, "r") as handle:
        state_key = choose_state_key(handle, spec, None)
        all_starts = valid_transition_starts(
            handle,
            spec,
            state_key,
            seq_len,
            frameskip,
            0,
            seed,
        )
        episode_ids = episode_ids_at_starts(data_file, all_starts)
    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(unique_episodes)
    train_n = int(round(len(perm) * train_fraction))
    train_episodes = set(map(int, perm[:train_n]))
    holdout_episodes = set(map(int, perm[train_n:]))
    return train_episodes, holdout_episodes


def _starts_for_episodes(
    data_file: Path,
    dataset_name: str,
    *,
    history_size: int,
    num_preds: int,
    frameskip: int,
    seed: int,
    episode_ids: set[int],
) -> np.ndarray:
    spec = DATASETS[dataset_name]
    seq_len = history_size + num_preds
    with h5py.File(data_file, "r") as handle:
        state_key = choose_state_key(handle, spec, None)
        all_starts = valid_transition_starts(
            handle,
            spec,
            state_key,
            seq_len,
            frameskip,
            0,
            seed,
        )
        starts_episode_ids = episode_ids_at_starts(data_file, all_starts)
    mask = np.asarray([int(ep) in episode_ids for ep in starts_episode_ids], dtype=bool)
    return np.sort(all_starts[mask].astype(np.int64))


def encode_eval_cache(
    *,
    data_file: Path,
    starts: np.ndarray,
    action_norm_starts: Path,
    pretrained_model: Path,
    output: Path,
    history_size: int,
    num_preds: int,
    frameskip: int,
    device: str,
    batch_size: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    starts_path = output.with_suffix(".starts.npy")
    np.save(starts_path, starts)
    dataset = make_hdf5_transition_dataset(
        data_file=str(data_file),
        starts=str(starts_path),
        action_norm_starts=str(action_norm_starts),
        history_size=history_size,
        num_preds=num_preds,
        frameskip=0,
    )
    encoder = LeWMEncoderAdapter(
        img_size=224,
        frameskip=0,
        checkpoint_cache_dir=None,
    )
    config = FastEncodingConfig(
        device=device,
        transition_batch_size=batch_size,
        frame_batch_size=512,
        exact_batch_shapes=True,
        num_workers=2,
        cpu_threads=4,
    )
    FastLatentCacheEncoder(config).encode(
        dataset=dataset,
        encoder=encoder,
        pretrained_model=str(pretrained_model),
        output=output,
    )


def resolve_run_dir(root: Path, train_seed: int, *, patterns: tuple[str, ...]) -> Path:
    for pattern in patterns:
        candidate = root / pattern.format(seed=train_seed)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no run directory for train_seed={train_seed} under {root}")


def _resolve_partition_manifest(partition_dir: Path) -> Path | None:
    partition_dir = partition_dir.resolve()
    manifest = partition_dir / "manifest.json"
    if manifest.exists():
        return manifest
    parent_manifest = partition_dir.parent / "manifest.json"
    if parent_manifest.exists():
        return parent_manifest
    return None


def resolve_global_checkpoint(run_dir: Path, name: str) -> Path:
    if name == "auto":
        for candidate in (
            "P_train_global_ft_object.ckpt",
            "P_train_cluster0_object.ckpt",
        ):
            path = run_dir / candidate
            if path.exists():
                return path
        raise FileNotFoundError(f"no global checkpoint found in {run_dir}")
    return run_dir / name


def load_expert_bundle(
    *,
    train_seed: int,
    regional_run: Path,
    global_run: Path,
    regional_template: str,
    global_name: str,
    num_regions: int,
    history_size: int,
    num_preds: int,
    latent_cache_hash: str,
    pretrained_hash: str,
    partition_hash: str,
    global_partition_hash: str | None = None,
    split_manifest_hash: str | None = None,
    device: torch.device,
) -> ExpertBundle:
    manifest_path = regional_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(
        manifest,
        train_seed=train_seed,
        history_size=history_size,
        num_preds=num_preds,
        latent_cache_hash=latent_cache_hash,
        pretrained_hash=pretrained_hash,
        partition_hash=partition_hash,
        split_manifest_hash=split_manifest_hash,
    )
    global_manifest_path = global_run / "manifest.json"
    global_manifest: dict[str, Any] | None = None
    if global_manifest_path.exists():
        global_manifest = json.loads(global_manifest_path.read_text(encoding="utf-8"))
        validate_manifest(
            global_manifest,
            train_seed=train_seed,
            history_size=history_size,
            num_preds=num_preds,
            latent_cache_hash=latent_cache_hash,
            pretrained_hash=pretrained_hash,
            partition_hash=global_partition_hash or partition_hash,
            split_manifest_hash=split_manifest_hash,
        )
    global_ckpt = resolve_global_checkpoint(global_run, global_name)
    checkpoint_hashes = {"global": sha256_file(global_ckpt)}
    for region in range(num_regions):
        ckpt = regional_run / regional_template.format(region=region)
        checkpoint_hashes[f"cluster{region}"] = sha256_file(ckpt)
    return ExpertBundle(
        train_seed=train_seed,
        global_model=load_model(global_ckpt).to(device),
        regional_models={
            region: load_model(regional_run / regional_template.format(region=region)).to(
                device
            )
            for region in range(num_regions)
        },
        checkpoint_hashes=checkpoint_hashes,
        manifest=manifest,
        global_manifest=global_manifest,
    )


def limit_episodes(
    anchors: np.ndarray,
    episode_ids: np.ndarray,
    *,
    max_episodes: int,
) -> np.ndarray:
    if max_episodes <= 0:
        return anchors
    keep: list[int] = []
    seen: set[int] = set()
    for start, episode in zip(anchors, episode_ids):
        if int(episode) not in seen:
            if len(seen) >= max_episodes:
                break
            seen.add(int(episode))
        keep.append(int(start))
    return np.asarray(keep, dtype=np.int64)


def expert_losses_from_matrix(
    loss_matrix: np.ndarray,
    anchor_regions: np.ndarray,
    num_regions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global_loss = loss_matrix[:, 0]
    correct_loss = np.asarray(
        [loss_matrix[row, 1 + anchor_regions[row]] for row in range(len(anchor_regions))],
        dtype=np.float64,
    )
    wrong_mean, wrong_best = wrong_expert_losses(
        loss_matrix[:, 1:],
        anchor_regions,
        num_regions,
    )
    return global_loss, correct_loss, wrong_mean, wrong_best


def collect_limited_horizon_anchors(
    eval_starts: np.ndarray,
    *,
    horizon: int,
    contract: Any,
    eval_start_map: Mapping[int, int],
    episode_lookup: Mapping[int, int],
    max_anchors: int,
    max_episodes: int,
) -> np.ndarray:
    anchors = collect_rollout_anchors(
        eval_starts,
        horizon=horizon,
        frameskip=contract.frameskip,
        history_size=contract.history_size,
        start_map=eval_start_map,
        episode_lookup=episode_lookup,
    )
    if max_anchors > 0:
        anchors = anchors[:max_anchors]
    if max_episodes > 0:
        anchors = limit_episodes(
            anchors,
            np.asarray([episode_lookup[int(start)] for start in anchors], dtype=np.int64),
            max_episodes=max_episodes,
        )
    return anchors


def horizon_rollout_losses(
    *,
    horizon: int,
    models: list[torch.nn.Module],
    eval_cache: LeWMLatentCache,
    anchors: np.ndarray,
    contract: Any,
    eval_start_map: Mapping[int, int],
    device: torch.device,
    batch_size: int,
    rollout_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if horizon == 1:
        rows = np.asarray([eval_start_map[int(start)] for start in anchors], dtype=np.int64)
        mean_traj_matrix = one_step_losses(
            models,
            eval_cache.emb[rows],
            eval_cache.act_emb[rows],
            history_size=contract.history_size,
            num_preds=contract.num_preds,
            device=device,
            batch_size=batch_size,
        )
        return mean_traj_matrix, mean_traj_matrix
    rollout = multi_horizon_open_loop_rollout_losses(
        models,
        eval_cache,
        anchors,
        horizons=[horizon],
        contract=contract,
        start_map=eval_start_map,
        device=device,
        batch_size=rollout_batch_size,
    ).by_horizon[horizon]
    return rollout.mean_trajectory_mse, rollout.terminal_mse


def append_horizon_metrics(
    *,
    task: str,
    train_seed: int,
    horizon: int,
    anchors: np.ndarray,
    anchor_regions: np.ndarray,
    anchor_episodes: np.ndarray,
    mean_traj_matrix: np.ndarray,
    terminal_matrix: np.ndarray,
    artifact: PartitionArtifact,
    anchor_support: str,
    sample_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    region_summary_rows: list[dict[str, Any]],
    weighted_rows: list[dict[str, Any]],
    seed_blocks: list[dict[str, Any]],
) -> None:
    global_loss, correct_loss, wrong_mean, wrong_best = expert_losses_from_matrix(
        mean_traj_matrix,
        anchor_regions,
        artifact.num_regions,
    )
    (
        terminal_global_loss,
        terminal_correct_loss,
        terminal_wrong_mean,
        terminal_wrong_best,
    ) = expert_losses_from_matrix(
        terminal_matrix,
        anchor_regions,
        artifact.num_regions,
    )

    seed_blocks.append(
        {
            "horizon": horizon,
            "train_seed": train_seed,
            "anchor_support": anchor_support,
            "global": global_loss,
            "correct": correct_loss,
            "wrong_mean": wrong_mean,
            "wrong_best": wrong_best,
            "terminal_global": terminal_global_loss,
            "terminal_correct": terminal_correct_loss,
            "terminal_wrong_mean": terminal_wrong_mean,
            "terminal_wrong_best": terminal_wrong_best,
            "episode_ids": anchor_episodes,
        }
    )

    for index, start in enumerate(anchors):
        sample_records.append(
            {
                "sample_id": int(start),
                "episode_id": int(anchor_episodes[index]),
                "region": int(anchor_regions[index]),
                "train_seed": train_seed,
                "horizon": horizon,
                "anchor_support": anchor_support,
                "global_loss": float(global_loss[index]),
                "correct_loss": float(correct_loss[index]),
                "wrong_mean_loss": float(wrong_mean[index]),
                "wrong_best_loss": float(wrong_best[index]),
                "terminal_global_loss": float(terminal_global_loss[index]),
                "terminal_correct_loss": float(terminal_correct_loss[index]),
                "terminal_wrong_mean_loss": float(terminal_wrong_mean[index]),
                "terminal_wrong_best_loss": float(terminal_wrong_best[index]),
                "mean_trajectory_global_loss": float(global_loss[index]),
                "mean_trajectory_correct_loss": float(correct_loss[index]),
                "mean_trajectory_wrong_mean_loss": float(wrong_mean[index]),
                "mean_trajectory_wrong_best_loss": float(wrong_best[index]),
            }
        )

    for episode in np.unique(anchor_episodes):
        mask = anchor_episodes == episode
        episode_records.append(
            {
                "task": task,
                "train_seed": train_seed,
                "horizon": horizon,
                "anchor_support": anchor_support,
                "episode_id": int(episode),
                "num_anchors": int(mask.sum()),
                "global_mse": float(global_loss[mask].mean()),
                "correct_mse": float(correct_loss[mask].mean()),
                "wrong_mean_mse": float(wrong_mean[mask].mean()),
                "wrong_best_mse": float(wrong_best[mask].mean()),
                "terminal_global_mse": float(terminal_global_loss[mask].mean()),
                "terminal_correct_mse": float(terminal_correct_loss[mask].mean()),
                "terminal_wrong_mean_mse": float(terminal_wrong_mean[mask].mean()),
                "terminal_wrong_best_mse": float(terminal_wrong_best[mask].mean()),
                "mean_trajectory_global_mse": float(global_loss[mask].mean()),
                "mean_trajectory_correct_mse": float(correct_loss[mask].mean()),
                "mean_trajectory_wrong_mean_mse": float(wrong_mean[mask].mean()),
                "mean_trajectory_wrong_best_mse": float(wrong_best[mask].mean()),
            }
        )

    region_rows = aggregate_region_metrics(
        task=task,
        train_seed=train_seed,
        horizon=horizon,
        regions=anchor_regions,
        episode_ids=anchor_episodes,
        global_loss=global_loss,
        correct_loss=correct_loss,
        wrong_mean_loss=wrong_mean,
        wrong_best_loss=wrong_best,
        num_regions=artifact.num_regions,
        terminal_global_loss=terminal_global_loss,
        terminal_correct_loss=terminal_correct_loss,
        terminal_wrong_mean_loss=terminal_wrong_mean,
        terminal_wrong_best_loss=terminal_wrong_best,
    )
    if not region_rows or not any(row["num_anchors"] > 0 for row in region_rows):
        return
    for row in region_rows:
        row["anchor_support"] = anchor_support
    region_summary_rows.extend(region_rows)
    summary = weighted_summary(
        region_rows,
        task=task,
        train_seed=train_seed,
        horizon=horizon,
    )
    summary["anchor_support"] = anchor_support
    weighted_rows.append(summary)


def resolve_paper_eligible(
    *,
    smoke_only: bool,
    split_manifest: Mapping[str, Any] | None,
    max_eval_starts: int,
    max_anchors: int,
    max_episodes: int,
    formal: bool,
    posttraining_train_only_valid: bool,
) -> bool:
    eval_untruncated = max_eval_starts <= 0 and max_anchors <= 0 and max_episodes <= 0
    split_unsubsampled = (
        split_manifest is not None and split_manifest_is_unsubsampled(split_manifest)
    )
    return (
        not smoke_only
        and eval_untruncated
        and split_unsubsampled
        and (not formal or posttraining_train_only_valid)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_region_risk(
    region_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    out_path: Path,
    *,
    horizons: list[int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), squeeze=False)
    fig.suptitle(PUBLIC_ANALYSIS_SHORT_NAME)
    for axis_index, horizon in enumerate(horizons):
        ax = axes[0, axis_index]
        subset = [row for row in region_rows if row["horizon"] == horizon]
        regions = sorted({row["region"] for row in subset})
        x = np.arange(len(regions))
        width = 0.22
        for offset, key, label in [
            (0.0, "global_mse", "Global"),
            (width, "correct_mse", "Correct"),
            (2 * width, "wrong_mean_mse", "Wrong mean"),
        ]:
            values = [
                next(row[key] for row in subset if row["region"] == region)
                for region in regions
            ]
            ax.bar(x + offset, values, width=width, label=label)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"R{r}" for r in regions])
        ax.set_title(f"Horizon {horizon}")
        ax.set_ylabel("MSE")
        if axis_index == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(
        out_path,
        bbox_inches="tight",
        metadata={"Title": PUBLIC_ANALYSIS_SHORT_NAME},
    )
    plt.close(fig)

    forest_path = out_path.with_name(out_path.stem + "_forest.pdf")
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in bootstrap_rows:
        if row["metric"] != "correct_minus_global":
            continue
        if row.get("loss_kind", "mean_trajectory") != "mean_trajectory":
            continue
        estimate = -row["estimate"]
        low = -row["ci_high"]
        high = -row["ci_low"]
        ax.errorbar(estimate, 0, xerr=[[estimate - low], [high - estimate]], fmt="o")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Global - Correct (positive = regional gain)")
    ax.set_title(PUBLIC_ANALYSIS_SHORT_NAME)
    fig.tight_layout()
    fig.savefig(
        forest_path,
        bbox_inches="tight",
        metadata={"Title": PUBLIC_ANALYSIS_SHORT_NAME},
    )
    plt.close(fig)


def _legacy_main() -> None:
    args = parse_args()
    if args.formal and args.allow_in_cache:
        raise ValueError("--formal cannot be combined with --allow-in-cache")
    device = resolve_device(args.device)
    train_seeds = [int(value) for value in args.train_seeds.split(",") if value]
    horizons = [int(value) for value in args.horizons.split(",") if value]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    partition_run = args.partition_dir.resolve(strict=True)
    artifact = PartitionArtifact.load(
        partition_run / "partition"
        if (partition_run / "partition").exists()
        else partition_run
    )
    partition_manifest = partition_run / "manifest.json"
    if not partition_manifest.exists() and (partition_run.parent / "manifest.json").exists():
        partition_manifest = partition_run.parent / "manifest.json"
    partition_hash = sha256_file(partition_manifest) if partition_manifest.exists() else ""
    global_partition_run = (
        args.global_partition_dir.resolve()
        if args.global_partition_dir is not None
        else partition_run
    )
    global_partition_manifest = global_partition_run / "manifest.json"
    if not global_partition_manifest.exists() and (
        global_partition_run.parent / "manifest.json"
    ).exists():
        global_partition_manifest = global_partition_run.parent / "manifest.json"
    global_partition_hash = (
        sha256_file(global_partition_manifest) if global_partition_manifest.exists() else ""
    )

    train_cache = load_lewm_cache(args.train_latent_cache, route_index=args.route_index)
    contract = load_cache_contract(
        train_cache,
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
    )

    train_cache_hash = sha256_file(args.train_latent_cache.resolve(strict=True))
    split_manifest_payload: dict[str, Any] | None = None
    split_manifest_hash = ""
    if args.split_manifest is not None:
        split_manifest_payload = load_split_manifest(args.split_manifest)
        split_manifest_hash = sha256_file(args.split_manifest.resolve(strict=True))
    action_norm_starts = resolve_action_norm_starts(
        args.train_latent_cache.resolve(strict=True),
        override=args.action_norm_starts,
    )

    eval_cache_path = args.eval_latent_cache.resolve()
    if args.build_eval_cache or not eval_cache_path.exists():
        if split_manifest_payload is not None:
            split_paths = split_paths_from_manifest(split_manifest_payload)
            test_starts = np.load(split_paths["eval_starts"])
        else:
            test_starts = episode_level_test_starts(
                args.data_file.resolve(strict=True),
                args.task,
                history_size=args.history_size,
                num_preds=args.num_preds,
                frameskip=args.frameskip,
                train_fraction=args.train_fraction,
                split_seed=args.split_seed,
                seed=0,
            )
        if args.max_eval_starts > 0 and len(test_starts) > args.max_eval_starts:
            test_starts = test_starts[: args.max_eval_starts]
        encode_eval_cache(
            data_file=args.data_file.resolve(strict=True),
            starts=test_starts,
            action_norm_starts=action_norm_starts,
            pretrained_model=args.pretrained_model.resolve(strict=True),
            output=eval_cache_path,
            history_size=args.history_size,
            num_preds=args.num_preds,
            frameskip=args.frameskip,
            device=str(device),
            batch_size=args.encoding_batch_size,
        )

    eval_cache = load_lewm_cache(eval_cache_path, route_index=args.route_index)
    load_cache_contract(
        eval_cache,
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
    )
    eval_cache_hash = sha256_file(eval_cache_path)

    audit = audit_episode_disjointness(
        data_file=args.data_file.resolve(strict=True),
        train_starts=train_cache.sample_ids,
        eval_starts=eval_cache.sample_ids,
        require_disjoint=args.formal or not args.allow_in_cache,
    )
    if args.formal:
        audit["mode"] = "formal_held_out_posttraining"
    elif not audit["episode_disjoint"] and args.allow_in_cache:
        audit["mode"] = "descriptive_in_cache"
    else:
        audit["mode"] = "episode_disjoint_held_out"

    pretrained_hash = sha256_file(args.pretrained_model.resolve(strict=True))
    action_norm_starts_hash = sha256_file(action_norm_starts)
    if split_manifest_payload is not None:
        nominal_train_episodes = set(map(int, split_manifest_payload["train_episode_ids"]))
        _holdout_episodes = set(map(int, split_manifest_payload["eval_episode_ids"]))
    else:
        nominal_train_episodes, _holdout_episodes = episode_level_split_episodes(
            args.data_file.resolve(strict=True),
            args.task,
            history_size=args.history_size,
            num_preds=args.num_preds,
            frameskip=args.frameskip,
            train_fraction=args.train_fraction,
            split_seed=args.split_seed,
            seed=0,
        )
    partition_contract = audit_partition_train_contract(
        data_file=args.data_file.resolve(strict=True),
        train_starts=train_cache.sample_ids,
        eval_starts=eval_cache.sample_ids,
        train_cache_hash=train_cache_hash,
        partition_manifest_path=_resolve_partition_manifest(
            args.audit_partition_dir or args.partition_dir
        ),
        nominal_train_episode_ids=nominal_train_episodes,
        require_train_only=args.formal or not args.allow_in_cache,
    )
    forced_spectral_partition_dir = (
        args.forced_spectral_partition_dir or args.partition_dir
    ).resolve()
    forced_spectral_contract = audit_partition_train_contract(
        data_file=args.data_file.resolve(strict=True),
        train_starts=train_cache.sample_ids,
        eval_starts=eval_cache.sample_ids,
        train_cache_hash=train_cache_hash,
        partition_manifest_path=_resolve_partition_manifest(forced_spectral_partition_dir),
        nominal_train_episode_ids=nominal_train_episodes,
        require_train_only=args.formal or not args.allow_in_cache,
    )
    global_partition_contract = audit_partition_train_contract(
        data_file=args.data_file.resolve(strict=True),
        train_starts=train_cache.sample_ids,
        eval_starts=eval_cache.sample_ids,
        train_cache_hash=train_cache_hash,
        partition_manifest_path=_resolve_partition_manifest(global_partition_run),
        nominal_train_episode_ids=nominal_train_episodes,
        require_train_only=args.formal or not args.allow_in_cache,
    )
    partition_contracts = {
        "auto": partition_contract,
        "forced_spectral": forced_spectral_contract,
        "global": global_partition_contract,
    }
    cache_start_audits: dict[str, dict[str, Any]] = {}
    if split_manifest_payload is not None:
        split_paths = split_paths_from_manifest(split_manifest_payload)
        cache_start_audits["train"] = audit_cache_starts_exact(
            cache_starts=train_cache.sample_ids,
            expected_starts=np.load(split_paths["train_starts"]),
            label="train_cache",
            require_exact=args.formal,
        )
        cache_start_audits["eval"] = audit_cache_starts_exact(
            cache_starts=eval_cache.sample_ids,
            expected_starts=np.load(split_paths["eval_starts"]),
            label="eval_cache",
            require_exact=args.formal,
        )

    eval_regions = route_regions_from_cache(
        eval_cache, artifact, device=device, batch_size=args.batch_size
    )
    eval_episode_ids = episode_ids_at_starts(
        args.data_file.resolve(strict=True), eval_cache.sample_ids
    )
    eval_start_map = start_index_map(eval_cache.sample_ids)
    episode_lookup = {
        int(start): int(episode)
        for start, episode in zip(eval_cache.sample_ids, eval_episode_ids)
    }

    sample_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    region_summary_rows: list[dict[str, Any]] = []
    weighted_rows: list[dict[str, Any]] = []
    common_h10_region_rows: list[dict[str, Any]] = []
    common_h10_weighted_rows: list[dict[str, Any]] = []
    seed_blocks: list[dict[str, Any]] = []
    checkpoint_manifests: list[dict[str, Any]] = []

    anchors_by_horizon = {
        horizon: collect_limited_horizon_anchors(
            eval_cache.sample_ids,
            horizon=horizon,
            contract=contract,
            eval_start_map=eval_start_map,
            episode_lookup=episode_lookup,
            max_anchors=args.max_anchors,
            max_episodes=args.max_episodes,
        )
        for horizon in horizons
    }
    horizon_anchor_counts = {
        str(horizon): int(len(anchors)) for horizon, anchors in anchors_by_horizon.items()
    }
    common_h10_anchors = (
        anchors_by_horizon[10] if 10 in anchors_by_horizon else np.asarray([], dtype=np.int64)
    )

    for train_seed in train_seeds:
        regional_run = resolve_run_dir(
            args.regional_runs,
            train_seed,
            patterns=FORMAL_REGIONAL_RUN_PATTERNS,
        )
        global_run = resolve_run_dir(
            args.global_runs,
            train_seed,
            patterns=(
                "tworoom_geometry_train_global_ft_50ep_trainseed{seed}",
                "tworoom_geometry_train_global_ft_45ep_trainseed{seed}",
                "trainseed{seed}",
                "train{seed}",
            ),
        )
        bundle = load_expert_bundle(
            train_seed=train_seed,
            regional_run=regional_run,
            global_run=global_run,
            regional_template=args.regional_checkpoint_template,
            global_name=args.global_checkpoint_name,
            num_regions=artifact.num_regions,
            history_size=args.history_size,
            num_preds=args.num_preds,
            latent_cache_hash=train_cache_hash,
            pretrained_hash=pretrained_hash,
            partition_hash=partition_hash,
            global_partition_hash=global_partition_hash,
            split_manifest_hash=split_manifest_hash or None,
            device=device,
        )
        checkpoint_manifests.append(bundle.manifest)
        if bundle.global_manifest is not None:
            checkpoint_manifests.append(bundle.global_manifest)
        models = [bundle.global_model] + [
            bundle.regional_models[region] for region in range(artifact.num_regions)
        ]

        for horizon in horizons:
            anchors = anchors_by_horizon[horizon]
            if len(anchors) == 0:
                continue
            anchor_regions = np.asarray(
                [eval_regions[eval_start_map[int(start)]] for start in anchors], dtype=np.int64
            )
            anchor_episodes = np.asarray(
                [episode_lookup[int(start)] for start in anchors], dtype=np.int64
            )
            mean_traj_matrix, terminal_matrix = horizon_rollout_losses(
                horizon=horizon,
                models=models,
                eval_cache=eval_cache,
                anchors=anchors,
                contract=contract,
                eval_start_map=eval_start_map,
                device=device,
                batch_size=args.batch_size,
                rollout_batch_size=args.rollout_batch_size,
            )
            append_horizon_metrics(
                task=args.task,
                train_seed=train_seed,
                horizon=horizon,
                anchors=anchors,
                anchor_regions=anchor_regions,
                anchor_episodes=anchor_episodes,
                mean_traj_matrix=mean_traj_matrix,
                terminal_matrix=terminal_matrix,
                artifact=artifact,
                anchor_support="horizon_valid",
                sample_records=sample_records,
                episode_records=episode_records,
                region_summary_rows=region_summary_rows,
                weighted_rows=weighted_rows,
                seed_blocks=seed_blocks,
            )

        if len(common_h10_anchors) > 0 and len(horizons) > 1:
            common_regions = np.asarray(
                [eval_regions[eval_start_map[int(start)]] for start in common_h10_anchors],
                dtype=np.int64,
            )
            common_episodes = np.asarray(
                [episode_lookup[int(start)] for start in common_h10_anchors], dtype=np.int64
            )
            for horizon in horizons:
                mean_traj_matrix, terminal_matrix = horizon_rollout_losses(
                    horizon=horizon,
                    models=models,
                    eval_cache=eval_cache,
                    anchors=common_h10_anchors,
                    contract=contract,
                    eval_start_map=eval_start_map,
                    device=device,
                    batch_size=args.batch_size,
                    rollout_batch_size=args.rollout_batch_size,
                )
                append_horizon_metrics(
                    task=args.task,
                    train_seed=train_seed,
                    horizon=horizon,
                    anchors=common_h10_anchors,
                    anchor_regions=common_regions,
                    anchor_episodes=common_episodes,
                    mean_traj_matrix=mean_traj_matrix,
                    terminal_matrix=terminal_matrix,
                    artifact=artifact,
                    anchor_support="common_h10",
                    sample_records=sample_records,
                    episode_records=episode_records,
                    region_summary_rows=common_h10_region_rows,
                    weighted_rows=common_h10_weighted_rows,
                    seed_blocks=[],
                )

    bootstrap_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        blocks = [
            block
            for block in seed_blocks
            if block["horizon"] == horizon and block.get("anchor_support") == "horizon_valid"
        ]
        if not blocks:
            continue
        bootstrap_rows.extend(
            nested_paired_bootstrap_ci(
                blocks,
                reps=args.bootstrap_reps,
                seed=args.bootstrap_seed + horizon,
                metric_label="mean_trajectory",
            )
        )
        bootstrap_rows.extend(
            nested_paired_bootstrap_ci(
                blocks,
                reps=args.bootstrap_reps,
                seed=args.bootstrap_seed + horizon + 1000,
                metric_keys=(
                    "terminal_global",
                    "terminal_correct",
                    "terminal_wrong_mean",
                    "terminal_wrong_best",
                ),
                metric_label="terminal",
                pairs=(
                    ("correct_minus_global", "terminal_correct", "terminal_global"),
                    ("correct_minus_wrong_mean", "terminal_correct", "terminal_wrong_mean"),
                    ("correct_minus_wrong_best", "terminal_correct", "terminal_wrong_best"),
                ),
            )
        )

    formal_audit: dict[str, Any] = {}
    if args.formal:
        if split_manifest_payload is None:
            raise ValueError("--formal requires --split-manifest")
        formal_audit = audit_formal_posttraining(
            episode_audit=audit,
            partition_contracts=partition_contracts,
            split_manifest=split_manifest_payload,
            split_manifest_sha256=split_manifest_hash,
            train_cache_hash=train_cache_hash,
            eval_cache_hash=eval_cache_hash,
            action_norm_starts_hash=action_norm_starts_hash,
            cache_start_audits=cache_start_audits,
            checkpoint_manifests=checkpoint_manifests,
            require_valid=True,
        )

    audit_payload = {
        **audit,
        **formal_audit,
        **cache_start_audits.get("train", {}),
        **cache_start_audits.get("eval", {}),
        "task": args.task,
        "formal": args.formal,
        "smoke_only": args.smoke_only,
        "paper_eligible": resolve_paper_eligible(
            smoke_only=args.smoke_only,
            split_manifest=split_manifest_payload,
            max_eval_starts=args.max_eval_starts,
            max_anchors=args.max_anchors,
            max_episodes=args.max_episodes,
            formal=args.formal,
            posttraining_train_only_valid=formal_audit.get(
                "posttraining_train_only_valid", False
            ),
        ),
        "claim_scope": FORMAL_CLAIM_SCOPE if args.formal else None,
        "base_pretraining_episode_disjoint": (
            FORMAL_BASE_PRETRAINING_EPISODE_DISJOINT if args.formal else None
        ),
        "split_manifest_unsubsampled_valid": (
            split_manifest_payload is not None
            and split_manifest_is_unsubsampled(split_manifest_payload)
        ),
        "horizon_anchor_counts": horizon_anchor_counts,
        "common_h10_anchor_count": int(len(common_h10_anchors)),
        "formal_episode_disjoint_valid": audit.get("episode_disjoint", False)
        and audit.get("region_start_disjoint", False),
        "auto_gate_train_only_valid": partition_contracts["auto"].get(
            "gate_partition_train_only_valid", False
        ),
        "forced_spectral_train_only_valid": partition_contracts["forced_spectral"].get(
            "gate_partition_train_only_valid", False
        ),
        "global_partition_train_only_valid": partition_contracts["global"].get(
            "gate_partition_train_only_valid", False
        ),
        "train_fraction": args.train_fraction,
        "split_seed": args.split_seed,
        "split_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None,
        "split_manifest_sha256": split_manifest_hash or None,
        "nominal_train_num_episodes": len(nominal_train_episodes),
        "nominal_holdout_num_episodes": len(_holdout_episodes),
        "train_cache": str(args.train_latent_cache.resolve()),
        "eval_cache": str(eval_cache_path),
        "train_cache_sha256": train_cache_hash,
        "eval_cache_sha256": eval_cache_hash,
        "partition_dir": str(partition_run),
        "partition_sha256": partition_hash,
        "pretrained_model_sha256": pretrained_hash,
        "action_norm_starts": str(action_norm_starts),
        "action_norm_starts_sha256": action_norm_starts_hash,
        "git_commit": git_commit(),
        "runtime": runtime_provenance(python_executable=sys.executable),
        "command": " ".join(sys.argv),
        "frameskip": contract.frameskip,
        "history_size": contract.history_size,
        "route_index": contract.route_index,
    }
    atomic_write_json(out_dir / "audit.json", audit_payload)
    if args.gate_summary is not None and args.gate_summary.exists():
        atomic_write_json(
            out_dir / "gate_summary.json",
            json.loads(args.gate_summary.read_text(encoding="utf-8")),
        )

    np.savez_compressed(
        out_dir / "sample_metrics.npz",
        sample_id=np.asarray([row["sample_id"] for row in sample_records], dtype=np.int64),
        episode_id=np.asarray([row["episode_id"] for row in sample_records], dtype=np.int64),
        region=np.asarray([row["region"] for row in sample_records], dtype=np.int64),
        train_seed=np.asarray([row["train_seed"] for row in sample_records], dtype=np.int64),
        horizon=np.asarray([row["horizon"] for row in sample_records], dtype=np.int64),
        anchor_support=np.asarray(
            [row["anchor_support"] for row in sample_records], dtype="<U16"
        ),
        global_loss=np.asarray([row["global_loss"] for row in sample_records], dtype=np.float64),
        correct_loss=np.asarray([row["correct_loss"] for row in sample_records], dtype=np.float64),
        wrong_mean_loss=np.asarray(
            [row["wrong_mean_loss"] for row in sample_records], dtype=np.float64
        ),
        wrong_best_loss=np.asarray(
            [row["wrong_best_loss"] for row in sample_records], dtype=np.float64
        ),
        terminal_global_loss=np.asarray(
            [row["terminal_global_loss"] for row in sample_records], dtype=np.float64
        ),
        terminal_correct_loss=np.asarray(
            [row["terminal_correct_loss"] for row in sample_records], dtype=np.float64
        ),
        terminal_wrong_mean_loss=np.asarray(
            [row["terminal_wrong_mean_loss"] for row in sample_records], dtype=np.float64
        ),
        terminal_wrong_best_loss=np.asarray(
            [row["terminal_wrong_best_loss"] for row in sample_records], dtype=np.float64
        ),
    )
    write_csv(out_dir / "episode_metrics.csv", episode_records)
    write_csv(out_dir / "region_summary.csv", region_summary_rows)
    write_csv(out_dir / "weighted_summary.csv", weighted_rows)
    write_csv(out_dir / "common_h10_support_summary.csv", common_h10_weighted_rows)
    write_csv(out_dir / "bootstrap_summary.csv", bootstrap_rows)
    main_region_rows = [
        row for row in region_summary_rows if row.get("anchor_support") == "horizon_valid"
    ]
    plot_region_risk(main_region_rows, bootstrap_rows, out_dir / "region_risk.pdf", horizons=horizons)

    manifest = {
        "task": args.task,
        "audit": audit_payload,
        "files": {
            name: sha256_file(path)
            for name, path in {
                "audit.json": out_dir / "audit.json",
                "sample_metrics.npz": out_dir / "sample_metrics.npz",
                "episode_metrics.csv": out_dir / "episode_metrics.csv",
                "region_summary.csv": out_dir / "region_summary.csv",
                "weighted_summary.csv": out_dir / "weighted_summary.csv",
                "bootstrap_summary.csv": out_dir / "bootstrap_summary.csv",
                "region_risk.pdf": out_dir / "region_risk.pdf",
            }.items()
            if path.exists()
        },
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    print(f"[done] {PUBLIC_ANALYSIS_SHORT_NAME} -> {out_dir}", flush=True)


def _npz_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"].item()))


def _anchors_sha256(anchors: np.ndarray) -> str:
    values = np.ascontiguousarray(anchors, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _raw_path(raw_dir: Path, train_seed: int, support_horizon: int) -> Path:
    return raw_dir / f"trainseed{train_seed}_h{support_horizon}_valid.npz"


def _resume_matches(path: Path, fingerprint: str, *, resume: bool, kind: str) -> bool:
    if not path.exists():
        return False
    actual = _npz_metadata(path).get("fingerprint")
    if actual != fingerprint:
        raise ValueError(
            f"{kind} fingerprint mismatch for {path}: expected={fingerprint} actual={actual}"
        )
    if not resume:
        raise FileExistsError(f"{kind} already exists: {path}; pass --resume to reuse it")
    print(f"[resume] {PUBLIC_ANALYSIS_SHORT_NAME}: {path}", flush=True)
    return True


def _resolve_bundle_provenance(
    args: argparse.Namespace,
    *,
    train_seed: int,
    num_regions: int,
    train_cache_hash: str,
    pretrained_hash: str,
    partition_hash: str,
    global_partition_hash: str,
    split_manifest_hash: str,
) -> dict[str, Any]:
    regional_run = resolve_run_dir(
        args.regional_runs, train_seed, patterns=FORMAL_REGIONAL_RUN_PATTERNS
    )
    global_run = resolve_run_dir(
        args.global_runs,
        train_seed,
        patterns=(
            "tworoom_geometry_train_global_ft_50ep_trainseed{seed}",
            "tworoom_geometry_train_global_ft_45ep_trainseed{seed}",
            "trainseed{seed}",
            "train{seed}",
        ),
    )
    regional_manifest = json.loads((regional_run / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(
        regional_manifest,
        train_seed=train_seed,
        history_size=args.history_size,
        num_preds=args.num_preds,
        latent_cache_hash=train_cache_hash,
        pretrained_hash=pretrained_hash,
        partition_hash=partition_hash,
        split_manifest_hash=split_manifest_hash or None,
    )
    global_manifest_path = global_run / "manifest.json"
    global_manifest = (
        json.loads(global_manifest_path.read_text(encoding="utf-8"))
        if global_manifest_path.exists()
        else None
    )
    if global_manifest is not None:
        validate_manifest(
            global_manifest,
            train_seed=train_seed,
            history_size=args.history_size,
            num_preds=args.num_preds,
            latent_cache_hash=train_cache_hash,
            pretrained_hash=pretrained_hash,
            partition_hash=global_partition_hash or partition_hash,
            split_manifest_hash=split_manifest_hash or None,
        )
    checkpoint_paths = {"global": resolve_global_checkpoint(global_run, args.global_checkpoint_name)}
    checkpoint_paths.update(
        {
            f"cluster{region}": regional_run
            / args.regional_checkpoint_template.format(region=region)
            for region in range(num_regions)
        }
    )
    return {
        "regional_run": regional_run,
        "global_run": global_run,
        "regional_manifest": regional_manifest,
        "global_manifest": global_manifest,
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_hashes": {
            name: sha256_file(path.resolve(strict=True)) for name, path in checkpoint_paths.items()
        },
    }


def _load_models_from_provenance(
    provenance: Mapping[str, Any], *, device: torch.device, num_regions: int
) -> list[torch.nn.Module]:
    paths = provenance["checkpoint_paths"]
    return [load_model(paths["global"]).to(device)] + [
        load_model(paths[f"cluster{region}"]).to(device) for region in range(num_regions)
    ]


def _prepare_rollout_context(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    partition_run = args.partition_dir.resolve(strict=True)
    artifact = PartitionArtifact.load(
        partition_run / "partition" if (partition_run / "partition").exists() else partition_run
    )
    partition_manifest = _resolve_partition_manifest(partition_run)
    partition_hash = sha256_file(partition_manifest) if partition_manifest else ""
    global_partition_run = (
        args.global_partition_dir.resolve()
        if args.global_partition_dir is not None
        else partition_run
    )
    global_partition_manifest = _resolve_partition_manifest(global_partition_run)
    global_partition_hash = (
        sha256_file(global_partition_manifest) if global_partition_manifest else ""
    )
    train_cache = load_lewm_cache(args.train_latent_cache, route_index=args.route_index)
    contract = load_cache_contract(
        train_cache,
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
    )
    train_cache_hash = sha256_file(args.train_latent_cache.resolve(strict=True))
    split_manifest_payload: dict[str, Any] | None = None
    split_manifest_hash = ""
    if args.split_manifest is not None:
        split_manifest_payload = load_split_manifest(args.split_manifest)
        split_manifest_hash = sha256_file(args.split_manifest.resolve(strict=True))
    action_norm_starts = resolve_action_norm_starts(
        args.train_latent_cache.resolve(strict=True), override=args.action_norm_starts
    )
    eval_cache_path = args.eval_latent_cache.resolve()
    if args.build_eval_cache or not eval_cache_path.exists():
        if split_manifest_payload is not None:
            test_starts = np.load(split_paths_from_manifest(split_manifest_payload)["eval_starts"])
        else:
            test_starts = episode_level_test_starts(
                args.data_file.resolve(strict=True),
                args.task,
                history_size=args.history_size,
                num_preds=args.num_preds,
                frameskip=args.frameskip,
                train_fraction=args.train_fraction,
                split_seed=args.split_seed,
                seed=0,
            )
        if args.max_eval_starts > 0:
            test_starts = test_starts[: args.max_eval_starts]
        encode_eval_cache(
            data_file=args.data_file.resolve(strict=True),
            starts=test_starts,
            action_norm_starts=action_norm_starts,
            pretrained_model=args.pretrained_model.resolve(strict=True),
            output=eval_cache_path,
            history_size=args.history_size,
            num_preds=args.num_preds,
            frameskip=args.frameskip,
            device=str(device),
            batch_size=args.encoding_batch_size,
        )
    eval_cache = load_lewm_cache(eval_cache_path, route_index=args.route_index)
    load_cache_contract(
        eval_cache,
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
    )
    eval_cache_hash = sha256_file(eval_cache_path)
    audit = audit_episode_disjointness(
        data_file=args.data_file.resolve(strict=True),
        train_starts=train_cache.sample_ids,
        eval_starts=eval_cache.sample_ids,
        require_disjoint=args.formal or not args.allow_in_cache,
    )
    if args.formal:
        audit["mode"] = "formal_held_out_posttraining"
    elif not audit["episode_disjoint"] and args.allow_in_cache:
        audit["mode"] = "descriptive_in_cache"
    else:
        audit["mode"] = "episode_disjoint_held_out"
    pretrained_hash = sha256_file(args.pretrained_model.resolve(strict=True))
    action_norm_starts_hash = sha256_file(action_norm_starts)
    if split_manifest_payload is not None:
        nominal_train_episodes = set(map(int, split_manifest_payload["train_episode_ids"]))
        holdout_episodes = set(map(int, split_manifest_payload["eval_episode_ids"]))
    else:
        nominal_train_episodes, holdout_episodes = episode_level_split_episodes(
            args.data_file.resolve(strict=True),
            args.task,
            history_size=args.history_size,
            num_preds=args.num_preds,
            frameskip=args.frameskip,
            train_fraction=args.train_fraction,
            split_seed=args.split_seed,
            seed=0,
        )
    partition_contracts = {
        "auto": audit_partition_train_contract(
            data_file=args.data_file.resolve(strict=True),
            train_starts=train_cache.sample_ids,
            eval_starts=eval_cache.sample_ids,
            train_cache_hash=train_cache_hash,
            partition_manifest_path=_resolve_partition_manifest(
                args.audit_partition_dir or args.partition_dir
            ),
            nominal_train_episode_ids=nominal_train_episodes,
            require_train_only=args.formal or not args.allow_in_cache,
        ),
        "forced_spectral": audit_partition_train_contract(
            data_file=args.data_file.resolve(strict=True),
            train_starts=train_cache.sample_ids,
            eval_starts=eval_cache.sample_ids,
            train_cache_hash=train_cache_hash,
            partition_manifest_path=_resolve_partition_manifest(
                args.forced_spectral_partition_dir or args.partition_dir
            ),
            nominal_train_episode_ids=nominal_train_episodes,
            require_train_only=args.formal or not args.allow_in_cache,
        ),
        "global": audit_partition_train_contract(
            data_file=args.data_file.resolve(strict=True),
            train_starts=train_cache.sample_ids,
            eval_starts=eval_cache.sample_ids,
            train_cache_hash=train_cache_hash,
            partition_manifest_path=_resolve_partition_manifest(global_partition_run),
            nominal_train_episode_ids=nominal_train_episodes,
            require_train_only=args.formal or not args.allow_in_cache,
        ),
    }
    cache_start_audits: dict[str, dict[str, Any]] = {}
    if split_manifest_payload is not None:
        paths = split_paths_from_manifest(split_manifest_payload)
        cache_start_audits = {
            "train": audit_cache_starts_exact(
                cache_starts=train_cache.sample_ids,
                expected_starts=np.load(paths["train_starts"]),
                label="train_cache",
                require_exact=args.formal,
            ),
            "eval": audit_cache_starts_exact(
                cache_starts=eval_cache.sample_ids,
                expected_starts=np.load(paths["eval_starts"]),
                label="eval_cache",
                require_exact=args.formal,
            ),
        }
    eval_regions = route_regions_from_cache(
        eval_cache, artifact, device=device, batch_size=args.batch_size
    )
    eval_episode_ids = episode_ids_at_starts(
        args.data_file.resolve(strict=True), eval_cache.sample_ids
    )
    eval_start_map = start_index_map(eval_cache.sample_ids)
    episode_lookup = {
        int(start): int(episode)
        for start, episode in zip(eval_cache.sample_ids, eval_episode_ids)
    }
    horizons = sorted({int(value) for value in args.horizons.split(",") if value})
    anchors_by_horizon = {
        horizon: collect_limited_horizon_anchors(
            eval_cache.sample_ids,
            horizon=horizon,
            contract=contract,
            eval_start_map=eval_start_map,
            episode_lookup=episode_lookup,
            max_anchors=args.max_anchors,
            max_episodes=args.max_episodes,
        )
        for horizon in horizons
    }
    return {
        "device": device,
        "out_dir": out_dir,
        "artifact": artifact,
        "partition_run": partition_run,
        "partition_hash": partition_hash,
        "global_partition_hash": global_partition_hash,
        "train_cache": train_cache,
        "eval_cache": eval_cache,
        "contract": contract,
        "train_cache_hash": train_cache_hash,
        "eval_cache_hash": eval_cache_hash,
        "pretrained_hash": pretrained_hash,
        "action_norm_starts": action_norm_starts,
        "action_norm_starts_hash": action_norm_starts_hash,
        "split_manifest": split_manifest_payload,
        "split_manifest_hash": split_manifest_hash,
        "nominal_train_episodes": nominal_train_episodes,
        "holdout_episodes": holdout_episodes,
        "audit": audit,
        "partition_contracts": partition_contracts,
        "cache_start_audits": cache_start_audits,
        "eval_regions": eval_regions,
        "eval_start_map": eval_start_map,
        "episode_lookup": episode_lookup,
        "horizons": horizons,
        "anchors_by_horizon": anchors_by_horizon,
    }


def run_rollout_stage(args: argparse.Namespace) -> None:
    context = _prepare_rollout_context(args)
    raw_dir = context["out_dir"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = [int(value) for value in args.train_seeds.split(",") if value]
    checkpoint_manifests: list[dict[str, Any]] = []
    raw_files: list[dict[str, Any]] = []
    for train_seed in train_seeds:
        provenance = _resolve_bundle_provenance(
            args,
            train_seed=train_seed,
            num_regions=context["artifact"].num_regions,
            train_cache_hash=context["train_cache_hash"],
            pretrained_hash=context["pretrained_hash"],
            partition_hash=context["partition_hash"],
            global_partition_hash=context["global_partition_hash"],
            split_manifest_hash=context["split_manifest_hash"],
        )
        checkpoint_manifests.append(provenance["regional_manifest"])
        if provenance["global_manifest"] is not None:
            checkpoint_manifests.append(provenance["global_manifest"])
        pending: list[tuple[int, list[int], np.ndarray, Path, dict[str, Any]]] = []
        for support_horizon in context["horizons"]:
            anchors = context["anchors_by_horizon"][support_horizon]
            requested = (
                [value for value in context["horizons"] if value <= 10]
                if support_horizon == 10
                else [support_horizon]
            )
            metadata = {
                "schema_version": 1,
                "task": args.task,
                "train_seed": train_seed,
                "anchor_support": f"h{support_horizon}_valid",
                "support_horizon": support_horizon,
                "requested_horizons": requested,
                "anchors_sha256": _anchors_sha256(anchors),
                "checkpoint_hashes": provenance["checkpoint_hashes"],
                "cache_hash": context["eval_cache_hash"],
                "partition_hash": context["partition_hash"],
                "history_size": context["contract"].history_size,
                "frameskip": context["contract"].frameskip,
                "route_index": context["contract"].route_index,
                "num_regions": context["artifact"].num_regions,
            }
            metadata["fingerprint"] = stable_json_sha256(metadata)
            path = _raw_path(raw_dir, train_seed, support_horizon)
            if _resume_matches(
                path, metadata["fingerprint"], resume=args.resume, kind="raw rollout"
            ):
                raw_files.append({"path": str(path), "sha256": sha256_file(path)})
                continue
            pending.append((support_horizon, requested, anchors, path, metadata))
        if not pending:
            continue
        models = _load_models_from_provenance(
            provenance, device=context["device"], num_regions=context["artifact"].num_regions
        )
        for support_horizon, requested, anchors, path, metadata in pending:
            rollout = multi_horizon_open_loop_rollout_losses(
                models,
                context["eval_cache"],
                anchors,
                horizons=requested,
                contract=context["contract"],
                start_map=context["eval_start_map"],
                device=context["device"],
                batch_size=args.rollout_batch_size,
            )
            rows = np.asarray(
                [context["eval_start_map"][int(start)] for start in anchors], dtype=np.int64
            )
            payload: dict[str, Any] = {
                "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
                "train_seed": np.asarray(train_seed, dtype=np.int64),
                "anchor_support": np.asarray(f"h{support_horizon}_valid"),
                "requested_horizons": np.asarray(requested, dtype=np.int64),
                "anchors": anchors.astype(np.int64),
                "episode_ids": np.asarray(
                    [context["episode_lookup"][int(start)] for start in anchors], dtype=np.int64
                ),
                "region_ids": context["eval_regions"][rows].astype(np.int64),
            }
            for horizon in requested:
                payload[f"h{horizon}_mean_loss_matrix"] = rollout.by_horizon[
                    horizon
                ].mean_trajectory_mse
                payload[f"h{horizon}_terminal_loss_matrix"] = rollout.by_horizon[
                    horizon
                ].terminal_mse
            atomic_savez_compressed(path, **payload)
            print(
                f"[rollout] {PUBLIC_ANALYSIS_SHORT_NAME}: seed={train_seed} "
                f"support=h{support_horizon}_valid -> {path}",
                flush=True,
            )
            raw_files.append({"path": str(path), "sha256": sha256_file(path)})
    formal_audit: dict[str, Any] = {}
    if args.formal:
        if context["split_manifest"] is None:
            raise ValueError("--formal requires --split-manifest")
        formal_audit = audit_formal_posttraining(
            episode_audit=context["audit"],
            partition_contracts=context["partition_contracts"],
            split_manifest=context["split_manifest"],
            split_manifest_sha256=context["split_manifest_hash"],
            train_cache_hash=context["train_cache_hash"],
            eval_cache_hash=context["eval_cache_hash"],
            action_norm_starts_hash=context["action_norm_starts_hash"],
            cache_start_audits=context["cache_start_audits"],
            checkpoint_manifests=checkpoint_manifests,
            require_valid=True,
        )
    audit_payload = {
        **context["audit"],
        **formal_audit,
        **context["cache_start_audits"].get("train", {}),
        **context["cache_start_audits"].get("eval", {}),
        "task": args.task,
        "formal": args.formal,
        "smoke_only": args.smoke_only,
        "paper_eligible": resolve_paper_eligible(
            smoke_only=args.smoke_only,
            split_manifest=context["split_manifest"],
            max_eval_starts=args.max_eval_starts,
            max_anchors=args.max_anchors,
            max_episodes=args.max_episodes,
            formal=args.formal,
            posttraining_train_only_valid=formal_audit.get(
                "posttraining_train_only_valid", False
            ),
        ),
        "claim_scope": FORMAL_CLAIM_SCOPE if args.formal else None,
        "base_pretraining_episode_disjoint": (
            FORMAL_BASE_PRETRAINING_EPISODE_DISJOINT if args.formal else None
        ),
        "split_manifest_unsubsampled_valid": (
            context["split_manifest"] is not None
            and split_manifest_is_unsubsampled(context["split_manifest"])
        ),
        "horizon_anchor_counts": {
            str(h): int(len(a)) for h, a in context["anchors_by_horizon"].items()
        },
        "common_h10_anchor_count": int(
            len(context["anchors_by_horizon"].get(10, np.asarray([], dtype=np.int64)))
        ),
        "formal_episode_disjoint_valid": context["audit"].get("episode_disjoint", False)
        and context["audit"].get("region_start_disjoint", False),
        "auto_gate_train_only_valid": context["partition_contracts"]["auto"].get(
            "gate_partition_train_only_valid", False
        ),
        "forced_spectral_train_only_valid": context["partition_contracts"][
            "forced_spectral"
        ].get("gate_partition_train_only_valid", False),
        "global_partition_train_only_valid": context["partition_contracts"]["global"].get(
            "gate_partition_train_only_valid", False
        ),
        "train_fraction": args.train_fraction,
        "split_seed": args.split_seed,
        "split_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None,
        "split_manifest_sha256": context["split_manifest_hash"] or None,
        "nominal_train_num_episodes": len(context["nominal_train_episodes"]),
        "nominal_holdout_num_episodes": len(context["holdout_episodes"]),
        "train_cache": str(args.train_latent_cache.resolve()),
        "eval_cache": str(args.eval_latent_cache.resolve()),
        "train_cache_sha256": context["train_cache_hash"],
        "eval_cache_sha256": context["eval_cache_hash"],
        "partition_dir": str(context["partition_run"]),
        "partition_sha256": context["partition_hash"],
        "pretrained_model_sha256": context["pretrained_hash"],
        "action_norm_starts": str(context["action_norm_starts"]),
        "action_norm_starts_sha256": context["action_norm_starts_hash"],
        "git_commit": git_commit(),
        "runtime": runtime_provenance(python_executable=sys.executable),
        "command": " ".join(sys.argv),
        "frameskip": context["contract"].frameskip,
        "history_size": context["contract"].history_size,
        "route_index": context["contract"].route_index,
    }
    atomic_write_json(
        context["out_dir"] / "rollout_manifest.json",
        {
            "public_analysis_name": PUBLIC_ANALYSIS_NAME,
            "public_analysis_short_name": PUBLIC_ANALYSIS_SHORT_NAME,
            "internal_pipeline_name": "formal_region_risk",
            "audit": audit_payload,
            "num_regions": context["artifact"].num_regions,
            "horizons": context["horizons"],
            "raw_files": sorted(raw_files, key=lambda row: row["path"]),
        },
    )


def _seed_blocks_from_raw(
    raw_dir: Path, *, horizon: int, loss_kind: str
) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = sorted(raw_dir.glob(f"trainseed*_h{horizon}_valid.npz"))
    blocks: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            regions = np.asarray(data["region_ids"], dtype=np.int64)
            matrix = np.asarray(
                data[f"h{horizon}_{'mean' if loss_kind == 'mean_trajectory' else 'terminal'}_loss_matrix"],
                dtype=np.float64,
            )
            global_loss, correct, wrong_mean, wrong_best = expert_losses_from_matrix(
                matrix, regions, matrix.shape[1] - 1
            )
            blocks.append(
                {
                    "global": global_loss,
                    "correct": correct,
                    "wrong_mean": wrong_mean,
                    "wrong_best": wrong_best,
                    "episode_ids": np.asarray(data["episode_ids"], dtype=np.int64),
                }
            )
    return blocks, paths


def _bootstrap_chunk_job(job: Mapping[str, Any]) -> str:
    pair_specs = (
        ("correct_minus_global", "correct", "global"),
        ("correct_minus_wrong_mean", "correct", "wrong_mean"),
        ("correct_minus_wrong_best", "correct", "wrong_best"),
    )
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(job["bootstrap_seed"]),
                int(job["horizon"]),
                int(job["loss_kind_id"]),
                int(job["chunk_id"]),
            ]
        )
    )
    draws = nested_paired_bootstrap_draws_from_summaries(
        job["summaries"], reps=int(job["reps"]), rng=rng, pairs=pair_specs
    )
    metadata = dict(job["metadata"])
    metadata["estimates"] = job["estimates"]
    atomic_savez_compressed(
        Path(job["path"]),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **draws,
    )
    return str(job["path"])


def run_bootstrap_stage(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    raw_dir = out_dir / "raw"
    if not (out_dir / "rollout_manifest.json").exists():
        raise FileNotFoundError("rollout_manifest.json is required before bootstrap")
    if args.bootstrap_chunk_size <= 0 or args.bootstrap_workers <= 0:
        raise ValueError("bootstrap chunk size and workers must be positive")
    horizons = [int(value) for value in args.horizons.split(",") if value]
    chunk_dir = out_dir / "bootstrap_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    index_entries: list[dict[str, Any]] = []
    for horizon in horizons:
        loss_kinds = ("mean_trajectory",) if horizon == 1 else ("mean_trajectory", "terminal")
        for loss_kind in loss_kinds:
            blocks, raw_paths = _seed_blocks_from_raw(
                raw_dir, horizon=horizon, loss_kind=loss_kind
            )
            if not blocks:
                raise FileNotFoundError(f"no raw rollout blocks for horizon={horizon}")
            metric_keys = ("global", "correct", "wrong_mean", "wrong_best")
            summaries = precompute_episode_summaries(blocks, metric_keys=metric_keys)
            estimates = {
                key: float(
                    np.mean(
                        [np.mean(np.asarray(block[key], dtype=np.float64)) for block in blocks]
                    )
                )
                for key in metric_keys
            }
            raw_hashes = {str(path): sha256_file(path) for path in raw_paths}
            chunks: list[str] = []
            for chunk_id, begin in enumerate(
                range(0, args.bootstrap_reps, args.bootstrap_chunk_size)
            ):
                reps = min(args.bootstrap_chunk_size, args.bootstrap_reps - begin)
                loss_kind_id = 0 if loss_kind == "mean_trajectory" else 1
                metadata = {
                    "schema_version": 1,
                    "task": args.task,
                    "horizon": horizon,
                    "loss_kind": loss_kind,
                    "loss_kind_id": loss_kind_id,
                    "chunk_id": chunk_id,
                    "replicate_begin": begin,
                    "reps": reps,
                    "bootstrap_seed": args.bootstrap_seed,
                    "raw_sha256": raw_hashes,
                }
                metadata["fingerprint"] = stable_json_sha256(metadata)
                path = chunk_dir / f"h{horizon}_{loss_kind}_chunk{chunk_id:05d}.npz"
                chunks.append(str(path))
                if _resume_matches(
                    path,
                    metadata["fingerprint"],
                    resume=args.resume,
                    kind="bootstrap chunk",
                ):
                    continue
                jobs.append(
                    {
                        "path": str(path),
                        "summaries": summaries,
                        "estimates": estimates,
                        "reps": reps,
                        "bootstrap_seed": args.bootstrap_seed,
                        "horizon": horizon,
                        "loss_kind_id": loss_kind_id,
                        "chunk_id": chunk_id,
                        "metadata": metadata,
                    }
                )
            index_entries.append(
                {"horizon": horizon, "loss_kind": loss_kind, "chunks": chunks}
            )
    if args.bootstrap_workers == 1:
        for job in jobs:
            print(f"[bootstrap] {PUBLIC_ANALYSIS_SHORT_NAME}: {job['path']}", flush=True)
            _bootstrap_chunk_job(job)
    elif jobs:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.bootstrap_workers
        ) as executor:
            futures = [executor.submit(_bootstrap_chunk_job, job) for job in jobs]
            for future in futures:
                print(
                    f"[bootstrap] {PUBLIC_ANALYSIS_SHORT_NAME}: {future.result()}",
                    flush=True,
                )
    atomic_write_json(
        out_dir / "bootstrap_index.json",
        {
            "public_analysis_short_name": PUBLIC_ANALYSIS_SHORT_NAME,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_chunk_size": args.bootstrap_chunk_size,
            "entries": index_entries,
        },
    )


def _bootstrap_rows_from_chunks(out_dir: Path) -> list[dict[str, Any]]:
    index = json.loads((out_dir / "bootstrap_index.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in index["entries"]:
        draws_by_metric: dict[str, list[np.ndarray]] = {}
        estimates: dict[str, float] | None = None
        for chunk_name in entry["chunks"]:
            path = Path(chunk_name)
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                estimates = metadata["estimates"]
                for key in data.files:
                    if key != "metadata_json":
                        draws_by_metric.setdefault(key, []).append(
                            np.asarray(data[key], dtype=np.float64)
                        )
        if estimates is None:
            raise RuntimeError(f"no bootstrap chunks for {entry}")
        for label, left, right in (
            ("correct_minus_global", "correct", "global"),
            ("correct_minus_wrong_mean", "correct", "wrong_mean"),
            ("correct_minus_wrong_best", "correct", "wrong_best"),
        ):
            delta = np.concatenate(draws_by_metric[label])
            row = {
                "metric": label,
                "loss_kind": entry["loss_kind"],
                "estimate": float(estimates[left] - estimates[right]),
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
            }
            rows.append(row)
            if int(entry["horizon"]) == 1 and entry["loss_kind"] == "mean_trajectory":
                rows.append({**row, "loss_kind": "terminal"})
            rows[-1]["horizon"] = int(entry["horizon"])
            if len(rows) >= 2 and int(entry["horizon"]) == 1:
                rows[-2]["horizon"] = int(entry["horizon"])
    return rows


def run_finalize_stage(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    rollout_manifest = json.loads(
        (out_dir / "rollout_manifest.json").read_text(encoding="utf-8")
    )
    bootstrap_rows = _bootstrap_rows_from_chunks(out_dir)
    num_regions = int(rollout_manifest["num_regions"])
    artifact_view = SimpleNamespace(num_regions=num_regions)
    sample_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    weighted_rows: list[dict[str, Any]] = []
    common_region_rows: list[dict[str, Any]] = []
    common_weighted_rows: list[dict[str, Any]] = []
    raw_paths = sorted((out_dir / "raw").glob("trainseed*_h*_valid.npz"))
    for path in raw_paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            train_seed = int(metadata["train_seed"])
            support_horizon = int(metadata["support_horizon"])
            anchors = np.asarray(data["anchors"], dtype=np.int64)
            regions = np.asarray(data["region_ids"], dtype=np.int64)
            episodes = np.asarray(data["episode_ids"], dtype=np.int64)
            append_horizon_metrics(
                task=args.task,
                train_seed=train_seed,
                horizon=support_horizon,
                anchors=anchors,
                anchor_regions=regions,
                anchor_episodes=episodes,
                mean_traj_matrix=np.asarray(
                    data[f"h{support_horizon}_mean_loss_matrix"], dtype=np.float64
                ),
                terminal_matrix=np.asarray(
                    data[f"h{support_horizon}_terminal_loss_matrix"], dtype=np.float64
                ),
                artifact=artifact_view,
                anchor_support="horizon_valid",
                sample_records=sample_records,
                episode_records=episode_records,
                region_summary_rows=region_rows,
                weighted_rows=weighted_rows,
                seed_blocks=[],
            )
            if support_horizon == 10:
                for horizon in metadata["requested_horizons"]:
                    append_horizon_metrics(
                        task=args.task,
                        train_seed=train_seed,
                        horizon=int(horizon),
                        anchors=anchors,
                        anchor_regions=regions,
                        anchor_episodes=episodes,
                        mean_traj_matrix=np.asarray(
                            data[f"h{horizon}_mean_loss_matrix"], dtype=np.float64
                        ),
                        terminal_matrix=np.asarray(
                            data[f"h{horizon}_terminal_loss_matrix"], dtype=np.float64
                        ),
                        artifact=artifact_view,
                        anchor_support="common_h10",
                        sample_records=sample_records,
                        episode_records=episode_records,
                        region_summary_rows=common_region_rows,
                        weighted_rows=common_weighted_rows,
                        seed_blocks=[],
                    )
    audit_payload = rollout_manifest["audit"]
    atomic_write_json(out_dir / "audit.json", audit_payload)
    if args.gate_summary is not None and args.gate_summary.exists():
        atomic_write_json(
            out_dir / "gate_summary.json",
            json.loads(args.gate_summary.read_text(encoding="utf-8")),
        )
    atomic_savez_compressed(
        out_dir / "sample_metrics.npz",
        sample_id=np.asarray([row["sample_id"] for row in sample_records], dtype=np.int64),
        episode_id=np.asarray([row["episode_id"] for row in sample_records], dtype=np.int64),
        region=np.asarray([row["region"] for row in sample_records], dtype=np.int64),
        train_seed=np.asarray([row["train_seed"] for row in sample_records], dtype=np.int64),
        horizon=np.asarray([row["horizon"] for row in sample_records], dtype=np.int64),
        anchor_support=np.asarray(
            [row["anchor_support"] for row in sample_records], dtype="<U16"
        ),
        global_loss=np.asarray([row["global_loss"] for row in sample_records], dtype=np.float64),
        correct_loss=np.asarray([row["correct_loss"] for row in sample_records], dtype=np.float64),
        wrong_mean_loss=np.asarray(
            [row["wrong_mean_loss"] for row in sample_records], dtype=np.float64
        ),
        wrong_best_loss=np.asarray(
            [row["wrong_best_loss"] for row in sample_records], dtype=np.float64
        ),
        terminal_global_loss=np.asarray(
            [row["terminal_global_loss"] for row in sample_records], dtype=np.float64
        ),
        terminal_correct_loss=np.asarray(
            [row["terminal_correct_loss"] for row in sample_records], dtype=np.float64
        ),
        terminal_wrong_mean_loss=np.asarray(
            [row["terminal_wrong_mean_loss"] for row in sample_records], dtype=np.float64
        ),
        terminal_wrong_best_loss=np.asarray(
            [row["terminal_wrong_best_loss"] for row in sample_records], dtype=np.float64
        ),
    )
    write_csv(out_dir / "episode_metrics.csv", episode_records)
    write_csv(out_dir / "region_summary.csv", region_rows)
    write_csv(out_dir / "weighted_summary.csv", weighted_rows)
    write_csv(out_dir / "common_h10_support_summary.csv", common_weighted_rows)
    write_csv(out_dir / "bootstrap_summary.csv", bootstrap_rows)
    plot_region_risk(
        region_rows,
        bootstrap_rows,
        out_dir / "region_risk.pdf",
        horizons=list(map(int, rollout_manifest["horizons"])),
    )
    file_names = (
        "audit.json",
        "sample_metrics.npz",
        "episode_metrics.csv",
        "region_summary.csv",
        "weighted_summary.csv",
        "common_h10_support_summary.csv",
        "bootstrap_summary.csv",
        "region_risk.pdf",
        "region_risk_forest.pdf",
    )
    manifest = {
        "task": args.task,
        "public_analysis_name": PUBLIC_ANALYSIS_NAME,
        "public_analysis_short_name": PUBLIC_ANALYSIS_SHORT_NAME,
        "internal_pipeline_name": "formal_region_risk",
        "audit": audit_payload,
        "files": {
            name: sha256_file(out_dir / name)
            for name in file_names
            if (out_dir / name).exists()
        },
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    print(f"[done] {PUBLIC_ANALYSIS_SHORT_NAME} -> {out_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.formal and args.allow_in_cache:
        raise ValueError("--formal cannot be combined with --allow-in-cache")
    stages = ("rollout", "bootstrap", "finalize") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"[stage] {PUBLIC_ANALYSIS_SHORT_NAME}: {stage}", flush=True)
        if stage == "rollout":
            run_rollout_stage(args)
        elif stage == "bootstrap":
            run_bootstrap_stage(args)
        elif stage == "finalize":
            run_finalize_stage(args)


if __name__ == "__main__":
    main()
