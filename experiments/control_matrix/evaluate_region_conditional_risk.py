#!/usr/bin/env python3
"""Evaluate region-conditioned prediction risk on episode-disjoint held-out caches."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

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
    split_paths_from_manifest,
)
from experiments.control_matrix.region_risk_lib import (  # noqa: E402
    ExpertBundle,
    aggregate_region_metrics,
    atomic_write_json,
    audit_cache_starts_exact,
    audit_episode_disjointness,
    audit_formal_posttraining,
    audit_partition_train_contract,
    collect_rollout_anchors,
    episode_ids_at_starts,
    git_commit,
    load_cache_contract,
    load_lewm_cache,
    load_model,
    multi_horizon_open_loop_rollout_losses,
    nested_paired_bootstrap_ci,
    one_step_losses,
    resolve_action_norm_starts,
    route_regions_from_cache,
    sha256_file,
    start_index_map,
    validate_manifest,
    weighted_summary,
    wrong_expert_losses,
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
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Formal split manifest produced by formal_region_risk_pipeline.py.",
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Require held-out post-training audits; incompatible with --allow-in-cache.",
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
    fig.savefig(out_path, bbox_inches="tight")
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
    ax.set_title("Paired bootstrap")
    fig.tight_layout()
    fig.savefig(forest_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
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
        "paper_eligible": (
            not args.smoke_only
            and args.paper_eligible
            and (not args.formal or formal_audit.get("posttraining_train_only_valid", False))
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
    print(f"[done] region risk analysis -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
