#!/usr/bin/env python3
"""Compare Official vs Global-FT held-out latent prediction loss."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "tworoom"))

from backends.lewm.cache import LeWMLatentCache  # noqa: E402
from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint  # noqa: E402
from backends.lewm.encoding import LeWMEncoderAdapter, make_hdf5_transition_dataset  # noqa: E402
from experiments.control_matrix.evaluate_region_conditional_risk import (  # noqa: E402
    collect_limited_horizon_anchors,
    horizon_rollout_losses,
)
from experiments.control_matrix.region_risk_lib import (  # noqa: E402
    audit_episode_disjointness,
    episode_ids_at_starts,
    load_cache_contract,
    load_lewm_cache,
    start_index_map,
)
from experiments.tworoom.gauge_drift import DATASETS, choose_state_key  # noqa: E402
from experiments.tworoom.predictor_rule_drift import valid_transition_starts  # noqa: E402
from experiments.tworoom.trajectory import TrainConfig, train_global_pool  # noqa: E402
from lap.encoding.fast import FastEncodingConfig, FastLatentCacheEncoder  # noqa: E402


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def holdout_starts(
    data_file: Path,
    *,
    dataset_name: str,
    split_seed: int,
    train_fraction: float,
) -> np.ndarray:
    spec = DATASETS[dataset_name]
    cfg = TrainConfig(
        history_size=3,
        num_preds=1,
        frameskip=5,
        train_fraction=train_fraction,
        split_seed=split_seed,
    )
    with h5py.File(data_file, "r") as handle:
        state_key = choose_state_key(handle, spec, None)
        all_starts = valid_transition_starts(
            handle, spec, state_key, cfg.history_size + cfg.num_preds, cfg.frameskip, 0, 0
        )
    train_starts = train_global_pool(all_starts, cfg)
    return np.sort(np.setdiff1d(all_starts, train_starts).astype(np.int64))


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
    model_family: str,
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
        frameskip=frameskip,
    )
    encoder = LeWMEncoderAdapter(
        img_size=224,
        frameskip=frameskip,
        model_family=model_family,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("/data/sicong/weitao/datasets/lewm/reacher.h5"),
    )
    parser.add_argument("--dataset-name", default="reacher", choices=sorted(DATASETS))
    parser.add_argument(
        "--pretrained-model",
        type=Path,
        default=Path("/data/sicong/weitao/.stable_worldmodel/reacher/lewm_object.ckpt"),
    )
    parser.add_argument(
        "--train-starts",
        type=Path,
        default=PROJECT_ROOT / "experiments/reacher/matrix/preparation/train_global_reference_starts.npy",
    )
    parser.add_argument(
        "--global-ft-ckpt",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/reacher/matrix/training/global/train42/P_train_cluster0_object.ckpt",
    )
    parser.add_argument(
        "--eval-cache",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/reacher/matrix/preparation/reacher_heldout_eval_latent_cache.npz",
    )
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--model-family", default="lewm")
    parser.add_argument("--horizons", default="1,5,10")
    parser.add_argument("--encoding-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--build-eval-cache", action="store_true")
    parser.add_argument("--max-eval-starts", type=int, default=0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/reacher/matrix/logs/heldout_prediction_loss_official_vs_global_ft50.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    horizons = [int(value) for value in args.horizons.split(",") if value]

    holdout = holdout_starts(
        args.data_file.resolve(),
        dataset_name=args.dataset_name,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
    )
    train_starts = np.load(args.train_starts)
    overlap = np.intersect1d(train_starts, holdout)
    if len(overlap):
        raise RuntimeError(f"train/holdout overlap: {len(overlap)}")

    if args.max_eval_starts > 0 and len(holdout) > args.max_eval_starts:
        holdout = holdout[: args.max_eval_starts]

    eval_cache_path = args.eval_cache.resolve()
    if args.build_eval_cache or not eval_cache_path.exists():
        print(f"[encode] {len(holdout)} held-out transitions -> {eval_cache_path}", flush=True)
        encode_eval_cache(
            data_file=args.data_file.resolve(),
            starts=holdout,
            action_norm_starts=args.train_starts.resolve(),
            pretrained_model=args.pretrained_model.resolve(),
            output=eval_cache_path,
            history_size=args.history_size,
            num_preds=args.num_preds,
            frameskip=args.frameskip,
            model_family=args.model_family,
            device=str(device),
            batch_size=args.encoding_batch_size,
        )

    eval_cache = load_lewm_cache(eval_cache_path)
    contract = load_cache_contract(
        eval_cache,
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
    )
    eval_start_map = start_index_map(eval_cache.sample_ids)
    eval_episode_ids = episode_ids_at_starts(args.data_file.resolve(), eval_cache.sample_ids)
    episode_lookup = {
        int(start): int(episode)
        for start, episode in zip(eval_cache.sample_ids, eval_episode_ids)
    }

    audit = audit_episode_disjointness(
        data_file=args.data_file.resolve(),
        train_starts=train_starts,
        eval_starts=eval_cache.sample_ids,
        require_disjoint=False,
    )

    official = load_jepa_object_checkpoint(
        args.pretrained_model.resolve(),
        model_family=args.model_family,
        map_location=device,
    ).to(device)
    global_ft = load_jepa_object_checkpoint(
        args.global_ft_ckpt.resolve(),
        model_family=args.model_family,
        map_location=device,
    ).to(device)
    models = [official, global_ft]
    model_names = ["official", "global_ft50"]

    anchors_by_horizon = {
        horizon: collect_limited_horizon_anchors(
            eval_cache.sample_ids,
            horizon=horizon,
            contract=contract,
            eval_start_map=eval_start_map,
            episode_lookup=episode_lookup,
            max_anchors=0,
            max_episodes=0,
        )
        for horizon in horizons
    }

    results: dict[str, object] = {
        "protocol": {
            "split": "transition_level_90_10",
            "dataset_name": args.dataset_name,
            "split_seed": args.split_seed,
            "train_fraction": args.train_fraction,
            "num_holdout_transitions": int(len(eval_cache.sample_ids)),
            "train_starts_path": str(args.train_starts.resolve()),
            "eval_cache": str(eval_cache_path),
            "official_ckpt": str(args.pretrained_model.resolve()),
            "global_ft_ckpt": str(args.global_ft_ckpt.resolve()),
            "model_family": args.model_family,
            "frameskip": args.frameskip,
            "history_size": args.history_size,
            "audit_episode_disjointness": audit,
            "note": (
                "Same 90/10 transition split as train_global_reference_starts; "
                "not episode-disjoint (matches matrix training contract)."
            ),
        },
        "horizons": {},
    }

    for horizon in horizons:
        anchors = anchors_by_horizon[horizon]
        mean_traj, terminal = horizon_rollout_losses(
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
        official_mean = float(mean_traj[:, 0].mean())
        ft_mean = float(mean_traj[:, 1].mean())
        official_terminal = float(terminal[:, 0].mean())
        ft_terminal = float(terminal[:, 1].mean())
        results["horizons"][str(horizon)] = {
            "num_anchors": int(len(anchors)),
            "mean_trajectory_mse": {
                "official": official_mean,
                "global_ft50": ft_mean,
                "delta_ft_minus_official": ft_mean - official_mean,
                "ratio_ft_over_official": ft_mean / official_mean if official_mean else None,
            },
            "terminal_step_mse": {
                "official": official_terminal,
                "global_ft50": ft_terminal,
                "delta_ft_minus_official": ft_terminal - official_terminal,
                "ratio_ft_over_official": ft_terminal / official_terminal
                if official_terminal
                else None,
            },
        }
        print(
            f"[H={horizon}] anchors={len(anchors)} "
            f"official={official_mean:.6f} global_ft50={ft_mean:.6f} "
            f"delta={ft_mean - official_mean:+.6f}",
            flush=True,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
