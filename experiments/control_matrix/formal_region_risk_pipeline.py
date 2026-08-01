#!/usr/bin/env python3
"""Strict 90/10 formal region-conditioned prediction risk experiment pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "tworoom"))

from backends.lewm.encoding import LeWMEncoderAdapter, make_hdf5_transition_dataset  # noqa: E402
from experiments.control_matrix.episode_split import (  # noqa: E402
    FORMAL_BOOTSTRAP_SEED,
    FORMAL_SPLIT_SEED,
    FORMAL_TRAIN_FRACTION,
    compute_episode_split,
    load_split_manifest,
    split_manifest_is_unsubsampled,
    split_paths_from_manifest,
    write_split_artifacts,
)
from experiments.control_matrix.region_risk_lib import atomic_write_json, sha256_file  # noqa: E402
from lap.encoding.fast import FastEncodingConfig, FastLatentCacheEncoder  # noqa: E402


DEFAULT_TASKS: dict[str, dict[str, str]] = {
    "pusht": {
        "dataset_name": "pusht",
        "data_file": "/data/sicong/weitao/datasets/lewm/pusht_expert_train.h5",
        "pretrained_model": "/data/sicong/weitao/.stable_worldmodel/pusht/lewm_object.ckpt",
    },
    "tworoom": {
        "dataset_name": "tworoom",
        "data_file": "/data/sicong/weitao/datasets/lewm/tworoom.h5",
        "pretrained_model": "/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt",
    },
}


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    split_manifest: Path
    train_cache: Path
    eval_cache: Path
    partition_auto: Path
    partition_global: Path
    partition_forced_spectral: Path
    training_global: Path
    training_forced_spectral: Path
    evaluation: Path
    gate_summary: Path

    @classmethod
    def from_root(cls, root: Path) -> "PipelinePaths":
        root = root.resolve()
        return cls(
            root=root,
            split_manifest=root / "split_manifest.json",
            train_cache=root / "train_latent_cache.npz",
            eval_cache=root / "eval_latent_cache.npz",
            partition_auto=root / "partition" / "auto",
            partition_global=root / "partition" / "global",
            partition_forced_spectral=root / "partition" / "forced_spectral_negative_control",
            training_global=root / "training" / "global",
            training_forced_spectral=root / "training" / "forced_spectral_negative_control",
            evaluation=root / "evaluation",
            gate_summary=root / "gate_summary.json",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(DEFAULT_TASKS))
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--pretrained-model", type=Path, default=None)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("split", "encode", "partition", "train", "evaluate", "smoke", "dry-run", "all"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-batch-size", type=int, default=128)
    parser.add_argument("--train-seeds", default="0,42,625")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bootstrap-reps", type=int, default=50000)
    parser.add_argument("--max-train-starts", type=int, default=0)
    parser.add_argument("--max-eval-starts", type=int, default=0)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def task_config(args: argparse.Namespace) -> dict[str, str]:
    base = dict(DEFAULT_TASKS[args.task])
    if args.data_file is not None:
        base["data_file"] = str(args.data_file.resolve())
    if args.pretrained_model is not None:
        base["pretrained_model"] = str(args.pretrained_model.resolve())
    return base


def run_command(command: Sequence[str], *, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"[cmd] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def encode_latent_cache(
    *,
    data_file: Path,
    starts: np.ndarray,
    action_norm_starts: Path,
    pretrained_model: Path,
    output: Path,
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
        history_size=3,
        num_preds=1,
        frameskip=0,
    )
    encoder = LeWMEncoderAdapter(img_size=224, frameskip=0, checkpoint_cache_dir=None)
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


def fit_partition_command(
    args: argparse.Namespace,
    cfg: dict[str, str],
    paths: PipelinePaths,
    *,
    method: str,
    out_dir: Path,
    seed: int = 0,
) -> list[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "experiments/control_matrix/fit_partition.py"),
        "--method",
        method,
        "--latent-cache",
        str(paths.train_cache),
        "--dataset-name",
        cfg["dataset_name"],
        "--data-file",
        cfg["data_file"],
        "--frameskip",
        "5",
        "--num-clusters",
        "3",
        "--seed",
        str(seed),
        "--num-landmarks",
        "20000",
        "--knn",
        "30",
        "--perturb-knn",
        "27,33",
        "--diagnostic-seeds",
        "0,1,2",
        "--deployment-seed",
        "0",
        "--gate-retention-threshold",
        "0.5",
        "--gate-background-gap-count",
        "10",
        "--device",
        args.device,
        "--out-dir",
        str(out_dir),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def phase_split(args: argparse.Namespace, paths: PipelinePaths, cfg: dict[str, str]) -> None:
    split = compute_episode_split(
        Path(cfg["data_file"]),
        cfg["dataset_name"],
        split_seed=FORMAL_SPLIT_SEED,
        train_fraction=FORMAL_TRAIN_FRACTION,
    )
    write_split_artifacts(
        split,
        paths.root,
        max_train_starts=args.max_train_starts,
        max_eval_starts=args.max_eval_starts,
    )
    print(
        f"[split] train_episodes={len(split.train_episode_ids)} "
        f"eval_episodes={len(split.eval_episode_ids)} -> {paths.root}",
        flush=True,
    )


def phase_encode(args: argparse.Namespace, paths: PipelinePaths, cfg: dict[str, str]) -> None:
    manifest = load_split_manifest(paths.split_manifest)
    split_paths = split_paths_from_manifest(manifest)
    train_starts = np.load(split_paths["train_starts"])
    eval_starts = np.load(split_paths["eval_starts"])
    if args.max_train_starts > 0:
        train_starts = train_starts[: args.max_train_starts]
    if args.max_eval_starts > 0:
        eval_starts = eval_starts[: args.max_eval_starts]
    encode_latent_cache(
        data_file=Path(cfg["data_file"]),
        starts=train_starts,
        action_norm_starts=split_paths["action_norm_starts"],
        pretrained_model=Path(cfg["pretrained_model"]),
        output=paths.train_cache,
        device=args.device,
        batch_size=args.encoding_batch_size,
    )
    encode_latent_cache(
        data_file=Path(cfg["data_file"]),
        starts=eval_starts,
        action_norm_starts=split_paths["action_norm_starts"],
        pretrained_model=Path(cfg["pretrained_model"]),
        output=paths.eval_cache,
        device=args.device,
        batch_size=args.encoding_batch_size,
    )
    atomic_write_json(
        paths.root / "encoding_manifest.json",
        {
            "train_cache": str(paths.train_cache),
            "eval_cache": str(paths.eval_cache),
            "action_norm_starts": str(split_paths["action_norm_starts"]),
            "sha256": {
                "train_cache": sha256_file(paths.train_cache),
                "eval_cache": sha256_file(paths.eval_cache),
                "action_norm_starts": sha256_file(split_paths["action_norm_starts"]),
            },
        },
    )


def phase_partition(
    args: argparse.Namespace,
    paths: PipelinePaths,
    cfg: dict[str, str],
    *,
    dry_run: bool,
) -> None:
    for method, out_dir, seed in (
        ("auto", paths.partition_auto, 0),
        ("global", paths.partition_global, 0),
        ("spectral", paths.partition_forced_spectral, 0),
    ):
        if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
            print(f"[skip] partition exists: {out_dir}", flush=True)
            continue
        run_command(
            fit_partition_command(args, cfg, paths, method=method, out_dir=out_dir, seed=seed),
            dry_run=dry_run,
        )
    if dry_run:
        return
    auto_manifest = json.loads((paths.partition_auto / "manifest.json").read_text(encoding="utf-8"))
    gate = auto_manifest.get("method_metadata", {}).get("automatic_gate", {})
    atomic_write_json(
        paths.gate_summary,
        {
            "task": args.task,
            "partition_manifest": str(paths.partition_auto / "manifest.json"),
            "selected_method": auto_manifest.get("selected_method"),
            "automatic_gate": gate,
            "forced_spectral_negative_control_partition": str(paths.partition_forced_spectral),
            "train_only_cache_sha256": sha256_file(paths.train_cache),
            "partition_latent_cache_sha256": auto_manifest.get("latent_cache_sha256"),
        },
    )


def train_command(
    args: argparse.Namespace,
    cfg: dict[str, str],
    paths: PipelinePaths,
    *,
    partition_dir: Path,
    out_dir: Path,
    train_seed: int,
    training_role: str,
) -> list[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "experiments/control_matrix/train_predictors.py"),
        "--dataset-name",
        cfg["dataset_name"],
        "--latent-cache",
        str(paths.train_cache),
        "--pretrained-model",
        cfg["pretrained_model"],
        "--partition-dir",
        str(partition_dir),
        "--out-dir",
        str(out_dir),
        "--train-seed",
        str(train_seed),
        "--epochs",
        str(args.epochs),
        "--split-manifest",
        str(paths.split_manifest),
        "--training-role",
        training_role,
        "--device",
        args.device,
    ]
    return command


def phase_train(
    args: argparse.Namespace,
    paths: PipelinePaths,
    cfg: dict[str, str],
    *,
    dry_run: bool,
) -> None:
    seeds = [int(value) for value in args.train_seeds.split(",") if value]
    for train_seed in seeds:
        global_out = paths.training_global / f"train{train_seed}"
        regional_out = paths.training_forced_spectral / f"train{train_seed}"
        if (global_out / "manifest.json").exists() and not args.overwrite:
            print(f"[skip] global training exists: {global_out}", flush=True)
        else:
            run_command(
                train_command(
                    args,
                    cfg,
                    paths,
                    partition_dir=paths.partition_global,
                    out_dir=global_out,
                    train_seed=train_seed,
                    training_role="global",
                ),
                dry_run=dry_run,
            )
        if (regional_out / "manifest.json").exists() and not args.overwrite:
            print(f"[skip] regional training exists: {regional_out}", flush=True)
        else:
            run_command(
                train_command(
                    args,
                    cfg,
                    paths,
                    partition_dir=paths.partition_forced_spectral,
                    out_dir=regional_out,
                    train_seed=train_seed,
                    training_role="forced_spectral_negative_control",
                ),
                dry_run=dry_run,
            )


def is_full_formal_run(
    args: argparse.Namespace,
    *,
    split_manifest: Mapping[str, Any] | None = None,
) -> bool:
    cli_untruncated = (
        args.phase in {"all", "evaluate"}
        and args.max_train_starts <= 0
        and args.max_eval_starts <= 0
        and args.max_anchors <= 0
        and args.max_episodes <= 0
    )
    if not cli_untruncated:
        return False
    if split_manifest is None:
        return args.phase == "all"
    return split_manifest_is_unsubsampled(split_manifest)


def evaluate_command(
    args: argparse.Namespace,
    cfg: dict[str, str],
    paths: PipelinePaths,
) -> list[str]:
    split_manifest = load_split_manifest(paths.split_manifest)
    split_paths = split_paths_from_manifest(split_manifest)
    command = [
        args.python,
        str(PROJECT_ROOT / "experiments/control_matrix/evaluate_region_conditional_risk.py"),
        "--task",
        args.task,
        "--data-file",
        cfg["data_file"],
        "--train-latent-cache",
        str(paths.train_cache),
        "--eval-latent-cache",
        str(paths.eval_cache),
        "--partition-dir",
        str(paths.partition_forced_spectral),
        "--audit-partition-dir",
        str(paths.partition_auto),
        "--forced-spectral-partition-dir",
        str(paths.partition_forced_spectral),
        "--global-partition-dir",
        str(paths.partition_global),
        "--regional-runs",
        str(paths.training_forced_spectral),
        "--global-runs",
        str(paths.training_global),
        "--pretrained-model",
        cfg["pretrained_model"],
        "--split-manifest",
        str(paths.split_manifest),
        "--action-norm-starts",
        str(split_paths["action_norm_starts"]),
        "--gate-summary",
        str(paths.gate_summary),
        "--formal",
        "--train-seeds",
        args.train_seeds,
        "--horizons",
        "1,5,10",
        "--split-seed",
        str(FORMAL_SPLIT_SEED),
        "--train-fraction",
        str(FORMAL_TRAIN_FRACTION),
        "--bootstrap-reps",
        str(args.bootstrap_reps),
        "--bootstrap-seed",
        str(FORMAL_BOOTSTRAP_SEED),
        "--encoding-batch-size",
        str(args.encoding_batch_size),
        "--device",
        args.device,
        "--out-dir",
        str(paths.evaluation),
    ]
    if args.max_anchors > 0:
        command.extend(["--max-anchors", str(args.max_anchors)])
    if args.max_episodes > 0:
        command.extend(["--max-episodes", str(args.max_episodes)])
    if args.phase == "smoke":
        command.extend(["--smoke-only"])
    elif is_full_formal_run(args, split_manifest=split_manifest):
        command.extend(["--paper-eligible"])
    return command


def phase_evaluate(
    args: argparse.Namespace,
    paths: PipelinePaths,
    cfg: dict[str, str],
    *,
    dry_run: bool,
) -> None:
    run_command(evaluate_command(args, cfg, paths), dry_run=dry_run)


def main() -> None:
    args = parse_args()
    cfg = task_config(args)
    paths = PipelinePaths.from_root(args.work_root)
    paths.root.mkdir(parents=True, exist_ok=True)
    dry_run = args.phase == "dry-run"

    if args.phase in {"split", "smoke", "all"}:
        phase_split(args, paths, cfg)
    if args.phase in {"encode", "smoke", "all"}:
        phase_encode(args, paths, cfg)
    if args.phase in {"partition", "smoke", "all"}:
        phase_partition(args, paths, cfg, dry_run=dry_run)
    if args.phase in {"train", "smoke", "all"}:
        phase_train(args, paths, cfg, dry_run=dry_run)
    if args.phase in {"evaluate", "smoke", "all"}:
        phase_evaluate(args, paths, cfg, dry_run=dry_run)
    if args.phase == "dry-run":
        phase_partition(args, paths, cfg, dry_run=True)
        phase_train(args, paths, cfg, dry_run=True)
        phase_evaluate(args, paths, cfg, dry_run=True)
    print(f"[done] phase={args.phase} work_root={paths.root}", flush=True)


if __name__ == "__main__":
    main()
