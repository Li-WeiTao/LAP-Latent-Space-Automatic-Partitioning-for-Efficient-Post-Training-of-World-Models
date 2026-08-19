#!/usr/bin/env python3
"""Unified LAP efficiency benchmark entrypoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from efficiency_lib.config import ANCHOR_TRAINING, inference_tasks
from efficiency_lib.gate_partition import measure_gate_and_partition
from efficiency_lib.inference import benchmark_inference_task
from efficiency_lib.metadata import collect_metadata, sha256_file, write_metadata
from efficiency_lib.report import build_reports
from efficiency_lib.training import benchmark_joint_training, benchmark_lap_regional_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--anchor-model", default="lewm")
    parser.add_argument("--anchor-task", default="tworoom")
    parser.add_argument("--inference-tasks", default="tworoom,pusht,reacher,cube")
    parser.add_argument(
        "--measure",
        default="train,gate,partition,inference",
        help="Comma-separated phases: train,gate,partition,inference",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--joint-epochs", type=int, default=5)
    parser.add_argument("--lap-epochs", type=int, default=5)
    parser.add_argument("--discard-warmup-epochs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/efficiency_results"))
    parser.add_argument("--skip-gate-rerun", action="store_true")
    parser.add_argument(
        "--training-methods",
        default="joint,lap",
        help="Comma-separated training benchmarks: joint,lap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    scratch = output_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    measures = {part.strip() for part in args.measure.split(",") if part.strip()}
    timing_epochs = max(args.joint_epochs, args.lap_epochs)
    train_cfg = replace(
        ANCHOR_TRAINING,
        timing_epochs=timing_epochs,
        discard_warmup_epochs=args.discard_warmup_epochs,
        seed=args.seed,
    )

    metadata = collect_metadata(repo_root, seed=args.seed, device=args.device)
    metadata["dataset"] = {
        "task": train_cfg.task,
        "dataset_file": str(train_cfg.dataset_file),
        "dataset_sha256": sha256_file(train_cfg.dataset_file),
        "checkpoint": str(train_cfg.checkpoint),
        "checkpoint_sha256": sha256_file(train_cfg.checkpoint),
        "latent_cache": str(train_cfg.latent_cache),
        "training_latent_cache": str(train_cfg.training_latent_cache),
    }
    metadata["benchmark"] = {
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_epochs": timing_epochs,
        "discard_warmup_epochs": args.discard_warmup_epochs,
        "measures": sorted(measures),
    }

    joint_result = None
    lap_result = None
    gate_result = None
    inference_results: list[dict] = []

    if "train" in measures:
        train_methods = {part.strip() for part in args.training_methods.split(",") if part.strip()}
        if "joint" in train_methods:
            print("[train] Joint training benchmark...", flush=True)
            joint_result = benchmark_joint_training(
                repo_root, train_cfg, device=args.device, scratch_dir=scratch / "training"
            )
        if "lap" in train_methods:
            print("[train] LAP Regional-FT benchmark...", flush=True)
            lap_result = benchmark_lap_regional_training(
                repo_root, train_cfg, device=args.device, scratch_dir=scratch / "training"
            )

    if "gate" in measures or "partition" in measures:
        print("[gate/partition] Measuring one-time LAP costs...", flush=True)
        gate_result = measure_gate_and_partition(
            repo_root,
            train_cfg,
            scratch_dir=scratch / "gate_partition",
            rerun=not args.skip_gate_rerun,
        )

    if "inference" in measures:
        task_map = inference_tasks(repo_root)
        for task_name in [t.strip() for t in args.inference_tasks.split(",") if t.strip()]:
            task_cfg = task_map[task_name]
            print(f"[inference] {task_name} baseline...", flush=True)
            inference_results.append(
                benchmark_inference_task(
                    repo_root,
                    task_cfg,
                    mode="baseline",
                    device=args.device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    scratch_dir=scratch / "inference",
                )
            )
            print(f"[inference] {task_name} lap...", flush=True)
            inference_results.append(
                benchmark_inference_task(
                    repo_root,
                    task_cfg,
                    mode="lap",
                    device=args.device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    scratch_dir=scratch / "inference",
                )
            )

    payload = {
        "joint_training": joint_result,
        "lap_regional_training": lap_result,
        "gate_partition": gate_result,
        "inference": inference_results,
    }
    (output_dir / "efficiency_payload.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    build_reports(
        output_dir=output_dir,
        joint=joint_result,
        lap=lap_result,
        gate_partition=gate_result,
        inference=inference_results,
    )
    write_metadata(output_dir / "metadata.json", metadata)
    print(f"[done] wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
