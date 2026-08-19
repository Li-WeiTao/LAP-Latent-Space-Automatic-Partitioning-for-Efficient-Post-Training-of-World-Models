#!/usr/bin/env python3
"""CPU-only provenance checks before formal GPU efficiency benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/efficiency_results/scratch/dry_run/dry_run_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    scripts = repo_root / "experiments" / "scripts"
    sys.path.insert(0, str(scripts))

    from efficiency_lib.config import ANCHOR_TRAINING, inference_tasks
    from efficiency_lib.validation import (
        materialize_tworoom_lap_run_dir,
        validate_joint_train_pool_dataset,
        validate_lap_predictor_manifest,
        validate_task_checkpoint,
        validate_training_latent_cache,
        validate_tworoom_predictor_partition_provenance,
    )

    cfg = ANCHOR_TRAINING
    scratch = (repo_root / args.output).parent
    scratch.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def record(name: str, fn) -> None:
        try:
            detail = fn()
            checks.append({"name": name, "status": "ok", "detail": detail})
        except Exception as exc:  # noqa: BLE001 - aggregate dry-run failures
            checks.append({"name": name, "status": "fail", "error": str(exc)})

    record("tworoom_checkpoint", lambda: validate_task_checkpoint("tworoom", cfg.checkpoint) or True)
    record(
        "pusht_checkpoint",
        lambda: validate_task_checkpoint("pusht", inference_tasks(repo_root)["pusht"].checkpoint)
        or True,
    )
    record(
        "training_cache",
        lambda: validate_training_latent_cache(
            cfg.training_latent_cache,
            partition_dir=cfg.partition_dir,
            expected_model=cfg.model,
            checkpoint=cfg.checkpoint,
            task=cfg.task,
        ),
    )
    record(
        "cache_pool_alignment",
        lambda: validate_joint_train_pool_dataset(
            train_pool_starts=cfg.train_pool_starts,
            training_latent_cache=cfg.training_latent_cache,
            data_file=cfg.dataset_file,
            dataset_name=cfg.task,
            history_size=cfg.history_size,
            num_preds=cfg.num_preds,
            frameskip=cfg.frameskip,
            img_size=cfg.img_size,
        ),
    )
    record(
        "joint_pool_indexing",
        lambda: validate_joint_train_pool_dataset(
            train_pool_starts=cfg.train_pool_starts,
            training_latent_cache=cfg.training_latent_cache,
            data_file=cfg.dataset_file,
            dataset_name=cfg.task,
            history_size=cfg.history_size,
            num_preds=cfg.num_preds,
            frameskip=cfg.frameskip,
            img_size=cfg.img_size,
        ),
    )
    record(
        "tworoom_partition_provenance",
        lambda: validate_tworoom_predictor_partition_provenance(
            cfg.training_latent_cache.parent,
            cfg.partition_dir,
        )
        or True,
    )
    record(
        "tworoom_lap_assembly",
        lambda: {
            "dir": str(
                materialize_tworoom_lap_run_dir(
                    predictor_dir=cfg.training_latent_cache.parent,
                    partition_root=cfg.partition_dir,
                    scratch_dir=scratch,
                )
            )
        },
    )

    for task_name, task_cfg in inference_tasks(repo_root).items():
        record(
            f"lap_manifest_{task_name}",
            lambda task=task_name, task_cfg=task_cfg: validate_lap_predictor_manifest(
                task_cfg.lap_run_dir,
                task=task_name,
                checkpoint=task_cfg.checkpoint,
                expect_regional=task_name == "tworoom",
                require_partition=False,
            ),
        )

    report = {
        "status": "ok" if all(row["status"] == "ok" for row in checks) else "fail",
        "checks": checks,
    }
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
