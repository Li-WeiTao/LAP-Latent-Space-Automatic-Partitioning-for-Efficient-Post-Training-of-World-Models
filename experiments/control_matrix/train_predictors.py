#!/usr/bin/env python3
"""Train Global-FT or one LeWM predictor per precomputed latent region."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backends.lewm import (  # noqa: E402
    LeWMBackendFactory,
    LeWMLatentCache,
    LeWMRegionalPredictorTrainer,
)
from experiments.control_matrix.backend_registry import backend_metadata, normalize_model_family  # noqa: E402
from lap import LAP, LAPConfig  # noqa: E402
from lap.interfaces import RegionalTrainingConfig  # noqa: E402
from lap.partition import IndexedPartitioner, PartitionArtifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=Path, required=True)
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-family", default="lewm")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--min-region-samples", type=int, default=256)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Formal split manifest for checkpoint provenance.",
    )
    parser.add_argument(
        "--training-role",
        default="standard",
        choices=("standard", "global", "forced_spectral_negative_control"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: str | Path, *, model_family: str = "lewm") -> torch.nn.Module:
    from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint

    return load_jepa_object_checkpoint(path, model_family=model_family, map_location="cpu")


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    cache_path = args.latent_cache.resolve(strict=True)
    model_path = args.pretrained_model.resolve(strict=True)
    partition_run = args.partition_dir.resolve(strict=True)
    artifact = PartitionArtifact.load(partition_run / "partition")
    with np.load(partition_run / "cluster_labels.npz", allow_pickle=False) as data:
        id_key = "sample_ids" if "sample_ids" in data.files else "global_idx"
        assignment_ids = np.asarray(data[id_key], dtype=np.int64)
        assignment_labels = np.asarray(data["labels"], dtype=np.int64)

    training = RegionalTrainingConfig(
        train_seed=args.train_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        min_region_samples=args.min_region_samples,
        options={
            "history_size": args.history_size,
            "num_preds": args.num_preds,
        },
    )
    family = normalize_model_family(args.model_family)

    def _load_pretrained(path: str | Path) -> torch.nn.Module:
        return load_model(path, model_family=family)

    method = LAP(
        backend_factory=LeWMBackendFactory(_load_pretrained),
        partitioner=IndexedPartitioner(
            artifact, assignment_ids, assignment_labels
        ),
        trainer=LeWMRegionalPredictorTrainer(
            torch.device(args.device), select_best_by_eval=True
        ),
        config=LAPConfig(training=training, output_directory=args.out_dir),
    )
    result = method.fit(LeWMLatentCache.from_npz(cache_path), model_path)

    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_manifest_hash = (
        sha256_file(args.split_manifest.resolve(strict=True))
        if args.split_manifest is not None
        else None
    )
    manifest.update(
        {
            **backend_metadata(family),
            "dataset": args.dataset_name,
            "training_role": args.training_role,
            "partition_source": str(partition_run),
            "partition_source_manifest_sha256": sha256_file(
                partition_run / "manifest.json"
            ),
            "latent_cache": str(cache_path),
            "latent_cache_sha256": sha256_file(cache_path),
            "pretrained_model": str(model_path),
            "pretrained_model_sha256": sha256_file(model_path),
            "train_seed": args.train_seed,
            "split_manifest": str(args.split_manifest.resolve())
            if args.split_manifest is not None
            else None,
            "split_manifest_sha256": split_manifest_hash,
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "history_size": args.history_size,
                "num_preds": args.num_preds,
                "cpu_threads": args.cpu_threads,
                "precision": "fp32",
                "checkpoint_selection": "minimum loss on same-region training cache",
            },
            "predictor_checkpoint_sha256": {
                f"cluster{region_id}": sha256_file(
                    args.out_dir / f"P_train_cluster{region_id}_object.ckpt"
                )
                for region_id in result.regional_predictors
            },
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] dataset={args.dataset_name} train_seed={args.train_seed} "
        f"regions={len(result.regional_predictors)} -> {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
