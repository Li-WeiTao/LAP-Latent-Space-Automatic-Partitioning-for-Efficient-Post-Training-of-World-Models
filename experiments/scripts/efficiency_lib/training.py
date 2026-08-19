"""Training efficiency benchmarks (Joint vs LAP Regional-FT)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import stable_pretraining as spt
import torch
from torch.utils.data import DataLoader

from .config import TrainingAnchorConfig
from .memory import GPUMemorySnapshot, read_peak_memory, reset_peak_memory


def _ensure_import_paths(repo_root: Path) -> None:
    tworoom = repo_root / "experiments" / "tworoom"
    for path in (repo_root, tworoom):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _merge_peak(a: GPUMemorySnapshot, b: GPUMemorySnapshot) -> GPUMemorySnapshot:
    return GPUMemorySnapshot(
        max(a.peak_allocated_bytes, b.peak_allocated_bytes),
        max(a.peak_reserved_bytes, b.peak_reserved_bytes),
    )


def benchmark_joint_training(
    repo_root: Path,
    cfg: TrainingAnchorConfig,
    *,
    device: str,
    scratch_dir: Path,
) -> dict[str, Any]:
    _ensure_import_paths(repo_root)
    from gauge_drift import load_encoder
    from joint_continue_tworoom import component_parameter_counts, prepare_dataset, set_seed
    from module import SIGReg

    scratch_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)
    dev = torch.device(device)

    dataset_t0 = time.perf_counter()
    args_ns = type(
        "Args",
        (),
        {
            "data_file": cfg.dataset_file,
            "dataset_name": cfg.task,
            "history_size": cfg.history_size,
            "num_preds": cfg.num_preds,
            "frameskip": cfg.frameskip,
            "img_size": cfg.img_size,
        },
    )()
    dataset = prepare_dataset(args_ns)
    split_generator = torch.Generator().manual_seed(cfg.split_seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_fraction, 1.0 - cfg.train_fraction],
        generator=split_generator,
    )
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        pin_memory=dev.type == "cuda",
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    dataset_setup_sec = time.perf_counter() - dataset_t0

    reset_peak_memory(dev)
    model = load_encoder(str(cfg.checkpoint), dev, None, model_family=cfg.model)
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.train()
    sigreg = SIGReg(knots=17, num_proj=1024).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

    epoch_rows: list[dict[str, Any]] = []
    training_t0 = time.perf_counter()
    global_peak = read_peak_memory(dev)
    for epoch in range(cfg.joint_epochs):
        reset_peak_memory(dev)
        epoch_t0 = time.perf_counter()
        epoch_steps = 0
        processed_samples = 0
        for batch in loader:
            batch = {
                key: value.to(dev, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            optimizer.zero_grad(set_to_none=True)
            output = model.encode(batch)
            emb = output["emb"]
            act_emb = output["act_emb"]
            ctx_emb = emb[:, : cfg.history_size]
            ctx_act = act_emb[:, : cfg.history_size]
            tgt_emb = emb[:, cfg.num_preds :]
            pred_emb = model.predict(ctx_emb, ctx_act)
            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            sigreg_loss = sigreg(emb.transpose(0, 1))
            loss = pred_loss + 0.09 * sigreg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_steps += 1
            processed_samples += int(emb.shape[0])

        epoch_peak = read_peak_memory(dev)
        global_peak = _merge_peak(global_peak, epoch_peak)
        elapsed = time.perf_counter() - epoch_t0
        epoch_rows.append(
            {
                "method": "joint",
                "epoch": epoch + 1,
                "expert_id": None,
                "region_sample_count": processed_samples,
                "optimizer_steps": epoch_steps,
                "epoch_wall_sec": elapsed,
                "transitions_per_sec": processed_samples / max(elapsed, 1e-9),
                **epoch_peak.as_dict(),
            }
        )

    training_sec = time.perf_counter() - training_t0
    stable_epochs = epoch_rows[1:] if len(epoch_rows) > 1 else epoch_rows
    stable_times = [row["epoch_wall_sec"] for row in stable_epochs]
    result = {
        "method": "joint",
        "dataset_file": str(cfg.dataset_file),
        "train_num_windows": len(train_set),
        "validation_num_windows_unused": len(val_set),
        "component_parameter_counts": component_parameter_counts(model),
        "dataset_setup_sec": dataset_setup_sec,
        "training_wall_sec": training_sec,
        "epochs": epoch_rows,
        "peak_memory": global_peak.as_dict(),
        "stable_epoch_summary": {
            "mean_sec": float(np.mean(stable_times)),
            "median_sec": float(np.median(stable_times)),
            "std_sec": float(np.std(stable_times)),
        },
    }
    (scratch_dir / "joint_training.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def benchmark_lap_regional_training(
    repo_root: Path,
    cfg: TrainingAnchorConfig,
    *,
    device: str,
    scratch_dir: Path,
) -> dict[str, Any]:
    _ensure_import_paths(repo_root)
    from backends.lewm import LeWMBackendFactory, LeWMLatentCache, LeWMRegionalPredictorTrainer
    from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint
    from lap.partition import IndexedPartitioner, PartitionArtifact

    scratch_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    torch.set_num_threads(cfg.cpu_threads)

    cache_path = cfg.training_latent_cache.resolve(strict=True)
    partition_dir = cfg.partition_dir.resolve(strict=True)
    artifact = PartitionArtifact.load(partition_dir / "partition")
    with np.load(partition_dir / "cluster_labels.npz", allow_pickle=False) as data:
        id_key = "sample_ids" if "sample_ids" in data.files else "global_idx"
        assignment_ids = np.asarray(data[id_key], dtype=np.int64)
        assignment_labels = np.asarray(data["labels"], dtype=np.int64)

    partitioner = IndexedPartitioner(artifact, assignment_ids, assignment_labels)
    cache = LeWMLatentCache.from_npz(cache_path)
    transitions = cache.load()
    partition_result = partitioner.fit(
        transitions.routing_latents,
        sample_ids=transitions.sample_ids,
    )
    region_labels = partition_result.labels

    per_expert_rows: list[dict[str, Any]] = []
    lap_epoch_rows: list[dict[str, Any]] = []
    global_peak = read_peak_memory(dev)

    def load_pretrained(path: str | Path) -> torch.nn.Module:
        return load_jepa_object_checkpoint(path, model_family=cfg.model, map_location="cpu")

    from lap.interfaces import RegionalTrainingConfig

    for lap_epoch in range(1, cfg.lap_epochs_per_expert + 1):
        lap_epoch_t0 = time.perf_counter()
        lap_epoch_peak = read_peak_memory(dev)
        for region_id in range(artifact.num_regions):
            reset_peak_memory(dev)
            expert_t0 = time.perf_counter()
            training = RegionalTrainingConfig(
                train_seed=cfg.seed,
                epochs=1,
                batch_size=cfg.batch_size,
                learning_rate=5e-5,
                weight_decay=1e-3,
                min_region_samples=256,
                options={
                    "history_size": cfg.history_size,
                    "num_preds": cfg.num_preds,
                },
            )
            indices = np.flatnonzero(region_labels == region_id)
            region_transitions = transitions.subset(indices)
            backend = LeWMBackendFactory(load_pretrained).load(
                cfg.checkpoint.resolve(strict=True)
            )
            trainer = LeWMRegionalPredictorTrainer(dev, select_best_by_eval=True)
            trainer.fit_region(backend, region_id, region_transitions, training)
            expert_peak = read_peak_memory(dev)
            lap_epoch_peak = _merge_peak(lap_epoch_peak, expert_peak)
            global_peak = _merge_peak(global_peak, expert_peak)
            elapsed = time.perf_counter() - expert_t0
            per_expert_rows.append(
                {
                    "method": "lap_regional",
                    "lap_epoch": lap_epoch,
                    "expert_id": region_id,
                    "region_sample_count": int(len(indices)),
                    "optimizer_steps": None,
                    "epoch_wall_sec": elapsed,
                    "transitions_per_sec": len(indices) / max(elapsed, 1e-9),
                    **expert_peak.as_dict(),
                }
            )
            del backend, trainer
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        lap_epoch_rows.append(
            {
                "method": "lap_regional",
                "lap_epoch": lap_epoch,
                "total_wall_sec": time.perf_counter() - lap_epoch_t0,
                "experts": [row for row in per_expert_rows if row["lap_epoch"] == lap_epoch],
                **lap_epoch_peak.as_dict(),
            }
        )

    stable_times = [row["total_wall_sec"] for row in lap_epoch_rows]
    result = {
        "method": "lap_regional",
        "latent_cache": str(cache_path),
        "partition_dir": str(partition_dir),
        "num_regions": artifact.num_regions,
        "lap_epochs": lap_epoch_rows,
        "expert_epochs": per_expert_rows,
        "peak_memory": global_peak.as_dict(),
        "stable_epoch_summary": {
            "mean_sec": float(np.mean(stable_times)),
            "median_sec": float(np.median(stable_times)),
            "std_sec": float(np.std(stable_times)),
        },
    }
    (scratch_dir / "lap_regional_training.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
