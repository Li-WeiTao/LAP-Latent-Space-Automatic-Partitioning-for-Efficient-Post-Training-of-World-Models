"""Training efficiency benchmarks (Joint vs LAP Regional-FT)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainingAnchorConfig
from .memory import GPUMemorySnapshot, read_peak_memory, reset_peak_memory
from .train_pool import build_joint_train_pool_dataset
from .validation import (
    assert_cache_matches_train_pool,
    validate_task_checkpoint,
    validate_training_latent_cache,
)


def _ensure_import_paths(repo_root: Path) -> None:
    tworoom = repo_root / "experiments" / "tworoom"
    lewm_root = repo_root.parent / "le-wm"
    for path in (repo_root, tworoom, lewm_root):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _merge_peak(a: GPUMemorySnapshot, b: GPUMemorySnapshot) -> GPUMemorySnapshot:
    return GPUMemorySnapshot(
        max(a.peak_allocated_bytes, b.peak_allocated_bytes),
        max(a.peak_reserved_bytes, b.peak_reserved_bytes),
    )


def _stable_epoch_summary(
    epoch_rows: list[dict[str, Any]],
    *,
    discard_warmup_epochs: int,
) -> dict[str, float]:
    stable = epoch_rows[discard_warmup_epochs:] if len(epoch_rows) > discard_warmup_epochs else epoch_rows
    stable_times = [row["epoch_wall_sec"] for row in stable]
    return {
        "mean_sec": float(np.mean(stable_times)),
        "median_sec": float(np.median(stable_times)),
        "std_sec": float(np.std(stable_times)),
        "count_epochs": len(stable_times),
        "discarded_warmup_epochs": discard_warmup_epochs,
    }


def benchmark_joint_training(
    repo_root: Path,
    cfg: TrainingAnchorConfig,
    *,
    device: str,
    scratch_dir: Path,
) -> dict[str, Any]:
    _ensure_import_paths(repo_root)
    from gauge_drift import load_encoder
    from joint_continue_tworoom import component_parameter_counts, resolve_state_key, set_seed
    from module import SIGReg

    validate_task_checkpoint(cfg.task, cfg.checkpoint)
    cache_provenance = validate_training_latent_cache(
        cfg.training_latent_cache,
        partition_dir=cfg.partition_dir,
        expected_model=cfg.model,
        checkpoint=cfg.checkpoint,
        task=cfg.task,
    )
    assert_cache_matches_train_pool(
        cfg.training_latent_cache,
        cfg.train_pool_starts,
    )

    scratch_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)
    dev = torch.device(device)

    dataset_t0 = time.perf_counter()
    train_set = build_joint_train_pool_dataset(
        data_file=cfg.dataset_file,
        dataset_name=cfg.task,
        train_pool_starts=cfg.train_pool_starts,
        history_size=cfg.history_size,
        num_preds=cfg.num_preds,
        frameskip=cfg.frameskip,
        img_size=cfg.img_size,
        resolve_state_key=resolve_state_key,
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

    if len(train_set) != cache_provenance["num_transitions"]:
        raise ValueError(
            "Joint training windows must match LAP cache transitions for a fair "
            f"benchmark: joint={len(train_set)} lap_cache="
            f"{cache_provenance['num_transitions']}. "
            "Configure the same curated train pool for both methods."
        )

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
    for epoch in range(cfg.timing_epochs):
        reset_peak_memory(dev)
        _sync_device(dev)
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

        _sync_device(dev)
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
    result = {
        "method": "joint",
        "dataset_file": str(cfg.dataset_file),
        "train_num_windows": len(train_set),
        "train_pool_starts": str(cfg.train_pool_starts),
        "train_dataset": "GlobalReferenceStartDataset",
        "component_parameter_counts": component_parameter_counts(model),
        "dataset_setup_sec": dataset_setup_sec,
        "training_wall_sec": training_sec,
        "epochs": epoch_rows,
        "peak_memory": global_peak.as_dict(),
        "stable_epoch_summary": _stable_epoch_summary(
            epoch_rows, discard_warmup_epochs=cfg.discard_warmup_epochs
        ),
        "timing_protocol": {
            "primary_metric": "pure_training_epoch_sec",
            "epochs": cfg.timing_epochs,
            "discard_warmup_epochs": cfg.discard_warmup_epochs,
            "cuda_synchronize": True,
            "excludes": ["dataset_setup", "model_load", "optimizer_init"],
        },
        "cache_provenance": cache_provenance,
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
    from lap.interfaces import RegionalTrainingConfig

    validate_task_checkpoint(cfg.task, cfg.checkpoint)
    cache_provenance = validate_training_latent_cache(
        cfg.training_latent_cache,
        partition_dir=cfg.partition_dir,
        expected_model=cfg.model,
        checkpoint=cfg.checkpoint,
        task=cfg.task,
    )
    assert_cache_matches_train_pool(
        cfg.training_latent_cache,
        cfg.train_pool_starts,
    )

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
    setup_rows: list[dict[str, Any]] = []
    global_peak = read_peak_memory(dev)
    expert_epoch_times: dict[int, list[float]] = {
        region_id: [] for region_id in range(artifact.num_regions)
    }

    def load_pretrained(path: str | Path) -> torch.nn.Module:
        return load_jepa_object_checkpoint(path, model_family=cfg.model, map_location="cpu")

    for region_id in range(artifact.num_regions):
        reset_peak_memory(dev)
        setup_t0 = time.perf_counter()
        indices = np.flatnonzero(region_labels == region_id)
        region_transitions = transitions.subset(indices)
        backend = LeWMBackendFactory(load_pretrained).load(
            cfg.checkpoint.resolve(strict=True)
        )
        trainer = LeWMRegionalPredictorTrainer(
            dev,
            select_best_by_eval=False,
            eval_each_epoch=False,
            benchmark_pure_epochs=True,
        )
        training = RegionalTrainingConfig(
            train_seed=cfg.seed,
            epochs=cfg.timing_epochs,
            batch_size=cfg.batch_size,
            learning_rate=5e-5,
            weight_decay=1e-3,
            min_region_samples=256,
            options={
                "history_size": cfg.history_size,
                "num_preds": cfg.num_preds,
            },
        )
        setup_sec = time.perf_counter() - setup_t0
        setup_rows.append(
            {
                "expert_id": region_id,
                "setup_wall_sec": setup_sec,
                "region_sample_count": int(len(indices)),
            }
        )

        reset_peak_memory(dev)
        fit_result = trainer.fit_region(backend, region_id, region_transitions, training)
        expert_peak = read_peak_memory(dev)
        global_peak = _merge_peak(global_peak, expert_peak)
        epoch_timings = list(fit_result.metrics.get("epoch_timings", []))
        if len(epoch_timings) != cfg.timing_epochs:
            raise RuntimeError(
                f"region {region_id} returned {len(epoch_timings)} epoch timings, "
                f"expected {cfg.timing_epochs}"
            )
        for row in epoch_timings:
            epoch_no = int(row["epoch"])
            elapsed = float(row["epoch_wall_sec"])
            optimizer_steps = int(row.get("optimizer_steps") or 0)
            processed_samples = optimizer_steps * cfg.batch_size
            expert_epoch_times[region_id].append(elapsed)
            per_expert_rows.append(
                {
                    "method": "lap_regional",
                    "lap_epoch": epoch_no,
                    "expert_id": region_id,
                    "region_sample_count": int(len(indices)),
                    "optimizer_steps": optimizer_steps,
                    "processed_samples": processed_samples,
                    "setup_wall_sec": setup_sec if epoch_no == 1 else 0.0,
                    "epoch_wall_sec": elapsed,
                    "transitions_per_sec": processed_samples / max(elapsed, 1e-9),
                    **expert_peak.as_dict(),
                }
            )
        del fit_result
        del backend
        del trainer
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    lap_epoch_rows: list[dict[str, Any]] = []
    for epoch_no in range(1, cfg.timing_epochs + 1):
        total = sum(
            expert_epoch_times[region_id][epoch_no - 1]
            for region_id in range(artifact.num_regions)
        )
        lap_epoch_rows.append(
            {
                "method": "lap_regional",
                "lap_epoch": epoch_no,
                "total_wall_sec": total,
                "experts": [
                    row
                    for row in per_expert_rows
                    if row["lap_epoch"] == epoch_no
                ],
            }
        )

    stable_rows = [{"epoch_wall_sec": row["total_wall_sec"]} for row in lap_epoch_rows]
    result = {
        "method": "lap_regional",
        "latent_cache": str(cache_path),
        "cache_provenance": cache_provenance,
        "partition_dir": str(partition_dir),
        "num_regions": artifact.num_regions,
        "lap_epochs": lap_epoch_rows,
        "expert_epochs": per_expert_rows,
        "expert_setup": setup_rows,
        "peak_memory": global_peak.as_dict(),
        "stable_epoch_summary": _stable_epoch_summary(
            stable_rows, discard_warmup_epochs=cfg.discard_warmup_epochs
        ),
        "timing_protocol": {
            "primary_metric": "sum_k pure_predictor_training_epoch_sec",
            "peak_memory_metric": "max_over_experts_after_predictor_release",
            "epochs": cfg.timing_epochs,
            "discard_warmup_epochs": cfg.discard_warmup_epochs,
            "cuda_synchronize": True,
            "expert_training": "setup_once_then_continuous_epochs",
            "excludes": [
                "checkpoint_load",
                "backend_construction",
                "trainer_construction",
                "per_epoch_eval",
                "final_eval",
                "checkpoint_selection",
            ],
        },
    }
    (scratch_dir / "lap_regional_training.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
