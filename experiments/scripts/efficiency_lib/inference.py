"""Inference planning latency benchmarks (original LeWM vs LAP)."""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import torch

from stable_worldmodel.solver import CEMSolver

from .config import InferenceTaskConfig
from .memory import read_peak_memory, reset_peak_memory
from .stats import summarize
from .validation import (
    materialize_tworoom_lap_run_dir,
    read_gate_branch,
    validate_lap_predictor_manifest,
    validate_lewm_checkpoint,
)

_CAPTURED_INFO: dict[str, Any] | None = None
_ORIG_SOLVE = CEMSolver.solve


def _patched_solve(self, info_dict, init_action=None):
    global _CAPTURED_INFO
    if _CAPTURED_INFO is None:
        _CAPTURED_INFO = {
            key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in info_dict.items()
        }
    return _ORIG_SOLVE(self, info_dict, init_action)


def _ensure_import_paths(repo_root: Path) -> None:
    tworoom = repo_root / "experiments" / "tworoom"
    for path in (repo_root, tworoom):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _sync(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _clone_info_fresh(source: dict[str, Any]) -> dict[str, Any]:
    """Clone planning inputs with new tensor storage (fresh MPC observation identity)."""
    return {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in source.items()
    }


def _task_action_space(repo_root: Path, cfg) -> tuple[Any, int]:
    import stable_worldmodel as swm

    cfg = cfg.copy() if hasattr(cfg, "copy") else cfg
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    return world.envs.action_space, int(world.envs.num_envs)


def _resolve_lap_run_dir(
    repo_root: Path,
    task_cfg: InferenceTaskConfig,
    *,
    scratch_dir: Path,
) -> Path:
    if task_cfg.lap_partition_root is not None:
        return materialize_tworoom_lap_run_dir(
            predictor_dir=task_cfg.lap_run_dir,
            partition_root=task_cfg.lap_partition_root,
            scratch_dir=scratch_dir / "lap_assembly",
        )
    return task_cfg.lap_run_dir.resolve(strict=True)


def _resolve_lap_model(
    task_cfg: InferenceTaskConfig,
    checkpoint: Path,
    device: torch.device,
    proprio_processor,
    *,
    lap_run_dir: Path,
):
    from tworoom_success_rate_eval import load_object_checkpoint, resolve_model

    gate_branch = read_gate_branch(task_cfg.gate_manifest)
    if gate_branch == "global_predictor":
        validate_lap_predictor_manifest(
            lap_run_dir,
            checkpoint=checkpoint,
            expect_regional=False,
        )
        global_ckpt = lap_run_dir / "P_train_cluster0_object.ckpt"
        model = load_object_checkpoint(
            global_ckpt, device, model_family="lewm"
        )
        meta = {
            "mode": "global_ft",
            "gate_branch": gate_branch,
            "lap_run_dir": str(lap_run_dir),
            "predictor_checkpoint": str(global_ckpt),
            "router": None,
        }
        return model, meta

    validate_lap_predictor_manifest(
        lap_run_dir,
        checkpoint=checkpoint,
        expect_regional=True,
    )
    model, meta = resolve_model(
        "lap",
        checkpoint,
        device,
        proprio_processor,
        lap_run_dir=lap_run_dir,
        latent_routing="mpc",
        model_family="lewm",
    )
    meta["gate_branch"] = gate_branch
    return model, meta


def _capture_planning_info(
    repo_root: Path,
    task_cfg: InferenceTaskConfig,
    *,
    mode: str,
    device: str,
    lap_run_dir: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    global _CAPTURED_INFO
    _CAPTURED_INFO = None
    CEMSolver.solve = _patched_solve
    try:
        from tworoom_success_rate_eval import run_eval

        with hydra.initialize_config_dir(
            version_base=None, config_dir=str(repo_root / "config" / "eval"
        )
        ):
            cfg = hydra.compose(config_name=task_cfg.config_name)
        cfg.seed = 42
        cfg.solver.device = device
        scratch = repo_root / "experiments/efficiency_results/scratch/inference_capture"
        scratch.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "eval_start_indices_path": task_cfg.eval_starts,
            "latent_routing": "mpc",
            "model_family": "lewm",
        }
        capture_mode = mode
        capture_checkpoint = task_cfg.checkpoint
        if mode == "lap":
            resolved = lap_run_dir or _resolve_lap_run_dir(
                repo_root, task_cfg, scratch_dir=scratch / "lap_assembly"
            )
            gate_branch = read_gate_branch(task_cfg.gate_manifest)
            if gate_branch == "global_predictor":
                capture_mode = "baseline"
                capture_checkpoint = resolved / "P_train_cluster0_object.ckpt"
            else:
                kwargs["lap_run_dir"] = resolved
        kwargs["checkpoint"] = capture_checkpoint
        run_eval(cfg, scratch / f"{task_cfg.task}_{mode}", capture_mode, **kwargs)
    finally:
        CEMSolver.solve = _ORIG_SOLVE
    if _CAPTURED_INFO is None:
        raise RuntimeError("Failed to capture planning info_dict from first CEM solve")
    return cfg, _CAPTURED_INFO


def benchmark_inference_task(
    repo_root: Path,
    task_cfg: InferenceTaskConfig,
    *,
    mode: str,
    device: str,
    warmup: int,
    repeats: int,
    scratch_dir: Path,
) -> dict[str, Any]:
    _ensure_import_paths(repo_root)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    validate_lewm_checkpoint(task_cfg.checkpoint)
    if not task_cfg.checkpoint.is_file():
        return {"task": task_cfg.task, "mode": mode, "status": "pending", "reason": "missing checkpoint"}

    lap_run_dir: Path | None = None
    if mode == "lap":
        if not task_cfg.lap_run_dir.is_dir():
            return {
                "task": task_cfg.task,
                "mode": mode,
                "status": "pending",
                "reason": "missing lap run dir",
            }
        lap_run_dir = _resolve_lap_run_dir(repo_root, task_cfg, scratch_dir=scratch_dir)

    import numpy as np
    import stable_worldmodel as swm
    from sklearn import preprocessing
    from tworoom_success_rate_eval import _state_processor

    reset_peak_memory(device)
    cfg, captured_info = _capture_planning_info(
        repo_root,
        task_cfg,
        mode=mode,
        device=device,
        lap_run_dir=lap_run_dir,
    )

    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(repo_root / "config" / "eval")
    ):
        cfg = hydra.compose(config_name=task_cfg.config_name)
    cfg.seed = 42
    cfg.solver.device = device
    dev = torch.device(device)

    cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        task_cfg.dataset_tag,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = processor

    if mode == "lap":
        assert lap_run_dir is not None
        model, model_meta = _resolve_lap_model(
            task_cfg,
            task_cfg.checkpoint,
            dev,
            _state_processor(process),
            lap_run_dir=lap_run_dir,
        )
    else:
        from tworoom_success_rate_eval import resolve_model

        model, model_meta = resolve_model(
            "baseline",
            task_cfg.checkpoint,
            dev,
            _state_processor(process),
            latent_routing="mpc",
            model_family="lewm",
        )
        model_meta["gate_branch"] = read_gate_branch(task_cfg.gate_manifest)

    model.eval()
    if hasattr(model, "classify_timing_sample_limit"):
        model.classify_timing_sample_limit = 0
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    action_space, n_envs = _task_action_space(repo_root, cfg)
    solver.configure(
        action_space=action_space,
        n_envs=n_envs,
        config=swm.PlanConfig(**cfg.plan_config),
    )

    reset_peak_memory(device)

    total_clones = warmup + repeats
    info_clones = [_clone_info_fresh(captured_info) for _ in range(total_clones)]

    for info in info_clones[:warmup]:
        solver.solve(info)

    solve_times = []
    for info in info_clones[warmup:]:
        _sync(device)
        t0 = time.perf_counter()
        solver.solve(info)
        _sync(device)
        solve_times.append(time.perf_counter() - t0)

    routing_times = []
    gate_branch = model_meta.get("gate_branch")
    if mode == "lap" and gate_branch != "global_predictor" and hasattr(model, "_current_latent"):
        route_clones = [_clone_info_fresh(captured_info) for _ in range(total_clones)]
        for info in route_clones[:warmup]:
            model._assign_clusters(model._current_latent(info))
        for info in route_clones[warmup:]:
            _sync(device)
            t0 = time.perf_counter()
            model._assign_clusters(model._current_latent(info))
            _sync(device)
            routing_times.append(time.perf_counter() - t0)

    peak = read_peak_memory(device)
    result = {
        "task": task_cfg.task,
        "mode": mode,
        "status": "ok",
        "checkpoint": str(task_cfg.checkpoint),
        "lap_run_dir": str(lap_run_dir) if mode == "lap" and lap_run_dir else None,
        "gate_branch": gate_branch,
        "cem": {
            "num_samples": int(cfg.solver.num_samples),
            "n_steps": int(cfg.solver.n_steps),
            "topk": int(cfg.solver.topk),
            "batch_size": int(cfg.solver.batch_size),
            "horizon": int(cfg.plan_config.horizon),
            "receding_horizon": int(cfg.plan_config.receding_horizon),
            "action_block": int(cfg.plan_config.action_block),
            "num_eval": int(cfg.eval.num_eval),
        },
        "planning_latency_sec": solve_times,
        "planning_summary": summarize(solve_times),
        "routing_latency_sec": routing_times,
        "routing_summary": summarize(routing_times) if routing_times else None,
        "peak_memory": peak.as_dict(),
        "model_meta": model_meta,
        "timing_protocol": {
            "fresh_observation_clones": total_clones,
            "routing_measured_per_mpc_cycle": gate_branch != "global_predictor",
        },
    }
    out = scratch_dir / f"inference_{task_cfg.task}_{mode}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
