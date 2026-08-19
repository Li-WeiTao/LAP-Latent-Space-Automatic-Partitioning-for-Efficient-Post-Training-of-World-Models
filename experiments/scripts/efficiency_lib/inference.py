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


def _time_call(fn) -> float:
    _sync(fn.__self__.device if hasattr(fn, "__self__") else "cuda")
    t0 = time.perf_counter()
    fn()
    _sync("cuda")
    return time.perf_counter() - t0


def _task_action_space(repo_root: Path, cfg) -> tuple[Any, int]:
    import stable_worldmodel as swm

    cfg = cfg.copy() if hasattr(cfg, "copy") else cfg
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    return world.envs.action_space, int(world.envs.num_envs)


def _capture_planning_info(
    repo_root: Path,
    task_cfg: InferenceTaskConfig,
    *,
    mode: str,
    device: str,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    global _CAPTURED_INFO
    _CAPTURED_INFO = None
    CEMSolver.solve = _patched_solve
    try:
        from tworoom_success_rate_eval import run_eval

        with hydra.initialize_config_dir(
            version_base=None, config_dir=str(repo_root / "config" / "eval")
        ):
            cfg = hydra.compose(config_name=task_cfg.config_name)
        cfg.seed = 42
        cfg.solver.device = device
        scratch = repo_root / "experiments/efficiency_results/scratch/inference_capture"
        scratch.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "checkpoint": task_cfg.checkpoint,
            "eval_start_indices_path": task_cfg.eval_starts,
            "latent_routing": "mpc",
            "model_family": "lewm",
        }
        if mode == "lap":
            kwargs["lap_run_dir"] = task_cfg.lap_run_dir
        run_eval(cfg, scratch / f"{task_cfg.task}_{mode}", mode, **kwargs)
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
    if not task_cfg.checkpoint.is_file():
        return {"task": task_cfg.task, "mode": mode, "status": "pending", "reason": "missing checkpoint"}
    if mode == "lap" and not task_cfg.lap_run_dir.is_dir():
        return {"task": task_cfg.task, "mode": mode, "status": "pending", "reason": "missing lap run dir"}

    from tworoom_success_rate_eval import resolve_model, _state_processor, img_transform
    import numpy as np
    import stable_worldmodel as swm
    from sklearn import preprocessing

    reset_peak_memory(device)
    cfg, captured_info = _capture_planning_info(
        repo_root, task_cfg, mode=mode, device=device
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

    kwargs = {"latent_routing": "mpc", "model_family": "lewm"}
    if mode == "lap":
        kwargs["lap_run_dir"] = task_cfg.lap_run_dir
    model, model_meta = resolve_model(
        mode, task_cfg.checkpoint, dev, _state_processor(process), **kwargs
    )
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

    info = {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in captured_info.items()
    }

    for _ in range(warmup):
        solver.solve(info)

    solve_times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        solver.solve(info)
        _sync(device)
        solve_times.append(time.perf_counter() - t0)

    routing_times = []
    if mode == "lap" and hasattr(model, "_current_latent"):
        route_info = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in info.items()}
        for _ in range(warmup):
            latent = model._current_latent(route_info)
            model._assign_clusters(latent)
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            model._assign_clusters(model._current_latent(route_info))
            _sync(device)
            routing_times.append(time.perf_counter() - t0)

    peak = read_peak_memory(device)
    result = {
        "task": task_cfg.task,
        "mode": mode,
        "status": "ok",
        "checkpoint": str(task_cfg.checkpoint),
        "lap_run_dir": str(task_cfg.lap_run_dir) if mode == "lap" else None,
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
    }
    out = scratch_dir / f"inference_{task_cfg.task}_{mode}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
