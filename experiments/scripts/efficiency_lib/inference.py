"""Inference planning latency benchmarks (original LeWM vs LAP)."""

from __future__ import annotations

import builtins
import copy
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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
    validate_task_checkpoint,
)

_CAPTURED_INFO: dict[str, Any] | None = None
_ORIG_SOLVE = CEMSolver.solve
_EXIT_AFTER_FIRST_CAPTURED_SOLVE = False


class PlanningInfoCaptured(Exception):
    """Stop evaluator after the first CEM solve inputs are captured."""


@contextmanager
def suppress_solver_logging() -> Iterator[None]:
    """Drop solver timing prints from formal latency measurements."""
    real_print = builtins.print

    def filtered_print(*args: Any, **kwargs: Any) -> None:
        if args:
            message = " ".join(str(arg) for arg in args)
            if "CEM solve time" in message or "iCEM solve time" in message:
                return
        real_print(*args, **kwargs)

    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = real_print


def _patched_solve(self, info_dict, init_action=None):
    global _CAPTURED_INFO
    if _CAPTURED_INFO is None:
        _CAPTURED_INFO = {
            key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in info_dict.items()
        }
    result = _ORIG_SOLVE(self, info_dict, init_action)
    if _EXIT_AFTER_FIRST_CAPTURED_SOLVE:
        raise PlanningInfoCaptured()
    return result


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


def _source_eval_num_envs(info: dict[str, Any]) -> int:
    for value in info.values():
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    return 1


def _slice_info_single_env(info: dict[str, Any]) -> dict[str, Any]:
    """Reduce a batched evaluator info dict to a single-environment template."""
    sliced: dict[str, Any] = {}
    for key, value in info.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] > 1:
            sliced[key] = value[:1].detach().clone()
        else:
            sliced[key] = value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
    return sliced


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
            task=task_cfg.task,
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
        task=task_cfg.task,
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
    device: str,
    lap_run_dir: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    global _CAPTURED_INFO, _EXIT_AFTER_FIRST_CAPTURED_SOLVE
    _CAPTURED_INFO = None
    _EXIT_AFTER_FIRST_CAPTURED_SOLVE = True
    CEMSolver.solve = _patched_solve
    cfg = None
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
            "checkpoint": task_cfg.checkpoint,
        }
        try:
            run_eval(cfg, scratch / f"{task_cfg.task}_capture", "baseline", **kwargs)
        except PlanningInfoCaptured:
            pass
    finally:
        _EXIT_AFTER_FIRST_CAPTURED_SOLVE = False
        CEMSolver.solve = _ORIG_SOLVE
    if _CAPTURED_INFO is None:
        raise RuntimeError("Failed to capture planning info_dict from first CEM solve")
    return cfg, _CAPTURED_INFO


def _planning_info_cache_path(scratch_dir: Path, task: str) -> Path:
    return scratch_dir / f"planning_info_{task}.pt"


def load_or_capture_task_planning_info(
    repo_root: Path,
    task_cfg: InferenceTaskConfig,
    *,
    device: str,
    scratch_dir: Path,
) -> dict[str, Any]:
    cache_path = _planning_info_cache_path(scratch_dir, task_cfg.task)
    if cache_path.is_file():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    _, captured_info = _capture_planning_info(
        repo_root,
        task_cfg,
        device=device,
        lap_run_dir=None,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured_info, cache_path)
    return captured_info


def benchmark_inference_task(
    repo_root: Path,
    task_cfg: InferenceTaskConfig,
    *,
    mode: str,
    device: str,
    warmup: int,
    repeats: int,
    scratch_dir: Path,
    planning_info: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_import_paths(repo_root)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    if not task_cfg.checkpoint.is_file():
        return {"task": task_cfg.task, "mode": mode, "status": "pending", "reason": "missing checkpoint"}
    try:
        validate_task_checkpoint(task_cfg.task, task_cfg.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        return {"task": task_cfg.task, "mode": mode, "status": "pending", "reason": str(exc)}

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
    if planning_info is None:
        captured_info = load_or_capture_task_planning_info(
            repo_root,
            task_cfg,
            device=device,
            scratch_dir=scratch_dir,
        )
    else:
        captured_info = planning_info
    source_eval_num_envs = _source_eval_num_envs(captured_info)
    single_info_template = _slice_info_single_env(captured_info)

    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(repo_root / "config" / "eval")
    ):
        cfg = hydra.compose(config_name=task_cfg.config_name)
    cfg.seed = 42
    cfg.solver.device = device
    dev = torch.device(device)

    cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
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
    action_space, _ = _task_action_space(repo_root, cfg)
    timed_num_envs = 1
    solver.configure(
        action_space=action_space,
        n_envs=timed_num_envs,
        config=swm.PlanConfig(**cfg.plan_config),
    )

    reset_peak_memory(device)

    total_clones = warmup + repeats
    info_clones = [_clone_info_fresh(single_info_template) for _ in range(total_clones)]

    with suppress_solver_logging():
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
    if (
        mode == "lap"
        and gate_branch != "global_predictor"
        and hasattr(model, "_assign_clusters")
    ):
        with torch.no_grad():
            ref_info = _clone_info_fresh(single_info_template)
            encoded = model.encode(ref_info)
            routing_latent_template = encoded["emb"][:, 0, :].detach()
        route_clones = [
            routing_latent_template.detach().clone() for _ in range(total_clones)
        ]
        for latent in route_clones[:warmup]:
            model._assign_clusters(latent)
        for latent in route_clones[warmup:]:
            _sync(device)
            t0 = time.perf_counter()
            model._assign_clusters(latent)
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
            "source_eval_num_envs": source_eval_num_envs,
            "timed_num_envs": timed_num_envs,
            "single_env_planning_latency": True,
            "suppress_solver_logging": True,
            "routing_measured_per_mpc_cycle": gate_branch != "global_predictor",
            "routing_excludes_encoder": True,
            "eval_dataset_name": str(cfg.eval.dataset_name),
        },
    }
    if provenance is not None:
        result["provenance"] = provenance
    out = scratch_dir / f"inference_{task_cfg.task}_{mode}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
