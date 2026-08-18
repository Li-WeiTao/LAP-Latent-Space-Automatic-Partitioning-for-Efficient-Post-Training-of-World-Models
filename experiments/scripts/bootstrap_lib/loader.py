"""Load model-task-horizon cells into compact NumPy arrays."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PARTITION_METHODS = frozenset({"random_voronoi", "kmeanspp", "spectral"})


@dataclass
class RunRecord:
    path: Path
    success_rate: float
    episode_successes: np.ndarray | None
    eval_start_indices: list[int] | None
    train_seed: int | None
    partition_seed: int | None
    eval_seed: int


@dataclass
class MethodData:
    method_id: str
    label: str
    official: bool
    partition_policy: str
    deployment_seed: int | None
    blocks: np.ndarray
    episodes: np.ndarray | None
    files_read: list[str] = field(default_factory=list)
    n_partition_seeds: int = 0


@dataclass
class CellData:
    model: str
    task: str
    horizon: str
    status: str
    methods: dict[str, MethodData]
    train_seeds: tuple[int, ...]
    partition_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    goal_offset_steps: int
    gate_info: dict[str, Any]
    validation: dict[str, Any]
    reference_estimates: dict[str, float]
    episodes_per_block: int
    has_episode_data: bool


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _starts_digest(starts: list[int]) -> str:
    payload = json.dumps([int(v) for v in starts], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _read_run(path: Path, *, eval_seed: int, goal_offset: int) -> RunRecord:
    payload = _load_json(path)
    if payload.get("seed") != eval_seed:
        raise ValueError(f"{path}: seed mismatch expected {eval_seed}, got {payload.get('seed')}")
    if payload.get("goal_offset_steps") != goal_offset:
        raise ValueError(
            f"{path}: goal_offset_steps mismatch expected {goal_offset}, "
            f"got {payload.get('goal_offset_steps')}"
        )
    metrics = payload["metrics"]
    rate = float(metrics["success_rate"])
    episodes_raw = metrics.get("episode_successes")
    episodes = None
    if episodes_raw is not None:
        episodes = np.asarray(episodes_raw, dtype=np.float64)
        recomputed = float(episodes.mean() * 100.0)
        if abs(recomputed - rate) > 1e-9:
            raise ValueError(
                f"{path}: success_rate {rate} != mean(episode_successes)*100 {recomputed}"
            )
    starts = payload.get("eval_start_indices")
    starts_list = [int(v) for v in starts] if starts is not None else None
    return RunRecord(
        path=path,
        success_rate=rate,
        episode_successes=episodes,
        eval_start_indices=starts_list,
        train_seed=None,
        partition_seed=None,
        eval_seed=eval_seed,
    )


def load_gate_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"branch": None, "deployment_seed": 0, "source": None}
    payload = _load_json(path)
    meta = payload.get("method_metadata", {})
    auto = meta.get("automatic_gate", meta)
    branch = payload.get("selected_method") or auto.get("selected_method")
    deployment_seed = int(
        payload.get("partition_seed")
        or auto.get("deployment_seed")
        or meta.get("deployment_seed", 0)
    )
    post_training = auto.get("selected_post_training") or meta.get("selected_post_training")
    return {
        "branch": branch,
        "deployment_seed": deployment_seed,
        "selected_post_training": post_training,
        "source": str(path),
    }


def _method_label(config: dict[str, Any], model: str, method_id: str) -> str:
    labels = config["method_labels"]
    if method_id == "official" and model == "subjepa":
        return config["subjepa_official_label"]
    if method_id in {"official", "autolap"}:
        return labels[method_id]
    return labels.get(method_id, method_id)


def _aggregate_runs(
    runs: dict[tuple[int | None, int | None, int], RunRecord],
    *,
    train_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
    official: bool,
    partition_policy: str,
    deployment_seed: int | None,
    partition_seeds: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray | None, list[str], int]:
    files: list[str] = []
    has_episodes = all(r.episode_successes is not None for r in runs.values())
    if official:
        rates = np.empty(len(eval_seeds), dtype=np.float64)
        episodes = None
        if has_episodes:
            n_ep = next(iter(runs.values())).episode_successes.shape[0]  # type: ignore[union-attr]
            episodes = np.empty((len(eval_seeds), n_ep), dtype=np.float64)
        for ei, es in enumerate(eval_seeds):
            rec = runs[(None, None, es)]
            rates[ei] = rec.success_rate
            files.append(str(rec.path))
            if episodes is not None and rec.episode_successes is not None:
                episodes[ei] = rec.episode_successes
        return rates, episodes, files, 0

    n_part = 0
    blocks = np.empty((len(train_seeds), len(eval_seeds)), dtype=np.float64)
    ep_blocks = None
    if has_episodes:
        n_ep = next(iter(runs.values())).episode_successes.shape[0]  # type: ignore[union-attr]
        ep_blocks = np.empty((len(train_seeds), len(eval_seeds), n_ep), dtype=np.float64)

    if partition_policy == "average":
        n_part = len(partition_seeds)
        for ti, ts in enumerate(train_seeds):
            for ei, es in enumerate(eval_seeds):
                part_rates: list[float] = []
                part_eps: list[np.ndarray] = []
                for ps in partition_seeds:
                    rec = runs[(ts, ps, es)]
                    part_rates.append(rec.success_rate)
                    files.append(str(rec.path))
                    if has_episodes and rec.episode_successes is not None:
                        part_eps.append(rec.episode_successes)
                blocks[ti, ei] = statistics.fmean(part_rates)
                if ep_blocks is not None and part_eps:
                    ep_blocks[ti, ei] = np.mean(np.stack(part_eps, axis=0), axis=0)
    elif partition_policy == "deployment":
        n_part = 1
        assert deployment_seed is not None
        for ti, ts in enumerate(train_seeds):
            for ei, es in enumerate(eval_seeds):
                rec = runs[(ts, deployment_seed, es)]
                blocks[ti, ei] = rec.success_rate
                files.append(str(rec.path))
                if ep_blocks is not None and rec.episode_successes is not None:
                    ep_blocks[ti, ei] = rec.episode_successes
    else:
        for ti, ts in enumerate(train_seeds):
            for ei, es in enumerate(eval_seeds):
                rec = runs[(ts, None, es)]
                blocks[ti, ei] = rec.success_rate
                files.append(str(rec.path))
                if ep_blocks is not None and rec.episode_successes is not None:
                    ep_blocks[ti, ei] = rec.episode_successes
    return blocks, ep_blocks, files, n_part


def point_estimate(blocks: np.ndarray, *, official: bool) -> float:
    if official:
        return float(blocks.mean())
    return float(blocks.mean(axis=1).mean())


def _autolap_source_method(gate_info: dict[str, Any]) -> str:
    if gate_info.get("branch") == "spectral":
        return "spectral"
    return "global"


def _collect_matrix_eval(
    eval_root: Path,
    *,
    method_id: str,
    train_seeds: tuple[int, ...],
    partition_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
    goal_offset: int,
    partition_policy: str,
    deployment_seed: int | None,
    gate_info: dict[str, Any] | None = None,
    autolap_eval_root: Path | None = None,
) -> dict[tuple[int | None, int | None, int], RunRecord]:
    runs: dict[tuple[int | None, int | None, int], RunRecord] = {}
    if method_id == "official":
        for es in eval_seeds:
            path = eval_root / "official" / f"eval{es}" / "results.json"
            rec = _read_run(path, eval_seed=es, goal_offset=goal_offset)
            runs[(None, None, es)] = rec
        return runs

    if method_id == "autolap":
        auto_root = autolap_eval_root or eval_root
        auto_base = auto_root / "auto" / "eval"
        if not auto_base.is_dir():
            auto_base = auto_root.parent / "auto" / "eval"
        if auto_base.is_dir() and any(auto_base.rglob("results.json")):
            for ts in train_seeds:
                for es in eval_seeds:
                    path = auto_base / f"train{ts}" / f"eval{es}" / "results.json"
                    rec = _read_run(path, eval_seed=es, goal_offset=goal_offset)
                    rec.train_seed = ts
                    rec.partition_seed = deployment_seed
                    key = (ts, deployment_seed if partition_policy == "deployment" else None, es)
                    runs[key] = rec
            return runs
        source = _autolap_source_method(gate_info or {})
        return _collect_matrix_eval(
            eval_root,
            method_id=source,
            train_seeds=train_seeds,
            partition_seeds=partition_seeds,
            eval_seeds=eval_seeds,
            goal_offset=goal_offset,
            partition_policy="deployment" if source == "spectral" else "none",
            deployment_seed=deployment_seed,
            gate_info=gate_info,
            autolap_eval_root=autolap_eval_root,
        )

    method_dir = method_id
    base = eval_root / method_dir

    if method_id in PARTITION_METHODS:
        seeds_to_use = partition_seeds if partition_policy == "average" else (deployment_seed,)
        for ts in train_seeds:
            for ps in seeds_to_use:
                for es in eval_seeds:
                    path = base / f"partition{ps}_train{ts}" / f"eval{es}" / "results.json"
                    rec = _read_run(path, eval_seed=es, goal_offset=goal_offset)
                    rec.train_seed = ts
                    rec.partition_seed = ps
                    runs[(ts, ps, es)] = rec
        return runs

    for ts in train_seeds:
        for es in eval_seeds:
            path = base / f"train{ts}" / f"eval{es}" / "results.json"
            rec = _read_run(path, eval_seed=es, goal_offset=goal_offset)
            rec.train_seed = ts
            runs[(ts, None, es)] = rec
    return runs


def _import_tworoom_paths(repo_root: Path):
    module_path = repo_root / "experiments/tworoom/aggregate_tworoom_main.py"
    import sys

    module_name = "lap_aggregate_tworoom_main_bootstrap"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _tworoom_method_map(method_id: str) -> str:
    return {
        "official": "baseline",
        "joint": "joint3",
        "global": "globalft50",
        "random_voronoi": "random",
        "kmeanspp": "kmeans",
        "spectral": "spectral",
        "rooms3": "rooms3",
        "autolap": "autolap",
    }[method_id]


def _collect_tworoom_results(
    results_root: Path,
    *,
    method_id: str,
    horizon: str,
    train_seeds: tuple[int, ...],
    partition_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
    goal_offset: int,
    partition_policy: str,
    deployment_seed: int | None,
    repo_root: Path,
) -> dict[tuple[int | None, int | None, int], RunRecord]:
    agg = _import_tworoom_paths(repo_root)
    tw_method = _tworoom_method_map(method_id)
    runs: dict[tuple[int | None, int | None, int], RunRecord] = {}

    if tw_method == "baseline":
        for es in eval_seeds:
            path = agg.path_for(results_root, horizon, "baseline", None, es)
            rec = _read_run(path, eval_seed=es, goal_offset=goal_offset)
            runs[(None, None, es)] = rec
        return runs

    if tw_method == "autolap":
        part_seeds: tuple[int | None, ...] = (deployment_seed,)
    elif tw_method in {"random", "kmeans", "spectral"}:
        part_seeds = partition_seeds if partition_policy == "average" else (deployment_seed,)
    else:
        part_seeds = (None,)

    for ts in train_seeds:
        for ps in part_seeds:
            for es in eval_seeds:
                path = agg.path_for(
                    results_root,
                    horizon,
                    tw_method,
                    ts,
                    es,
                    ps if ps is not None else None,
                )
                rec = _read_run(path, eval_seed=es, goal_offset=goal_offset)
                rec.train_seed = ts
                rec.partition_seed = ps
                key = (ts, ps, es) if ps is not None else (ts, None, es)
                runs[key] = rec
    return runs


def _normalize_horizon(value: str) -> str:
    return value.lower().replace(" horizon", "").strip()


def _load_reference_estimates(path: Path | None, *, horizon: str) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    mapping = {
        "baseline": "official",
        "joint3": "joint",
        "globalft50": "global",
        "random": "random_voronoi",
        "kmeans": "kmeanspp",
        "spectral": "spectral",
        "autolap": "autolap",
        "rooms3": "rooms3",
        "official_sub-jepa": "official",
    }
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "horizon" in row and row["horizon"]:
                if _normalize_horizon(row["horizon"]) != horizon:
                    continue
            raw = (row.get("method_id") or row.get("method") or "").strip().lower()
            raw = raw.replace("\n", " ").replace("  ", " ")
            mid = mapping.get(raw.replace(" ", "_"), raw.replace(" ", "_"))
            if raw in mapping:
                mid = mapping[raw]
            elif raw.startswith("official"):
                mid = "official"
            elif "joint" in raw:
                mid = "joint"
            elif "global" in raw:
                mid = "global"
            elif "random" in raw:
                mid = "random_voronoi"
            elif "k-means" in raw or "kmeans" in raw:
                mid = "kmeanspp"
            elif "spectral" in raw:
                mid = "spectral"
            elif "auto" in raw:
                mid = "autolap"
            elif "rooms3" in raw or "human" in raw:
                mid = "rooms3"
            val = row.get("mean_percent")
            if val and val not in {"NA", ""}:
                out[mid] = float(val)
    return out


def _resolve_eval_root(cell_cfg: dict[str, Any], horizon: str, repo_root: Path) -> Path | None:
    loader = cell_cfg["loader"]
    if loader == "tworoom_results":
        rel = cell_cfg.get("results_root")
        return repo_root / rel if rel else None
    if loader == "matrix_eval":
        rel = cell_cfg.get("matrix_roots", {}).get(horizon)
        if not rel:
            return None
        root = repo_root / rel
        eval_dir = root / "eval"
        return eval_dir if eval_dir.is_dir() else root
    if loader == "subjepa_matrix":
        return repo_root / cell_cfg["matrix_root"] / f"eval_{horizon}"
    return None


def _cell_pending(cell_cfg: dict[str, Any], eval_root: Path | None) -> bool:
    if not cell_cfg.get("pending_if_roots_missing"):
        return False
    if eval_root is None or not eval_root.is_dir():
        return True
    return not any(eval_root.rglob("results.json"))


def _partition_policy(method_id: str, gate_info: dict[str, Any]) -> str:
    if method_id == "official":
        return "none"
    if method_id == "autolap":
        return "deployment" if gate_info.get("branch") == "spectral" else "none"
    if method_id in PARTITION_METHODS:
        return "average"
    return "none"


def load_cell(
    *,
    repo_root: Path,
    config: dict[str, Any],
    model: str,
    task: str,
    horizon: str,
) -> CellData:
    cell_cfg = config["cells"][model][task]
    train_seeds = tuple(config["train_seeds"])
    partition_seeds = tuple(config["partition_seeds"])
    eval_seeds = tuple(config["eval_seeds"])
    goal_offset = config["horizons"][horizon]["goal_offset_steps"]
    eval_root = _resolve_eval_root(cell_cfg, horizon, repo_root)

    if _cell_pending(cell_cfg, eval_root):
        return CellData(
            model=model,
            task=task,
            horizon=horizon,
            status="pending",
            methods={},
            train_seeds=train_seeds,
            partition_seeds=partition_seeds,
            eval_seeds=eval_seeds,
            goal_offset_steps=goal_offset,
            gate_info={},
            validation={"reason": "matrix root missing or empty"},
            reference_estimates={},
            episodes_per_block=config["num_eval"],
            has_episode_data=False,
        )

    gate_path = cell_cfg.get("gate_manifest")
    gate_info = load_gate_manifest(repo_root / gate_path if gate_path else None)
    deployment_seed = int(gate_info.get("deployment_seed", 0))

    ref_path = cell_cfg.get("reference_summary")
    if isinstance(ref_path, dict):
        ref_path = ref_path.get(horizon)
    reference = _load_reference_estimates(
        repo_root / ref_path if ref_path else None,
        horizon=horizon,
    )

    methods: dict[str, MethodData] = {}
    pairing_digests: dict[int, set[str]] = {es: set() for es in eval_seeds}
    validation_issues: list[str] = []

    for method_id in cell_cfg.get("main_methods", []):
        pp = _partition_policy(method_id, gate_info)
        try:
            if cell_cfg["loader"] == "tworoom_results":
                runs = _collect_tworoom_results(
                    repo_root / cell_cfg["results_root"],
                    method_id=method_id,
                    horizon=horizon,
                    train_seeds=train_seeds,
                    partition_seeds=partition_seeds,
                    eval_seeds=eval_seeds,
                    goal_offset=goal_offset,
                    partition_policy=pp,
                    deployment_seed=deployment_seed,
                    repo_root=repo_root,
                )
            else:
                autolap_root = None
                if cell_cfg["loader"] == "subjepa_matrix":
                    matrix_root = repo_root / cell_cfg["matrix_root"]
                    autolap_root = matrix_root / "eval" if (matrix_root / "eval").is_dir() else matrix_root
                runs = _collect_matrix_eval(
                    eval_root,  # type: ignore[arg-type]
                    method_id=method_id,
                    train_seeds=train_seeds,
                    partition_seeds=partition_seeds,
                    eval_seeds=eval_seeds,
                    goal_offset=goal_offset,
                    partition_policy=pp,
                    deployment_seed=deployment_seed,
                    gate_info=gate_info,
                    autolap_eval_root=autolap_root,
                )
        except (FileNotFoundError, ValueError) as exc:
            validation_issues.append(f"{method_id}: {exc}")
            continue

        official = method_id == "official"
        blocks, episodes, files, n_part = _aggregate_runs(
            runs,
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            official=official,
            partition_policy=pp,
            deployment_seed=deployment_seed,
            partition_seeds=partition_seeds,
        )

        for rec in runs.values():
            if rec.eval_start_indices is not None:
                pairing_digests[rec.eval_seed].add(_starts_digest(rec.eval_start_indices))

        methods[method_id] = MethodData(
            method_id=method_id,
            label=_method_label(config, model, method_id),
            official=official,
            partition_policy=pp,
            deployment_seed=deployment_seed if method_id == "autolap" else None,
            blocks=blocks,
            episodes=episodes,
            files_read=files,
            n_partition_seeds=n_part,
        )

    pairing_issues = [
        f"eval_seed {es}: {len(d)} distinct eval_start_indices digests"
        for es, d in pairing_digests.items()
        if len(d) > 1
    ]
    validation_issues.extend(pairing_issues)

    for mid, mdata in methods.items():
        est = point_estimate(mdata.blocks, official=mdata.official)
        ref = reference.get(mid)
        if ref is not None and abs(est - ref) > 0.01:
            validation_issues.append(
                f"{mid}: recomputed {est:.4f} vs reference {ref:.4f} "
                f"(diff {abs(est - ref):.4f} pp)"
            )

    status = "ok"
    if not methods:
        status = "failed"
    elif pairing_issues:
        status = "incomplete"

    return CellData(
        model=model,
        task=task,
        horizon=horizon,
        status=status,
        methods=methods,
        train_seeds=train_seeds,
        partition_seeds=partition_seeds,
        eval_seeds=eval_seeds,
        goal_offset_steps=goal_offset,
        gate_info=gate_info,
        validation={
            "issues": validation_issues,
            "pairing_by_eval_seed": {str(k): len(v) for k, v in pairing_digests.items()},
        },
        reference_estimates=reference,
        episodes_per_block=config["num_eval"],
        has_episode_data=any(m.episodes is not None for m in methods.values()),
    )
