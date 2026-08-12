#!/usr/bin/env python3
"""Resolve generic JEPA matrix configuration with explicit precedence rules."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.control_matrix.backend_registry import (  # noqa: E402
    DEFAULT_MODEL_FAMILY,
    backend_metadata,
    normalize_model_family,
)
from experiments.control_matrix.task_spec import load_task_spec  # noqa: E402


CONFLICT_FIELDS = (
    "task_name",
    "dataset_name",
    "eval_config_name",
    "frameskip",
    "short_goal_offset",
    "long_goal_offset",
    "eval_budget",
)


@dataclass
class ResolvedMatrixConfig:
    model_family: str
    implementation_backend: str
    task_name: str
    dataset_name: str
    dataset: str | None
    checkpoint: str | None
    eval_config_name: str
    eval_config: str | None
    eval_dataset_name: str
    work_root: str
    cache_dir: str
    paired_start_root: str | None
    paired_start_root_short: str | None
    paired_start_root_long: str | None
    phase: str
    frameskip: int
    history_size: int
    num_preds: int
    img_size: int
    short_goal_offset: int
    long_goal_offset: int
    eval_budget: int
    num_eval: int
    max_train_starts: int
    dry_run: bool
    task_spec_path: str | None
    train_seeds: str
    partition_seeds: str
    eval_seeds: str
    methods: str
    skip_joint: bool
    python: str
    cpu_threads: int
    gpu_id: str
    sources: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _source(name: str, value: Any, sources: dict[str, str]) -> Any:
    sources[name] = sources.get(name, "default")
    return value


def _assign(
    name: str,
    value: Any,
    *,
    cli_value: Any,
    env_value: Any,
    spec_value: Any,
    default_value: Any,
    sources: dict[str, str],
) -> Any:
    if cli_value is not None and cli_value != "":
        sources[name] = "cli"
        return cli_value
    if env_value is not None and env_value != "":
        sources[name] = "env"
        return env_value
    if spec_value is not None and spec_value != "":
        sources[name] = "task_spec"
        return spec_value
    sources[name] = "default"
    return default_value


def _collect_conflicts(
    *,
    cli_map: dict[str, Any],
    env_map: dict[str, Any],
    spec_map: dict[str, Any],
) -> list[str]:
    conflicts: list[str] = []
    for key in CONFLICT_FIELDS:
        values = {
            "cli": cli_map.get(key),
            "env": env_map.get(key),
            "task_spec": spec_map.get(key),
        }
        present = {
            label: value
            for label, value in values.items()
            if value is not None and value != ""
        }
        if len({str(value) for value in present.values()}) > 1:
            conflicts.append(f"{key}: {present}")
    return conflicts


def resolve_config(args: argparse.Namespace) -> ResolvedMatrixConfig:
    sources: dict[str, str] = {}
    spec = load_task_spec(args.task_spec) if args.task_spec else {}

    cli_map = {
        "task_name": args.task_name,
        "dataset_name": args.dataset_name,
        "eval_config_name": args.eval_config_name,
        "frameskip": args.frameskip,
        "short_goal_offset": args.short_goal_offset,
        "long_goal_offset": args.long_goal_offset,
        "eval_budget": args.eval_budget,
    }
    env_map = {
        "task_name": os.environ.get("DATASET_NAME"),
        "dataset_name": os.environ.get("DATASET_NAME"),
        "eval_config_name": os.environ.get("EVAL_CONFIG"),
        "frameskip": os.environ.get("FRAMESKIP"),
        "short_goal_offset": os.environ.get("GOAL_OFFSET"),
        "long_goal_offset": os.environ.get("LONG_GOAL_OFFSET"),
        "eval_budget": os.environ.get("EVAL_BUDGET"),
    }
    spec_map = {
        "task_name": spec.get("task_name"),
        "dataset_name": spec.get("dataset_name"),
        "eval_config_name": spec.get("eval_config_name"),
        "frameskip": spec.get("frameskip"),
        "short_goal_offset": spec.get("short_goal_offset"),
        "long_goal_offset": spec.get("long_goal_offset"),
        "eval_budget": spec.get("eval_budget"),
    }
    conflicts = _collect_conflicts(cli_map=cli_map, env_map=env_map, spec_map=spec_map)
    if conflicts:
        raise ValueError(
            "conflicting task parameters across CLI/env/task-spec:\n"
            + "\n".join(f"  - {item}" for item in conflicts)
        )

    model_family = normalize_model_family(
        _assign(
            "model_family",
            args.model_family,
            cli_value=args.model_family,
            env_value=os.environ.get("MODEL_FAMILY"),
            spec_value=None,
            default_value=DEFAULT_MODEL_FAMILY,
            sources=sources,
        )
    )
    meta = backend_metadata(model_family)

    dataset_name = str(
        _assign(
            "dataset_name",
            None,
            cli_value=args.task_name or args.dataset_name,
            env_value=os.environ.get("DATASET_NAME"),
            spec_value=spec.get("dataset_name"),
            default_value="pusht",
            sources=sources,
        )
    )
    task_name = str(
        _assign(
            "task_name",
            None,
            cli_value=args.task_name,
            env_value=os.environ.get("DATASET_NAME"),
            spec_value=spec.get("task_name"),
            default_value=dataset_name,
            sources=sources,
        )
    )

    eval_config_name = str(
        _assign(
            "eval_config_name",
            None,
            cli_value=args.eval_config_name,
            env_value=os.environ.get("EVAL_CONFIG"),
            spec_value=spec.get("eval_config_name"),
            default_value=dataset_name,
            sources=sources,
        )
    )
    eval_dataset_name = str(
        _assign(
            "eval_dataset_name",
            None,
            cli_value=None,
            env_value=os.environ.get("EVAL_DATASET_NAME"),
            spec_value=spec.get("eval_dataset_name"),
            default_value=f"{dataset_name}_expert_train"
            if dataset_name == "pusht"
            else dataset_name,
            sources=sources,
        )
    )

    work_root = str(
        _assign(
            "work_root",
            None,
            cli_value=args.work_root,
            env_value=os.environ.get("WORK_ROOT"),
            spec_value=None,
            default_value=f"experiments/{dataset_name}/matrix",
            sources=sources,
        )
    )
    cache_dir = str(
        _assign(
            "cache_dir",
            None,
            cli_value=args.cache_dir,
            env_value=os.environ.get("CACHE_DIR"),
            spec_value=None,
            default_value=str(Path.home() / ".stable_worldmodel"),
            sources=sources,
        )
    )

    dataset = _first_non_empty(
        args.dataset,
        os.environ.get("DATA_FILE"),
    )
    checkpoint = _first_non_empty(
        args.checkpoint,
        os.environ.get("CHECKPOINT"),
    )
    eval_config = _first_non_empty(
        args.eval_config,
        os.environ.get("EVAL_CONFIG_PATH"),
    )

    phase = str(
        _assign(
            "phase",
            None,
            cli_value=args.phase,
            env_value=os.environ.get("PHASE"),
            spec_value=None,
            default_value="all",
            sources=sources,
        )
    )

    skip_joint_default = model_family == "subjepa"
    skip_joint_env = os.environ.get("SKIP_JOINT")
    if args.skip_joint is not None:
        skip_joint = bool(args.skip_joint)
        sources["skip_joint"] = "cli"
    elif skip_joint_env is not None and skip_joint_env != "":
        skip_joint = skip_joint_env == "1"
        sources["skip_joint"] = "env"
    else:
        skip_joint = skip_joint_default
        sources["skip_joint"] = "default"

    return ResolvedMatrixConfig(
        model_family=model_family,
        implementation_backend=meta["implementation_backend"],
        task_name=task_name,
        dataset_name=dataset_name,
        dataset=dataset,
        checkpoint=checkpoint,
        eval_config_name=eval_config_name,
        eval_config=eval_config,
        eval_dataset_name=eval_dataset_name,
        work_root=work_root,
        cache_dir=cache_dir,
        paired_start_root=_first_non_empty(
            args.paired_start_root, os.environ.get("PAIRED_START_ROOT")
        ),
        paired_start_root_short=_first_non_empty(
            args.paired_start_root_short, os.environ.get("PAIRED_START_ROOT_SHORT")
        ),
        paired_start_root_long=_first_non_empty(
            args.paired_start_root_long, os.environ.get("PAIRED_START_ROOT_LONG")
        ),
        phase=phase,
        frameskip=int(
            _assign(
                "frameskip",
                None,
                cli_value=args.frameskip,
                env_value=os.environ.get("FRAMESKIP"),
                spec_value=spec.get("frameskip"),
                default_value=5,
                sources=sources,
            )
        ),
        history_size=int(spec.get("history_size", 3)),
        num_preds=int(spec.get("num_preds", 1)),
        img_size=int(spec.get("img_size", 224)),
        short_goal_offset=int(
            _assign(
                "short_goal_offset",
                None,
                cli_value=args.short_goal_offset,
                env_value=os.environ.get("SHORT_GOAL_OFFSET") or os.environ.get("GOAL_OFFSET"),
                spec_value=spec.get("short_goal_offset"),
                default_value=25,
                sources=sources,
            )
        ),
        long_goal_offset=int(
            _assign(
                "long_goal_offset",
                None,
                cli_value=args.long_goal_offset,
                env_value=os.environ.get("LONG_GOAL_OFFSET"),
                spec_value=spec.get("long_goal_offset"),
                default_value=50,
                sources=sources,
            )
        ),
        eval_budget=int(
            _assign(
                "eval_budget",
                None,
                cli_value=args.eval_budget,
                env_value=os.environ.get("EVAL_BUDGET"),
                spec_value=spec.get("eval_budget"),
                default_value=50,
                sources=sources,
            )
        ),
        num_eval=int(
            _assign(
                "num_eval",
                None,
                cli_value=args.num_eval,
                env_value=os.environ.get("NUM_EVAL"),
                spec_value=spec.get("num_eval"),
                default_value=50,
                sources=sources,
            )
        ),
        max_train_starts=int(args.max_train_starts or os.environ.get("PREPARE_MAX_STARTS") or 0),
        dry_run=bool(args.dry_run),
        task_spec_path=str(Path(args.task_spec).resolve()) if args.task_spec else None,
        train_seeds=str(
            _assign(
                "train_seeds",
                None,
                cli_value=args.train_seeds,
                env_value=os.environ.get("TRAIN_SEEDS"),
                spec_value=None,
                default_value="0,42,625",
                sources=sources,
            )
        ),
        partition_seeds=str(
            _assign(
                "partition_seeds",
                None,
                cli_value=args.partition_seeds,
                env_value=os.environ.get("PARTITION_SEEDS"),
                spec_value=None,
                default_value="0,1,2",
                sources=sources,
            )
        ),
        eval_seeds=str(
            _assign(
                "eval_seeds",
                None,
                cli_value=args.eval_seeds,
                env_value=os.environ.get("EVAL_SEEDS"),
                spec_value=None,
                default_value="0,1,2,3,4",
                sources=sources,
            )
        ),
        methods=str(
            _assign(
                "methods",
                None,
                cli_value=args.methods,
                env_value=os.environ.get("METHODS"),
                spec_value=None,
                default_value="random_voronoi,kmeanspp,spectral",
                sources=sources,
            )
        ),
        skip_joint=skip_joint,
        python=str(
            _assign(
                "python",
                None,
                cli_value=args.python,
                env_value=os.environ.get("PYTHON"),
                spec_value=None,
                default_value=sys.executable,
                sources=sources,
            )
        ),
        cpu_threads=int(
            _assign(
                "cpu_threads",
                None,
                cli_value=args.cpu_threads,
                env_value=os.environ.get("CPU_THREADS"),
                spec_value=None,
                default_value=4,
                sources=sources,
            )
        ),
        gpu_id=str(
            _assign(
                "gpu_id",
                None,
                cli_value=args.gpu_id,
                env_value=os.environ.get("GPU_ID"),
                spec_value=None,
                default_value="",
                sources=sources,
            )
        ),
        sources=sources,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", default=None)
    parser.add_argument("--task-spec", type=Path, default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--encoder-config", default=None)
    parser.add_argument("--eval-config-name", default=None)
    parser.add_argument("--eval-config", default=None)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--paired-start-root", default=None)
    parser.add_argument("--paired-start-root-short", default=None)
    parser.add_argument("--paired-start-root-long", default=None)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--frameskip", type=int, default=None)
    parser.add_argument("--short-goal-offset", type=int, default=None)
    parser.add_argument("--long-goal-offset", type=int, default=None)
    parser.add_argument("--eval-budget", type=int, default=None)
    parser.add_argument("--num-eval", type=int, default=None)
    parser.add_argument("--max-train-starts", type=int, default=0)
    parser.add_argument("--train-seeds", default=None)
    parser.add_argument("--partition-seeds", default=None)
    parser.add_argument("--eval-seeds", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--skip-joint", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--emit-shell",
        action="store_true",
        help="Print shell export statements for resolved configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for resolved configuration.",
    )
    return parser.parse_args(argv)


def emit_shell(resolved: ResolvedMatrixConfig) -> None:
    mapping = resolved.to_json()
    for key, value in mapping.items():
        if key == "sources":
            continue
        env_key = key.upper()
        if isinstance(value, bool):
            print(f"export {env_key}={'1' if value else '0'}")
        elif value is None:
            print(f"unset {env_key} 2>/dev/null || true")
        else:
            escaped = str(value).replace("'", "'\\''")
            print(f"export {env_key}='{escaped}'")
    print(f"export DATASET_NAME='{resolved.dataset_name}'")
    print(f"export DATA_FILE='{resolved.dataset}'")
    print(f"export CHECKPOINT='{resolved.checkpoint}'")
    print(f"export EVAL_CONFIG='{resolved.eval_config_name}'")
    if resolved.eval_config:
        print(f"export EVAL_CONFIG_PATH='{resolved.eval_config}'")
    print(f"export MODEL_FAMILY='{resolved.model_family}'")
    print(f"export IMPLEMENTATION_BACKEND='{resolved.implementation_backend}'")
    print(f"export PHASE='{resolved.phase}'")
    print(f"export SKIP_JOINT={'1' if resolved.skip_joint else '0'}")
    print(f"export FRAMESKIP='{resolved.frameskip}'")
    print(f"export HISTORY_SIZE='{resolved.history_size}'")
    print(f"export NUM_PREDS='{resolved.num_preds}'")
    print(f"export IMG_SIZE='{resolved.img_size}'")
    print(f"export SHORT_GOAL_OFFSET='{resolved.short_goal_offset}'")
    print(f"export LONG_GOAL_OFFSET='{resolved.long_goal_offset}'")
    print(f"export NUM_EVAL='{resolved.num_eval}'")
    print(f"export EVAL_DATASET_NAME='{resolved.eval_dataset_name}'")
    print(f"export DRY_RUN={'1' if resolved.dry_run else '0'}")
    print(f"export PREPARE_MAX_STARTS='{resolved.max_train_starts}'")


def main() -> None:
    args = parse_args()
    resolved = resolve_config(args)
    payload = resolved.to_json()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.emit_shell:
        emit_shell(resolved)
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
