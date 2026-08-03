"""Strict validator for version-controlled JEPA matrix task specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "task_name",
        "dataset_name",
        "eval_config_name",
        "eval_dataset_name",
        "frameskip",
        "history_size",
        "num_preds",
        "img_size",
        "short_goal_offset",
        "long_goal_offset",
        "eval_budget",
        "num_eval",
        "plan_horizon",
        "plan_receding_horizon",
        "plan_action_block",
    }
)

REQUIRED_KEYS = ALLOWED_KEYS - {"schema_version"}


def load_task_spec(path: Path | str) -> dict[str, Any]:
    spec_path = Path(path).resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"task spec must be a JSON object: {spec_path}")
    return validate_task_spec(payload, source=str(spec_path))


def validate_task_spec(payload: dict[str, Any], *, source: str = "<task-spec>") -> dict[str, Any]:
    unknown = set(payload) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"{source}: unknown fields: {sorted(unknown)}")

    missing = REQUIRED_KEYS - set(payload)
    if missing:
        raise ValueError(f"{source}: missing required fields: {sorted(missing)}")

    schema_version = int(payload.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(f"{source}: unsupported schema_version={schema_version}")

    for key in (
        "task_name",
        "dataset_name",
        "eval_config_name",
        "eval_dataset_name",
    ):
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source}: {key} must be a non-empty string")

    short_goal = int(payload["short_goal_offset"])
    long_goal = int(payload["long_goal_offset"])
    if short_goal <= 0 or long_goal <= 0:
        raise ValueError(f"{source}: goal offsets must be positive")
    if short_goal == long_goal:
        raise ValueError(f"{source}: short_goal_offset and long_goal_offset must differ")

    for key in ("eval_budget", "num_eval", "frameskip", "history_size", "num_preds"):
        value = int(payload[key])
        if value <= 0:
            raise ValueError(f"{source}: {key} must be positive")

    for key in (
        "img_size",
        "plan_horizon",
        "plan_receding_horizon",
        "plan_action_block",
    ):
        value = int(payload[key])
        if value <= 0:
            raise ValueError(f"{source}: {key} must be positive")

    return dict(payload)
