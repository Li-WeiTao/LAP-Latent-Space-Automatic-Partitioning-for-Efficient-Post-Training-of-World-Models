#!/usr/bin/env python3
"""List, preflight, or run the audited TwoRoom reproduction profiles."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("reproduction_manifest.json")


def load_manifest() -> dict:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = [item["id"] for item in payload["experiments"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate experiment ids in reproduction manifest")
    known = set(ids)
    for profile, members in payload["profiles"].items():
        missing = set(members) - known
        if missing:
            raise ValueError(f"profile {profile} references unknown ids: {sorted(missing)}")
    return payload


def experiment_map(payload: dict) -> dict[str, dict]:
    return {item["id"]: item for item in payload["experiments"]}


def selected(payload: dict, profile: str | None, ids: list[str]) -> list[dict]:
    by_id = experiment_map(payload)
    requested = list(payload["profiles"][profile]) if profile else []
    requested.extend(ids)
    if not requested:
        requested = list(payload["profiles"]["main"])
    unknown = [item for item in requested if item not in by_id]
    if unknown:
        raise SystemExit(f"unknown experiment ids: {', '.join(unknown)}")
    seen: set[str] = set()
    return [by_id[item] for item in requested if not (item in seen or seen.add(item))]


def command_entrypoint(command: str) -> Path | None:
    parts = shlex.split(command, posix=True)
    candidates = [part for part in parts if part.startswith("experiments/")]
    return ROOT / candidates[0] if candidates else None


def preflight(
    items: list[dict], *, require_outputs: bool = False, check_environment: bool = True
) -> list[str]:
    errors: list[str] = []
    for item in items:
        entrypoint = command_entrypoint(item["command"])
        if entrypoint is not None and not entrypoint.is_file():
            errors.append(f"{item['id']}: missing entrypoint {entrypoint.relative_to(ROOT)}")
        if check_environment:
            for name in item.get("required_env", []):
                value = os.environ.get(name)
                if not value:
                    errors.append(f"{item['id']}: missing environment variable {name}")
                elif name in {"LAP_TWOROOM_DATA", "LAP_LEWM_CHECKPOINT"} and not Path(value).is_file():
                    errors.append(f"{item['id']}: {name} does not exist: {value}")
        if require_outputs:
            for pattern in item.get("outputs", []):
                if not list(ROOT.glob(pattern)):
                    errors.append(f"{item['id']}: no committed output matches {pattern}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "check", "run"))
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--profile", choices=("main", "analysis", "ablations", "full"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-outputs", action="store_true")
    args = parser.parse_args()

    payload = load_manifest()
    items = selected(payload, args.profile, args.ids)
    if args.action == "list":
        for item in items:
            print(f"{item['id']:24s} {item['tier']:10s} {item['description']}")
        return

    errors = preflight(items, require_outputs=args.require_outputs)
    if errors:
        print("TwoRoom reproduction preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(2)
    print(f"preflight passed for {len(items)} experiment(s)")
    if args.action == "check":
        return

    for item in items:
        print(f"[{item['id']}] {item['command']}", flush=True)
        if not args.dry_run:
            subprocess.run(item["command"], cwd=ROOT, shell=True, check=True)


if __name__ == "__main__":
    main()
