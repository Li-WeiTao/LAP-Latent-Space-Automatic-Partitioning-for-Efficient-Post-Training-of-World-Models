#!/usr/bin/env python3
"""Statically inspect GLOBAL references inside a PyTorch zip checkpoint pickle."""

from __future__ import annotations

import argparse
import importlib
import json
import pickletools
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GlobalRef:
    module: str
    name: str
    opcode: str
    position: int
    count: int


def _parse_global_arg(arg: Any) -> tuple[str, str] | None:
    if isinstance(arg, tuple) and len(arg) == 2:
        return str(arg[0]), str(arg[1])
    if isinstance(arg, str):
        parts = arg.rsplit(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return None


def _resolve_pickle_path(zip_file: zipfile.ZipFile) -> str:
    candidates = [name for name in zip_file.namelist() if name.endswith(".pkl")]
    if not candidates:
        raise FileNotFoundError("checkpoint zip contains no .pkl files")
    if len(candidates) == 1:
        return candidates[0]
    preferred = [name for name in candidates if name.endswith("/data.pkl")]
    if len(preferred) == 1:
        return preferred[0]
    raise ValueError(f"multiple pickle files found: {candidates}")


def _import_status(module: str, name: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "module": module,
        "name": name,
        "module_importable": False,
        "attribute_exists": False,
        "resolved_type": None,
        "import_error": None,
    }
    try:
        mod = importlib.import_module(module)
        record["module_importable"] = True
        if hasattr(mod, name):
            record["attribute_exists"] = True
            obj = getattr(mod, name)
            record["resolved_type"] = f"{type(obj).__module__}.{type(obj).__qualname__}"
    except Exception as exc:  # noqa: BLE001 - audit must capture all failures
        record["import_error"] = repr(exc)
    return record


def inspect_checkpoint(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"checkpoint missing or empty: {checkpoint}")

    with zipfile.ZipFile(checkpoint) as archive:
        pickle_name = _resolve_pickle_path(archive)
        payload = archive.read(pickle_name)
        is_zip = True

    counter: Counter[tuple[str, str, str, int]] = Counter()
    for opcode, arg, position in pickletools.genops(payload):
        if opcode.name not in {"GLOBAL", "STACK_GLOBAL"}:
            continue
        parsed = _parse_global_arg(arg)
        if parsed is None:
            continue
        module, name = parsed
        counter[(module, name, opcode.name, position)] += 1

    grouped: Counter[tuple[str, str]] = Counter()
    entries: list[dict[str, Any]] = []
    for (module, name, opcode, position), count in sorted(counter.items()):
        grouped[(module, name)] += count
        entries.append(
            asdict(
                GlobalRef(
                    module=module,
                    name=name,
                    opcode=opcode,
                    position=position,
                    count=count,
                )
            )
        )

    statuses = [_import_status(module, name) for module, name in sorted(grouped)]
    missing = [item for item in statuses if not item["attribute_exists"]]
    importable = [item for item in statuses if item["attribute_exists"]]

    return {
        "checkpoint": str(checkpoint),
        "is_zip_archive": is_zip,
        "pickle_member": pickle_name,
        "unique_globals": len(grouped),
        "globals": entries,
        "grouped_globals": [
            {"module": module, "name": name, "count": grouped[(module, name)]}
            for module, name in sorted(grouped)
        ],
        "importable_globals": importable,
        "missing_globals": missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    missing_path = args.output.with_name("missing_globals.json")
    importable_path = args.output.with_name("importable_globals.json")
    missing_path.write_text(
        json.dumps(report["missing_globals"], indent=2) + "\n", encoding="utf-8"
    )
    importable_path.write_text(
        json.dumps(report["importable_globals"], indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "unique_globals": report["unique_globals"],
                "missing": len(report["missing_globals"]),
            },
            indent=2,
        )
    )
    if report["missing_globals"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
