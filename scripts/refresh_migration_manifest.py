#!/usr/bin/env python3
"""Refresh file hashes after an intentional LAP-side reproducibility repair."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MIGRATION_MANIFEST.json"


def repository_files() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    )
    return sorted(
        path
        for path in result.stdout.splitlines()
        if path and path != MANIFEST.name
    )


def main() -> None:
    current_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    try:
        committed_bytes = subprocess.check_output(
            ["git", "show", "HEAD:MIGRATION_MANIFEST.json"], cwd=ROOT
        )
        committed_payload = json.loads(committed_bytes)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        committed_payload = current_payload

    # The committed migration manifest retains source_path/source_sha256 and
    # relation fields for migrated files.  Always use it as the provenance
    # base, then merge LAP-side files already added by an earlier refresh.
    payload = committed_payload
    committed = {entry["path"]: entry for entry in committed_payload["files"]}
    current = {entry["path"]: entry for entry in current_payload["files"]}
    files = []
    changed = 0
    added = 0
    repair_changed = 0
    repair_added = 0
    for relative in repository_files():
        path = ROOT / relative
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        previous = current.get(relative) or committed.get(relative)
        committed_entry = committed.get(relative)
        if committed_entry is None:
            repair_added += 1
        elif (
            committed_entry["sha256"] != digest
            or int(committed_entry["bytes"]) != len(data)
        ):
            repair_changed += 1
        if previous is None:
            origin = "generated_for_lap"
            added += 1
        elif previous["sha256"] == digest and int(previous["bytes"]) == len(data):
            origin = previous["origin"]
        else:
            origin = (
                "migrated_repaired_for_lap"
                if previous["origin"] == "migrated"
                else previous["origin"]
            )
            changed += 1
        entry = dict(committed.get(relative, previous or {}))
        entry.update(
            {
                "bytes": len(data),
                "origin": origin,
                "path": relative.replace("\\", "/"),
                "sha256": digest,
            }
        )
        if origin == "migrated_repaired_for_lap":
            entry["relation"] = "adapted_for_independent_repo"
        files.append(entry)

    payload["files"] = files
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["repair_revision"] = {
        "purpose": "close independent-repository end-to-end reproducibility gaps",
        "changed_existing_files": repair_changed,
        "new_files": repair_added,
    }
    # Write bytes so this provenance artifact is stable on Windows and Linux.
    MANIFEST.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    print(f"Refreshed {len(files)} entries ({changed} changed, {added} new)")


if __name__ == "__main__":
    main()
