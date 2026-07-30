from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TWOROOM = ROOT / "experiments" / "tworoom"


def load_reproduce_module():
    path = TWOROOM / "reproduce.py"
    spec = importlib.util.spec_from_file_location("tworoom_reproduce", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_is_complete_and_entrypoints_exist():
    module = load_reproduce_module()
    payload = module.load_manifest()
    experiments = payload["experiments"]
    assert {item["tier"] for item in experiments} >= {
        "canonical",
        "analysis",
        "ablation",
        "validation",
    }
    assert module.preflight(experiments, check_environment=False) == []


def test_main_profile_has_single_canonical_entrypoint():
    payload = json.loads(
        (TWOROOM / "reproduction_manifest.json").read_text(encoding="utf-8")
    )
    assert payload["profiles"]["main"] == ["main_matrix"]
    by_id = {item["id"]: item for item in payload["experiments"]}
    assert by_id["main_matrix"]["tier"] == "canonical"


def test_canonical_shell_uses_parameterized_data_and_checkpoint():
    script = (TWOROOM / "scripts" / "canonical" / "run_tworoom_main_matrix.sh").read_text(
        encoding="utf-8"
    )
    assert "LAP_TWOROOM_DATA" in script
    assert "LAP_LEWM_CHECKPOINT" in script
    assert "/data/sicong" not in script


def test_script_tree_is_physically_partitioned():
    scripts = TWOROOM / "scripts"
    assert not list(scripts.glob("*.sh"))
    assert not list(scripts.glob("*.R"))
    for directory in ("canonical", "analysis", "ablations", "internal", "legacy"):
        assert (scripts / directory).is_dir()


def test_registered_shell_commands_match_their_physical_tiers():
    payload = json.loads(
        (TWOROOM / "reproduction_manifest.json").read_text(encoding="utf-8")
    )
    tier_directories = {
        "canonical": "canonical",
        "analysis": "analysis",
        "ablation": "ablations",
    }
    for experiment in payload["experiments"]:
        command = experiment["command"]
        directory = tier_directories.get(experiment["tier"])
        if directory is not None and command.startswith("bash "):
            assert f"/scripts/{directory}/" in command
