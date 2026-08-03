"""Unit tests for Sub-JEPA integration scaffolding."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from experiments.control_matrix.backend_registry import (
    DEFAULT_MODEL_FAMILY,
    IMPLEMENTATION_BACKEND,
    manifest_model_family,
    normalize_model_family,
    resolve_backend_bundle,
)
from experiments.control_matrix.resolve_jepa_matrix_config import resolve_config
from experiments.control_matrix.task_spec import load_task_spec, validate_task_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


class BackendRegistryTest(unittest.TestCase):
    def test_defaults_to_lewm(self):
        self.assertEqual(normalize_model_family(None), "lewm")
        self.assertEqual(manifest_model_family({}), "lewm")

    def test_subjepa_uses_same_implementation(self):
        bundle = resolve_backend_bundle("subjepa")
        self.assertEqual(bundle.implementation_backend, IMPLEMENTATION_BACKEND)
        self.assertEqual(bundle.model_family, "subjepa")

    def test_legacy_manifest_without_model_family(self):
        self.assertEqual(manifest_model_family({"checkpoint": "x.ckpt"}), "lewm")


class TaskSpecValidatorTest(unittest.TestCase):
    def test_committed_tworoom_spec(self):
        spec = load_task_spec(REPO_ROOT / "configs/experiments/tasks/tworoom.json")
        self.assertEqual(spec["short_goal_offset"], 25)
        self.assertEqual(spec["long_goal_offset"], 50)
        self.assertEqual(spec["eval_budget"], 50)

    def test_committed_pusht_spec(self):
        spec = load_task_spec(REPO_ROOT / "configs/experiments/tasks/pusht.json")
        self.assertEqual(spec["short_goal_offset"], 25)
        self.assertEqual(spec["long_goal_offset"], 50)

    def test_rejects_unknown_fields(self):
        payload = json.loads(
            (REPO_ROOT / "configs/experiments/tasks/pusht.json").read_text(encoding="utf-8")
        )
        payload["checkpoint"] = "/tmp/forbidden.ckpt"
        with self.assertRaises(ValueError):
            validate_task_spec(payload)

    def test_rejects_equal_horizons(self):
        payload = json.loads(
            (REPO_ROOT / "configs/experiments/tasks/pusht.json").read_text(encoding="utf-8")
        )
        payload["long_goal_offset"] = payload["short_goal_offset"]
        with self.assertRaises(ValueError):
            validate_task_spec(payload)


class ConfigPrecedenceTest(unittest.TestCase):
    def test_cli_overrides_task_spec(self):
        spec = REPO_ROOT / "configs/experiments/tasks/tworoom.json"
        resolved = resolve_config(
            Namespace(
                model_family="subjepa",
                task_spec=spec,
                task_name=None,
                dataset_name=None,
                dataset="/tmp/data.h5",
                checkpoint="/tmp/model.ckpt",
                dataset_config=None,
                encoder_config=None,
                eval_config_name=None,
                eval_config=None,
                work_root="experiments/tworoom/subjepa",
                cache_dir="/tmp/cache",
                paired_start_root=None,
                paired_start_root_short=None,
                paired_start_root_long=None,
                phase="smoke",
                frameskip=None,
                short_goal_offset=None,
                long_goal_offset=None,
                eval_budget=None,
                max_train_starts=4096,
                train_seeds=None,
                partition_seeds=None,
                eval_seeds=None,
                methods=None,
                skip_joint=None,
                python=None,
                cpu_threads=None,
                gpu_id=None,
                dry_run=False,
            )
        )
        self.assertEqual(resolved.model_family, "subjepa")
        self.assertEqual(resolved.short_goal_offset, 25)
        self.assertEqual(resolved.max_train_starts, 4096)

    def test_conflicting_cli_and_spec_fail(self):
        spec = REPO_ROOT / "configs/experiments/tasks/tworoom.json"
        with self.assertRaises(ValueError):
            resolve_config(
                Namespace(
                    model_family=None,
                    task_spec=spec,
                    task_name="other",
                    dataset_name=None,
                    dataset=None,
                    checkpoint=None,
                    dataset_config=None,
                    encoder_config=None,
                    eval_config_name=None,
                    eval_config=None,
                    work_root=None,
                    cache_dir=None,
                    paired_start_root=None,
                    paired_start_root_short=None,
                    paired_start_root_long=None,
                    phase="probe",
                    frameskip=None,
                    short_goal_offset=None,
                    long_goal_offset=None,
                    eval_budget=None,
                    max_train_starts=0,
                    train_seeds=None,
                    partition_seeds=None,
                    eval_seeds=None,
                    methods=None,
                    skip_joint=None,
                    python=None,
                    cpu_threads=None,
                    gpu_id=None,
                    dry_run=False,
                )
            )


class LegacyLeWMDefaultsTest(unittest.TestCase):
    def test_default_model_family_is_lewm(self):
        env = os.environ.copy()
        env.pop("WORK_ROOT", None)
        env.pop("CACHE_DIR", None)
        with mock.patch.dict(os.environ, env, clear=True):
            resolved = resolve_config(
                Namespace(
                    model_family=None,
                    task_spec=None,
                    task_name=None,
                    dataset_name=None,
                    dataset="/tmp/data.h5",
                    checkpoint="/tmp/lewm.ckpt",
                    dataset_config=None,
                    encoder_config=None,
                    eval_config_name=None,
                    eval_config=None,
                    work_root=None,
                    cache_dir=None,
                    paired_start_root=None,
                    paired_start_root_short=None,
                    paired_start_root_long=None,
                    phase="prepare",
                    frameskip=None,
                    short_goal_offset=None,
                    long_goal_offset=None,
                    eval_budget=None,
                    max_train_starts=0,
                    train_seeds=None,
                    partition_seeds=None,
                    eval_seeds=None,
                    methods=None,
                    skip_joint=None,
                    python=None,
                    cpu_threads=None,
                    gpu_id=None,
                    dry_run=False,
                )
            )
        self.assertEqual(resolved.model_family, DEFAULT_MODEL_FAMILY)
        self.assertEqual(resolved.work_root, "experiments/pusht/matrix")


class CheckpointLoaderTest(unittest.TestCase):
    def test_loads_lewm_object_checkpoint_when_present(self):
        checkpoint = Path(
            "/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt"
        )
        if not checkpoint.exists():
            self.skipTest("official LeWM checkpoint not available")
        from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint

        model = load_jepa_object_checkpoint(
            checkpoint, model_family="lewm", map_location="cpu"
        )
        self.assertTrue(hasattr(model, "encoder"))
        self.assertTrue(hasattr(model, "predictor"))

    def test_loads_subjepa_object_checkpoint_when_present(self):
        checkpoint = Path(
            "/data/sicong/weitao/.stable_worldmodel/tworoom/subjepa_object.ckpt"
        )
        if not checkpoint.exists():
            self.skipTest("official Sub-JEPA checkpoint not available")
        from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint

        model = load_jepa_object_checkpoint(
            checkpoint, model_family="subjepa", map_location="cpu"
        )
        self.assertEqual(type(model).__module__, "backends.lewm.vendor.jepa")
        self.assertTrue(hasattr(model.encoder, "encoder"))


if __name__ == "__main__":
    unittest.main()
