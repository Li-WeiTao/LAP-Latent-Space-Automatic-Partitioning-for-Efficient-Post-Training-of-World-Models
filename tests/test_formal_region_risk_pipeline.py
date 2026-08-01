from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from experiments.control_matrix.episode_split import (
    FORMAL_SPLIT_SEED,
    FORMAL_TRAIN_FRACTION,
    EpisodeSplit,
    compute_episode_split,
    load_split_manifest,
    split_manifest_is_unsubsampled,
    write_split_artifacts,
)
from experiments.control_matrix.evaluate_region_conditional_risk import (  # noqa: E402
    FORMAL_REGIONAL_RUN_PATTERNS,
    resolve_paper_eligible,
    resolve_run_dir,
)
from experiments.control_matrix.formal_region_risk_pipeline import (  # noqa: E402
    DEFAULT_PYTHON,
    DEFAULT_TASKS,
    LEWM_VENV_PYTHON,
    PipelinePaths,
    evaluate_command,
    is_full_formal_run,
    train_command,
)
from experiments.control_matrix.region_risk_lib import (
    audit_cache_starts_exact,
    audit_formal_posttraining,
    audit_partition_train_contract,
    FORMAL_BASE_PRETRAINING_EPISODE_DISJOINT,
    FORMAL_CLAIM_SCOPE,
    runtime_provenance,
    sha256_file,
)


class FormalRegionRiskPipelineTest(unittest.TestCase):
    def test_default_python_points_to_lewm_venv_when_present(self) -> None:
        if LEWM_VENV_PYTHON.exists():
            self.assertEqual(DEFAULT_PYTHON, str(LEWM_VENV_PYTHON))

    def test_runtime_provenance_includes_python_executable(self) -> None:
        payload = runtime_provenance(python_executable="/tmp/test-python")
        self.assertEqual(payload["python_executable"], "/tmp/test-python")
        self.assertIn("torch_version", payload)
        self.assertIn("git_commit", payload)

    def test_formal_claim_scope_constants(self) -> None:
        self.assertEqual(
            FORMAL_CLAIM_SCOPE,
            "held_out_from_LAP_partition_and_posttraining",
        )
        self.assertFalse(FORMAL_BASE_PRETRAINING_EPISODE_DISJOINT)

    def test_split_is_reproducible_and_episode_disjoint(self) -> None:
        data_file = Path("/data/sicong/weitao/datasets/lewm/pusht_expert_train.h5")
        if not data_file.exists():
            self.skipTest("PushT dataset unavailable")
        first = compute_episode_split(
            data_file,
            "pusht",
            split_seed=FORMAL_SPLIT_SEED,
            train_fraction=FORMAL_TRAIN_FRACTION,
        )
        second = compute_episode_split(
            data_file,
            "pusht",
            split_seed=FORMAL_SPLIT_SEED,
            train_fraction=FORMAL_TRAIN_FRACTION,
        )
        np.testing.assert_array_equal(first.train_starts, second.train_starts)
        np.testing.assert_array_equal(first.eval_starts, second.eval_starts)
        self.assertTrue(first.train_eval_episode_disjoint)

    def test_split_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = EpisodeSplit(
                train_episode_ids=(0, 1),
                eval_episode_ids=(2,),
                train_starts=np.asarray([0, 1, 5, 6], dtype=np.int64),
                eval_starts=np.asarray([10, 11], dtype=np.int64),
                split_seed=FORMAL_SPLIT_SEED,
                train_fraction=FORMAL_TRAIN_FRACTION,
                data_file=Path(directory) / "dummy.h5",
                dataset_name="pusht",
                history_size=3,
                num_preds=1,
                frameskip=5,
                valid_start_seed=0,
            )
            out_dir = Path(directory) / "formal"
            manifest = write_split_artifacts(split, out_dir)
            loaded = load_split_manifest(out_dir / "split_manifest.json")
            self.assertEqual(manifest["sha256"]["train_starts"], loaded["sha256"]["train_starts"])
            self.assertEqual(
                manifest["sha256"]["action_norm_starts"],
                sha256_file(out_dir / "action_norm_starts.npy"),
            )
            self.assertFalse(manifest["subsampled"])
            self.assertEqual(manifest["nominal_train_num_transitions"], 4)
            self.assertEqual(manifest["written_train_num_transitions"], 4)

    def test_split_manifest_records_subsampled_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = EpisodeSplit(
                train_episode_ids=(0, 1),
                eval_episode_ids=(2,),
                train_starts=np.asarray([0, 1, 5, 6, 7], dtype=np.int64),
                eval_starts=np.asarray([10, 11, 12], dtype=np.int64),
                split_seed=FORMAL_SPLIT_SEED,
                train_fraction=FORMAL_TRAIN_FRACTION,
                data_file=Path(directory) / "dummy.h5",
                dataset_name="pusht",
                history_size=3,
                num_preds=1,
                frameskip=5,
                valid_start_seed=0,
            )
            out_dir = Path(directory) / "formal"
            manifest = write_split_artifacts(split, out_dir, max_train_starts=2, max_eval_starts=1)
            self.assertTrue(manifest["subsampled"])
            self.assertEqual(manifest["nominal_train_num_transitions"], 5)
            self.assertEqual(manifest["written_train_num_transitions"], 2)
            self.assertEqual(manifest["nominal_eval_num_transitions"], 3)
            self.assertEqual(manifest["written_eval_num_transitions"], 1)

    def test_is_full_formal_run(self) -> None:
        full = argparse.Namespace(
            smoke_only=False,
            max_train_starts=0,
            max_eval_starts=0,
            max_anchors=0,
            max_episodes=0,
        )
        smoke = argparse.Namespace(
            smoke_only=True,
            max_train_starts=4096,
            max_eval_starts=512,
            max_anchors=32,
            max_episodes=4,
        )
        unsubsampled = {
            "subsampled": False,
            "nominal_train_num_transitions": 100,
            "written_train_num_transitions": 100,
            "nominal_eval_num_transitions": 10,
            "written_eval_num_transitions": 10,
        }
        subsampled = {
            "subsampled": True,
            "nominal_train_num_transitions": 100,
            "written_train_num_transitions": 10,
            "nominal_eval_num_transitions": 10,
            "written_eval_num_transitions": 1,
        }
        evaluate_full = argparse.Namespace(
            smoke_only=False,
            max_train_starts=0,
            max_eval_starts=0,
            max_anchors=0,
            max_episodes=0,
        )
        self.assertTrue(is_full_formal_run(full))
        self.assertFalse(is_full_formal_run(smoke))
        self.assertTrue(is_full_formal_run(evaluate_full, split_manifest=unsubsampled))
        self.assertFalse(is_full_formal_run(evaluate_full, split_manifest=subsampled))

    def test_split_manifest_is_unsubsampled(self) -> None:
        self.assertTrue(
            split_manifest_is_unsubsampled(
                {
                    "subsampled": False,
                    "nominal_train_num_transitions": 5,
                    "written_train_num_transitions": 5,
                    "nominal_eval_num_transitions": 2,
                    "written_eval_num_transitions": 2,
                }
            )
        )
        self.assertFalse(
            split_manifest_is_unsubsampled(
                {
                    "subsampled": True,
                    "nominal_train_num_transitions": 5,
                    "written_train_num_transitions": 2,
                    "nominal_eval_num_transitions": 2,
                    "written_eval_num_transitions": 2,
                }
            )
        )

    def test_resolve_paper_eligible_rejects_subsampled_split(self) -> None:
        self.assertFalse(
            resolve_paper_eligible(
                smoke_only=False,
                split_manifest={
                    "subsampled": True,
                    "nominal_train_num_transitions": 5,
                    "written_train_num_transitions": 2,
                    "nominal_eval_num_transitions": 2,
                    "written_eval_num_transitions": 1,
                },
                max_eval_starts=0,
                max_anchors=0,
                max_episodes=0,
                formal=True,
                posttraining_train_only_valid=True,
            )
        )

    def test_formal_posttraining_audit_passes_on_valid_contract(self) -> None:
        split_manifest = {
            "train_eval_episode_disjoint": True,
            "sha256": {"action_norm_starts": "abc"},
        }
        partition_contracts = {
            "auto": {"gate_partition_train_only_valid": True},
            "forced_spectral": {"gate_partition_train_only_valid": True},
            "global": {"gate_partition_train_only_valid": True},
        }
        cache_start_audits = {
            "train": {"train_cache_starts_exact_match": True},
            "eval": {"eval_cache_starts_exact_match": True},
        }
        audit = audit_formal_posttraining(
            episode_audit={"episode_disjoint": True, "region_start_disjoint": True},
            partition_contracts=partition_contracts,
            split_manifest=split_manifest,
            split_manifest_sha256="split",
            train_cache_hash="train",
            eval_cache_hash="eval",
            action_norm_starts_hash="abc",
            cache_start_audits=cache_start_audits,
            checkpoint_manifests=[
                {"latent_cache_sha256": "train", "split_manifest_sha256": "split"}
            ],
            require_valid=True,
        )
        self.assertTrue(audit["posttraining_train_only_valid"])
        self.assertTrue(audit["auto_gate_train_only_valid"])
        self.assertTrue(audit["forced_spectral_train_only_valid"])

    def test_formal_posttraining_audit_fails_on_action_norm_mismatch(self) -> None:
        with self.assertRaises(RuntimeError):
            audit_formal_posttraining(
                episode_audit={"episode_disjoint": True, "region_start_disjoint": True},
                partition_contracts={
                    "auto": {"gate_partition_train_only_valid": True},
                    "forced_spectral": {"gate_partition_train_only_valid": True},
                    "global": {"gate_partition_train_only_valid": True},
                },
                split_manifest={"train_eval_episode_disjoint": True, "sha256": {"action_norm_starts": "abc"}},
                split_manifest_sha256="split",
                train_cache_hash="train",
                eval_cache_hash="eval",
                action_norm_starts_hash="wrong",
                cache_start_audits={
                    "train": {"train_cache_starts_exact_match": True},
                    "eval": {"eval_cache_starts_exact_match": True},
                },
                checkpoint_manifests=[
                    {"latent_cache_sha256": "train", "split_manifest_sha256": "split"}
                ],
                require_valid=True,
            )

    def test_formal_posttraining_audit_fails_when_forced_spectral_invalid(self) -> None:
        with self.assertRaises(RuntimeError):
            audit_formal_posttraining(
                episode_audit={"episode_disjoint": True, "region_start_disjoint": True},
                partition_contracts={
                    "auto": {"gate_partition_train_only_valid": True},
                    "forced_spectral": {"gate_partition_train_only_valid": False},
                    "global": {"gate_partition_train_only_valid": True},
                },
                split_manifest={"train_eval_episode_disjoint": True, "sha256": {"action_norm_starts": "abc"}},
                split_manifest_sha256="split",
                train_cache_hash="train",
                eval_cache_hash="eval",
                action_norm_starts_hash="abc",
                cache_start_audits={
                    "train": {"train_cache_starts_exact_match": True},
                    "eval": {"eval_cache_starts_exact_match": True},
                },
                checkpoint_manifests=[
                    {"latent_cache_sha256": "train", "split_manifest_sha256": "split"}
                ],
                require_valid=True,
            )

    def test_orchestrator_regional_run_dir_resolves_train_seed_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "training" / "forced_spectral_negative_control"
            run_dir = root / "train42"
            run_dir.mkdir(parents=True)
            resolved = resolve_run_dir(root, 42, patterns=FORMAL_REGIONAL_RUN_PATTERNS)
            self.assertEqual(resolved, run_dir)

    def test_evaluate_command_includes_forced_spectral_audit_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            paths = PipelinePaths.from_root(root)
            paths.root.mkdir(parents=True)
            split_manifest = {
                "schema_version": 1,
                "train_episode_ids": [0],
                "eval_episode_ids": [1],
                "sha256": {},
                "paths": {
                    "train_starts": str(paths.root / "train_starts.npy"),
                    "eval_starts": str(paths.root / "eval_starts.npy"),
                    "action_norm_starts": str(paths.root / "action_norm_starts.npy"),
                },
            }
            (paths.root / "split_manifest.json").write_text(
                json.dumps(split_manifest), encoding="utf-8"
            )
            for name in ("train_starts.npy", "eval_starts.npy", "action_norm_starts.npy"):
                np.save(paths.root / name, np.asarray([0], dtype=np.int64))
            args = argparse.Namespace(
                python=sys.executable,
                task="pusht",
                train_seeds="0",
                bootstrap_reps=100,
                encoding_batch_size=128,
                device="cuda",
                smoke_only=True,
                max_train_starts=4096,
                max_eval_starts=512,
                max_anchors=0,
                max_episodes=0,
            )
            command = evaluate_command(args, DEFAULT_TASKS["pusht"], paths)
            self.assertIn("--forced-spectral-partition-dir", command)
            self.assertIn(str(paths.partition_forced_spectral), command)
            self.assertIn("--smoke-only", command)

    def test_evaluate_command_sets_paper_eligible_for_full_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            paths = PipelinePaths.from_root(root)
            paths.root.mkdir(parents=True)
            split_manifest = {
                "schema_version": 1,
                "train_episode_ids": [0],
                "eval_episode_ids": [1],
                "subsampled": False,
                "nominal_train_num_transitions": 1,
                "written_train_num_transitions": 1,
                "nominal_eval_num_transitions": 1,
                "written_eval_num_transitions": 1,
                "sha256": {},
                "paths": {
                    "train_starts": str(paths.root / "train_starts.npy"),
                    "eval_starts": str(paths.root / "eval_starts.npy"),
                    "action_norm_starts": str(paths.root / "action_norm_starts.npy"),
                },
            }
            (paths.root / "split_manifest.json").write_text(
                json.dumps(split_manifest), encoding="utf-8"
            )
            for name in ("train_starts.npy", "eval_starts.npy", "action_norm_starts.npy"):
                np.save(paths.root / name, np.asarray([0], dtype=np.int64))
            args = argparse.Namespace(
                python=sys.executable,
                task="pusht",
                train_seeds="0",
                bootstrap_reps=50000,
                encoding_batch_size=128,
                device="cuda",
                smoke_only=False,
                max_train_starts=0,
                max_eval_starts=0,
                max_anchors=0,
                max_episodes=0,
            )
            command = evaluate_command(args, DEFAULT_TASKS["pusht"], paths)
            self.assertIn("--paper-eligible", command)
            self.assertNotIn("--smoke-only", command)

    def test_evaluate_command_rejects_paper_eligible_on_subsampled_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            paths = PipelinePaths.from_root(root)
            paths.root.mkdir(parents=True)
            split_manifest = {
                "schema_version": 1,
                "train_episode_ids": [0],
                "eval_episode_ids": [1],
                "subsampled": True,
                "nominal_train_num_transitions": 10,
                "written_train_num_transitions": 2,
                "nominal_eval_num_transitions": 5,
                "written_eval_num_transitions": 1,
                "sha256": {},
                "paths": {
                    "train_starts": str(paths.root / "train_starts.npy"),
                    "eval_starts": str(paths.root / "eval_starts.npy"),
                    "action_norm_starts": str(paths.root / "action_norm_starts.npy"),
                },
            }
            (paths.root / "split_manifest.json").write_text(
                json.dumps(split_manifest), encoding="utf-8"
            )
            for name in ("train_starts.npy", "eval_starts.npy", "action_norm_starts.npy"):
                np.save(paths.root / name, np.asarray([0], dtype=np.int64))
            args = argparse.Namespace(
                python=sys.executable,
                task="pusht",
                train_seeds="0",
                bootstrap_reps=50000,
                encoding_batch_size=128,
                device="cuda",
                smoke_only=False,
                max_train_starts=0,
                max_eval_starts=0,
                max_anchors=0,
                max_episodes=0,
            )
            command = evaluate_command(args, DEFAULT_TASKS["pusht"], paths)
            self.assertNotIn("--paper-eligible", command)

    def test_train_command_includes_device(self) -> None:
        args = argparse.Namespace(
            python=sys.executable,
            epochs=1,
            device="cpu",
        )
        paths = PipelinePaths.from_root(Path("/tmp/unused"))
        command = train_command(
            args,
            DEFAULT_TASKS["pusht"],
            paths,
            partition_dir=paths.partition_global,
            out_dir=Path("/tmp/out"),
            train_seed=0,
            training_role="global",
        )
        self.assertIn("--device", command)
        self.assertIn("cpu", command)

    def test_partition_provenance_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("ep_offset", data=np.asarray([0, 5, 10], dtype=np.int64))
                handle.create_dataset("ep_len", data=np.asarray([5, 5, 5], dtype=np.int64))
            manifest = Path(directory) / "partition_manifest.json"
            manifest.write_text(json.dumps({"latent_cache_sha256": "trainhash"}), encoding="utf-8")
            result = audit_partition_train_contract(
                data_file=path,
                train_starts=np.asarray([0, 1], dtype=np.int64),
                eval_starts=np.asarray([10, 11], dtype=np.int64),
                train_cache_hash="trainhash",
                partition_manifest_path=manifest,
                nominal_train_episode_ids={0},
                require_train_only=False,
            )
            self.assertTrue(result["partition_latent_cache_hash_match"])
            self.assertTrue(result["gate_partition_train_only_valid"])


if __name__ == "__main__":
    unittest.main()
