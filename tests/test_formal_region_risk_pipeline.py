from __future__ import annotations

import json
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
    write_split_artifacts,
)
from experiments.control_matrix.region_risk_lib import (
    audit_formal_posttraining,
    audit_partition_train_contract,
    sha256_file,
)


class FormalRegionRiskPipelineTest(unittest.TestCase):
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

    def test_formal_posttraining_audit_passes_on_valid_contract(self) -> None:
        split_manifest = {
            "train_eval_episode_disjoint": True,
            "sha256": {"action_norm_starts": "abc"},
        }
        audit = audit_formal_posttraining(
            episode_audit={"episode_disjoint": True, "region_start_disjoint": True},
            partition_contract={"gate_partition_train_only_valid": True, "partition_latent_cache_hash_match": True},
            split_manifest=split_manifest,
            split_manifest_sha256="split",
            train_cache_hash="train",
            eval_cache_hash="eval",
            action_norm_starts_hash="abc",
            checkpoint_manifests=[
                {"latent_cache_sha256": "train", "split_manifest_sha256": "split"}
            ],
            require_valid=True,
        )
        self.assertTrue(audit["posttraining_train_only_valid"])

    def test_formal_posttraining_audit_fails_on_action_norm_mismatch(self) -> None:
        with self.assertRaises(RuntimeError):
            audit_formal_posttraining(
                episode_audit={"episode_disjoint": True, "region_start_disjoint": True},
                partition_contract={"gate_partition_train_only_valid": True, "partition_latent_cache_hash_match": True},
                split_manifest={"train_eval_episode_disjoint": True, "sha256": {"action_norm_starts": "abc"}},
                split_manifest_sha256="split",
                train_cache_hash="train",
                eval_cache_hash="eval",
                action_norm_starts_hash="wrong",
                checkpoint_manifests=[
                    {"latent_cache_sha256": "train", "split_manifest_sha256": "split"}
                ],
                require_valid=True,
            )

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
