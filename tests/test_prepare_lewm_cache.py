from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.control_matrix import prepare_lewm_cache as prep


class PrepareLeWMCacheTest(unittest.TestCase):
    def test_build_spectral_embedding_cache_matches_partition_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "embedding_cache.npz"
            emb = np.asarray(
                [
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [2.0, 0.0]],
                    [[2.1, 0.0], [3.0, 0.0]],
                ],
                dtype=np.float32,
            )
            np.savez(
                cache,
                emb=emb,
                act_emb=np.zeros_like(emb),
                region_starts=np.asarray([0, 1, 2], dtype=np.int64),
            )
            paths = prep.PreparePaths.from_out_dir(root)
            stats = prep.build_spectral_embedding_cache(
                paths, frameskip=1, overwrite=True
            )
            with np.load(paths.spectral_cache, allow_pickle=False) as data:
                self.assertEqual(list(data.files), ["emb", "global_timestep_ids"])
                self.assertEqual(int(data["emb"].shape[0]), stats["num_unique_timesteps"])

    def test_audit_against_reference_detects_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "generated.npz"
            reference = root / "reference.npz"
            np.savez(
                generated,
                emb=np.zeros((2, 4, 3), dtype=np.float32),
                act_emb=np.zeros((2, 4, 3), dtype=np.float32),
                region_starts=np.asarray([0, 1], dtype=np.int64),
            )
            np.savez(
                reference,
                emb=np.zeros((3, 4, 3), dtype=np.float32),
                act_emb=np.zeros((3, 4, 3), dtype=np.float32),
                region_starts=np.asarray([0, 1, 2], dtype=np.int64),
            )
            report = prep.audit_against_reference(generated, reference)
            self.assertFalse(report["passed"])

    def test_write_representation_manifest_records_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = prep.PreparePaths.from_out_dir(root / "preparation", dataset_name="pusht")
            paths.out_dir.mkdir(parents=True, exist_ok=True)
            np.save(paths.starts, np.asarray([0, 5, 10], dtype=np.int64))
            np.savez(
                paths.embedding_cache,
                emb=np.zeros((3, 4, 2), dtype=np.float32),
                act_emb=np.zeros((3, 4, 2), dtype=np.float32),
                region_starts=np.asarray([0, 5, 10], dtype=np.int64),
            )
            np.savez(
                paths.spectral_cache,
                emb=np.zeros((3, 2), dtype=np.float32),
                global_timestep_ids=np.asarray([0, 5, 10], dtype=np.int64),
            )
            np.savez(
                paths.action_norm_stats,
                action_mean=np.zeros((1, 2), dtype=np.float32),
                action_std=np.ones((1, 2), dtype=np.float32),
                frameskip=np.int64(5),
                normalization_samples=np.int64(3),
            )
            paths.action_norm_manifest.write_text("{}", encoding="utf-8")
            data_file = root / "data.h5"
            checkpoint = root / "model.ckpt"
            data_file.write_bytes(b"h5")
            checkpoint.write_bytes(b"ckpt")
            args = mock.Mock(
                dataset_name="pusht",
                data_file=data_file,
                checkpoint=checkpoint,
                history_size=3,
                num_preds=1,
                frameskip=5,
            )
            prep.write_representation_manifest(
                paths,
                args=args,
                encode_report={"method": "test", "selection": {}, "arrays": {}},
                spectral_stats={"num_unique_timesteps": 3},
                reference_audit=None,
            )
            manifest = json.loads(paths.representation_manifest.read_text())
            self.assertIn("sha256", manifest)
            self.assertEqual(
                set(manifest["sha256"]),
                {
                    "data_file",
                    "checkpoint",
                    "starts",
                    "embedding_cache",
                    "spectral_embedding_cache",
                    "action_norm_stats",
                },
            )


if __name__ == "__main__":
    unittest.main()
