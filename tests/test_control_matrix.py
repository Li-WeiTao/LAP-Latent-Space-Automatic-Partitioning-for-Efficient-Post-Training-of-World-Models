from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.control_matrix.aggregate_matrix import rate
from experiments.control_matrix.fit_partition import load_unique_latents


class ControlMatrixAggregationTest(unittest.TestCase):
    def test_official_success_rate_is_already_a_percentage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(
                json.dumps({"metrics": {"success_rate": 61.25}}),
                encoding="utf-8",
            )
            self.assertEqual(rate(path), 61.25)

    def test_rejects_values_outside_official_percentage_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(
                json.dumps({"metrics": {"success_rate": 6100.0}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                rate(path)


class CanonicalFirstPartitionCacheTest(unittest.TestCase):
    def _write_cache(self, root: Path) -> Path:
        cache = root / "latent_cache.npz"
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
        return cache

    def test_nonidentical_duplicate_uses_first_occurrence(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self._write_cache(Path(directory))
            unique, ids, stats = load_unique_latents(cache, frameskip=1)
            self.assertEqual(unique.shape, (4, 2))
            np.testing.assert_array_equal(ids, [0, 1, 2, 3])
            np.testing.assert_array_equal(
                unique, [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
            )
            self.assertEqual(stats["num_unique_timesteps"], 4)
            self.assertEqual(stats["discarded_repeated_window_slots"], 2)
            self.assertEqual(
                stats["duplicate_policy"], "stable_first_occurrence_per_timestep"
            )
