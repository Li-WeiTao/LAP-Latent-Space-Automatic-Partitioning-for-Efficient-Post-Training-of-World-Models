from __future__ import annotations

import numpy as np
import unittest

from lap.partition import LandmarkSpectralConfig, LandmarkSpectralPartitioner


class LandmarkPartitionerTest(unittest.TestCase):
    def test_produces_deployable_regions(self):
        rng = np.random.default_rng(5)
        values = np.concatenate(
            [
                rng.normal((-4.0, 0.0), 0.25, size=(30, 2)),
                rng.normal((0.0, 4.0), 0.25, size=(30, 2)),
                rng.normal((4.0, 0.0), 0.25, size=(30, 2)),
            ]
        ).astype(np.float32)
        result = LandmarkSpectralPartitioner(
            LandmarkSpectralConfig(
                num_regions=3,
                num_landmarks=90,
                knn=8,
                prototypes_per_region=2,
                seed=3,
                spectral_n_init=10,
                prototype_n_init=3,
                cpu_threads=1,
            )
        ).fit(values, sample_ids=np.arange(len(values)))
        result.validate(len(values))
        self.assertEqual(set(result.labels.tolist()), {0, 1, 2})
        self.assertEqual(result.artifact.prototypes.shape, (6, 2))
