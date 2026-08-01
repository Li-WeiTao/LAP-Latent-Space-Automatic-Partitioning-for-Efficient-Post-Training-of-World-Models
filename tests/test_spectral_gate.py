from __future__ import annotations

import unittest

import numpy as np

from lap.partition import (
    GlobalPartitioner,
    SpectralGateConfig,
    evaluate_spectral_gate_spectra,
)


def spectrum_for_candidate(
    relative_gap: float, *, background_step: float = 1e-3
) -> np.ndarray:
    # For K=3, lambda_3=1-E and lambda_4=1 makes the relative gap E
    # up to the configured numerical epsilon.  Eleven tail values provide the
    # q_bg=10 background gaps required by Appendix A.12.
    values = [0.0, 0.2, 1.0 - relative_gap, 1.0]
    values.extend(1.0 + background_step * index for index in range(1, 11))
    return np.asarray(values, dtype=np.float64)


class SpectralDegeneracyGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SpectralGateConfig(
            num_regions=3,
            num_landmarks=100,
            nominal_knn=30,
            perturb_knn=(27, 33),
            diagnostic_seeds=(0, 1, 2),
            deployment_seed=0,
            background_gap_count=10,
        )

    def test_tworoom_like_stable_gap_selects_spectral(self):
        values = {
            0: {27: 0.573606, 30: 0.575072, 33: 0.575568},
            1: {27: 0.574116, 30: 0.574852, 33: 0.574832},
            2: {27: 0.573157, 30: 0.573556, 33: 0.573931},
        }
        spectra = {
            seed: {
                knn: spectrum_for_candidate(gap) for knn, gap in rows.items()
            }
            for seed, rows in values.items()
        }
        result = evaluate_spectral_gate_spectra(spectra, self.config)
        self.assertTrue(result.use_partition)
        self.assertEqual(result.selected_method, "spectral")
        self.assertEqual(result.deployment_seed, 0)
        self.assertGreater(result.retained_safety_fraction, 0.99)
        self.assertGreater(result.robust_residual_gap, result.background_threshold)

    def test_pusht_like_unstable_gap_falls_back_to_global(self):
        values = {
            0: {27: 0.156646, 30: 0.162002, 33: 0.173558},
            1: {27: 0.120671, 30: 0.142223, 33: 0.150228},
            2: {27: 0.222046, 30: 0.151789, 33: 0.128625},
        }
        spectra = {
            seed: {
                knn: spectrum_for_candidate(gap) for knn, gap in rows.items()
            }
            for seed, rows in values.items()
        }
        result = evaluate_spectral_gate_spectra(spectra, self.config)
        self.assertFalse(result.use_partition)
        self.assertEqual(result.selected_method, "global")
        self.assertLess(result.retained_safety_fraction, 0.5)

    def test_small_stable_gap_is_rejected_by_spectral_background(self):
        spectra = {
            seed: {
                knn: spectrum_for_candidate(0.03, background_step=0.05)
                for knn in (27, 30, 33)
            }
            for seed in (0, 1, 2)
        }
        result = evaluate_spectral_gate_spectra(spectra, self.config)
        self.assertFalse(result.use_partition)
        self.assertEqual(result.reason, "residual_gap_not_above_background")

    def test_fewer_than_three_draws_are_rejected(self):
        config = SpectralGateConfig(
            num_landmarks=100,
            diagnostic_seeds=(0, 1),
            deployment_seed=0,
        )
        with self.assertRaisesRegex(ValueError, "at least three"):
            config.validate()

    def test_global_fallback_is_a_constant_one_region_partition(self):
        latents = np.asarray(
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
        )
        result = GlobalPartitioner().fit(
            latents, sample_ids=np.asarray([10, 11, 12], dtype=np.int64)
        )
        result.validate(3)
        self.assertEqual(result.artifact.num_regions, 1)
        np.testing.assert_array_equal(result.labels, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
