#!/usr/bin/env python3
"""Tests for gate-only sensitivity audit helpers."""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "tworoom"))

from experiments.control_matrix.gate_audit_lib import (  # noqa: E402
    B_SEED_PREFIXES,
    OATScenario,
    build_oat_scenarios,
    compare_to_manifest,
    config_from_manifest_dict,
    enumerate_seed_subsets,
    evaluate_config,
    knn_center_sweep,
    landmark_sweep_values,
    margins_from_result,
    resolve_pair_inputs,
    result_row,
    safety_margin,
    spectrum_cache_key,
)
from lap.partition import SpectralGateConfig, evaluate_spectral_gate_spectra  # noqa: E402
from tests.test_spectral_gate import spectrum_for_candidate  # noqa: E402


class GateSensitivityAuditTests(unittest.TestCase):
    def test_oat_is_not_cartesian_product(self) -> None:
        baseline = SpectralGateConfig(num_landmarks=100)
        scenarios = build_oat_scenarios(baseline)
        factors = {s.varied_factor for s in scenarios}
        self.assertIn("baseline", factors)
        self.assertIn("K", factors)
        self.assertIn("rho", factors)
        # One-at-a-time: each non-baseline scenario changes one family.
        non_base = [s for s in scenarios if not s.is_baseline]
        self.assertGreater(len(non_base), 10)
        self.assertEqual(sum(1 for s in scenarios if s.varied_factor == "baseline"), 1)

    def test_k_excluded_from_non_k_agreement_denominator(self) -> None:
        baseline = SpectralGateConfig(num_landmarks=100)
        scenarios = build_oat_scenarios(baseline)
        k_alts = [s for s in scenarios if s.varied_factor == "K" and not s.is_baseline]
        non_k_alts = [s for s in scenarios if s.varied_factor != "K" and not s.is_baseline]
        self.assertEqual(len(k_alts), 3)
        self.assertGreater(len(non_k_alts), 0)

    def test_margin_calculations(self) -> None:
        values = {
            0: {27: 0.573606, 30: 0.575072, 33: 0.575568},
            1: {27: 0.574116, 30: 0.574852, 33: 0.574832},
            2: {27: 0.573157, 30: 0.573556, 33: 0.573931},
        }
        spectra = {
            seed: {knn: spectrum_for_candidate(gap) for knn, gap in rows.items()}
            for seed, rows in values.items()
        }
        config = SpectralGateConfig(num_landmarks=100)
        result = evaluate_spectral_gate_spectra(spectra, config)
        margins = margins_from_result(result)
        self.assertAlmostEqual(safety_margin(result), margins["safety_margin"])
        self.assertGreater(margins["safety_margin"], 0.0)
        self.assertGreater(margins["prominence_margin"], 0.0)

    def test_seed_prefixes_and_subsets(self) -> None:
        self.assertEqual(B_SEED_PREFIXES[3], (0, 1, 2))
        self.assertEqual(len(B_SEED_PREFIXES[10]), 10)
        subsets = enumerate_seed_subsets(universe=range(10), subset_size=3, required_seed=0)
        self.assertTrue(all(0 in subset for subset in subsets))
        self.assertEqual(len(subsets), 36)

    def test_knn_pm10_percent(self) -> None:
        centers = [center for center, _ in knn_center_sweep(30)]
        self.assertEqual(centers, [27, 30, 33])
        _, perturb = knn_center_sweep(30)[1]
        self.assertEqual(len(set((30, *perturb))), 3)

    def test_landmark_fractions(self) -> None:
        self.assertEqual(landmark_sweep_values(20_000), [10_000, 15_000, 20_000])

    def test_threshold_only_reuses_spectra(self) -> None:
        baseline = SpectralGateConfig(num_landmarks=100)
        values = {
            0: {27: 0.573606, 30: 0.575072, 33: 0.575568},
            1: {27: 0.574116, 30: 0.574852, 33: 0.574832},
            2: {27: 0.573157, 30: 0.573556, 33: 0.573931},
        }
        spectra = {
            seed: {knn: spectrum_for_candidate(gap) for knn, gap in rows.items()}
            for seed, rows in values.items()
        }
        alt = replace(baseline, perturbation_multiplier=3.0)
        base_result = evaluate_config(baseline, spectra)
        alt_result = evaluate_config(alt, spectra)
        self.assertEqual(base_result.candidate_gap_min, alt_result.candidate_gap_min)
        self.assertNotEqual(
            base_result.perturbation_threshold_max, alt_result.perturbation_threshold_max
        )

    def test_spectrum_cache_key_pair_scoped(self) -> None:
        k1 = spectrum_cache_key(
            cache_hash="abc", model="lewm", task="tworoom", num_landmarks=100, seed=0, knn=30
        )
        k2 = spectrum_cache_key(
            cache_hash="abc", model="subjepa", task="tworoom", num_landmarks=100, seed=0, knn=30
        )
        self.assertNotEqual(k1, k2)

    def test_result_row_schema(self) -> None:
        baseline = SpectralGateConfig(num_landmarks=100)
        scenario = OATScenario("rho", "0.4", replace(baseline, retention_threshold=0.4), False, False, False)
        spectra = {
            seed: {knn: spectrum_for_candidate(0.5) for knn in baseline.graph_knn_values}
            for seed in baseline.diagnostic_seeds
        }
        result = evaluate_config(baseline, spectra)
        row = result_row(
            model="lewm",
            task="tworoom",
            scenario=scenario,
            result=result,
            baseline_decision="spectral",
            elapsed_sec=0.1,
            cache_hit=True,
        )
        for key in ("model", "task", "decision", "agreement_with_baseline", "cache_hit"):
            self.assertIn(key, row)

    def test_manifest_baseline_config_roundtrip(self) -> None:
        manifest_path = REPO / "experiments/tworoom/results/auto_gate_complete_k3/auto/partition/manifest.json"
        if not manifest_path.is_file():
            self.skipTest("baseline manifest unavailable")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = manifest["method_metadata"]["automatic_gate"]["configuration"]
        cfg = config_from_manifest_dict(raw)
        self.assertEqual(cfg.num_regions, 3)
        self.assertEqual(cfg.diagnostic_seeds, (0, 1, 2))

    def test_missing_cache_preflight(self) -> None:
        from experiments.control_matrix.gate_audit_lib import PairSpec

        spec = PairSpec("subjepa", "pusht", "experiments/pusht/subjepa/formal/gate/partition/manifest.json")
        resolved, issues = resolve_pair_inputs(REPO, spec)
        if resolved is not None:
            self.skipTest("pusht subjepa cache unexpectedly present")
        self.assertTrue(any("missing latent cache" in issue for issue in issues))

    def test_audit_source_has_no_predictor_training(self) -> None:
        source = (REPO / "experiments/control_matrix/gate_sensitivity_audit.py").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "train_predictors",
            "train_auto",
            "fit_partition.main",
            "GatedSpectralPartitioner",
        ):
            self.assertNotIn(forbidden, source)

    def test_synthetic_boundary_case(self) -> None:
        config = SpectralGateConfig(num_landmarks=100, retention_threshold=0.5)
        tight = {
            seed: {knn: spectrum_for_candidate(0.03, background_step=0.05) for knn in (27, 30, 33)}
            for seed in (0, 1, 2)
        }
        result = evaluate_config(config, tight)
        self.assertFalse(result.use_partition)
        margins = margins_from_result(result)
        self.assertLess(margins["prominence_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
