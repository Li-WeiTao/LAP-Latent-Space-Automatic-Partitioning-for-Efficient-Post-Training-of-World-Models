#!/usr/bin/env python3
"""Minimal tests for main-experiment bootstrap pipeline."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "experiments" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bootstrap_lib.loader import (  # noqa: E402
    load_cell,
    point_estimate,
)
from bootstrap_lib.resample import bootstrap_cell_with_contrasts  # noqa: E402


class BootstrapMainResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((SCRIPTS / "bootstrap_config.json").read_text(encoding="utf-8"))

    def test_subjepa_tworoom_short_loads_and_matches_reference(self) -> None:
        cell = load_cell(
            repo_root=REPO_ROOT,
            config=self.config,
            model="subjepa",
            task="tworoom",
            horizon="short",
        )
        self.assertEqual(cell.status, "ok")
        self.assertIn("autolap", cell.methods)
        autolap = cell.methods["autolap"]
        est = point_estimate(autolap.blocks, official=False)
        ref = cell.reference_estimates["autolap"]
        self.assertLessEqual(abs(est - ref), 0.01)
        self.assertEqual(autolap.partition_policy, "deployment")

    def test_lewm_cube_loads_and_matches_matrix_summary(self) -> None:
        cell = load_cell(
            repo_root=REPO_ROOT,
            config=self.config,
            model="lewm",
            task="cube",
            horizon="short",
        )
        self.assertEqual(cell.status, "ok")
        self.assertIn("autolap", cell.methods)
        for method_id in ("official", "global", "autolap", "random_voronoi"):
            est = point_estimate(cell.methods[method_id].blocks, official=method_id == "official")
            if method_id == "official":
                self.assertAlmostEqual(est, 64.8, places=1)
            elif method_id in ("global", "autolap"):
                self.assertAlmostEqual(est, 64.53333333333333, places=1)
            elif method_id == "random_voronoi":
                self.assertAlmostEqual(est, 66.26666666666667, places=1)
        self.assertEqual(
            cell.methods["autolap"].partition_policy,
            cell.methods["global"].partition_policy,
        )
        self.assertEqual(cell.gate_info.get("branch"), "global")
        self.assertEqual(cell.gate_info.get("deployment_seed"), 0)
        self.assertAlmostEqual(cell.reference_estimates["official"], 64.8, places=1)
        self.assertAlmostEqual(cell.reference_estimates["autolap"], 64.53333333333333, places=1)

    def test_hierarchical_bootstrap_reproducible(self) -> None:
        cell = load_cell(
            repo_root=REPO_ROOT,
            config=self.config,
            model="subjepa",
            task="pusht",
            horizon="short",
        )
        r1, c1 = bootstrap_cell_with_contrasts(
            cell, n_bootstrap=200, seed=123, batch_size=64, resampling_unit="eval-block"
        )
        r2, _ = bootstrap_cell_with_contrasts(
            cell, n_bootstrap=200, seed=123, batch_size=64, resampling_unit="eval-block"
        )
        self.assertEqual(r1["global"].bootstrap_mean, r2["global"].bootstrap_mean)
        self.assertEqual(r1["autolap"].ci_low, r2["autolap"].ci_low)
        self.assertTrue(c1)

    def test_paired_contrast_uses_shared_indices(self) -> None:
        cell = load_cell(
            repo_root=REPO_ROOT,
            config=self.config,
            model="subjepa",
            task="reacher",
            horizon="long",
        )
        results, contrasts = bootstrap_cell_with_contrasts(
            cell,
            n_bootstrap=500,
            seed=999,
            batch_size=128,
            resampling_unit="eval-block",
            save_draws=True,
        )
        auto = results["autolap"].draws
        base = results["global"].draws
        self.assertIsNotNone(auto)
        self.assertIsNotNone(base)
        delta = auto - base  # type: ignore[operator]
        match = [c for c in contrasts if c.baseline_method == "global"][0]
        np.testing.assert_allclose(delta, match.draws, rtol=0, atol=1e-12)

    def test_synthetic_loader_point_estimate(self) -> None:
        blocks = np.array([[80.0, 90.0], [85.0, 95.0]], dtype=np.float64)
        self.assertAlmostEqual(point_estimate(blocks, official=False), 87.5)

    def test_lewm_cube_config_has_main_methods(self) -> None:
        cube = self.config["cells"]["lewm"]["cube"]
        self.assertIn("main_methods", cube)
        self.assertIn("gate_manifest", cube)
        self.assertIn("autolap", cube["main_methods"])

    def test_episode_mode_runs_with_shared_pairing(self) -> None:
        cell = load_cell(
            repo_root=REPO_ROOT,
            config=self.config,
            model="subjepa",
            task="tworoom",
            horizon="short",
        )
        self.assertTrue(cell.has_episode_data)
        results, contrasts = bootstrap_cell_with_contrasts(
            cell,
            n_bootstrap=64,
            seed=42,
            batch_size=32,
            resampling_unit="episode",
            save_draws=True,
        )
        self.assertIn("autolap", results)
        auto = results["autolap"].draws
        base = results["global"].draws
        self.assertIsNotNone(auto)
        self.assertIsNotNone(base)
        match = [c for c in contrasts if c.baseline_method == "global"][0]
        np.testing.assert_allclose(auto - base, match.draws, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
