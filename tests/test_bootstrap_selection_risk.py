#!/usr/bin/env python3
"""Tests for branch-selection uncertainty diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "experiments" / "scripts"
RESULTS = REPO_ROOT / "experiments" / "bootstrap_results"
sys.path.insert(0, str(SCRIPTS))

from bootstrap_lib.loader import load_cell, point_estimate  # noqa: E402
from bootstrap_lib.resample import bootstrap_cell_with_contrasts  # noqa: E402
from bootstrap_lib.selection_risk import (  # noqa: E402
    classify_selection_risk,
    resolve_rejected_method,
    selection_delta_draws,
    summarize_selection_risk,
    verify_point_estimate_agreement,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SelectionRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((SCRIPTS / "bootstrap_config.json").read_text(encoding="utf-8"))
        cls.contrast_hash_before = _sha256(RESULTS / "bootstrap_contrasts.csv")

    @classmethod
    def tearDownClass(cls) -> None:
        self = cls()
        self.assertEqual(
            _sha256(RESULTS / "bootstrap_contrasts.csv"),
            cls.contrast_hash_before,
            "bootstrap_contrasts.csv must remain unchanged",
        )

    def test_resolve_rejected_method_from_gate_manifest(self) -> None:
        cell_global = load_cell(
            repo_root=REPO_ROOT,
            config=self.config,
            model="lewm",
            task="pusht",
            horizon="long",
        )
        self.assertEqual(cell_global.gate_info["branch"], "global")
        self.assertEqual(resolve_rejected_method(cell_global.gate_info), "spectral")

        cell_spectral = load_cell(
            repo_root=REPO_ROOT,
            config=self.config,
            model="subjepa",
            task="tworoom",
            horizon="long",
        )
        self.assertEqual(cell_spectral.gate_info["branch"], "spectral")
        self.assertEqual(resolve_rejected_method(cell_spectral.gate_info), "global")

    def test_delta_uses_shared_paired_bootstrap_indices(self) -> None:
        cell = load_cell(
            repo_root=REPO_ROOT,
            config=self.config,
            model="subjepa",
            task="tworoom",
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
        raw = {mid: res.draws for mid, res in results.items()}
        delta = selection_delta_draws(cell, raw)
        rejected = resolve_rejected_method(cell.gate_info)
        expected = raw["autolap"] - raw[rejected]
        np.testing.assert_allclose(delta, expected, rtol=0, atol=1e-12)

        match = [c for c in contrasts if c.baseline_method == rejected][0]
        np.testing.assert_allclose(delta, match.draws, rtol=0, atol=1e-12)

    def test_summarize_selection_risk_bounds_and_eol(self) -> None:
        delta = np.array([1.0, 0.5, -0.5, -3.0], dtype=np.float64)
        cell = load_cell(
            repo_root=REPO_ROOT,
            config=self.config,
            model="lewm",
            task="pusht",
            horizon="long",
        )
        summary = summarize_selection_risk(
            delta,
            margin_pp=2.0,
            model="lewm",
            task="pusht",
            horizon="long",
            gate_info=cell.gate_info,
            cell=cell,
            n_bootstrap=4,
            rng_seed=1,
        )
        self.assertGreaterEqual(summary.p_harm, 0.0)
        self.assertLessEqual(summary.p_harm, 1.0)
        self.assertGreaterEqual(summary.eol_pp, 0.0)
        self.assertGreaterEqual(summary.practical_eol_pp, 0.0)

        all_nonneg = summarize_selection_risk(
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            margin_pp=2.0,
            model="lewm",
            task="pusht",
            horizon="long",
            gate_info=cell.gate_info,
            cell=cell,
            n_bootstrap=3,
            rng_seed=1,
        )
        self.assertEqual(all_nonneg.eol_pp, 0.0)

    def test_classification_priority_order(self) -> None:
        self.assertEqual(
            classify_selection_risk(3.0, 5.0, margin_pp=2.0),
            "materially_better",
        )
        self.assertEqual(
            classify_selection_risk(-1.0, 4.0, margin_pp=2.0),
            "practically_noninferior",
        )
        self.assertEqual(
            classify_selection_risk(-5.0, -3.0, margin_pp=2.0),
            "materially_worse",
        )
        self.assertEqual(
            classify_selection_risk(-3.0, 3.0, margin_pp=2.0),
            "statistically_unresolved",
        )

    def test_point_estimate_agreement_reproduces_seven_of_eight(self) -> None:
        long_rows: list[dict] = []
        for model in ("lewm", "subjepa"):
            for task in ("tworoom", "pusht", "reacher", "cube"):
                cell = load_cell(
                    repo_root=REPO_ROOT,
                    config=self.config,
                    model=model,
                    task=task,
                    horizon="long",
                )
                if cell.status != "ok":
                    continue
                results, _ = bootstrap_cell_with_contrasts(
                    cell,
                    n_bootstrap=256,
                    seed=20260818,
                    batch_size=64,
                    resampling_unit="eval-block",
                    save_draws=True,
                )
                raw = {mid: res.draws for mid, res in results.items()}
                delta = selection_delta_draws(cell, raw)
                summary = summarize_selection_risk(
                    delta,
                    margin_pp=2.0,
                    model=model,
                    task=task,
                    horizon="long",
                    gate_info=cell.gate_info,
                    cell=cell,
                    n_bootstrap=256,
                    rng_seed=20260818,
                )
                self.assertTrue(verify_point_estimate_agreement(summary, cell))
                long_rows.append(summary.to_row())

        self.assertEqual(len(long_rows), 8)
        pilot = [r for r in long_rows if r["pilot_status"] == "pilot"]
        non_pilot = [r for r in long_rows if r["pilot_status"] == "non-pilot"]
        self.assertEqual(len(pilot), 2)
        self.assertEqual(len(non_pilot), 6)
        agreement = sum(1 for r in long_rows if r["point_estimate_favors_selected"])
        self.assertEqual(agreement, 7)

    def test_subjepa_tworoom_long_interval_crosses_zero(self) -> None:
        cell = load_cell(
            repo_root=REPO_ROOT,
            config=self.config,
            model="subjepa",
            task="tworoom",
            horizon="long",
        )
        results, _ = bootstrap_cell_with_contrasts(
            cell,
            n_bootstrap=50_000,
            seed=20260818,
            batch_size=4096,
            resampling_unit="eval-block",
            save_draws=True,
        )
        raw = {mid: res.draws for mid, res in results.items()}
        delta = selection_delta_draws(cell, raw)
        summary = summarize_selection_risk(
            delta,
            margin_pp=2.0,
            model="subjepa",
            task="tworoom",
            horizon="long",
            gate_info=cell.gate_info,
            cell=cell,
            n_bootstrap=50_000,
            rng_seed=20260818,
        )
        self.assertLess(summary.ci_low_pp, 0.0)
        self.assertGreater(summary.ci_high_pp, 0.0)
        self.assertEqual(summary.classification, "statistically_unresolved")

    def test_selection_risk_script_is_reproducible(self) -> None:
        cmd = [
            sys.executable,
            str(SCRIPTS / "bootstrap_selection_risk.py"),
            "--repo-root",
            str(REPO_ROOT),
            "--smoke-test",
        ]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        first = (RESULTS / "selection_risk.csv").read_text(encoding="utf-8")
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        second = (RESULTS / "selection_risk.csv").read_text(encoding="utf-8")
        self.assertEqual(first, second)

    def test_existing_contrast_values_unchanged(self) -> None:
        with (RESULTS / "bootstrap_contrasts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        subjepa_tworoom_global = next(
            r
            for r in rows
            if r["model"] == "subjepa"
            and r["task"] == "tworoom"
            and r["horizon"] == "long"
            and r["method_b"] == "Global-FT50"
        )
        self.assertAlmostEqual(
            float(subjepa_tworoom_global["point_difference_pp"]),
            0.13333333333333286,
            places=10,
        )
        self.assertAlmostEqual(
            float(subjepa_tworoom_global["ci_low_pp"]),
            -2.3999999999999986,
            places=10,
        )
        self.assertAlmostEqual(
            float(subjepa_tworoom_global["ci_high_pp"]),
            2.799999999999997,
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
