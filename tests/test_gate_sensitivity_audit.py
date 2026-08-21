#!/usr/bin/env python3
"""Tests for gate-only sensitivity audit helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "tworoom"))

from experiments.control_matrix.gate_audit_lib import (  # noqa: E402
    AUDIT_SOURCE_REL_PATHS,
    B_SEED_PREFIXES,
    CACHE_SCHEMA_VERSION,
    ComputationProvenance,
    MAX_EIGENVALUES,
    OATScenario,
    PREPROCESSING_VERSION,
    SpectrumCacheIdentity,
    VersionedSpectrumCache,
    audit_source_hashes,
    baseline_draw_bank,
    build_excluded_landmark_indices,
    build_oat_scenarios,
    build_spectra_by_m,
    choose_held_out_audit_rows,
    collect_minimal_spectrum_requests,
    collect_non_k_boundary_cases,
    combined_source_digest,
    config_from_manifest_dict,
    decision_agreement_rows,
    enumerate_seed_subsets,
    evaluate_config,
    git_info,
    k_behavior_rows,
    knn_center_sweep,
    landmark_sweep_values,
    latex_escape,
    margins_from_result,
    NeighborDrawCache,
    resolve_pair_inputs,
    result_row,
    scenario_spectra_bank,
    NeighborDrawCache,
)
from experiments.control_matrix.gate_sensitivity_audit import (  # noqa: E402
    promote_staging,
    render_paper_table,
)
from lap.partition import SpectralGateConfig, evaluate_spectral_gate_spectra  # noqa: E402
from tests.test_spectral_gate import spectrum_for_candidate  # noqa: E402


def synthetic_bank(gap: float, *, m: int) -> dict[int, dict[int, dict[int, np.ndarray]]]:
    cfg = SpectralGateConfig(num_landmarks=100)
    spectrum = spectrum_for_candidate(gap)
    values = {
        seed: {knn: spectrum.copy() for knn in cfg.graph_knn_values}
        for seed in (0, 1, 2)
    }
    return {m: values}


class GateSensitivityAuditTests(unittest.TestCase):
    def test_different_m_use_different_banks(self) -> None:
        baseline = SpectralGateConfig(num_landmarks=100)
        scenarios = build_oat_scenarios(baseline)
        m10 = replace(baseline, num_landmarks=50)
        m20 = replace(baseline, num_landmarks=100)
        bank = {
            50: {
                0: {30: spectrum_for_candidate(0.05)},
                1: {30: spectrum_for_candidate(0.05)},
                2: {30: spectrum_for_candidate(0.05)},
            },
            100: {
                0: {30: spectrum_for_candidate(0.8)},
                1: {30: spectrum_for_candidate(0.8)},
                2: {30: spectrum_for_candidate(0.8)},
            },
        }
        for knn in (27, 33):
            bank[50][0][knn] = bank[50][1][knn] = bank[50][2][knn] = spectrum_for_candidate(0.05)
            bank[100][0][knn] = bank[100][1][knn] = bank[100][2][knn] = spectrum_for_candidate(0.8)
        r10 = evaluate_config(m10, scenario_spectra_bank(bank, m10))
        r20 = evaluate_config(m20, scenario_spectra_bank(bank, m20))
        self.assertNotAlmostEqual(r10.candidate_gap_min, r20.candidate_gap_min)
        m_scenario = next(s for s in scenarios if s.varied_factor == "M" and s.config.num_landmarks == 50)
        used = scenario_spectra_bank(bank, m_scenario.config)[0][30]
        expected = spectrum_for_candidate(0.05)
        self.assertTrue(np.allclose(used, expected))

    def test_draw_subsets_use_baseline_m_bank_only(self) -> None:
        baseline = SpectralGateConfig(num_landmarks=100)
        bank = synthetic_bank(0.6, m=100)
        bank[50] = {
            seed: {knn: spectrum_for_candidate(0.01) for knn in baseline.graph_knn_values}
            for seed in range(10)
        }
        draw_bank = baseline_draw_bank(bank, baseline)
        self.assertIn(0, draw_bank)
        self.assertAlmostEqual(
            evaluate_config(baseline, draw_bank).candidate_gap_min,
            evaluate_config(baseline, scenario_spectra_bank(bank, baseline)).candidate_gap_min,
        )

    def test_cache_invalidation_on_metadata_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = VersionedSpectrumCache(Path(tmp))
            base = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "latent_cache_sha256": "abc",
                "group_ids_hash": "grp",
                "model": "lewm",
                "task": "tworoom",
                "num_landmarks": 100,
                "landmark_seed": 0,
                "knn": 30,
                "eigenvalue_count": MAX_EIGENVALUES,
                "eig_tol": 1e-4,
                "eig_maxiter": 20_000,
                "preprocessing_version": PREPROCESSING_VERSION,
                "source_digest": "digest_v3",
            }
            identity = SpectrumCacheIdentity(**base)
            values = np.arange(MAX_EIGENVALUES, dtype=np.float64)
            cache._write(identity, values)
            self.assertIsNotNone(cache._load(identity))
            for field in ("latent_cache_sha256", "group_ids_hash", "eigenvalue_count", "eig_tol", "schema_version"):
                mutated = dict(base)
                mutated[field] = mutated[field] + "_x" if isinstance(mutated[field], str) else mutated[field] + 1
                self.assertIsNone(cache._load(SpectrumCacheIdentity(**mutated)))

    def test_cold_warm_cache_smoke(self) -> None:
        rng = np.random.default_rng(0)
        transformed = rng.standard_normal((400, 12)).astype(np.float32)
        group_ids = np.repeat(np.arange(4), 100)
        baseline = SpectralGateConfig(num_landmarks=80, diagnostic_seeds=(0, 1, 2))
        requests = {
            (80, seed, knn)
            for seed in baseline.diagnostic_seeds
            for knn in baseline.graph_knn_values
        }
        source_digest = combined_source_digest(audit_source_hashes(REPO))
        identity_base = {
            "latent_cache_sha256": "latent_smoke",
            "group_ids_hash": "group_smoke",
            "model": "lewm",
            "task": "smoke",
            "source_digest": source_digest,
        }

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            cold_cache = VersionedSpectrumCache(cache_root)
            neighbor_cache = NeighborDrawCache()
            cold_bank, cold_hits = build_spectra_by_m(
                cold_cache,
                identity_base=identity_base,
                transformed=transformed,
                group_ids=group_ids,
                baseline_config=baseline,
                requests=requests,
                neighbor_cache=neighbor_cache,
                max_k=max(baseline.graph_knn_values),
            )
            self.assertEqual(cold_cache.hits, 0)
            self.assertEqual(cold_cache.misses, len(requests))
            self.assertEqual(cold_cache.eigensolves, len(requests))
            self.assertFalse(any(cold_hits.values()))

            warm_cache = VersionedSpectrumCache(cache_root)
            warm_bank, warm_hits = build_spectra_by_m(
                warm_cache,
                identity_base=identity_base,
                transformed=transformed,
                group_ids=group_ids,
                baseline_config=baseline,
                requests=requests,
                neighbor_cache=NeighborDrawCache(),
                max_k=max(baseline.graph_knn_values),
            )
            self.assertEqual(warm_cache.hits, len(requests))
            self.assertEqual(warm_cache.misses, 0)
            self.assertEqual(warm_cache.eigensolves, 0)
            self.assertTrue(all(warm_hits.values()))

            cold_result = evaluate_config(baseline, scenario_spectra_bank(cold_bank, baseline))
            warm_result = evaluate_config(baseline, scenario_spectra_bank(warm_bank, baseline))
            self.assertEqual(cold_result.selected_method, warm_result.selected_method)
            self.assertEqual(cold_result.reason, warm_result.reason)
            self.assertAlmostEqual(
                cold_result.candidate_gap_min,
                warm_result.candidate_gap_min,
            )

    def test_load_does_not_change_cache_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = VersionedSpectrumCache(Path(tmp))
            identity = SpectrumCacheIdentity(
                schema_version=CACHE_SCHEMA_VERSION,
                latent_cache_sha256="abc",
                group_ids_hash="grp",
                model="lewm",
                task="tworoom",
                num_landmarks=50,
                landmark_seed=0,
                knn=30,
                eigenvalue_count=MAX_EIGENVALUES,
                eig_tol=1e-4,
                eig_maxiter=20_000,
                preprocessing_version=PREPROCESSING_VERSION,
                source_digest="digest",
            )
            values = np.arange(MAX_EIGENVALUES, dtype=np.float64)
            cache._write(identity, values)
            self.assertEqual(cache.hits, 0)
            self.assertIsNotNone(cache._load(identity))
            self.assertEqual(cache.hits, 0)

    def test_computation_provenance_lists_all_audit_sources(self) -> None:
        provenance = ComputationProvenance.capture(REPO)
        self.assertEqual(set(provenance.source_hashes), set(AUDIT_SOURCE_REL_PATHS))
        self.assertEqual(
            provenance.combined_source_digest,
            combined_source_digest(provenance.source_hashes),
        )
        self.assertTrue(provenance.combined_source_digest)

    def test_promote_staging_leaves_final_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            final = root / "final"
            staging.mkdir()
            (staging / "baseline_margins.csv").write_text("model,task\n", encoding="utf-8")
            promote_staging(staging, final)
            self.assertTrue(final.is_dir())
            self.assertFalse(staging.exists())
            self.assertTrue((final / "baseline_margins.csv").is_file())

    def test_k_excluded_from_agreement_and_boundary(self) -> None:
        baseline = SpectralGateConfig(num_landmarks=100)
        scenarios = build_oat_scenarios(baseline)
        rows = [
            result_row(
                model="lewm",
                task="tworoom",
                scenario=OATScenario("baseline", "baseline", baseline, True, False, False),
                result=evaluate_config(baseline, synthetic_bank(0.6, m=100)[100]),
                baseline_decision="spectral",
                elapsed_sec=0.0,
                cache_hit=True,
            ),
            result_row(
                model="lewm",
                task="tworoom",
                scenario=OATScenario("K", "4", replace(baseline, num_regions=4), False, False, False),
                result=evaluate_config(
                    replace(baseline, num_regions=4),
                    {
                        seed: {
                            knn: np.concatenate(
                                [spectrum_for_candidate(0.01), np.asarray([1.02])]
                            )
                            for knn in baseline.graph_knn_values
                        }
                        for seed in baseline.diagnostic_seeds
                    },
                ),
                baseline_decision="spectral",
                elapsed_sec=0.0,
                cache_hit=True,
            ),
        ]
        decision_rows, overall = decision_agreement_rows(rows, scenarios)
        self.assertEqual(overall["non_k_overall_agreement"], "0/0")
        self.assertFalse(any(r["varied_factor"] == "K" for r in decision_rows))
        self.assertEqual(collect_non_k_boundary_cases(rows), [])
        k_rows = k_behavior_rows(rows)
        self.assertEqual(len(k_rows), 1)
        self.assertFalse(k_rows[0]["agreement_with_baseline_K"])

    def test_held_out_audit_rows_disjoint_from_landmarks(self) -> None:
        baseline = SpectralGateConfig(num_landmarks=20, diagnostic_seeds=(0, 1, 2))
        alt = replace(baseline, num_landmarks=10)
        scenarios = [
            OATScenario("baseline", "baseline", baseline, True, False, False),
            OATScenario("M", "10", alt, False, True, True),
        ]
        group_ids = np.repeat(np.arange(5), 40)
        excluded = build_excluded_landmark_indices(200, baseline, scenarios, group_ids)
        audit_rows = choose_held_out_audit_rows(200, excluded_indices=excluded, size=10, seed=1)
        self.assertTrue(set(audit_rows.tolist()).isdisjoint(excluded))

    def test_latex_escape_and_compiles(self) -> None:
        self.assertEqual(latex_escape("residual_gap_not_above_background"), r"residual\_gap\_not\_above\_background")
        tex = render_paper_table(
            [
                {
                    "model": "lewm",
                    "task": "reacher",
                    "decision": "global",
                    "reason": "residual_gap_not_above_background",
                    "safety_margin": 0.1,
                    "prominence_margin": -0.2,
                }
            ]
        )
        doc = (
            "\\documentclass{article}\n\\usepackage{booktabs}\n"
            "\\begin{document}\n" + tex + "\\end{document}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.tex"
            path.write_text(doc, encoding="utf-8")
            if shutil.which("pdflatex"):
                proc = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "table.tex"],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_git_info_dirty_ignores_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "audit@test.local"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "audit-test"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            source_dir = root / "experiments/control_matrix"
            source_dir.mkdir(parents=True)
            source_file = source_dir / "gate_audit_lib.py"
            source_file.write_text("a", encoding="utf-8")
            other = root / "results.csv"
            other.write_text("a", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)
            (root / "run.log").write_text("noise", encoding="utf-8")
            other.write_text("b", encoding="utf-8")
            self.assertFalse(git_info(root)["dirty"])
            source_file.write_text("b", encoding="utf-8")
            self.assertTrue(git_info(root)["dirty"])

    def test_shell_script_has_no_private_paths(self) -> None:
        text = (REPO / "experiments/control_matrix/scripts/run_gate_sensitivity_audit.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("/data/", text)
        self.assertIn("${REPO_ROOT}", text)
        self.assertIn("${CACHE_DIR}", text)
        self.assertIn("--cache-dir", text)

    def test_minimal_spectrum_requests_skip_threshold_only(self) -> None:
        baseline = SpectralGateConfig(num_landmarks=100)
        scenarios = build_oat_scenarios(baseline)
        requests = collect_minimal_spectrum_requests(scenarios, baseline)
        self.assertIn((100, 0, 30), requests)
        self.assertIn((100, 9, 30), requests)  # baseline-M draw-subset seeds
        m_alt = next(s.config.num_landmarks for s in scenarios if s.varied_factor == "M" and not s.is_baseline)
        self.assertIn((m_alt, 0, 30), requests)
        self.assertNotIn((m_alt, 9, 30), requests)

    def test_audit_source_has_no_predictor_training(self) -> None:
        source = (REPO / "experiments/control_matrix/gate_sensitivity_audit.py").read_text(encoding="utf-8")
        for forbidden in ("train_predictors", "train_auto", "fit_partition.main", "GatedSpectralPartitioner"):
            self.assertNotIn(forbidden, source)

    def test_seed_prefixes_and_knn(self) -> None:
        self.assertEqual(B_SEED_PREFIXES[3], (0, 1, 2))
        self.assertEqual(landmark_sweep_values(20_000), [10_000, 15_000, 20_000])
        self.assertEqual([c for c, _ in knn_center_sweep(30)], [27, 30, 33])

    def test_missing_cache_preflight(self) -> None:
        from experiments.control_matrix.gate_audit_lib import PairSpec

        spec = PairSpec("subjepa", "pusht", "experiments/pusht/subjepa/formal/gate/partition/manifest.json")
        resolved, issues = resolve_pair_inputs(REPO, spec)
        if resolved is not None:
            self.skipTest("pusht subjepa cache unexpectedly present")
        self.assertTrue(any("missing latent cache" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
