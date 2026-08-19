"""Smoke tests for LAP efficiency benchmark helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "experiments" / "scripts"
import sys

sys.path.insert(0, str(SCRIPTS))

from efficiency_lib.config import ANCHOR_TRAINING, inference_tasks  # noqa: E402
from efficiency_lib.inference import _clone_info_fresh  # noqa: E402
from efficiency_lib.report import build_reports  # noqa: E402
from efficiency_lib.stats import bootstrap_ci, summarize  # noqa: E402
from efficiency_lib.validation import (  # noqa: E402
    read_gate_branch,
    validate_training_latent_cache,
)


class EfficiencyBenchmarkTests(unittest.TestCase):
    def test_summarize_and_bootstrap(self) -> None:
        values = [1.0, 1.1, 0.9, 1.05, 1.0]
        summary = summarize(values, seed=0)
        self.assertAlmostEqual(summary["mean"], 1.01, places=2)
        low, high = bootstrap_ci(values, seed=0)
        self.assertLess(low, summary["mean"])
        self.assertGreater(high, summary["mean"])

    def test_anchor_training_uses_lewm_cache_not_subjepa(self) -> None:
        self.assertNotIn("subjepa", str(ANCHOR_TRAINING.training_latent_cache))
        self.assertIn("P_train_global_merged_embeddings.npz", str(ANCHOR_TRAINING.training_latent_cache))
        self.assertIn("tworoom_lewm_train_latent_cache.npz", str(ANCHOR_TRAINING.latent_cache))

    def test_tworoom_inference_points_to_lewm_spectral_seed0(self) -> None:
        tworoom = inference_tasks(REPO_ROOT)["tworoom"]
        self.assertIn("tworoom_latent_spectral", str(tworoom.lap_run_dir))
        self.assertNotIn("subjepa", str(tworoom.lap_run_dir))
        self.assertIsNotNone(tworoom.lap_partition_root)

    def test_validate_training_latent_cache_rejects_subjepa(self) -> None:
        subjepa_cache = (
            REPO_ROOT / "experiments/tworoom/subjepa/formal/preparation/embedding_cache.npz"
        )
        if not subjepa_cache.is_file():
            self.skipTest("subjepa cache missing")
        with self.assertRaises(ValueError):
            validate_training_latent_cache(
                subjepa_cache,
                partition_dir=ANCHOR_TRAINING.partition_dir,
            )

    def test_validate_training_latent_cache_accepts_lewm_merged(self) -> None:
        cache = ANCHOR_TRAINING.training_latent_cache
        if not cache.is_file():
            self.skipTest("LeWM merged training cache missing")
        report = validate_training_latent_cache(
            cache,
            partition_dir=ANCHOR_TRAINING.partition_dir,
            checkpoint=ANCHOR_TRAINING.checkpoint,
        )
        self.assertEqual(report["num_transitions"], 693728)
        self.assertEqual(report["emb_shape"][1], 4)

    def test_gate_branch_labels(self) -> None:
        tworoom = read_gate_branch(ANCHOR_TRAINING.gate_manifest)
        pusht = read_gate_branch(inference_tasks(REPO_ROOT)["pusht"].gate_manifest)
        self.assertEqual(tworoom, "regional_predictors")
        self.assertEqual(pusht, "global_predictor")

    def test_fresh_info_clone_changes_tensor_storage(self) -> None:
        import torch

        source = {"pixels": torch.zeros(2, 3, 4, 4)}
        clone_a = _clone_info_fresh(source)
        clone_b = _clone_info_fresh(source)
        self.assertTrue(torch.equal(source["pixels"], clone_a["pixels"]))
        self.assertFalse(clone_a["pixels"].data_ptr() == clone_b["pixels"].data_ptr())

    def test_build_reports_writes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            joint = {
                "stable_epoch_summary": {"mean_sec": 100.0, "median_sec": 99.0, "std_sec": 1.0},
                "peak_memory": {"peak_allocated_gb": 10.0, "peak_reserved_gb": 11.0},
                "epochs": [{"epoch": 1, "epoch_wall_sec": 100.0}],
            }
            lap = {
                "stable_epoch_summary": {"mean_sec": 50.0, "median_sec": 49.0, "std_sec": 1.0},
                "peak_memory": {"peak_allocated_gb": 5.0, "peak_reserved_gb": 6.0},
                "lap_epochs": [{"lap_epoch": 1, "total_wall_sec": 50.0}],
                "expert_epochs": [{"lap_epoch": 1, "expert_id": 0, "epoch_wall_sec": 20.0}],
            }
            inference = [
                {
                    "task": "tworoom",
                    "mode": "baseline",
                    "status": "ok",
                    "planning_summary": {"mean": 1.0, "median": 1.0, "std": 0.0, "p5": 1.0, "p95": 1.0, "ci_low": 1.0, "ci_high": 1.0, "count": 1.0},
                    "planning_latency_sec": [1.0],
                    "routing_latency_sec": [],
                    "peak_memory": {"peak_allocated_gb": 2.0, "peak_reserved_gb": 2.5},
                },
                {
                    "task": "tworoom",
                    "mode": "lap",
                    "status": "ok",
                    "planning_summary": {"mean": 1.05, "median": 1.05, "std": 0.0, "p5": 1.05, "p95": 1.05, "ci_low": 1.05, "ci_high": 1.05, "count": 1.0},
                    "routing_summary": {"mean": 0.001, "median": 0.001, "std": 0.0, "p5": 0.001, "p95": 0.001, "ci_low": 0.001, "ci_high": 0.001, "count": 1.0},
                    "planning_latency_sec": [1.05],
                    "routing_latency_sec": [0.001],
                    "peak_memory": {"peak_allocated_gb": 2.2, "peak_reserved_gb": 2.7},
                },
            ]
            build_reports(
                output_dir=out,
                joint=joint,
                lap=lap,
                gate_partition={"gate_wall_sec": 22.0, "partition_wall_sec": 23.0, "selected_branch": "spectral"},
                inference=inference,
            )
            self.assertTrue((out / "training_comparison.csv").is_file())
            self.assertTrue((out / "inference_comparison.csv").is_file())
            self.assertTrue((out / "efficiency_table.tex").is_file())
            payload = (out / "efficiency_raw.jsonl").read_text(encoding="utf-8")
            self.assertIn("training", payload)
            self.assertIn("gate_partition", payload)


if __name__ == "__main__":
    unittest.main()
