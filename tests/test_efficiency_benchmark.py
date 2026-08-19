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

from efficiency_lib.stats import bootstrap_ci, summarize  # noqa: E402
from efficiency_lib.report import build_reports  # noqa: E402


class EfficiencyBenchmarkTests(unittest.TestCase):
    def test_summarize_and_bootstrap(self) -> None:
        values = [1.0, 1.1, 0.9, 1.05, 1.0]
        summary = summarize(values, seed=0)
        self.assertAlmostEqual(summary["mean"], 1.01, places=2)
        low, high = bootstrap_ci(values, seed=0)
        self.assertLess(low, summary["mean"])
        self.assertGreater(high, summary["mean"])

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
                gate_partition={"gate_wall_sec": 22.0, "partition_wall_sec": 23.0},
                inference=inference,
            )
            self.assertTrue((out / "training_comparison.csv").is_file())
            self.assertTrue((out / "inference_comparison.csv").is_file())
            self.assertTrue((out / "efficiency_table.tex").is_file())
            payload = (out / "efficiency_raw.jsonl").read_text(encoding="utf-8")
            self.assertIn("training", payload)


if __name__ == "__main__":
    unittest.main()
