from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.control_matrix.aggregate_matrix import rate


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
