from __future__ import annotations

import unittest
from pathlib import Path

from experiments.control_matrix.region_risk_lib import (
    PUBLIC_ANALYSIS_NAME,
    PUBLIC_ANALYSIS_SHORT_NAME,
)


class RegionRiskPublicTerminologyTest(unittest.TestCase):
    def test_public_docs_use_region_risk_name(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        public_files = (
            project_root / "README.md",
            project_root / "experiments/control_matrix/REGION_RISK_ANALYSIS.md",
            project_root / "experiments/control_matrix/evaluate_region_conditional_risk.py",
            project_root / "experiments/control_matrix/formal_region_risk_pipeline.py",
            project_root / "experiments/control_matrix/verify_formal_region_risk.py",
        )
        combined = "\n".join(path.read_text(encoding="utf-8") for path in public_files)
        self.assertIn(PUBLIC_ANALYSIS_NAME, combined)
        self.assertIn(PUBLIC_ANALYSIS_SHORT_NAME, combined)
        lowered = combined.lower()
        self.assertNotIn("pusht formal experiment", lowered)
        self.assertNotIn("formal experiment", lowered)
        self.assertNotIn("second main experiment", lowered)


if __name__ == "__main__":
    unittest.main()
