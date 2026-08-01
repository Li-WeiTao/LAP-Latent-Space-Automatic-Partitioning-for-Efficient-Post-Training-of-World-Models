#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

echo "[formal-smoke] unit tests"
$PYTHON -m pytest tests/test_region_conditional_risk.py tests/test_formal_region_risk_pipeline.py -q

WORK="experiments/control_matrix/assets/formal_region_risk/smoke/pusht"
echo "[formal-smoke] end-to-end GPU smoke for PushT -> $WORK"
$PYTHON experiments/control_matrix/formal_region_risk_pipeline.py \
  --task pusht \
  --work-root "$WORK" \
  --phase smoke \
  --device cuda \
  --max-train-starts 4096 \
  --max-eval-starts 512 \
  --train-seeds 0 \
  --epochs 1 \
  --bootstrap-reps 100 \
  --max-anchors 32 \
  --max-episodes 4 \
  --overwrite

echo "[formal-smoke] verify audit flags and outputs"
$PYTHON - <<'PY'
import csv
import json
import math
from pathlib import Path

import numpy as np

work = Path("experiments/control_matrix/assets/formal_region_risk/smoke/pusht")
audit = json.loads((work / "evaluation/audit.json").read_text())

assert audit.get("smoke_only") is True, audit
assert audit.get("paper_eligible") is False, audit
assert audit.get("split_manifest_unsubsampled_valid") is False, audit
assert audit.get("auto_gate_train_only_valid") is True, audit
assert audit.get("forced_spectral_train_only_valid") is True, audit
assert audit.get("global_partition_train_only_valid") is True, audit
assert audit.get("posttraining_train_only_valid") is True, audit
assert audit.get("cache_starts_exact_valid") is True, audit
assert audit.get("action_norm_starts_hash_match") is True, audit
assert audit.get("checkpoint_provenance_valid") is True, audit

horizon_counts = audit.get("horizon_anchor_counts", {})
for horizon in ("1", "5", "10"):
    assert int(horizon_counts.get(horizon, 0)) > 0, horizon_counts
assert int(audit.get("common_h10_anchor_count", 0)) > 0, audit

split = json.loads((work / "split_manifest.json").read_text())
assert split.get("subsampled") is True, split
assert split["written_train_num_transitions"] <= split["nominal_train_num_transitions"], split
assert split["written_eval_num_transitions"] <= split["nominal_eval_num_transitions"], split

metrics = np.load(work / "evaluation/sample_metrics.npz")
h1_main = (metrics["horizon"] == 1) & (metrics["anchor_support"] == "horizon_valid")
np.testing.assert_allclose(
    metrics["terminal_global_loss"][h1_main],
    metrics["global_loss"][h1_main],
)
np.testing.assert_allclose(
    metrics["terminal_correct_loss"][h1_main],
    metrics["correct_loss"][h1_main],
)

for csv_name in (
    "episode_metrics.csv",
    "region_summary.csv",
    "weighted_summary.csv",
    "bootstrap_summary.csv",
    "common_h10_support_summary.csv",
):
    csv_path = work / "evaluation" / csv_name
    if not csv_path.exists():
        continue
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for value in row.values():
                if value in ("", "nan", "inf", "-inf"):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                assert math.isfinite(number), (csv_name, row)

print("[formal-smoke] audit ok")
print("[formal-smoke] split manifest ok")
print("[formal-smoke] output checks ok")
PY

echo "[formal-smoke] done -> $WORK"
