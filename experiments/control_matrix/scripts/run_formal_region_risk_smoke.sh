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

echo "[formal-smoke] verify audit flags"
$PYTHON - <<'PY'
import json
from pathlib import Path

audit = json.loads(
    Path("experiments/control_matrix/assets/formal_region_risk/smoke/pusht/evaluation/audit.json").read_text()
)
assert audit.get("smoke_only") is True, audit
assert audit.get("paper_eligible") is False, audit
assert audit.get("auto_gate_train_only_valid") is True, audit
assert audit.get("forced_spectral_train_only_valid") is True, audit
assert audit.get("global_partition_train_only_valid") is True, audit
assert audit.get("posttraining_train_only_valid") is True, audit
assert audit.get("cache_starts_exact_valid") is True, audit
assert audit.get("action_norm_starts_hash_match") is True, audit
assert audit.get("checkpoint_provenance_valid") is True, audit

split = json.loads(
    Path("experiments/control_matrix/assets/formal_region_risk/smoke/pusht/split_manifest.json").read_text()
)
assert split.get("subsampled") is True, split
assert split["written_train_num_transitions"] <= split["nominal_train_num_transitions"], split
assert split["written_eval_num_transitions"] <= split["nominal_eval_num_transitions"], split
print("[formal-smoke] audit ok")
print("[formal-smoke] split manifest ok")
PY

echo "[formal-smoke] done -> $WORK"
