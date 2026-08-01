#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models"
source "$ROOT/experiments/control_matrix/scripts/run_formal_region_risk_env.sh"
cd "$ROOT"

echo "[formal-smoke] unit tests"
$PYTHON -m pytest tests/test_region_conditional_risk.py tests/test_formal_region_risk_pipeline.py -q

export TASK=pusht
export WORK_ROOT="experiments/control_matrix/assets/formal_region_risk/smoke/pusht"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export SMOKE_ONLY=1

exec "$ROOT/experiments/control_matrix/scripts/run_formal_region_risk_parallel.sh"
