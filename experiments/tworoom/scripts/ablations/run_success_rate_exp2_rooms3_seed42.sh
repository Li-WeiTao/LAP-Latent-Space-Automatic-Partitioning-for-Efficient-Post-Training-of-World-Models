#!/usr/bin/env bash
# Experiment 2: geometry room predictors (left/doorway/right) switched per MPC replan.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

OUT_DIR="experiments/tworoom/results/tworoom_success_rate_rooms3_seed42"
BASELINE_STARTS="experiments/tworoom/results/tworoom_success_rate_baseline_seed42/results.json"

mkdir -p "${OUT_DIR}"

echo "==== exp2 rooms3 success rate (seed=42) started at $(date) ===="
/usr/bin/time -p python experiments/tworoom/tworoom_success_rate_eval.py \
  --mode rooms3 \
  --seed 42 \
  --eval-start-indices "${BASELINE_STARTS}" \
  --out-dir "${OUT_DIR}" \
  > "${OUT_DIR}/run.log" 2>&1

echo "==== exp2 finished at $(date) ===="
