#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models"
export TASK=tworoom
export WORK_ROOT="${WORK_ROOT:-experiments/control_matrix/assets/formal_region_risk/tworoom}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-0,42,625}"
export EPOCHS="${EPOCHS:-50}"
export BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-50000}"
export SMOKE_ONLY=0

exec "$ROOT/experiments/control_matrix/scripts/run_formal_region_risk_parallel.sh"
