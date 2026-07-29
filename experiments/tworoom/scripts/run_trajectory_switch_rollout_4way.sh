#!/usr/bin/env bash
# Offline 4-way latent rollout MSE: Official / Global-FT 50ep / rooms3 switch / cluster switch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT="experiments/tworoom"
OUT="${ROOT}/results/tworoom_trajectory_switch_rollout_4way"
CACHE="${ROOT}/cache/tworoom_trajectory_test_full_transitions.npz"
LOG="${OUT}/run_full_test.log"

mkdir -p "${OUT}"
echo "==== 4-way switch rollout MSE started at $(date) ====" | tee "${LOG}"

/usr/bin/time -p python "${ROOT}/trajectory_switch_rollout.py" \
  --load-test-cache "${CACHE}" \
  --history-size 3 \
  --max-steps 10 \
  --batch-size 64 \
  --device cuda \
  --out-dir "${OUT}" \
  >> "${LOG}" 2>&1

echo "==== 4-way switch rollout MSE finished at $(date) ====" | tee -a "${LOG}"
