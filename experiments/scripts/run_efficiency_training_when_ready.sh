#!/usr/bin/env bash
# Wait until a GPU is idle, then benchmark Joint + LAP training on the same device.
set -euo pipefail

REPO="/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models"
PYTHON="/data/sicong/weitao/le-wm/.venv/bin/python"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
GPU_INDEX="${GPU_INDEX:-0}"
LOG="${REPO}/experiments/efficiency_results/training_wait.log"

cd "${REPO}"
export PYTHONPATH="experiments/scripts:.:experiments/tworoom:/data/sicong/weitao/le-wm"

echo "[training-wait] waiting for idle GPU (free>=${MIN_FREE_MIB} MiB, util<=${MAX_UTIL_PCT}%)" | tee -a "${LOG}"
selected="$(
  MIN_FREE_MIB="${MIN_FREE_MIB}" \
  MAX_UTIL_PCT="${MAX_UTIL_PCT}" \
  GPU_INDEX="${GPU_INDEX}" \
  LOG="${LOG}" \
  bash "${REPO}/experiments/scripts/wait_for_gpu.sh"
)"

echo "[training-wait] starting Joint + LAP training on cuda:${selected}" | tee -a "${LOG}"
"${PYTHON}" experiments/scripts/benchmark_efficiency.py \
  --measure train \
  --training-methods joint,lap \
  --skip-gate-rerun \
  --joint-epochs 5 \
  --lap-epochs 5 \
  --discard-warmup-epochs 1 \
  --seed 20260819 \
  --device "cuda:${selected}" \
  --output-dir experiments/efficiency_results \
  2>&1 | tee -a "${REPO}/experiments/efficiency_results/training_run.log"

echo "[training-wait] done on cuda:${selected}" | tee -a "${LOG}"
