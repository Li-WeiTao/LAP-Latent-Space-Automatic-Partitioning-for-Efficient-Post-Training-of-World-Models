#!/usr/bin/env bash
# Wait for an idle GPU, then run the formal 50-repeat inference benchmark.
set -euo pipefail

REPO="/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models"
PYTHON="/data/sicong/weitao/le-wm/.venv/bin/python"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
GPU_INDEX="${GPU_INDEX:-}"
LOG="${REPO}/experiments/efficiency_results/inference_wait.log"

cd "${REPO}"
export PYTHONPATH="experiments/scripts:.:experiments/tworoom:/data/sicong/weitao/le-wm"

if [[ -f "${REPO}/experiments/efficiency_results/inference_run.log" ]]; then
  mv -f "${REPO}/experiments/efficiency_results/inference_run.log" \
    "${REPO}/experiments/efficiency_results/inference_run_shared_gpu.partial.log" 2>/dev/null || true
fi

selected="$(
  MIN_FREE_MIB="${MIN_FREE_MIB}" \
  MAX_UTIL_PCT="${MAX_UTIL_PCT}" \
  GPU_INDEX="${GPU_INDEX}" \
  LOG="${LOG}" \
  bash "${REPO}/experiments/scripts/wait_for_gpu.sh"
)"

echo "[inference-wait] starting formal inference on cuda:${selected}" | tee -a "${LOG}"
"${PYTHON}" experiments/scripts/benchmark_efficiency.py \
  --measure inference \
  --skip-gate-rerun \
  --warmup 20 \
  --repeats 50 \
  --seed 20260819 \
  --device "cuda:${selected}" \
  --inference-tasks tworoom,pusht,reacher,cube \
  --output-dir experiments/efficiency_results \
  2>&1 | tee "${REPO}/experiments/efficiency_results/inference_run.log"

"${PYTHON}" experiments/scripts/benchmark_efficiency.py \
  --aggregate-only \
  --output-dir experiments/efficiency_results

echo "[inference-wait] done on cuda:${selected}" | tee -a "${LOG}"
