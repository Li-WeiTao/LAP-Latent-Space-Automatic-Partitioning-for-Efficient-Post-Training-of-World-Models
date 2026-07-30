#!/usr/bin/env bash
# Re-run the three predictor-train seeds after the inference-tensor cache fix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
ROOT="experiments/tworoom"
RUNNER="${ROOT}/scripts/legacy/run_latent_spectral_seed_eval_after_train.sh"
JOB_ROOT="/tmp/lewm_spectral_k3_50_eval_rerun_inference_fix"
mkdir -p "${JOB_ROOT}"

run_seed() {
  local train_seed="$1"
  local gpu_list="$2"
  local job_dir="${JOB_ROOT}/trainseed${train_seed}"
  mkdir -p "${job_dir}"
  env TRAIN_SEED="${train_seed}" GPU_LIST="${gpu_list}" JOB_DIR="${job_dir}" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    timeout 7200s bash "${RUNNER}" >"${job_dir}/dispatch.log" 2>&1
}

echo "[start] train seeds 0 and 42 at $(date -Is)" | tee "${JOB_ROOT}/status.log"
run_seed 0 "3 4 6" &
pid0="$!"
run_seed 42 "0 1 2" &
pid42="$!"

failed=0
if ! wait "${pid0}"; then
  echo "[failed] train seed 0 evaluation; no further retry attempted" | tee -a "${JOB_ROOT}/status.log" >&2
  failed=1
fi
if ! wait "${pid42}"; then
  echo "[failed] train seed 42 evaluation; no further retry attempted" | tee -a "${JOB_ROOT}/status.log" >&2
  failed=1
fi
(( failed == 0 )) || exit 1

echo "[start] train seed 625 at $(date -Is)" | tee -a "${JOB_ROOT}/status.log"
if ! run_seed 625 "0 1 2"; then
  echo "[failed] train seed 625 evaluation; no further retry attempted" | tee -a "${JOB_ROOT}/status.log" >&2
  exit 2
fi
echo "[done] all three train seeds at $(date -Is)" | tee -a "${JOB_ROOT}/status.log"
