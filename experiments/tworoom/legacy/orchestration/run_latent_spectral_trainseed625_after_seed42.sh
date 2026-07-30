#!/usr/bin/env bash
# Start train seed 625 immediately after the seed-42 training releases GPUs 0/1/2.
# Cache loading is serialized; GPU training remains parallel after each load.
# The seed-42 and seed-625 evaluations run only after all predictor training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
ROOT="experiments/tworoom"
JOB_DIR="/tmp/lewm_spectral_k3_50_trainseed625"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-30}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-21600}"
LOAD_TIMEOUT_SEC="${LOAD_TIMEOUT_SEC:-600}"
mkdir -p "${JOB_DIR}"

seed42_complete() {
  local spectral_seed
  for spectral_seed in 0 1 2; do
    [[ -f "${ROOT}/results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed${spectral_seed}_trainseed42/manifest.json" ]] || return 1
  done
}

start_time="$(date +%s)"
echo "[wait] all three seed-42 predictor manifests" | tee "${JOB_DIR}/train_watcher.log"
while ! seed42_complete; do
  now="$(date +%s)"
  elapsed="$((now - start_time))"
  if (( elapsed >= WAIT_TIMEOUT_SEC )); then
    echo "[timeout] seed-42 predictor training incomplete after ${elapsed}s" | tee -a "${JOB_DIR}/train_watcher.log" >&2
    exit 3
  fi
  sleep "${WAIT_INTERVAL_SEC}"
done

pids=()
for spectral_seed in 0 1 2; do
  gpu="${spectral_seed}"
  out_dir="${ROOT}/results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed${spectral_seed}_trainseed625"
  load_marker="${out_dir}/.cluster0.training.lock"
  manifest="${out_dir}/manifest.json"
  [[ -f "${out_dir}/P_train_global_merged_embeddings.npz" ]] || { echo "missing cache: ${out_dir}" >&2; exit 4; }
  [[ ! -f "${manifest}" ]] || { echo "refusing to overwrite completed run: ${manifest}" >&2; exit 5; }
  [[ ! -f "${load_marker}" ]] || { echo "stale load marker: ${load_marker}" >&2; exit 6; }

  env GPU="${gpu}" SPECTRAL_SEED="${spectral_seed}" TRAIN_SEED=625 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    bash "${ROOT}/scripts/run_latent_spectral_train_predictors_50ep.sh" \
    >"${JOB_DIR}/seed${spectral_seed}.dispatch.log" 2>&1 &
  pids+=("$!")
  echo "[train] spectral_seed=${spectral_seed} gpu=${gpu} pid=${pids[spectral_seed]}" | tee -a "${JOB_DIR}/train_watcher.log"

  load_start="$(date +%s)"
  while [[ ! -f "${load_marker}" ]]; do
    if ! kill -0 "${pids[spectral_seed]}" 2>/dev/null; then
      echo "[failed] spectral_seed=${spectral_seed} exited during cache load; no retry attempted" | tee -a "${JOB_DIR}/train_watcher.log" >&2
      exit 7
    fi
    now="$(date +%s)"
    if (( now - load_start >= LOAD_TIMEOUT_SEC )); then
      echo "[timeout] spectral_seed=${spectral_seed} cache load exceeded ${LOAD_TIMEOUT_SEC}s" | tee -a "${JOB_DIR}/train_watcher.log" >&2
      exit 8
    fi
    sleep 2
  done
  echo "[loaded] spectral_seed=${spectral_seed}; next cache reader may start" | tee -a "${JOB_DIR}/train_watcher.log"
done

failed=0
for index in 0 1 2; do
  if ! wait "${pids[index]}"; then
    echo "[failed] predictor spectral_seed=${index}; no retry attempted" | tee -a "${JOB_DIR}/train_watcher.log" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 9

echo "[ready] all predictor training complete; evaluating train seed 42" | tee -a "${JOB_DIR}/train_watcher.log"
env TRAIN_SEED=42 GPU_LIST="0 1 2" JOB_DIR="/tmp/lewm_spectral_k3_50_trainseed42" \
  bash "${ROOT}/scripts/run_latent_spectral_seed_eval_after_train.sh" \
  >>"${JOB_DIR}/train_watcher.log" 2>&1

echo "[ready] train seed 42 evaluation complete; evaluating train seed 625" | tee -a "${JOB_DIR}/train_watcher.log"
env TRAIN_SEED=625 GPU_LIST="0 1 2" JOB_DIR="${JOB_DIR}" \
  bash "${ROOT}/scripts/run_latent_spectral_seed_eval_after_train.sh" \
  >>"${JOB_DIR}/train_watcher.log" 2>&1
