#!/usr/bin/env bash
# Wait for one predictor train seed, run three spectral partitions on separate GPUs,
# and compare against the matching K-means++ train-seed results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

ROOT="experiments/tworoom"
TRAIN_SEED="${TRAIN_SEED:?set TRAIN_SEED}"
LATENT_ROUTING="${LATENT_ROUTING:-mpc}"
GPU_LIST="${GPU_LIST:?set GPU_LIST, e.g. '3 4 6'}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-30}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-18000}"
EVAL_STAGGER_SEC="${EVAL_STAGGER_SEC:-20}"
JOB_DIR="${JOB_DIR:-/tmp/lewm_spectral_k3_50_trainseed${TRAIN_SEED}}"
mkdir -p "${JOB_DIR}"
read -r -a gpus <<<"${GPU_LIST}"
[[ "${#gpus[@]}" -eq 3 ]] || { echo "GPU_LIST must contain exactly three ids" >&2; exit 2; }

predictor_dirs=()
for spectral_seed in 0 1 2; do
  predictor_dirs+=("${ROOT}/results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed${spectral_seed}_trainseed${TRAIN_SEED}")
done

all_manifests_complete() {
  local directory
  for directory in "${predictor_dirs[@]}"; do
    [[ -f "${directory}/manifest.json" ]] || return 1
  done
}

training_is_alive() {
  pgrep -f "latent_cluster_train_predictors.py.*tworoom_latent_spectral_spectral_M20000_k30_P16_seed[012]_trainseed${TRAIN_SEED}" >/dev/null
}

start_time="$(date +%s)"
echo "[wait] train_seed=${TRAIN_SEED}; timeout=${WAIT_TIMEOUT_SEC}s" | tee "${JOB_DIR}/seed_eval_watcher.log"
while ! all_manifests_complete; do
  now="$(date +%s)"
  elapsed="$((now - start_time))"
  if (( elapsed >= WAIT_TIMEOUT_SEC )); then
    echo "[timeout] predictors incomplete after ${elapsed}s; evaluation not started" | tee -a "${JOB_DIR}/seed_eval_watcher.log" >&2
    exit 3
  fi
  if ! training_is_alive; then
    echo "[failed] predictor process exited before all manifests were committed" | tee -a "${JOB_DIR}/seed_eval_watcher.log" >&2
    exit 4
  fi
  echo "[wait] elapsed=${elapsed}s" | tee -a "${JOB_DIR}/seed_eval_watcher.log"
  sleep "${WAIT_INTERVAL_SEC}"
done

echo "[ready] all predictor manifests committed" | tee -a "${JOB_DIR}/seed_eval_watcher.log"
pids=()
for spectral_seed in 0 1 2; do
  gpu="${gpus[spectral_seed]}"
  log="${JOB_DIR}/eval_seed${spectral_seed}.dispatch.log"
  env GPU="${gpu}" SPECTRAL_SEED="${spectral_seed}" TRAIN_SEED="${TRAIN_SEED}" LATENT_ROUTING="${LATENT_ROUTING}" \
    bash "${ROOT}/scripts/internal/run_success_rate_5seed_latent_spectral_longrange.sh" >"${log}" 2>&1 &
  pids+=("$!")
  echo "[eval] spectral_seed=${spectral_seed} gpu=${gpu} pid=${pids[spectral_seed]}" | tee -a "${JOB_DIR}/seed_eval_watcher.log"
  if (( spectral_seed < 2 )); then sleep "${EVAL_STAGGER_SEC}"; fi
done

failed=0
for index in 0 1 2; do
  if ! wait "${pids[index]}"; then
    echo "[failed] eval spectral_seed=${index}; no retry attempted" | tee -a "${JOB_DIR}/seed_eval_watcher.log" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 5

method_prefix="${ROOT}/results/tworoom_success_rate_latent_spectral_spectral_M20000_k30_P16"
comparison_prefix="${ROOT}/results/tworoom_success_rate_latent_spectral_k3_50_vs_kmeanspp_R50_trainseed${TRAIN_SEED}_mpc"
reference_suffix="_trainseed${TRAIN_SEED}"
if [[ "${TRAIN_SEED}" == "42" ]]; then reference_suffix=""; fi
python "${ROOT}/aggregate_partition_seed_success.py" \
  --method-name "spectral_k3_50_trainseed${TRAIN_SEED}_mpc" \
  --method-summary "${method_prefix}_seed0_trainseed${TRAIN_SEED}_mpc_5seed_summary.csv" \
  --method-summary "${method_prefix}_seed1_trainseed${TRAIN_SEED}_mpc_5seed_summary.csv" \
  --method-summary "${method_prefix}_seed2_trainseed${TRAIN_SEED}_mpc_5seed_summary.csv" \
  --reference-name "kmeanspp_R50_k3_50_trainseed${TRAIN_SEED}_mpc" \
  --reference-summary "${ROOT}/results/tworoom_success_rate_latent_kmeanspp_kmeanspp_R50_outer0${reference_suffix}_5seed_summary.csv" \
  --reference-summary "${ROOT}/results/tworoom_success_rate_latent_kmeanspp_kmeanspp_R50_outer1${reference_suffix}_5seed_summary.csv" \
  --reference-summary "${ROOT}/results/tworoom_success_rate_latent_kmeanspp_kmeanspp_R50_outer2${reference_suffix}_5seed_summary.csv" \
  --out-json "${comparison_prefix}.json" \
  --out-csv "${comparison_prefix}.csv" \
  | tee -a "${JOB_DIR}/seed_eval_watcher.log"
echo "[done] ${comparison_prefix}.{json,csv}" | tee -a "${JOB_DIR}/seed_eval_watcher.log"
