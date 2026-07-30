#!/usr/bin/env bash
# Wait for the three Spectral K3-50 predictor jobs, run matched 5-seed MPC
# evaluation in parallel, then compare against the existing K-means++ R50 run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

ROOT="experiments/tworoom"
TRAIN_SEED="${TRAIN_SEED:-42}"
LATENT_ROUTING="${LATENT_ROUTING:-mpc}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-60}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-14400}"
JOB_DIR="${JOB_DIR:-/tmp/lewm_spectral_k3_50_trainseed${TRAIN_SEED}}"
mkdir -p "${JOB_DIR}"

if [[ "${TRAIN_SEED}" != "42" || "${LATENT_ROUTING}" != "mpc" ]]; then
  echo "This fair-comparison launcher is pre-registered for TRAIN_SEED=42 and LATENT_ROUTING=mpc" >&2
  exit 2
fi

predictor_dirs=(
  "${ROOT}/results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed0_trainseed42"
  "${ROOT}/results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed1_trainseed42"
  "${ROOT}/results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed2_trainseed42"
)

all_manifests_complete() {
  local directory
  for directory in "${predictor_dirs[@]}"; do
    [[ -f "${directory}/manifest.json" ]] || return 1
  done
  return 0
}

training_is_alive() {
  pgrep -f "latent_cluster_train_predictors.py.*tworoom_latent_spectral_spectral_M20000_k30_P16_seed[012]_trainseed42" >/dev/null
}

start_time="$(date +%s)"
echo "[wait] Spectral K3-50 manifests; timeout=${WAIT_TIMEOUT_SEC}s" | tee "${JOB_DIR}/eval_watcher.log"
while ! all_manifests_complete; do
  now="$(date +%s)"
  elapsed="$((now - start_time))"
  if (( elapsed >= WAIT_TIMEOUT_SEC )); then
    echo "[timeout] predictors incomplete after ${elapsed}s; evaluation was not started" | tee -a "${JOB_DIR}/eval_watcher.log" >&2
    exit 3
  fi
  if ! training_is_alive; then
    echo "[failed] predictor process exited before all manifests were committed; evaluation was not started" | tee -a "${JOB_DIR}/eval_watcher.log" >&2
    exit 4
  fi
  echo "[wait] elapsed=${elapsed}s" | tee -a "${JOB_DIR}/eval_watcher.log"
  sleep "${WAIT_INTERVAL_SEC}"
done

echo "[ready] all schema-v2 predictor manifests committed" | tee -a "${JOB_DIR}/eval_watcher.log"

pids=()
for spectral_seed in 0 1 2; do
  gpu="${spectral_seed}"
  log="${JOB_DIR}/eval_seed${spectral_seed}.dispatch.log"
  env \
    GPU="${gpu}" \
    SPECTRAL_SEED="${spectral_seed}" \
    TRAIN_SEED="${TRAIN_SEED}" \
    LATENT_ROUTING="${LATENT_ROUTING}" \
    bash "${ROOT}/scripts/run_success_rate_5seed_latent_spectral_longrange.sh" \
    >"${log}" 2>&1 &
  pids+=("$!")
  echo "[eval] spectral_seed=${spectral_seed} gpu=${gpu} pid=${pids[spectral_seed]}" | tee -a "${JOB_DIR}/eval_watcher.log"
done

failed=0
for index in 0 1 2; do
  if ! wait "${pids[index]}"; then
    echo "[failed] eval spectral_seed=${index}; no retry was attempted" | tee -a "${JOB_DIR}/eval_watcher.log" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 5
fi

method_prefix="${ROOT}/results/tworoom_success_rate_latent_spectral_spectral_M20000_k30_P16"
comparison_prefix="${ROOT}/results/tworoom_success_rate_latent_spectral_k3_50_vs_kmeanspp_R50_trainseed42_mpc"
python "${ROOT}/aggregate_partition_seed_success.py" \
  --method-name spectral_k3_50_trainseed42_mpc \
  --method-summary "${method_prefix}_seed0_trainseed42_mpc_5seed_summary.csv" \
  --method-summary "${method_prefix}_seed1_trainseed42_mpc_5seed_summary.csv" \
  --method-summary "${method_prefix}_seed2_trainseed42_mpc_5seed_summary.csv" \
  --reference-name kmeanspp_R50_k3_50_trainseed42_mpc \
  --reference-summary "${ROOT}/results/tworoom_success_rate_latent_kmeanspp_kmeanspp_R50_outer0_5seed_summary.csv" \
  --reference-summary "${ROOT}/results/tworoom_success_rate_latent_kmeanspp_kmeanspp_R50_outer1_5seed_summary.csv" \
  --reference-summary "${ROOT}/results/tworoom_success_rate_latent_kmeanspp_kmeanspp_R50_outer2_5seed_summary.csv" \
  --out-json "${comparison_prefix}.json" \
  --out-csv "${comparison_prefix}.csv" \
  | tee -a "${JOB_DIR}/eval_watcher.log"

echo "[done] fair comparison -> ${comparison_prefix}.{json,csv}" | tee -a "${JOB_DIR}/eval_watcher.log"
