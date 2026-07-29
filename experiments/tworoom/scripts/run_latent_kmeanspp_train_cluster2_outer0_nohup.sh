#!/usr/bin/env bash
# Resume outer=0 cluster2 predictor FT (server restart left 0-byte ckpt).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${SCRIPT_DIR}/run_latent_kmeanspp_train_predictors_50ep.sh"
LOG="experiments/tworoom/results/latent_kmeanspp_cluster2_outer0_seed42.log"
PIDFILE="experiments/tworoom/results/latent_kmeanspp_cluster2_outer0_seed42.pid"

chmod +x "${RUN}"

if [[ -f "${PIDFILE}" ]]; then
  old="$(cat "${PIDFILE}")"
  if kill -0 "${old}" 2>/dev/null; then
    echo "Already running pid=${old}. Log: ${LOG}"
    exit 0
  fi
fi

export GPU="${GPU:-0}"
export OUTER_SEED=0
export ONLY_CLUSTERS=2
setsid nohup env GPU="${GPU}" OUTER_SEED="${OUTER_SEED}" ONLY_CLUSTERS="${ONLY_CLUSTERS}" bash "${RUN}" >> "${LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PIDFILE}"
disown "${pid}" 2>/dev/null || true
echo "Detached pid=${pid} GPU=${GPU} OUTER_SEED=${OUTER_SEED} ONLY_CLUSTERS=${ONLY_CLUSTERS}"
echo "Log: ${LOG}"
