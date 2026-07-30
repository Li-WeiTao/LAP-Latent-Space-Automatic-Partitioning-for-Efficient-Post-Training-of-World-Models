#!/usr/bin/env bash
# Detached launcher for Global-FT 50ep (latent main experiment strong baseline).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${SCRIPT_DIR}/run_geometry_train_global_ft_50ep.sh"
LOG="experiments/tworoom/results/tworoom_geometry_train_global_ft_50ep/train_nohup.log"
PIDFILE="experiments/tworoom/results/tworoom_geometry_train_global_ft_50ep/train.pid"

mkdir -p "$(dirname "${LOG}")"
chmod +x "${RUN}"

if [[ -f "${PIDFILE}" ]]; then
  old="$(cat "${PIDFILE}")"
  if kill -0 "${old}" 2>/dev/null; then
    echo "Already running pid=${old}. Log: ${LOG}"
    exit 0
  fi
fi

export GPU="${GPU:-0}"
setsid nohup env GPU="${GPU}" bash "${RUN}" >> "${LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PIDFILE}"
disown "${pid}" 2>/dev/null || true
echo "Detached pid=${pid} GPU=${GPU}"
echo "Log: ${LOG}"
echo "Train: results/tworoom_geometry_train_global_ft_50ep/train_global_ft_50ep.log"
