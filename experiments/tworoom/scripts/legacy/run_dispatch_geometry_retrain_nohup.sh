#!/usr/bin/env bash
# Launch GPU-pool retrain dispatcher detached from IDE terminals (survives Cursor close).
#
# Usage:
#   bash experiments/tworoom/scripts/legacy/run_dispatch_geometry_retrain_nohup.sh
#   GPUS="0 1 2 3 4 6 7" bash ...   # omit GPU 5 if manually occupied
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISPATCH="${SCRIPT_DIR}/dispatch_geometry_retrain.sh"
LOG="experiments/tworoom/results/retrain_dispatch.log"
PIDFILE="experiments/tworoom/results/retrain_dispatch.pid"

if [[ ! -x "${DISPATCH}" ]]; then
  chmod +x "${DISPATCH}"
fi

if [[ -f "${PIDFILE}" ]]; then
  old_pid="$(cat "${PIDFILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "Dispatcher already running (pid=${old_pid}). Log: ${LOG}"
    exit 0
  fi
fi

export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
export SEED="${SEED:-42}"

setsid nohup env GPUS="${GPUS}" SEED="${SEED}" bash "${DISPATCH}" >> "${LOG}" 2>&1 < /dev/null &
disp_pid=$!
echo "${disp_pid}" > "${PIDFILE}"
disown "${disp_pid}" 2>/dev/null || true

echo "Detached dispatcher pid=${disp_pid}"
echo "Log: ${LOG}"
echo "GPUs: ${GPUS}"
