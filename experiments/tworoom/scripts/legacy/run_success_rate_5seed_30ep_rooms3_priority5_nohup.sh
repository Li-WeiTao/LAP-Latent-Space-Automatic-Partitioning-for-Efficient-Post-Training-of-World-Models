#!/usr/bin/env bash
# Detached launcher for short-range 30ep rooms3/priority5 rerun (epoch30 ckpt override).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${SCRIPT_DIR}/../ablations/run_success_rate_5seed_30ep_rooms3_priority5.sh"
LOG="experiments/tworoom/results/success_rate_30ep_short_seed42.log"
PIDFILE="experiments/tworoom/results/success_rate_30ep_short_seed42.pid"

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
