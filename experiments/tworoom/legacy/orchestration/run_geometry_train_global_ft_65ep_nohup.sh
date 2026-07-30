#!/usr/bin/env bash
# Detached Global-FT training: survives closing Cursor/terminal.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ROOT="$REPO_ROOT"
OUT="${ROOT}/experiments/tworoom/results/tworoom_geometry_train_global_ft_65ep"
NOHUP_LOG="${OUT}/nohup.out"
PID_FILE="${OUT}/train.pid"

mkdir -p "${OUT}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "Already running (pid=${old_pid}). Stop it first or delete ${PID_FILE} if stale."
    exit 1
  fi
fi

cd "${ROOT}"
nohup bash experiments/tworoom/scripts/run_geometry_train_global_ft_65ep.sh \
  >> "${NOHUP_LOG}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started Global-FT training in background."
echo "  pid:      $(cat "${PID_FILE}")"
echo "  nohup:    ${NOHUP_LOG}"
echo "  train:    ${OUT}/train_global_ft_65ep.log"
echo "Check: tail -f ${OUT}/train_global_ft_65ep.log"
echo "Stop:  kill \$(cat ${PID_FILE})"
