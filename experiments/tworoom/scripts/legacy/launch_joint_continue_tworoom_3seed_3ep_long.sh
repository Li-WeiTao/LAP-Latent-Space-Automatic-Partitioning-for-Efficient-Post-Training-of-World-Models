#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
ROOT=experiments/tworoom
STATUS_DIR="${ROOT}/results/joint_continue_3seed_3ep_long_status"
MASTER_LOG="${STATUS_DIR}/master.log"
PID_FILE="${STATUS_DIR}/launcher.pid"

if [[ "${1:-}" != "--worker" ]]; then
  mkdir -p "${STATUS_DIR}"
  if [[ -s "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}")"
    if kill -0 "${old_pid}" 2>/dev/null; then
      echo "Already running with PID ${old_pid}" >&2
      exit 2
    fi
  fi
  if [[ -s "${MASTER_LOG}" ]]; then
    echo "Refusing to overwrite non-empty ${MASTER_LOG}" >&2
    exit 2
  fi
  nohup bash "$0" --worker > "${MASTER_LOG}" 2>&1 < /dev/null &
  launcher_pid=$!
  echo "${launcher_pid}" > "${PID_FILE}"
  echo "Launched PID ${launcher_pid}; log=${MASTER_LOG}"
  exit 0
fi

echo "[$(date --iso-8601=seconds)] start seeds=0,42,625 epochs=3 precision=fp32 GPUs=0,1,2"

run_one() {
  local gpu="$1"
  local seed="$2"
  local log="${STATUS_DIR}/trainseed${seed}.log"
  echo "[$(date --iso-8601=seconds)] seed=${seed} gpu=${gpu} train_start" > "${log}"
  GPU="${gpu}" TRAIN_SEED="${seed}" \
    bash "${ROOT}/scripts/internal/run_joint_continue_tworoom_3ep.sh" >> "${log}" 2>&1
  status=$?
  if (( status != 0 )); then
    echo "[$(date --iso-8601=seconds)] seed=${seed} train_exit=${status}" >> "${log}"
    return "${status}"
  fi

  echo "[$(date --iso-8601=seconds)] seed=${seed} train_complete eval_start" >> "${log}"
  GPU="${gpu}" TRAIN_SEED="${seed}" \
    bash "${ROOT}/scripts/ablations/run_joint_continue_tworoom_3ep_long_eval.sh" >> "${log}" 2>&1
  status=$?
  echo "[$(date --iso-8601=seconds)] seed=${seed} eval_exit=${status}" >> "${log}"
  return "${status}"
}

run_one 0 0 & p0=$!
run_one 1 42 & p1=$!
run_one 2 625 & p2=$!

status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
wait "${p2}" || status=1
echo "[$(date --iso-8601=seconds)] finished exit=${status}"
exit "${status}"
