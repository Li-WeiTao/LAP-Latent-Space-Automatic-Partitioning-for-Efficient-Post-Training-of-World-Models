#!/usr/bin/env bash
# Queue the initial seed-42 Joint-Continue and three Random-Voronoi partitions.
# A worker starts only after its assigned GPU has no substantial allocation.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
ROOT=experiments/tworoom
STATUS_DIR="${ROOT}/results/tworoom_control_baselines_queue"
mkdir -p "${STATUS_DIR}"

wait_for_free_gpu() {
  local gpu="$1"
  local label="$2"
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used < 1000 )); then
      echo "[$(date --iso-8601=seconds)] ${label}: GPU ${gpu} free (${used} MiB)" \
        >> "${STATUS_DIR}/queue.log"
      return
    fi
    sleep 60
  done
}

run_joint() {
  local gpu="$1"
  wait_for_free_gpu "${gpu}" joint_continue_seed42
  if GPU="${gpu}" TRAIN_SEED=42 bash "${ROOT}/scripts/run_joint_continue_tworoom_1ep.sh"; then
    GPU="${gpu}" TRAIN_SEED=42 HORIZON=long bash "${ROOT}/scripts/run_joint_continue_tworoom_eval.sh" && \
      GPU="${gpu}" TRAIN_SEED=42 HORIZON=short bash "${ROOT}/scripts/run_joint_continue_tworoom_eval.sh"
    status=$?
  else
    status=$?
  fi
  echo "[$(date --iso-8601=seconds)] joint_continue_seed42 exit=${status}" \
    >> "${STATUS_DIR}/queue.log"
  return "${status}"
}

run_random() {
  local gpu="$1"
  local partition_seed="$2"
  local label="random_voronoi_seed${partition_seed}_trainseed42"
  wait_for_free_gpu "${gpu}" "${label}"
  if GPU="${gpu}" RANDOM_SEED="${partition_seed}" TRAIN_SEED=42 \
    bash "${ROOT}/scripts/run_random_voronoi_train_predictors_50ep.sh"; then
    GPU="${gpu}" RANDOM_SEED="${partition_seed}" TRAIN_SEED=42 HORIZON=long \
      bash "${ROOT}/scripts/run_random_voronoi_eval.sh" && \
    GPU="${gpu}" RANDOM_SEED="${partition_seed}" TRAIN_SEED=42 HORIZON=short \
      bash "${ROOT}/scripts/run_random_voronoi_eval.sh"
    status=$?
  else
    status=$?
  fi
  echo "[$(date --iso-8601=seconds)] ${label} exit=${status}" \
    >> "${STATUS_DIR}/queue.log"
  return "${status}"
}

echo "[$(date --iso-8601=seconds)] queue started pid=$$" >> "${STATUS_DIR}/queue.log"
run_joint 0 & p0=$!
run_random 1 0 & p1=$!
run_random 2 1 & p2=$!
run_random 3 2 & p3=$!

status=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
  wait "${pid}" || status=1
done
echo "[$(date --iso-8601=seconds)] queue finished exit=${status}" \
  >> "${STATUS_DIR}/queue.log"
exit "${status}"
