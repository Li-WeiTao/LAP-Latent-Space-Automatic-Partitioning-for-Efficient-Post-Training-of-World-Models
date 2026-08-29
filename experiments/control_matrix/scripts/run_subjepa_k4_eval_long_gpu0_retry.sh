#!/usr/bin/env bash
# Retry Sub-JEPA K=4 eval-long for tworoom -> pusht on GPU 0 (serial).
#
# Usage:
#   bash experiments/control_matrix/scripts/run_subjepa_k4_eval_long_gpu0_retry.sh
#
# Env:
#   EVAL_RUN_ID   defaults to new UTC timestamp
#   GPU0=0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
CPU_THREADS="${CPU_THREADS:-4}"
NUM_CLUSTERS="${NUM_CLUSTERS:-4}"
EVAL_RUN_ID="${EVAL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU0="${GPU0:-0}"

LOG_ROOT="$REPO_ROOT/experiments/control_matrix/assets/subjepa_k4_logs"
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/eval_long_gpu0_retry_${EVAL_RUN_ID}.log"
PID_FILE="$LOG_ROOT/eval_long_gpu0_retry_${EVAL_RUN_ID}.pid"

run_eval_long_task() {
  local task=$1
  local ctrl="experiments/${task}/subjepa/matrix/scripts/run_full_matrix.sh"
  local task_log="$REPO_ROOT/experiments/${task}/subjepa/matrix_k${NUM_CLUSTERS}/logs/eval_long_${EVAL_RUN_ID}.log"
  mkdir -p "$(dirname "$task_log")"
  echo "[k4-gpu0-retry] start task=$task gpu=$GPU0 run_id=${EVAL_RUN_ID}_${task} log=$task_log"
  env \
    PYTHON="$PYTHON" \
    NUM_CLUSTERS="$NUM_CLUSTERS" \
    CPU_THREADS="$CPU_THREADS" \
    RUN_ID="${EVAL_RUN_ID}_${task}" \
    EVAL_GPU_IDS="$GPU0" \
    EVAL_GPU="$GPU0" \
    GPU_ID="$GPU0" \
    GPU_IDS="$GPU0" \
    PARTITION_GPU_IDS="$GPU0" \
    CUDA_VISIBLE_DEVICES="$GPU0" \
    bash "$ctrl" eval-long >>"$task_log" 2>&1
  echo "[k4-gpu0-retry] done task=$task gpu=$GPU0"
}

{
  echo "[k4-gpu0-retry] repo=$REPO_ROOT eval_run_id=$EVAL_RUN_ID gpu=$GPU0"
  echo "[k4-gpu0-retry] tasks=tworoom pusht (serial)"
  run_eval_long_task tworoom
  run_eval_long_task pusht
  echo "[k4-gpu0-retry] complete eval_run_id=$EVAL_RUN_ID"
} >>"$LOG" 2>&1 &

retry_pid=$!
echo "$retry_pid" >"$PID_FILE"
echo "[k4-gpu0-retry] scheduled pid=$retry_pid"
echo "[k4-gpu0-retry] log=$LOG"
echo "[k4-gpu0-retry] tail: tail -f $LOG"
