#!/usr/bin/env bash
# Wait for Sub-JEPA K=4 spectral training (36 jobs), then run eval-long on GPU 0/5.
#
# Layout (matches training):
#   GPU 0: tworoom -> pusht (serial)
#   GPU 5: reacher -> cube (serial)
#
# Usage:
#   bash experiments/control_matrix/scripts/run_subjepa_k4_eval_long_when_ready.sh
#
# Env:
#   TRAIN_RUN_ID=20260824T024548Z   training run to wait on
#   GPU0=0 GPU5=5                  eval GPUs (never GPU 4 by default)
#   POLL_SEC=60
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
CPU_THREADS="${CPU_THREADS:-4}"
NUM_CLUSTERS="${NUM_CLUSTERS:-4}"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-20260824T024548Z}"
EVAL_RUN_ID="${EVAL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU0="${GPU0:-0}"
GPU5="${GPU5:-5}"
POLL_SEC="${POLL_SEC:-60}"
REACHER_CKPT="${REACHER_CKPT:-/data/sicong/weitao/.stable_worldmodel/reacher/subjepa_object.ckpt}"

LOG_ROOT="$REPO_ROOT/experiments/control_matrix/assets/subjepa_k4_logs"
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/eval_long_when_ready_${EVAL_RUN_ID}.log"
PID_FILE="$LOG_ROOT/eval_long_when_ready_${EVAL_RUN_ID}.pid"

TASKS_GPU0=(tworoom pusht)
TASKS_GPU5=(reacher cube)
EXPECTED_JOBS=36
EXPECTED_PER_TASK=9

count_training_manifests() {
  local task=$1
  find "$REPO_ROOT/experiments/${task}/subjepa/matrix_k${NUM_CLUSTERS}/training/spectral" \
    -name manifest.json 2>/dev/null | wc -l
}

count_all_manifests() {
  local total=0 n
  for task in tworoom pusht reacher cube; do
    n="$(count_training_manifests "$task")"
    total=$((total + n))
  done
  echo "$total"
}

training_processes_running() {
  ps aux | rg 'matrix_k4' | rg 'train_predictors|phase train_regions' | rg -v rg >/dev/null
}

task_training_complete() {
  local task=$1
  local detached="$REPO_ROOT/experiments/${task}/subjepa/matrix_k${NUM_CLUSTERS}/logs/detached_${TRAIN_RUN_ID}.log"
  [[ -f "$detached" ]] && rg -q '\[complete\]' "$detached"
}

all_training_complete() {
  local total n task
  total="$(count_all_manifests)"
  [[ "$total" -ge "$EXPECTED_JOBS" ]] || return 1
  for task in tworoom pusht reacher cube; do
    n="$(count_training_manifests "$task")"
    [[ "$n" -ge "$EXPECTED_PER_TASK" ]] || return 1
    task_training_complete "$task" || return 1
  done
  if training_processes_running; then
    return 1
  fi
  return 0
}

run_eval_long_task() {
  local task=$1 gpu=$2
  local ctrl="experiments/${task}/subjepa/matrix/scripts/run_full_matrix.sh"
  local log="$REPO_ROOT/experiments/${task}/subjepa/matrix_k${NUM_CLUSTERS}/logs/eval_long_${EVAL_RUN_ID}.log"
  mkdir -p "$(dirname "$log")"
  local extra=(
    PYTHON="$PYTHON"
    NUM_CLUSTERS="$NUM_CLUSTERS"
    CPU_THREADS="$CPU_THREADS"
    RUN_ID="${EVAL_RUN_ID}_${task}"
    EVAL_GPU_IDS="$gpu"
    EVAL_GPU="$gpu"
    GPU_ID="$gpu"
    PARTITION_GPU_IDS="$gpu"
  )
  if [[ "$task" == "reacher" ]]; then
    extra+=(CHECKPOINT="$REACHER_CKPT")
  fi
  echo "[k4-eval-long] start task=$task gpu=$gpu run_id=${EVAL_RUN_ID}_${task} log=$log"
  env "${extra[@]}" bash "$ctrl" eval-long >>"$log" 2>&1
  echo "[k4-eval-long] done task=$task gpu=$gpu"
}

gpu0_lane() {
  local task
  for task in "${TASKS_GPU0[@]}"; do
    run_eval_long_task "$task" "$GPU0"
  done
}

gpu5_lane() {
  local task
  for task in "${TASKS_GPU5[@]}"; do
    run_eval_long_task "$task" "$GPU5"
  done
}

{
  echo "[k4-eval-wait] repo=$REPO_ROOT"
  echo "[k4-eval-wait] train_run_id=$TRAIN_RUN_ID eval_run_id=$EVAL_RUN_ID"
  echo "[k4-eval-wait] layout gpu${GPU0}=${TASKS_GPU0[*]} gpu${GPU5}=${TASKS_GPU5[*]}"
  echo "[k4-eval-wait] waiting for ${EXPECTED_JOBS} training manifests + no matrix_k4 trainers"

  while ! all_training_complete; do
    total="$(count_all_manifests)"
    echo "[k4-eval-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) manifests=${total}/${EXPECTED_JOBS} tworoom=$(count_training_manifests tworoom)/9 pusht=$(count_training_manifests pusht)/9 reacher=$(count_training_manifests reacher)/9 cube=$(count_training_manifests cube)/9 trainers_running=$(training_processes_running && echo yes || echo no)"
    sleep "$POLL_SEC"
  done

  echo "[k4-eval-wait] training complete; sleeping 30s for GPU cleanup"
  sleep 30

  echo "[k4-eval-long] launching eval-long run_id=$EVAL_RUN_ID"
  gpu0_lane &
  pid0=$!
  gpu5_lane &
  pid5=$!
  wait "$pid0"
  wait "$pid5"
  echo "[k4-eval-long] complete run_id=$EVAL_RUN_ID"
} >>"$LOG" 2>&1 &

wait_pid=$!
echo "$wait_pid" >"$PID_FILE"
echo "[k4-eval-wait] scheduled pid=$wait_pid"
echo "[k4-eval-wait] log=$LOG"
echo "[k4-eval-wait] tail: tail -f $LOG"