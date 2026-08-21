#!/usr/bin/env bash
# Sub-JEPA K=2 global+spectral post-train/eval for TwoRoom, PushT, Reacher, Cube.
# Reuses the existing per-task run_full_matrix.sh with NUM_CLUSTERS=2.
#
# Training packs multiple workers onto one GPU via repeated TRAIN_GPU_IDS
# (default 3,3,3 on GPU 3). Partition and eval stay single-GPU to avoid OOM.
# Tasks run sequentially by default so workers are not multiplied across tasks.
#
# Usage:
#   bash experiments/control_matrix/scripts/run_subjepa_k2.sh [training|eval|all]
#
# Env:
#   TRAIN_GPU=3          GPU for training workers (default 3)
#   TRAIN_WORKERS=3      parallel training jobs per task (default 3)
#   EVAL_GPU=3           GPU for partition+eval (default 3)
#   TASKS=tworoom,...    task order (default all four)
#   CPU_THREADS=4        threads per leaf job
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

NUM_CLUSTERS="${NUM_CLUSTERS:-2}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE="${1:-all}"

TRAIN_GPU="${TRAIN_GPU:-3}"
TRAIN_WORKERS="${TRAIN_WORKERS:-3}"
EVAL_GPU="${EVAL_GPU:-3}"
REACHER_CKPT="${REACHER_CKPT:-/data/sicong/weitao/.stable_worldmodel/reacher/subjepa_object.ckpt}"
TASKS="${TASKS:-tworoom,pusht,reacher,cube}"

build_repeated_gpu_ids() {
  local gpu=$1
  local workers=$2
  local i out=""
  for ((i = 0; i < workers; i++)); do
    out+="${gpu},"
  done
  echo "${out%,}"
}

TRAIN_GPU_IDS="$(build_repeated_gpu_ids "$TRAIN_GPU" "$TRAIN_WORKERS")"
EVAL_GPU_IDS="$EVAL_GPU"
PARTITION_GPU_IDS="$EVAL_GPU"

export TRAIN_GPU TRAIN_WORKERS EVAL_GPU TRAIN_GPU_IDS EVAL_GPU_IDS PARTITION_GPU_IDS

case "$STAGE" in
  training|eval|all) ;;
  *)
    echo "usage: $0 {training|eval|all}" >&2
    exit 2
    ;;
esac

LOG_ROOT="$REPO_ROOT/experiments/control_matrix/assets/subjepa_k2_logs"
mkdir -p "$LOG_ROOT"
ORCH_LOG="$LOG_ROOT/orchestrator_${RUN_ID}.log"
ORCH_PID="$LOG_ROOT/orchestrator_${RUN_ID}.pid"

run_task() {
  local task=$1
  local ctrl="experiments/${task}/subjepa/matrix/scripts/run_full_matrix.sh"
  local logdir="experiments/${task}/subjepa/matrix_k${NUM_CLUSTERS}/logs"
  mkdir -p "$logdir"
  local log="$logdir/k${NUM_CLUSTERS}_${STAGE}_${RUN_ID}.log"
  local extra_env=(
    PYTHON="$PYTHON"
    NUM_CLUSTERS="$NUM_CLUSTERS"
    GPU_ID="$EVAL_GPU"
    CPU_THREADS="$CPU_THREADS"
    RUN_ID="${RUN_ID}_${task}"
    TRAIN_GPU_IDS="$TRAIN_GPU_IDS"
    EVAL_GPU_IDS="$EVAL_GPU_IDS"
    PARTITION_GPU_IDS="$PARTITION_GPU_IDS"
  )
  if [[ "$task" == "reacher" ]]; then
    extra_env+=(CHECKPOINT="$REACHER_CKPT")
  fi

  echo "[k2] task=$task train_gpus=$TRAIN_GPU_IDS eval_gpu=$EVAL_GPU_IDS log=$log"
  if [[ "$STAGE" == "training" || "$STAGE" == "all" ]]; then
    env "${extra_env[@]}" bash "$ctrl" training >>"$log" 2>&1
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    env "${extra_env[@]}" bash "$ctrl" eval-short >>"$log" 2>&1
    env "${extra_env[@]}" bash "$ctrl" eval-long >>"$log" 2>&1
    env "${extra_env[@]}" bash "$ctrl" aggregate >>"$log" 2>&1
  fi
  echo "[k2] done task=$task"
}

{
  echo "[k2] repo=$REPO_ROOT"
  echo "[k2] commit=$(git rev-parse HEAD)"
  echo "[k2] stage=$STAGE"
  echo "[k2] train_gpu_ids=$TRAIN_GPU_IDS"
  echo "[k2] eval_gpu=$EVAL_GPU"
  echo "[k2] tasks=$TASKS"
  IFS=, read -r -a task_list <<< "$TASKS"
  for task in "${task_list[@]}"; do
    run_task "$task"
  done
  echo "[k2] complete stage=$STAGE"
} >>"$ORCH_LOG" 2>&1 &

orch_pid=$!
echo "$orch_pid" >"$ORCH_PID"
echo "[k2] orchestrator pid=$orch_pid"
echo "[k2] log=$ORCH_LOG"
echo "[k2] tail: tail -f $ORCH_LOG"
