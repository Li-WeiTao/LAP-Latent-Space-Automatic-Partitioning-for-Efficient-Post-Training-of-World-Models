#!/usr/bin/env bash
# Sub-JEPA K=2 global+spectral post-train/eval for TwoRoom, PushT, Reacher, Cube.
# Reuses the existing per-task run_full_matrix.sh with NUM_CLUSTERS=2.
#
# Usage:
#   bash experiments/control_matrix/scripts/run_subjepa_k2.sh [training|eval|all]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

NUM_CLUSTERS="${NUM_CLUSTERS:-2}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE="${1:-all}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
REACHER_CKPT="${REACHER_CKPT:-/data/sicong/weitao/.stable_worldmodel/reacher/subjepa_object.ckpt}"

case "$STAGE" in
  training|eval|all) ;;
  *)
    echo "usage: $0 {training|eval|all}" >&2
    exit 2
    ;;
esac

run_one() {
  local task=$1
  local gpu=$2
  local ctrl="experiments/${task}/subjepa/matrix/scripts/run_full_matrix.sh"
  local logdir="experiments/${task}/subjepa/matrix_k${NUM_CLUSTERS}/logs"
  mkdir -p "$logdir"
  local log="$logdir/k${NUM_CLUSTERS}_${STAGE}_${RUN_ID}.log"
  local pid_file="$logdir/k${NUM_CLUSTERS}_${STAGE}_${RUN_ID}.pid"
  local extra_env=(
    PYTHON="$PYTHON"
    NUM_CLUSTERS="$NUM_CLUSTERS"
    GPU_IDS="$gpu"
    GPU_ID="$gpu"
    CPU_THREADS="$CPU_THREADS"
    RUN_ID="${RUN_ID}_${task}"
  )
  if [[ "$task" == "reacher" ]]; then
    extra_env+=(CHECKPOINT="$REACHER_CKPT")
  fi

  echo "[k2] start task=$task gpu=$gpu log=$log"
  (
    set -euo pipefail
    if [[ "$STAGE" == "training" || "$STAGE" == "all" ]]; then
      env "${extra_env[@]}" bash "$ctrl" training
    fi
    if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
      env "${extra_env[@]}" bash "$ctrl" eval-short
      env "${extra_env[@]}" bash "$ctrl" eval-long
      env "${extra_env[@]}" bash "$ctrl" aggregate
    fi
    echo "[k2] done task=$task"
  ) >>"$log" 2>&1 &
  local pid=$!
  echo "$pid" >"$pid_file"
  echo "[k2] task=$task pid=$pid log=$log"
}

wait_wave() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "K=2 wave failed; inspect experiments/*/subjepa/matrix_k${NUM_CLUSTERS}/logs" >&2
    exit 1
  fi
}

echo "[k2] repo=$REPO_ROOT commit=$(git rev-parse HEAD) stage=$STAGE gpus=$GPU_A,$GPU_B"

run_one tworoom "$GPU_A"
pid_a=$!
run_one pusht "$GPU_B"
pid_b=$!
wait_wave "$pid_a" "$pid_b"

run_one reacher "$GPU_A"
pid_a=$!
run_one cube "$GPU_B"
pid_b=$!
wait_wave "$pid_a" "$pid_b"

echo "[k2] complete stage=$STAGE"
