#!/usr/bin/env bash
# Detached long-horizon eval only for OGBench Cube.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/cube/subjepa/env.sh"

GPU_IDS="${GPU_IDS:-0}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$REPO_ROOT/$MATRIX/logs"
LOG="$LOG_DIR/eval_long_${RUN_ID}.log"
PID_FILE="$LOG_DIR/eval_long_${RUN_ID}.pid"

mkdir -p "$LOG_DIR"

launch_cmd=(
  env
  PYTHON="$PYTHON"
  CHECKPOINT="$CHECKPOINT"
  GPU_IDS="$GPU_IDS"
  CPU_THREADS="$CPU_THREADS"
  RUN_ID="$RUN_ID"
  bash "$REPO_ROOT/experiments/cube/subjepa/matrix/scripts/run_full_matrix.sh" eval-long
)

setsid nohup "${launch_cmd[@]}" >>"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[cube-eval-long] pid=$pid"
echo "[cube-eval-long] gpu_ids=$GPU_IDS"
echo "[cube-eval-long] log=$LOG"
echo "[cube-eval-long] tail: tail -f $LOG"
