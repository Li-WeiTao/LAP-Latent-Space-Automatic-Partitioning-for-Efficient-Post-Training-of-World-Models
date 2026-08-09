#!/usr/bin/env bash
# Detached Reacher short+long eval only (official + Global-FT + K-means++ +
# Spectral + Auto-LAP). Created for future use only; not executed as part of
# this change.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/reacher/subjepa/env.sh"

GPU_IDS="${GPU_IDS:-0}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$REPO_ROOT/$MATRIX/logs"
LOG="$LOG_DIR/eval_only_${RUN_ID}.log"
PID_FILE="$LOG_DIR/eval_only_${RUN_ID}.pid"

mkdir -p "$LOG_DIR"

launch_cmd=(
  env
  PYTHON="$PYTHON"
  CHECKPOINT="$CHECKPOINT"
  GPU_IDS="$GPU_IDS"
  CPU_THREADS="$CPU_THREADS"
  RUN_ID="$RUN_ID"
  bash "$REPO_ROOT/experiments/reacher/subjepa/matrix/scripts/run_eval_only.sh"
)

setsid nohup "${launch_cmd[@]}" >>"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[reacher-eval] pid=$pid"
echo "[reacher-eval] gpu_ids=$GPU_IDS"
echo "[reacher-eval] log=$LOG"
echo "[reacher-eval] tail: tail -f $LOG"
