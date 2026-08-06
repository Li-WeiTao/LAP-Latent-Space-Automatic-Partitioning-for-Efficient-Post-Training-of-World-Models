#!/usr/bin/env bash
# Detached long-horizon eval only (official + Global-FT + K-means++ + Spectral), 7-GPU parallel.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
LOG_DIR="$REPO_ROOT/experiments/tworoom/subjepa/matrix/logs"
LOG="$LOG_DIR/eval_long_${RUN_ID}.log"
PID_FILE="$LOG_DIR/eval_long_${RUN_ID}.pid"

mkdir -p "$LOG_DIR"

launch_cmd=(
  env
  PYTHON="$PYTHON"
  GPU_IDS="$GPU_IDS"
  CPU_THREADS="$CPU_THREADS"
  RUN_ID="$RUN_ID"
  bash experiments/tworoom/subjepa/matrix/scripts/run_full_matrix.sh eval-long
)

setsid nohup "${launch_cmd[@]}" >>"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[eval-long-detached] pid=$pid"
echo "[eval-long-detached] gpu_ids=$GPU_IDS"
echo "[eval-long-detached] log=$LOG"
echo "[eval-long-detached] tail: tail -f $LOG"
