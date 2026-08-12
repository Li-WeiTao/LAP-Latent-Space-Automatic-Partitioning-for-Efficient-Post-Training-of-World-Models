#!/usr/bin/env bash
# Detached OGBench Cube Sub-JEPA matrix training (nohup + setsid).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/cube/subjepa/env.sh"

GPU_IDS="${GPU_IDS:-0}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$REPO_ROOT/$MATRIX/logs"
LOG="$LOG_DIR/detached_${RUN_ID}.log"
PID_FILE="$LOG_DIR/detached_${RUN_ID}.pid"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "detached run already active: pid=$old_pid log=$LOG" >&2
    exit 1
  fi
fi

launch_cmd=(
  env
  PYTHON="$PYTHON"
  CHECKPOINT="$CHECKPOINT"
  GPU_IDS="$GPU_IDS"
  GPU_ID="${GPU_ID:-${GPU_IDS%%,*}}"
  CPU_THREADS="$CPU_THREADS"
  RUN_ID="$RUN_ID"
  bash "$REPO_ROOT/experiments/cube/subjepa/matrix/scripts/run_full_matrix.sh" training
)

setsid nohup "${launch_cmd[@]}" >>"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[cube-matrix] pid=$pid"
echo "[cube-matrix] log=$LOG"
echo "[cube-matrix] tail: tail -f $LOG"
echo "[cube-matrix] stop: kill $pid"
