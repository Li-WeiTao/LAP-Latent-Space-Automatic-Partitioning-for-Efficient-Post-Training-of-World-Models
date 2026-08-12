#!/usr/bin/env bash
# Detached OGBench Cube Sub-JEPA formal gate (prepare -> gate).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/cube/subjepa/env.sh"

GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$REPO_ROOT/$FORMAL/logs"
LOG="$LOG_DIR/formal_${RUN_ID}.log"
PID_FILE="$LOG_DIR/formal_${RUN_ID}.pid"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "formal run already active: pid=$old_pid log=$LOG" >&2
    exit 1
  fi
fi

phase="${1:-all}"

launch_cmd=(
  env
  PYTHON="$PYTHON"
  CHECKPOINT="$CHECKPOINT"
  GPU_ID="$GPU_ID"
  bash "$REPO_ROOT/experiments/cube/subjepa/formal/scripts/run_cube_gate.sh" "$phase"
)

setsid nohup "${launch_cmd[@]}" >>"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[cube-formal] pid=$pid phase=$phase"
echo "[cube-formal] log=$LOG"
echo "[cube-formal] tail: tail -f $LOG"
echo "[cube-formal] stop: kill $pid"
