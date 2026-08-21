#!/usr/bin/env bash
# Launch Sub-JEPA TwoRoom matrix training detached from Cursor (nohup + setsid).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON="${PYTHON:-/data/sicong/weitao/le-wm/.venv/bin/python}"
LOG_DIR="$REPO_ROOT/experiments/tworoom/subjepa/matrix/logs"
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
  GPU_IDS="$GPU_IDS"
  CPU_THREADS="$CPU_THREADS"
  RUN_ID="$RUN_ID"
  NUM_CLUSTERS="${NUM_CLUSTERS:-3}"
  bash experiments/tworoom/subjepa/matrix/scripts/run_full_matrix.sh training
)

setsid nohup "${launch_cmd[@]}" >>"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[detached] pid=$pid"
echo "[detached] log=$LOG"
echo "[detached] pid_file=$PID_FILE"
echo "[detached] tail: tail -f $LOG"
echo "[detached] stop: kill $pid"
