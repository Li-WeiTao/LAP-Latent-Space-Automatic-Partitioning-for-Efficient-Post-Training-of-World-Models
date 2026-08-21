#!/usr/bin/env bash
# Detached PushT Sub-JEPA matrix training (nohup + setsid).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/pusht/subjepa/env.sh"

K3_MATRIX="${K3_MATRIX:-$MATRIX}"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/control_matrix/scripts/subjepa_k_variant.sh"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
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
  GPU_IDS="$GPU_IDS"
  CPU_THREADS="$CPU_THREADS"
  RUN_ID="$RUN_ID"
  NUM_CLUSTERS="$NUM_CLUSTERS"
  bash "$REPO_ROOT/experiments/pusht/subjepa/matrix/scripts/run_full_matrix.sh" training
)

setsid nohup "${launch_cmd[@]}" >>"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[pusht-matrix] pid=$pid"
echo "[pusht-matrix] log=$LOG"
echo "[pusht-matrix] tail: tail -f $LOG"
echo "[pusht-matrix] stop: kill $pid"
