#!/usr/bin/env bash
# Detached gate → passport after replay audit (file-based wait, no pgrep self-match).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/pusht/subjepa/env.sh"

pick_free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '
}
GPU_ID="${GPU_ID:-$(pick_free_gpu)}"
echo "[pusht-gate-passport] using GPU_ID=$GPU_ID (override with env GPU_ID=...)" >&2
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$REPO_ROOT/$FORMAL/logs"
LOG="$LOG_DIR/gate_passport_${RUN_ID}.log"
PID_FILE="$LOG_DIR/gate_passport_${RUN_ID}.pid"
WAIT_SCRIPT="$REPO_ROOT/experiments/pusht/subjepa/formal/scripts/wait_for_replay_audit.sh"
GATE_SCRIPT="$REPO_ROOT/experiments/pusht/subjepa/formal/scripts/run_formal_gate.sh"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "gate_passport run already active: pid=$old_pid log=$LOG" >&2
    exit 1
  fi
fi

launch_cmd=(
  env
  PYTHON="$PYTHON"
  GPU_ID="$GPU_ID"
  bash -c "
    set -euo pipefail
    bash \"$WAIT_SCRIPT\"
    echo \"[gate-wait] starting gate\"
    bash \"$GATE_SCRIPT\" gate
    echo \"[gate] finished, starting passport\"
    bash \"$GATE_SCRIPT\" passport
    echo \"[passport] done\"
  "
)

setsid nohup "${launch_cmd[@]}" >>"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[pusht-gate-passport] pid=$pid"
echo "[pusht-gate-passport] log=$LOG"
echo "[pusht-gate-passport] tail: tail -f $LOG"
