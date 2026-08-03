#!/usr/bin/env bash
# Stop duplicate/orphan matrix training workers. Keeps the newest parallel run only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
WORK_ROOT="${1:-experiments/tworoom/subjepa/matrix}"
LOG_DIR="$REPO_ROOT/$WORK_ROOT/logs"

latest_run=""
latest_mtime=0
for env_file in "$LOG_DIR"/parallel_*/run.env; do
  [[ -f "$env_file" ]] || continue
  mtime=$(stat -c %Y "$env_file")
  if [[ "$mtime" -gt "$latest_mtime" ]]; then
    latest_mtime=$mtime
    latest_run=$(basename "$(dirname "$env_file")")
  fi
done

if [[ -z "$latest_run" ]]; then
  echo "[stop] no parallel runs found under $LOG_DIR"
  exit 0
fi

keep_controller=""
keep_pid_file="$LOG_DIR/${latest_run}/controller.pid"
[[ -f "$keep_pid_file" ]] && keep_controller=$(cat "$keep_pid_file")

echo "[stop] keeping $latest_run controller=${keep_controller:-unknown}"

stopped=0
for env_file in "$LOG_DIR"/parallel_*/run.env; do
  run_id=$(basename "$(dirname "$env_file")")
  [[ "$run_id" == "$latest_run" ]] && continue
  pid_file="$LOG_DIR/${run_id}/controller.pid"
  if [[ -f "$pid_file" ]]; then
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop] killing stale controller pid=$pid run=$run_id"
      kill "$pid" 2>/dev/null || true
      pkill -P "$pid" 2>/dev/null || true
      stopped=$((stopped + 1))
    fi
  fi
done

if [[ -n "$keep_controller" ]] && kill -0 "$keep_controller" 2>/dev/null; then
  keep_tree=$(pstree -p "$keep_controller" 2>/dev/null || true)
  for pid in $(pgrep -f "$WORK_ROOT/preparation/embedding_cache.npz" 2>/dev/null || true); do
    cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
    [[ "$cmd" == *train_predictors.py* ]] || continue
    if [[ "$keep_tree" != *"($pid)"* ]]; then
      echo "[stop] killing orphan train_predictors pid=$pid"
      kill "$pid" 2>/dev/null || true
      stopped=$((stopped + 1))
    fi
  done
fi

echo "[stop] done (actions=$stopped, active_run=$latest_run)"
