#!/usr/bin/env bash
# Detached PushT Sub-JEPA smoke (probe → prepare → cache-equiv → train → eval smoke).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

DATASET="${DATASET:-/home/sicong/weitao/datasets/lewm/pusht_expert_train.h5}"
CHECKPOINT="${CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/pusht/subjepa_object.ckpt}"
WORK_ROOT="${WORK_ROOT:-experiments/pusht/subjepa}"
CACHE_DIR="${CACHE_DIR:-/data/sicong/weitao/.stableworldmodel/subjepa/pusht}"

LOG_DIR="$REPO_ROOT/$WORK_ROOT/logs"
LOG="$LOG_DIR/smoke_${RUN_ID}.log"
PID_FILE="$LOG_DIR/smoke_${RUN_ID}.pid"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "smoke already running: pid=$old_pid log=$LOG" >&2
    exit 1
  fi
fi

launch_cmd=(
  env
  PYTHON="$PYTHON"
  CPU_THREADS="$CPU_THREADS"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  PREPARE_OVERWRITE=1
  bash experiments/control_matrix/scripts/run_subjepa_matrix.sh
  --task-spec configs/experiments/tasks/pusht.json
  --dataset "$DATASET"
  --checkpoint "$CHECKPOINT"
  --eval-config-name pusht
  --work-root "$WORK_ROOT"
  --cache-dir "$CACHE_DIR"
  --max-train-starts 4096
  --phase smoke
)

setsid nohup "${launch_cmd[@]}" >>"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[pusht-smoke] pid=$pid"
echo "[pusht-smoke] log=$LOG"
echo "[pusht-smoke] tail: tail -f $LOG"
echo "[pusht-smoke] stop: kill $pid"
