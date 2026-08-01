#!/usr/bin/env bash
# Wait for the short-horizon reacher matrix controller to finish, then launch
# the long-horizon matrix exactly as PushT did: reuse cache/partitions/training
# via symlinks and re-evaluate with GOAL_OFFSET=50 / EVAL_BUDGET=50.
set -euo pipefail

ROOT="/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models"
cd "$ROOT"

SHORT_ROOT="experiments/reacher/matrix"
LONG_ROOT="experiments/reacher/matrix_long"
SHORT_PID_FILE="$SHORT_ROOT/logs/controller_reacher8gpu_20260801T100349Z.pid"
SHORT_LOG="$SHORT_ROOT/logs/controller_reacher8gpu_20260801T100349Z.log"
FOLLOW_LOG="$LONG_ROOT/logs/long_after_short.log"

mkdir -p "$LONG_ROOT/logs"

log() {
  echo "[$(date -Is)] $*" | tee -a "$FOLLOW_LOG"
}

SHORT_PID=$(cat "$SHORT_PID_FILE")
log "waiting for short controller pid=$SHORT_PID"

while kill -0 "$SHORT_PID" 2>/dev/null; do
  sleep 60
done

if ! rg -q '^\[complete\]' "$SHORT_LOG"; then
  log "short controller exited without [complete]; aborting long horizon"
  exit 1
fi
if [[ ! -f "$SHORT_ROOT/matrix_summary.csv" ]]; then
  log "missing short summary: $SHORT_ROOT/matrix_summary.csv"
  exit 1
fi

log "short matrix complete; preparing long-horizon work root"

ln -sfn ../matrix/preparation "$LONG_ROOT/preparation"
ln -sfn ../matrix/partitions "$LONG_ROOT/partitions"
ln -sfn ../matrix/training "$LONG_ROOT/training"
ln -sfn ../matrix/reacher_lewm_train_latent_cache.npz \
  "$LONG_ROOT/reacher_lewm_train_latent_cache.npz"

RUN_ID="reacher8gpu_long_$(date -u +%Y%m%dT%H%MZ)"
log "launching long-horizon matrix RUN_ID=$RUN_ID"

nohup env \
  DATASET_NAME=reacher \
  DATA_FILE=/data/sicong/weitao/datasets/lewm/reacher.h5 \
  CHECKPOINT=/data/sicong/weitao/.stable_worldmodel/reacher/lewm_object.ckpt \
  EVAL_CONFIG=reacher \
  EVAL_DATASET_NAME=reacher \
  CACHE_DIR=/data/sicong/weitao/.stable_worldmodel \
  WORK_ROOT="$LONG_ROOT" \
  PYTHON=/data/sicong/weitao/le-wm/.venv/bin/python \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  CPU_THREADS=4 \
  GOAL_OFFSET=50 \
  EVAL_BUDGET=50 \
  RUN_ID="$RUN_ID" \
  bash experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh \
  >"$LONG_ROOT/controller_${RUN_ID}.log" 2>&1 &

echo $! >"$LONG_ROOT/logs/controller_${RUN_ID}.pid"
echo "$RUN_ID" >"$LONG_ROOT/logs/current_run_id.txt"
log "long controller pid=$! log=$LONG_ROOT/controller_${RUN_ID}.log"
