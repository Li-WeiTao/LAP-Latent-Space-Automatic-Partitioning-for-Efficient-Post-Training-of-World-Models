#!/usr/bin/env bash
# Wait for the short-horizon reacher matrix controller to finish, then launch
# the long-horizon matrix exactly as PushT did: reuse cache/partitions/training
# via symlinks and re-evaluate with GOAL_OFFSET=50 / EVAL_BUDGET=50.
set -euo pipefail

ROOT="/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models"
cd "$ROOT"

SHORT_ROOT="experiments/reacher/matrix"
LONG_ROOT="experiments/reacher/matrix_long"
SHORT_SUMMARY="$SHORT_ROOT/matrix_summary.csv"
FOLLOW_LOG="$LONG_ROOT/logs/long_after_short.log"
SHORT_WAIT_LOG=${SHORT_WAIT_LOG:-}

mkdir -p "$LONG_ROOT/logs"

log() {
  echo "[$(date -Is)] $*" | tee -a "$FOLLOW_LOG"
}

if [[ -n "$SHORT_WAIT_LOG" && -f "$SHORT_WAIT_LOG" ]]; then
  log "waiting for short eval completion in $SHORT_WAIT_LOG"
  while ! rg -q '^\[complete\]' "$SHORT_WAIT_LOG" 2>/dev/null; do
    sleep 60
  done
else
  log "waiting for short summary: $SHORT_SUMMARY"
  while [[ ! -f "$SHORT_SUMMARY" ]]; do
    sleep 60
  done
fi

if [[ ! -f "$SHORT_SUMMARY" ]]; then
  log "missing short summary: $SHORT_SUMMARY"
  exit 1
fi

log "short matrix complete; preparing long-horizon work root"

ln -sfn ../matrix/preparation "$LONG_ROOT/preparation"
ln -sfn ../matrix/partitions "$LONG_ROOT/partitions"
ln -sfn ../matrix/training "$LONG_ROOT/training"
ln -sfn ../matrix/reacher_lewm_train_latent_cache.npz \
  "$LONG_ROOT/reacher_lewm_train_latent_cache.npz"

RUN_ID="reacher8gpu_long_$(date -u +%Y%m%dT%H%MZ)"
CONTROLLER_LOG="$LONG_ROOT/controller_${RUN_ID}.log"
log "launching long-horizon matrix RUN_ID=$RUN_ID"

common_long=(
  DATASET_NAME=reacher
  DATA_FILE=/data/sicong/weitao/datasets/lewm/reacher.h5
  CHECKPOINT=/data/sicong/weitao/.stable_worldmodel/reacher/lewm_object.ckpt
  EVAL_CONFIG=reacher
  EVAL_DATASET_NAME=reacher
  CACHE_DIR=/data/sicong/weitao/.stable_worldmodel
  WORK_ROOT="$LONG_ROOT"
  PYTHON=/data/sicong/weitao/le-wm/.venv/bin/python
  CPU_THREADS=1
  TRAIN_SEEDS=0,42,625
  PARTITION_SEEDS=0,1,2
  EVAL_SEEDS=0,1,2,3,4
  METHODS=random_voronoi,kmeanspp,spectral
  GOAL_OFFSET=50
  EVAL_BUDGET=50
  TASK_RETRIES=2
  MUJOCO_GL=egl
  PYOPENGL_PLATFORM=egl
)

(
  env "${common_long[@]}" \
    GPU_IDS=3 \
    START_STAGE=official_eval \
    END_STAGE=official_eval \
    RUN_ID="${RUN_ID}_official" \
    bash experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh \
    >>"${CONTROLLER_LOG}.official" 2>&1
  env "${common_long[@]}" \
    GPU_IDS=3,4,5,6 \
    START_STAGE=model_eval \
    END_STAGE=model_eval \
    RUN_ID="${RUN_ID}_model" \
    bash experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh \
    >>"${CONTROLLER_LOG}.model" 2>&1
  env "${common_long[@]}" \
    GPU_IDS=3 \
    START_STAGE=aggregate \
    END_STAGE=aggregate \
    RUN_ID="${RUN_ID}_aggregate" \
    bash experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh \
    >>"${CONTROLLER_LOG}.aggregate" 2>&1
  echo "[complete] run_id=$RUN_ID logs=$LONG_ROOT/logs"
) >"$CONTROLLER_LOG" 2>&1 &

echo $! >"$LONG_ROOT/logs/controller_${RUN_ID}.pid"
echo "$RUN_ID" >"$LONG_ROOT/logs/current_run_id.txt"
log "long controller pid=$! log=$CONTROLLER_LOG"
