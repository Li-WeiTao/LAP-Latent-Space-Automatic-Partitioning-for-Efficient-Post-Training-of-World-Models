#!/usr/bin/env bash
# Fresh short-horizon reacher matrix eval, then queue long-horizon eval.
# Official baselines run serially on one GPU; model eval uses four GPUs.
set -euo pipefail

ROOT="/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models"
cd "$ROOT"

SHORT_ROOT="experiments/reacher/matrix"
LONG_ROOT="experiments/reacher/matrix_long"
LONG_LOG_ROOT="$LONG_ROOT/logs"
SHORT_RUN_ID="reacher8gpu_eval_short_$(date -u +%Y%m%dT%H%MZ)"
SHORT_CONTROLLER_LOG="$SHORT_ROOT/controller_${SHORT_RUN_ID}.log"

mkdir -p "$SHORT_ROOT/logs" "$LONG_LOG_ROOT"

echo "[fresh] removing prior short eval artifacts under $SHORT_ROOT"
rm -rf "$SHORT_ROOT/eval"
rm -f "$SHORT_ROOT/matrix_summary.csv" "$SHORT_ROOT/matrix_summary.json" "$SHORT_ROOT/matrix_raw.csv"
rm -rf "$LONG_ROOT/eval"
rm -f "$LONG_ROOT/matrix_summary.csv" "$LONG_ROOT/matrix_summary.json" "$LONG_ROOT/matrix_raw.csv"

# Kill stale long-horizon watchers from earlier attempts.
if [[ -f "$LONG_LOG_ROOT/long_after_short.pid" ]]; then
  old_long=$(cat "$LONG_LOG_ROOT/long_after_short.pid" || true)
  [[ -n "$old_long" ]] && kill "$old_long" 2>/dev/null || true
fi

common_short=(
  DATASET_NAME=reacher
  DATA_FILE=/data/sicong/weitao/datasets/lewm/reacher.h5
  CHECKPOINT=/data/sicong/weitao/.stable_worldmodel/reacher/lewm_object.ckpt
  EVAL_CONFIG=reacher
  EVAL_DATASET_NAME=reacher
  CACHE_DIR=/data/sicong/weitao/.stable_worldmodel
  WORK_ROOT="$SHORT_ROOT"
  PYTHON=/data/sicong/weitao/le-wm/.venv/bin/python
  CPU_THREADS=1
  TRAIN_SEEDS=0,42,625
  PARTITION_SEEDS=0,1,2
  EVAL_SEEDS=0,1,2,3,4
  METHODS=random_voronoi,kmeanspp,spectral
  GOAL_OFFSET=
  EVAL_BUDGET=
  TASK_RETRIES=2
  SKIP_JOINT=0
  SKIP_REGIONS=0
  MUJOCO_GL=egl
  PYOPENGL_PLATFORM=egl
)

# Run official → model → aggregate sequentially in one background subshell.
# Do not start official in the parent and wait from a subshell: bash only lets
# you wait on direct child processes, so that pattern fails with
# "wait: pid … is not a child of this shell" and skips model_eval/aggregate.
# Each stage must set END_STAGE so parallel.sh does not run through aggregate
# inside the official job (the failure mode that hit line 238 EOF mid-edit).
(
  set -euo pipefail
  env "${common_short[@]}" \
    GPU_IDS=3 \
    START_STAGE=official_eval \
    END_STAGE=official_eval \
    RUN_ID="${SHORT_RUN_ID}_official" \
    bash experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh \
    >>"${SHORT_CONTROLLER_LOG}.official" 2>&1
  env "${common_short[@]}" \
    GPU_IDS=3,4,5,6 \
    START_STAGE=model_eval \
    END_STAGE=model_eval \
    RUN_ID="${SHORT_RUN_ID}_model" \
    bash experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh \
    >>"${SHORT_CONTROLLER_LOG}.model" 2>&1
  env "${common_short[@]}" \
    GPU_IDS=3 \
    START_STAGE=aggregate \
    END_STAGE=aggregate \
    RUN_ID="${SHORT_RUN_ID}_aggregate" \
    bash experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh \
    >>"${SHORT_CONTROLLER_LOG}.aggregate" 2>&1
  echo "[complete] run_id=$SHORT_RUN_ID logs=$SHORT_ROOT/logs"
) >"$SHORT_CONTROLLER_LOG" 2>&1 &

SHORT_PID=$!
echo "$SHORT_PID" >"$SHORT_ROOT/logs/${SHORT_RUN_ID}.pid"
echo "$SHORT_RUN_ID" >"$SHORT_ROOT/logs/current_eval_run_id.txt"

nohup env \
  SHORT_WAIT_LOG="$SHORT_CONTROLLER_LOG" \
  bash experiments/reacher/scripts/run_long_after_short.sh \
  >>"$LONG_LOG_ROOT/long_after_short.log" 2>&1 &

LONG_PID=$!
echo "$LONG_PID" >"$LONG_LOG_ROOT/long_after_short.pid"

echo "SHORT_RUN_ID=$SHORT_RUN_ID"
echo "SHORT_PID=$SHORT_PID"
echo "SHORT_LOG=$SHORT_CONTROLLER_LOG"
echo "LONG_PID=$LONG_PID"
