#!/usr/bin/env bash
# Resume Cube eval-long for failed/skipped tasks, then snapshot + aggregate.
# Does NOT wipe eval/; leaf driver skips completed eval seeds.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/cube/subjepa/env.sh"

export TASK_SPEC="configs/experiments/tasks/cube.json"
export MATRIX="experiments/cube/subjepa/matrix"
export DATASET="/home/sicong/weitao/datasets/lewm/cube_single_expert.h5"
export CHECKPOINT="/data/sicong/weitao/.stable_worldmodel/cube/subjepa_object.ckpt"
export CACHE_DIR="/data/sicong/weitao/.stableworldmodel/subjepa/cube"
require_checkpoint

# Use RESUME_GPU_IDS to override; ignore inherited GPU_IDS from other matrix jobs.
GPU_IDS="${RESUME_GPU_IDS:-0,2,3,5,6}"
CPU_THREADS="${RESUME_CPU_THREADS:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_ROOT="$MATRIX"
LOG="$WORK_ROOT/logs/resume_eval_long_${RUN_ID}.log"
PAIR_LONG="$(realpath "$WORK_ROOT/paired_starts/canon_long")"
LINK_AUTO_LAP="$REPO_ROOT/experiments/cube/subjepa/matrix/scripts/link_auto_lap.sh"
MATRIX_CTRL="$REPO_ROOT/experiments/cube/subjepa/matrix/scripts/run_full_matrix.sh"

# Hardcoded defaults; use RESUME_LONG_TASKS to override (not PARALLEL_TASKS).
RESUME_LONG_TASKS="${RESUME_LONG_TASKS:-official,global_t0,global_t625}"

mkdir -p "$WORK_ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[resume-long] run_id=$RUN_ID gpus=$GPU_IDS tasks=$RESUME_LONG_TASKS log=$LOG"

JEPA_BASE=(
  --task-spec "$TASK_SPEC"
  --dataset "$DATASET"
  --checkpoint "$CHECKPOINT"
  --eval-config-name cube
  --work-root "$WORK_ROOT"
  --cache-dir "$CACHE_DIR"
  --python "$PYTHON"
  --cpu-threads "$CPU_THREADS"
  --methods kmeanspp,spectral
)

env \
  GPU_IDS="$GPU_IDS" \
  CPU_THREADS="$CPU_THREADS" \
  RUN_ID="${RUN_ID}_long" \
  PARALLEL_TASKS="$RESUME_LONG_TASKS" \
  START_STAGE=eval_long \
  END_STAGE=eval_long \
  PAIRED_START_ROOT_LONG="$PAIR_LONG" \
  LONG_GOAL_OFFSET=50 \
  SKIP_JOINT=1 \
  TRAIN_SEEDS=0,42,625 \
  PARTITION_SEEDS=0,1,2 \
  EVAL_SEEDS=0,1,2,3,4 \
  METHODS=kmeanspp,spectral \
  bash experiments/control_matrix/scripts/run_jepa_matrix_parallel.sh \
  --model-family subjepa \
  "${JEPA_BASE[@]}" \
  --train-seeds 0,42,625 \
  --partition-seeds 0,1,2 \
  --eval-seeds 0,1,2,3,4

echo "[resume-long] eval-long complete; results=$(find "$WORK_ROOT/eval" -name results.json | wc -l)/110"
DEPLOYMENT_SEED="${DEPLOYMENT_SEED:-0}" bash "$LINK_AUTO_LAP" "$WORK_ROOT" eval

rm -rf "$WORK_ROOT/eval_long"
cp -a "$WORK_ROOT/eval" "$WORK_ROOT/eval_long"
echo "[resume-long] snapshot eval_long"

env PYTHON="$PYTHON" CHECKPOINT="$CHECKPOINT" bash "$MATRIX_CTRL" aggregate

echo "[resume-long] complete run_id=$RUN_ID"
