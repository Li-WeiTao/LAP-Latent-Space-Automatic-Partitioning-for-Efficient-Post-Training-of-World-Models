#!/usr/bin/env bash
# Resume Cube eval-short for failed/skipped tasks, then eval-long + aggregate.
# Uses run_jepa_matrix_parallel.sh (same as run_full_matrix.sh eval-short).
# Does NOT wipe existing eval/ results; leaf driver skips completed eval seeds.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/cube/subjepa/env.sh"

require_checkpoint

export TASK_SPEC="configs/experiments/tasks/cube.json"
export MATRIX="experiments/cube/subjepa/matrix"
export DATASET="/home/sicong/weitao/datasets/lewm/cube_single_expert.h5"
export CHECKPOINT="/data/sicong/weitao/.stable_worldmodel/cube/subjepa_object.ckpt"
export CACHE_DIR="/data/sicong/weitao/.stableworldmodel/subjepa/cube"

GPU_IDS="${GPU_IDS:-0,2,3,5,6}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_ROOT="$MATRIX"
LOG="$WORK_ROOT/logs/resume_eval_${RUN_ID}.log"
PAIR_SHORT="$(realpath "$WORK_ROOT/paired_starts/canon_short")"
LINK_AUTO_LAP="$REPO_ROOT/experiments/cube/subjepa/matrix/scripts/link_auto_lap.sh"

PARALLEL_TASKS="${PARALLEL_TASKS:-global_t42,kmeanspp_p2_t42,spectral_p1_t0,kmeanspp_p1_t625,kmeanspp_p2_t625,spectral_p0_t42,spectral_p2_t42,spectral_p1_t625}"

mkdir -p "$WORK_ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[resume] run_id=$RUN_ID gpus=$GPU_IDS tasks=$PARALLEL_TASKS log=$LOG"

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
  RUN_ID="${RUN_ID}_short" \
  PARALLEL_TASKS="$PARALLEL_TASKS" \
  START_STAGE=eval_short \
  END_STAGE=eval_short \
  PAIRED_START_ROOT_SHORT="$PAIR_SHORT" \
  SHORT_GOAL_OFFSET=25 \
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

echo "[resume] short eval补 complete; results=$(find "$WORK_ROOT/eval" -name results.json | wc -l)/110"
DEPLOYMENT_SEED="${DEPLOYMENT_SEED:-0}" bash "$LINK_AUTO_LAP" "$WORK_ROOT" eval

rm -rf "$WORK_ROOT/eval_short"
cp -a "$WORK_ROOT/eval" "$WORK_ROOT/eval_short"
echo "[resume] snapshot eval_short"

env \
  PYTHON="$PYTHON" \
  CHECKPOINT="$CHECKPOINT" \
  GPU_IDS="$GPU_IDS" \
  CPU_THREADS="$CPU_THREADS" \
  RUN_ID="${RUN_ID}_long" \
  bash "$REPO_ROOT/experiments/cube/subjepa/matrix/scripts/run_full_matrix.sh" eval-long

env \
  PYTHON="$PYTHON" \
  CHECKPOINT="$CHECKPOINT" \
  bash "$REPO_ROOT/experiments/cube/subjepa/matrix/scripts/run_full_matrix.sh" aggregate

echo "[resume] complete run_id=$RUN_ID"
