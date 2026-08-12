#!/usr/bin/env bash
# OGBench Cube Sub-JEPA restricted smoke test — validates the full
# prepare -> partition -> train -> eval -> aggregate pipeline end to end at
# tiny scale, entirely separate from formal/matrix outputs (writes only
# under $SMOKE_ROOT). Single GPU, single train/partition/eval seed, capped
# max train starts, minimal eval episodes, few training epochs.
#
# Deliberately does NOT call experiments/control_matrix/validate_jepa_smoke.py
# (cache-equivalence / frozen-audit / route-equivalence) and does NOT use
# run_jepa_matrix.sh's built-in PHASE=smoke (which invokes those same audit
# checks). This task explicitly excludes adding/copying audit workflows;
# PushT's audits already cover that method-correctness question. What this
# script validates is narrower and purely operational: do the generic
# prepare/partition/train/eval/aggregate CLIs run to completion for Cube
# with real (tiny) data, checkpoint, and config wiring.
#
# Every phase call below is a direct invocation of the generic, task-agnostic
# control-matrix driver (run_subjepa_matrix.sh -> run_jepa_matrix.sh) or of
# train_predictors.py directly (only needed because the driver's PHASE=train
# hardcodes 50 epochs; smoke overrides via --epochs). No generic Python logic
# is duplicated here.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/cube/subjepa/env.sh"

require_checkpoint
[[ -f "$DATASET" ]] || { echo "STOP: dataset not found: $DATASET" >&2; exit 1; }

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
CPU_THREADS="${CPU_THREADS:-4}"

SMOKE_WORK_ROOT="${SMOKE_WORK_ROOT:-$SMOKE_ROOT}"
SMOKE_MAX_TRAIN_STARTS="${SMOKE_MAX_TRAIN_STARTS:-300}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-2}"
SMOKE_NUM_EVAL="${SMOKE_NUM_EVAL:-3}"
SMOKE_METHOD="${SMOKE_METHOD:-spectral}"
# K=3 spectral partitioning of a tiny smoke-scale cache can leave a region
# with fewer than train_predictors.py's default --min-region-samples (256)
# transitions; override down for smoke only (never touches formal/matrix).
SMOKE_MIN_REGION_SAMPLES="${SMOKE_MIN_REGION_SAMPLES:-32}"

mkdir -p "$SMOKE_WORK_ROOT/logs"
LOG="$SMOKE_WORK_ROOT/logs/smoke.log"

log() { echo "[cube-subjepa-smoke] $*" | tee -a "$LOG"; }

JEPA_BASE=(
  --task-spec "$TASK_SPEC"
  --dataset "$DATASET"
  --checkpoint "$CHECKPOINT"
  --eval-config-name cube
  --work-root "$SMOKE_WORK_ROOT"
  --cache-dir "$CACHE_DIR"
  --python "$PYTHON"
  --cpu-threads "$CPU_THREADS"
  --train-seeds 0
  --partition-seeds 0
  --eval-seeds 0
  --methods "$SMOKE_METHOD"
)

run_driver() {
  local phase=$1
  shift
  env GPU_ID="$GPU_ID" \
    bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
    "${JEPA_BASE[@]}" \
    --phase "$phase" \
    "$@" \
    2>&1 | tee -a "$LOG"
}

log "phase=prepare (max_train_starts=$SMOKE_MAX_TRAIN_STARTS)"
if [[ -f "$SMOKE_WORK_ROOT/preparation/embedding_cache.npz" ]]; then
  log "phase=prepare skipped — smoke cache already present"
else
  run_driver prepare --max-train-starts "$SMOKE_MAX_TRAIN_STARTS"
fi
[[ -f "$SMOKE_WORK_ROOT/preparation/embedding_cache.npz" ]] || {
  echo "STOP: smoke embedding cache missing after prepare" >&2
  exit 1
}

log "phase=partition (global + $SMOKE_METHOD, seed 0)"
[[ -f "$SMOKE_WORK_ROOT/partitions/global/seed0/manifest.json" ]] || run_driver partition_global
[[ -f "$SMOKE_WORK_ROOT/partitions/$SMOKE_METHOD/seed0/manifest.json" ]] || run_driver partition_regions

log "phase=train (direct train_predictors.py calls, epochs=$SMOKE_EPOCHS — driver's PHASE=train hardcodes 50)"
GLOBAL_OUT="$SMOKE_WORK_ROOT/training/global/train0"
if [[ ! -f "$GLOBAL_OUT/manifest.json" ]]; then
  "$PYTHON" experiments/control_matrix/train_predictors.py \
    --dataset-name cube \
    --latent-cache "$SMOKE_WORK_ROOT/preparation/embedding_cache.npz" \
    --pretrained-model "$CHECKPOINT" \
    --partition-dir "$SMOKE_WORK_ROOT/partitions/global/seed0" \
    --out-dir "$GLOBAL_OUT" \
    --train-seed 0 --epochs "$SMOKE_EPOCHS" --model-family subjepa \
    --cpu-threads "$CPU_THREADS" \
    2>&1 | tee -a "$LOG"
fi
REGION_OUT="$SMOKE_WORK_ROOT/training/$SMOKE_METHOD/partition0_train0"
if [[ ! -f "$REGION_OUT/manifest.json" ]]; then
  "$PYTHON" experiments/control_matrix/train_predictors.py \
    --dataset-name cube \
    --latent-cache "$SMOKE_WORK_ROOT/preparation/embedding_cache.npz" \
    --pretrained-model "$CHECKPOINT" \
    --partition-dir "$SMOKE_WORK_ROOT/partitions/$SMOKE_METHOD/seed0" \
    --out-dir "$REGION_OUT" \
    --train-seed 0 --epochs "$SMOKE_EPOCHS" --model-family subjepa \
    --min-region-samples "$SMOKE_MIN_REGION_SAMPLES" \
    --cpu-threads "$CPU_THREADS" \
    2>&1 | tee -a "$LOG"
fi

log "phase=eval-short (num_eval=$SMOKE_NUM_EVAL, goal_offset=25, sample fresh — smoke-only starts)"
NUM_EVAL="$SMOKE_NUM_EVAL" run_driver eval-short

log "phase=aggregate"
"$PYTHON" experiments/control_matrix/aggregate_matrix.py \
  --root "$SMOKE_WORK_ROOT" \
  --dataset-name cube \
  --train-seeds 0 \
  --partition-seeds 0 \
  --eval-seeds 0 \
  --methods "$SMOKE_METHOD" \
  --skip-joint \
  2>&1 | tee -a "$LOG"

log "smoke complete — outputs under $SMOKE_WORK_ROOT (formal/matrix untouched)"
