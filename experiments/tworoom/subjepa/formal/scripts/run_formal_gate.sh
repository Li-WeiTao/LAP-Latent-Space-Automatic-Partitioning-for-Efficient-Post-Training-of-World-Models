#!/usr/bin/env bash
# Sub-JEPA TwoRoom formal gate (frozen at git 36f960a). Task-local orchestrator only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/data/sicong/weitao/le-wm/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

TASK_SPEC="configs/experiments/tasks/tworoom.json"
DATASET="/data/sicong/weitao/datasets/lewm/tworoom.h5"
CHECKPOINT="/data/sicong/weitao/.stable_worldmodel/tworoom/subjepa_object.ckpt"
WORK_ROOT="experiments/tworoom/subjepa/formal"
FORMAL_GIT_BASELINE="36f960a"
SMOKE_CACHE_SHA256="6828c6b5b7f87df33878ed43684821e975b4e5aa9e859a1ce00e1bf6f40ab3a7"

PREP="$WORK_ROOT/preparation"
CACHE="$PREP/embedding_cache.npz"
GATE_OUT="$WORK_ROOT/gate/partition"
LOG_DIR="$WORK_ROOT/logs"
mkdir -p "$LOG_DIR" "$WORK_ROOT/manifests"

log() { echo "[formal-gate] $*" | tee -a "$LOG_DIR/formal_gate.log"; }

verify_smoke_untouched() {
  local current
  current="$(sha256sum "$ROOT/experiments/tworoom/subjepa/preparation/embedding_cache.npz" | awk '{print $1}')"
  if [[ "$current" != "$SMOKE_CACHE_SHA256" ]]; then
    echo "STOP: smoke cache SHA changed ($current != $SMOKE_CACHE_SHA256)" >&2
    exit 1
  fi
}

verify_code_baseline() {
  local diff_lines
  diff_lines="$(git diff "$FORMAL_GIT_BASELINE" HEAD -- '*.py' '*.sh' | wc -l)"
  if [[ "$diff_lines" != "0" ]]; then
    echo "STOP: public Python/shell code differs from $FORMAL_GIT_BASELINE" >&2
    exit 1
  fi
}

phase_prepare() {
  log "phase=prepare (no max_train_starts cap)"
  verify_code_baseline
  verify_smoke_untouched
  env MODEL_FAMILY=subjepa GPU_ID="$GPU_ID" PYTHON="$PYTHON" \
    bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
    --task-spec "$TASK_SPEC" \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --eval-config-name tworoom \
    --work-root "$WORK_ROOT" \
    --phase prepare \
    2>&1 | tee "$LOG_DIR/prepare.log"
  "$PYTHON" "$WORK_ROOT/scripts/formal_cache_audit.py" \
    --phase augment-manifest \
    --work-root "$WORK_ROOT" \
    --git-baseline "$FORMAL_GIT_BASELINE"
  "$PYTHON" "$WORK_ROOT/scripts/formal_cache_audit.py" \
    --phase all-replay-audits \
    --work-root "$WORK_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET"
  verify_smoke_untouched
}

phase_gate() {
  log "phase=gate (method=auto, K=3, 9 graph configs)"
  verify_smoke_untouched
  "$PYTHON" experiments/control_matrix/fit_partition.py \
    --method auto \
    --dataset-name tworoom \
    --data-file "$DATASET" \
    --latent-cache "$PREP/embedding_cache.npz" \
    --frameskip 5 \
    --num-clusters 3 \
    --num-landmarks 20000 \
    --knn 30 \
    --perturb-knn 27,33 \
    --diagnostic-seeds 0,1,2 \
    --deployment-seed 0 \
    --gate-perturbation-multiplier 2.0 \
    --gate-retention-threshold 0.5 \
    --gate-background-gap-count 10 \
    --gate-background-mad-multiplier 3.0 \
    --gate-epsilon 1e-8 \
    --gpu-id 0 \
    --cpu-threads 4 \
    --out-dir "$GATE_OUT" \
    2>&1 | tee "$LOG_DIR/gate.log"
  "$PYTHON" "$WORK_ROOT/scripts/formal_post_gate_audit.py" \
    --work-root "$WORK_ROOT" \
    --data-file "$DATASET" \
    --latent-cache "$PREP/embedding_cache.npz" \
    --git-baseline "$FORMAL_GIT_BASELINE"
}

phase_passport() {
  "$PYTHON" "$WORK_ROOT/scripts/formal_post_gate_audit.py" \
    --work-root "$WORK_ROOT" \
    --data-file "$DATASET" \
    --latent-cache "$PREP/embedding_cache.npz" \
    --git-baseline "$FORMAL_GIT_BASELINE" \
    --emit-passport-only
}

case "${1:-all}" in
  prepare) phase_prepare ;;
  gate) phase_gate ;;
  passport) phase_passport ;;
  all)
    phase_prepare
    phase_gate
    phase_passport
    ;;
  *)
    echo "usage: $0 {prepare|gate|passport|all}" >&2
    exit 2
    ;;
esac
