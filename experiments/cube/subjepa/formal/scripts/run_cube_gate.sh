#!/usr/bin/env bash
# OGBench Cube Sub-JEPA formal stage: full-cache preparation + LAP empirical
# spectral gate. Mirrors experiments/reacher/subjepa/formal/scripts/run_reacher_gate.sh
# (audit-free: no smoke/passport/replay-audit steps). The LAP gate itself
# (fit_partition.py --method auto) is the experimental method, not an audit
# step, and is preserved as-is with the spectral-gate parameters specified in
# the task brief (K=3, 20000 landmarks, nominal kNN 30, perturbed kNN 27/33,
# retention threshold 0.5, background gap count 10, MAD multiplier 3.0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/experiments/cube/subjepa/env.sh"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
CPU_THREADS="${CPU_THREADS:-4}"

WORK_ROOT="$FORMAL"
PREP="$WORK_ROOT/preparation"
GATE_OUT="$WORK_ROOT/gate/partition"
LOG_DIR="$WORK_ROOT/logs"
mkdir -p "$LOG_DIR" "$WORK_ROOT/manifests"

log() { echo "[cube-formal-gate] $*" | tee -a "$LOG_DIR/formal_gate.log"; }

phase_prepare() {
  log "phase=prepare (full cache; writes only under $WORK_ROOT)"
  if [[ -f "$PREP/embedding_cache.npz" ]]; then
    log "phase=prepare skipped — embedding cache already present: $PREP/embedding_cache.npz"
    return 0
  fi
  require_checkpoint
  [[ -f "$DATASET" ]] || { echo "STOP: dataset not found: $DATASET" >&2; exit 1; }
  [[ -f "$TASK_SPEC" ]] || { echo "STOP: task spec not found: $TASK_SPEC" >&2; exit 1; }

  env MODEL_FAMILY=subjepa GPU_ID="$GPU_ID" PYTHON="$PYTHON" \
    bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
    --task-spec "$TASK_SPEC" \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --eval-config-name cube \
    --work-root "$WORK_ROOT" \
    --cache-dir "$CACHE_DIR" \
    --python "$PYTHON" \
    --cpu-threads "$CPU_THREADS" \
    --phase prepare \
    2>&1 | tee "$LOG_DIR/prepare.log"

  if [[ ! -f "$PREP/embedding_cache.npz" ]]; then
    echo "STOP: formal embedding cache missing after prepare: $PREP/embedding_cache.npz" >&2
    exit 1
  fi
}

phase_gate() {
  log "phase=gate (method=auto, K=3, LAP empirical spectral degeneracy gate)"
  if [[ -f "$GATE_OUT/manifest.json" ]]; then
    log "phase=gate skipped — existing gate manifest present: $GATE_OUT/manifest.json"
    return 0
  fi
  [[ -f "$PREP/embedding_cache.npz" ]] || {
    echo "STOP: missing embedding cache; run phase=prepare first: $PREP/embedding_cache.npz" >&2
    exit 1
  }
  [[ -f "$DATASET" ]] || { echo "STOP: dataset not found: $DATASET" >&2; exit 1; }

  "$PYTHON" experiments/control_matrix/fit_partition.py \
    --method auto \
    --dataset-name cube \
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
    --cpu-threads "$CPU_THREADS" \
    --out-dir "$GATE_OUT" \
    2>&1 | tee "$LOG_DIR/gate.log"

  [[ -f "$GATE_OUT/manifest.json" ]] || {
    echo "STOP: gate did not produce a manifest: $GATE_OUT/manifest.json" >&2
    exit 1
  }
  selected="$("$PYTHON" -c "import json; print(json.load(open('$GATE_OUT/manifest.json'))['selected_method'])")"
  log "phase=gate complete — selected_method=$selected"
}

case "${1:-all}" in
  prepare) phase_prepare ;;
  gate) phase_gate ;;
  all)
    phase_prepare
    phase_gate
    ;;
  *)
    echo "usage: $0 {prepare|gate|all}" >&2
    exit 2
    ;;
esac
