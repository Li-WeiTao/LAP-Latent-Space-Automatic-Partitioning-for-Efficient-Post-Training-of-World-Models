#!/usr/bin/env bash
# PushT Sub-JEPA formal gate — same stages as tworoom/subjepa/formal, task-local paths only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/experiments/pusht/subjepa/env.sh"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
FORMAL_GIT_BASELINE="${FORMAL_GIT_BASELINE:-$(git rev-parse HEAD)}"

WORK_ROOT="$FORMAL"
PREP="$WORK_ROOT/preparation"
GATE_OUT="$WORK_ROOT/gate/partition"
LOG_DIR="$WORK_ROOT/logs"
mkdir -p "$LOG_DIR" "$WORK_ROOT/manifests"

log() { echo "[pusht-formal-gate] $*" | tee -a "$LOG_DIR/formal_gate.log"; }

verify_smoke_untouched() {
  local current smoke_cache="$ROOT/$SMOKE_ROOT/preparation/embedding_cache.npz"
  current="$(sha256sum "$smoke_cache" | awk '{print $1}')"
  if [[ "$current" != "$SMOKE_CACHE_SHA256" ]]; then
    echo "STOP: smoke cache SHA changed ($current != $SMOKE_CACHE_SHA256)" >&2
    exit 1
  fi
}

require_smoke_verified() {
  local status_file="$ROOT/$SMOKE_ROOT/manifests/verification_status.json"
  [[ -f "$status_file" ]] || { echo "missing smoke verification: $status_file" >&2; exit 1; }
  "$PYTHON" -c "
import json, sys
s = json.load(open('$status_file'))
if s.get('status') != 'VERIFIED':
    sys.exit('smoke status is not VERIFIED: ' + str(s.get('status')))
"
}

phase_prepare() {
  log "phase=prepare (no max_train_starts cap)"
  require_smoke_verified
  verify_smoke_untouched
  env MODEL_FAMILY=subjepa GPU_ID="$GPU_ID" PYTHON="$PYTHON" \
    bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
    --task-spec "$TASK_SPEC" \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --eval-config-name pusht \
    --work-root "$WORK_ROOT" \
    --phase prepare \
    2>&1 | tee "$LOG_DIR/prepare.log"
  "$PYTHON" "$TWOROOM_FORMAL_SCRIPTS/formal_cache_audit.py" \
    --phase augment-manifest \
    --work-root "$WORK_ROOT" \
    --git-baseline "$FORMAL_GIT_BASELINE"
  "$PYTHON" "$TWOROOM_FORMAL_SCRIPTS/formal_cache_audit.py" \
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
    --dataset-name pusht \
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
  "$PYTHON" "$TWOROOM_FORMAL_SCRIPTS/formal_post_gate_audit.py" \
    --work-root "$WORK_ROOT" \
    --data-file "$DATASET" \
    --latent-cache "$PREP/embedding_cache.npz" \
    --git-baseline "$FORMAL_GIT_BASELINE"
}

phase_passport() {
  "$PYTHON" "$TWOROOM_FORMAL_SCRIPTS/formal_post_gate_audit.py" \
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
