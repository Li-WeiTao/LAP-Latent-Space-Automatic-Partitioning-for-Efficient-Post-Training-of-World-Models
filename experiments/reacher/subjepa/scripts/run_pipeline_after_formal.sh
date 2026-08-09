#!/usr/bin/env bash
# Wait for formal prepare+gate, then run matrix partition/training/eval/aggregate.
# GPU policy: use GPU_IDS (default 3,5,6) — never GPU 4 unless explicitly overridden.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/reacher/subjepa/env.sh"

require_checkpoint

# Default GPU set excludes GPU 4 per operator request.
export GPU_IDS="${GPU_IDS:-3,5,6}"
export GPU_ID="${GPU_ID:-3}"
export CPU_THREADS="${CPU_THREADS:-4}"

FORMAL_LOG_DIR="$REPO_ROOT/$FORMAL/logs"
PIPELINE_LOG="$FORMAL_LOG_DIR/pipeline_after_formal.log"
GATE_MANIFEST="$REPO_ROOT/$FORMAL/gate/partition/manifest.json"
CACHE="$REPO_ROOT/$FORMAL/preparation/embedding_cache.npz"

mkdir -p "$FORMAL_LOG_DIR"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPELINE_LOG"; }

log "waiting for formal cache: $CACHE"
while [[ ! -f "$CACHE" ]]; do
  sleep 120
done
log "formal cache ready"

log "waiting for gate manifest: $GATE_MANIFEST"
while [[ ! -f "$GATE_MANIFEST" ]]; do
  sleep 60
done
log "gate complete"

MATRIX_CTRL="$REPO_ROOT/experiments/reacher/subjepa/matrix/scripts/run_full_matrix.sh"

log "starting matrix partition (GPU_IDS=$GPU_IDS, GPU 4 excluded unless overridden)"
bash "$MATRIX_CTRL" partition 2>&1 | tee -a "$FORMAL_LOG_DIR/matrix_partition.log"

log "starting matrix training (GPU_IDS=$GPU_IDS)"
bash "$MATRIX_CTRL" training 2>&1 | tee -a "$FORMAL_LOG_DIR/matrix_training.log"

log "starting eval-short + eval-long + aggregate (GPU_IDS=$GPU_IDS)"
bash "$MATRIX_CTRL" all-post-train 2>&1 | tee -a "$FORMAL_LOG_DIR/matrix_posttrain.log"

log "pipeline complete"
