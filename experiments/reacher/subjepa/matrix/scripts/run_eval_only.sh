#!/usr/bin/env bash
# Reacher Sub-JEPA: run paired short + long evaluation only (assumes training
# already complete). official + Global-FT + K-means++ + Spectral + Auto-LAP.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/reacher/subjepa/env.sh"

export GPU_IDS="${GPU_IDS:-0}"
export CPU_THREADS="${CPU_THREADS:-4}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

RUN="$REPO_ROOT/experiments/reacher/subjepa/matrix/scripts/run_full_matrix.sh"
bash "$RUN" eval-short
bash "$RUN" eval-long

echo "[reacher-eval-only] short and long complete"
