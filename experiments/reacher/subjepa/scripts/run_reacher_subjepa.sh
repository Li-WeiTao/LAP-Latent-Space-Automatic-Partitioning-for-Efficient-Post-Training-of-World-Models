#!/usr/bin/env bash
# Reacher Sub-JEPA main stage controller. Single entry point dispatching to
# the formal (gate) and matrix (partition/training/eval/aggregate) scripts.
# Each stage is independently resumable and skips already-complete artifacts.
#
#   prepare        -> formal cache preparation (encode Reacher latents)
#   gate           -> LAP empirical spectral gate (method=auto)
#   partition      -> Global / K-means++ K3 / Spectral K3 region partitions
#   training       -> Global-FT50 / K-means++ K3-50 / Spectral K3-50 / Auto-LAP
#   eval-short     -> paired short-horizon eval (goal_offset=25)
#   eval-long      -> paired long-horizon eval (goal_offset=50)
#   aggregate      -> matrix_summary_{short,long}.json
#   all-post-train -> eval-short + eval-long + aggregate
#
# This script does not execute anything by itself beyond argument dispatch;
# no stage is invoked automatically.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

FORMAL_GATE="$REPO_ROOT/experiments/reacher/subjepa/formal/scripts/run_reacher_gate.sh"
MATRIX_CTRL="$REPO_ROOT/experiments/reacher/subjepa/matrix/scripts/run_full_matrix.sh"

case "${1:-}" in
  prepare) exec bash "$FORMAL_GATE" prepare ;;
  gate) exec bash "$FORMAL_GATE" gate ;;
  formal-all) exec bash "$FORMAL_GATE" all ;;
  partition) exec bash "$MATRIX_CTRL" partition ;;
  training) exec bash "$MATRIX_CTRL" training ;;
  eval-short) exec bash "$MATRIX_CTRL" eval-short ;;
  eval-long) exec bash "$MATRIX_CTRL" eval-long ;;
  aggregate) exec bash "$MATRIX_CTRL" aggregate ;;
  all-post-train) exec bash "$MATRIX_CTRL" all-post-train ;;
  *)
    echo "usage: $0 {prepare|gate|formal-all|partition|training|eval-short|eval-long|aggregate|all-post-train}" >&2
    exit 2
    ;;
esac
