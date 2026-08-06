#!/usr/bin/env bash
# Deprecated monolith — use formal + matrix scripts (aligned with tworoom/subjepa).
set -euo pipefail

echo "run_formal_pipeline.sh is deprecated." >&2
echo "Use the TwoRoom-aligned entry points instead:" >&2
echo "  1. bash experiments/pusht/subjepa/formal/scripts/run_formal_gate.sh {prepare|gate|passport|all}" >&2
echo "  2. bash experiments/pusht/subjepa/matrix/scripts/run_full_matrix.sh {training|eval-short|eval-long|all-post-train|...}" >&2
echo "Or detached:" >&2
echo "  bash experiments/pusht/subjepa/scripts/launch_formal_detached.sh [prepare|gate|passport|all]" >&2
echo "  bash experiments/pusht/subjepa/matrix/scripts/launch_matrix_detached.sh" >&2
echo >&2

case "${1:-}" in
  prepare|gate|passport|all)
    exec bash experiments/pusht/subjepa/formal/scripts/run_formal_gate.sh "$1"
    ;;
  matrix_setup|setup)
    exec bash experiments/pusht/subjepa/matrix/scripts/run_full_matrix.sh setup
    ;;
  matrix_train|training)
    exec bash experiments/pusht/subjepa/matrix/scripts/run_full_matrix.sh training
    ;;
  matrix_eval|all-post-train)
    exec bash experiments/pusht/subjepa/matrix/scripts/run_full_matrix.sh all-post-train
    ;;
  "")
    exit 2
    ;;
  *)
    echo "unknown phase: $1" >&2
    exit 2
    ;;
esac
