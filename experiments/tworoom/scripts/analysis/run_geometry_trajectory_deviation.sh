#!/usr/bin/env bash
# Compare each geometry region predictor vs downloaded P_train_global on full test pool.
# Does NOT start automatically — run manually when ready.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

PRED_DIR="experiments/tworoom/results/tworoom_geometry_trajectory_predictors"
GLOBAL_CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
TEST_CACHE="experiments/tworoom/cache/tworoom_trajectory_test_full_transitions.npz"

regions=(common doorway_corridor left_room near_wall right_room)

for region in "${regions[@]}"; do
  OUT="experiments/tworoom/results/tworoom_geometry_trajectory_deviation_${region}_vs_train"
  mkdir -p "${OUT}"
  echo "==== trajectory deviation ${region} vs P_train_global started at $(date) ===="
  /usr/bin/time -p python experiments/tworoom/trajectory_deviation.py \
    --predictor-a "${PRED_DIR}/P_${region}_object.ckpt" \
    --predictor-b "${GLOBAL_CKPT}" \
    --predictor-a-name "P_${region}" \
    --predictor-b-name P_train_global \
    --encoder-checkpoint "${GLOBAL_CKPT}" \
    --load-test-cache "${TEST_CACHE}" \
    --test-max-samples 0 \
    --history-size 3 \
    --max-steps 10 \
    --batch-size 64 \
    --device cuda \
    --out-dir "${OUT}" \
    > "${OUT}/run_full_test.log" 2>&1
  echo "==== trajectory deviation ${region} finished at $(date) ===="
done
