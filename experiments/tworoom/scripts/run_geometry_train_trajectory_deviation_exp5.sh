#!/usr/bin/env bash
# Trajectory deviation: exp5 geometry train∩region predictors vs downloaded P_train_global,
# on the full test pool (same protocol as the earlier quantile-region table).
# Predictors: common/near_wall 30ep, left/right 50ep best, doorway_corridor 80ep best.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

PRED_DIR="experiments/tworoom/results/tworoom_geometry_train_region_predictors"
DOORWAY80_DIR="experiments/tworoom/results/tworoom_geometry_train_region_predictors_doorway80ep"
GLOBAL_CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
TEST_CACHE="experiments/tworoom/cache/tworoom_trajectory_test_full_transitions.npz"

regions=(common doorway_corridor left_room near_wall right_room)

for region in "${regions[@]}"; do
  if [[ "${region}" == "doorway_corridor" ]]; then
    ckpt="${DOORWAY80_DIR}/P_train_${region}_object.ckpt"
  else
    ckpt="${PRED_DIR}/P_train_${region}_object.ckpt"
  fi

  OUT="experiments/tworoom/results/tworoom_geometry_train_trajectory_deviation_${region}_vs_train_exp5"
  mkdir -p "${OUT}"
  echo "==== trajectory deviation P_train_${region} (exp5) vs P_train_global started at $(date) ===="
  /usr/bin/time -p python experiments/tworoom/trajectory_deviation.py \
    --predictor-a "${ckpt}" \
    --predictor-b "${GLOBAL_CKPT}" \
    --predictor-a-name "P_train_${region}" \
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
  echo "==== trajectory deviation P_train_${region} finished at $(date) ===="
done

echo "==== exp5 trajectory deviation all done at $(date) ===="
