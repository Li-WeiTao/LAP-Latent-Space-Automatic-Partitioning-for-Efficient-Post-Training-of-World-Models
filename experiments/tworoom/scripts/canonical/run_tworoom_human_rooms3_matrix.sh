#!/usr/bin/env bash
# Reproduce the human rooms3-50 arm with the same train/eval seeds as LAP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

DATA_FILE="${LAP_TWOROOM_DATA:?set LAP_TWOROOM_DATA}"
CHECKPOINT="${LAP_LEWM_CHECKPOINT:?set LAP_LEWM_CHECKPOINT}"
GPU="${GPU:?set GPU}"
WORK_ROOT="${WORK_ROOT:-experiments/tworoom/matrix}"
EMBED_DIR="${EMBED_DIR:-$WORK_ROOT/human/geometry_cache}"
CACHE_DIR="${LAP_STABLEWM_HOME:-${STABLEWM_HOME:-$HOME/.stable_worldmodel}}"
PHASE="${PHASE:-all}"
PYTHON="${PYTHON:-python}"
IFS=, read -r -a train_seeds <<< "${TRAIN_SEEDS:-0,42,625}"
IFS=, read -r -a eval_seeds <<< "${EVAL_SEEDS:-0,1,2,3,4}"
regions=(doorway_corridor left_room right_room)

prepare() {
  env GPU="$GPU" LAP_TWOROOM_DATA="$DATA_FILE" LAP_LEWM_CHECKPOINT="$CHECKPOINT" \
    LAP_STABLEWM_HOME="$CACHE_DIR" EMBED_DIR="$EMBED_DIR" \
    bash experiments/tworoom/scripts/internal/prepare_tworoom_spectral_inputs.sh
}

train() {
  for train_seed in "${train_seeds[@]}"; do
    main_dir="$WORK_ROOT/training/human_rooms3/train${train_seed}"
    mkdir -p "$main_dir"
    for cache in "$EMBED_DIR"/P_train_*_embeddings.npz; do
      [[ -e "$cache" ]] || { echo "missing geometry cache under $EMBED_DIR" >&2; exit 1; }
      ln -sfn "$(realpath "$cache")" "$main_dir/$(basename "$cache")"
    done
    for region in "${regions[@]}"; do
      checkpoint="$main_dir/P_train_${region}_object.ckpt"
      [[ -f "$checkpoint" ]] && continue
      env GPU="$GPU" REGION="$region" EPOCHS=50 SEED="$train_seed" \
        SELECT_BEST=1 SAVE_EPOCHS=20,30,40,50 MAIN_DIR="$main_dir" \
        WORK_DIR="$main_dir/_work/${region}_50ep_seed${train_seed}" \
        LAP_LEWM_CHECKPOINT="$CHECKPOINT" \
        bash experiments/tworoom/scripts/internal/run_geometry_train_one_region.sh
    done
  done
}

evaluate() {
  for train_seed in "${train_seeds[@]}"; do
    pred_dir="$WORK_ROOT/training/human_rooms3/train${train_seed}"
    for region in "${regions[@]}"; do
      [[ -f "$pred_dir/P_train_${region}_object.ckpt" ]] || {
        echo "missing rooms3 predictor: $pred_dir/P_train_${region}_object.ckpt" >&2
        exit 1
      }
    done
    for eval_seed in "${eval_seeds[@]}"; do
      starts="$WORK_ROOT/eval/official/eval${eval_seed}/results.json"
      out="$WORK_ROOT/eval/human_rooms3/train${train_seed}/eval${eval_seed}"
      [[ -f "$starts" ]] || { echo "missing paired official starts: $starts" >&2; exit 1; }
      [[ -f "$out/results.json" ]] && continue
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiments/tworoom/tworoom_success_rate_eval.py \
        --mode rooms3 --seed "$eval_seed" --checkpoint "$CHECKPOINT" \
        --config-name tworoom --dataset-tag tworoom --cache-dir "$CACHE_DIR" \
        --out-dir "$out" --num-eval 50 --eval-start-indices "$starts" \
        --region-ckpt "doorway_corridor=$pred_dir/P_train_doorway_corridor_object.ckpt" \
        --region-ckpt "left_room=$pred_dir/P_train_left_room_object.ckpt" \
        --region-ckpt "right_room=$pred_dir/P_train_right_room_object.ckpt" \
        --device cuda
    done
  done
}

case "$PHASE" in
  prepare) prepare ;;
  train) train ;;
  eval) evaluate ;;
  all) prepare; train; evaluate ;;
  *) echo "unknown PHASE=$PHASE" >&2; exit 2 ;;
esac
