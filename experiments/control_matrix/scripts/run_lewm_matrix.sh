#!/usr/bin/env bash
set -euo pipefail

# Dataset/model-parameterized launcher for the six-method comparison matrix.
# Override any variable below from the environment; no task-specific Python
# script is required when moving from TwoRoom to Push-T.
DATASET_NAME=${DATASET_NAME:-pusht}
DATA_FILE=${DATA_FILE:?set DATA_FILE to the task HDF5 file}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the official LeWM object checkpoint}
EVAL_CONFIG=${EVAL_CONFIG:-pusht}
EVAL_DATASET_NAME=${EVAL_DATASET_NAME:-pusht_expert_train}
# stable_worldmodel interprets this as a cache root and appends
# datasets/<eval.dataset_name>.h5; it is not the HDF5 file's parent directory.
CACHE_DIR=${CACHE_DIR:-$HOME/.stable_worldmodel}
WORK_ROOT=${WORK_ROOT:-experiments/${DATASET_NAME}/matrix}
PHASE=${PHASE:-all}
GPU_ID=${GPU_ID:-}
TRAIN_SEEDS=${TRAIN_SEEDS:-0,42,625}
PARTITION_SEEDS=${PARTITION_SEEDS:-0,1,2}
EVAL_SEEDS=${EVAL_SEEDS:-0,1,2,3,4}
METHODS=${METHODS:-random_voronoi,kmeanspp,spectral}

PYTHON=${PYTHON:-python}
if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES=$GPU_ID
fi
mkdir -p "$WORK_ROOT"

IFS=, read -r -a train_seeds <<< "$TRAIN_SEEDS"
IFS=, read -r -a partition_seeds <<< "$PARTITION_SEEDS"
IFS=, read -r -a eval_seeds <<< "$EVAL_SEEDS"
IFS=, read -r -a methods <<< "$METHODS"

PREP="$WORK_ROOT/preparation"
STARTS="$PREP/train_global_reference_starts.npy"
LATENT_CACHE="$WORK_ROOT/${DATASET_NAME}_lewm_train_latent_cache.npz"

prepare() {
  mkdir -p "$PREP"
  local eval_dataset_path="$CACHE_DIR/datasets/$EVAL_DATASET_NAME.h5"
  mkdir -p "$(dirname "$eval_dataset_path")"
  if [[ ! -e "$eval_dataset_path" ]]; then
    ln -s "$(realpath "$DATA_FILE")" "$eval_dataset_path"
  elif [[ "$(realpath "$eval_dataset_path")" != "$(realpath "$DATA_FILE")" ]]; then
    echo "evaluation dataset path already resolves to another file: $eval_dataset_path" >&2
    exit 1
  fi
  if [[ ! -f "$STARTS" ]]; then
    "$PYTHON" experiments/tworoom/trajectory.py \
      --dataset "$DATASET_NAME" \
      --data-file "$DATA_FILE" \
      --checkpoint "$CHECKPOINT" \
      --out-dir "$PREP" \
      --prepare-starts-only \
      --regions common \
      --restrict-to-train-split \
      --predictor-prefix train_ \
      --frameskip 5
  fi
  if [[ ! -f "$LATENT_CACHE" ]]; then
    "$PYTHON" -m lap.encoding.cli encode \
      --dataset-factory backends.lewm.encoding:make_hdf5_transition_dataset \
      --dataset-arg "data_file=\"$DATA_FILE\"" \
      --dataset-arg "starts=\"$STARTS\"" \
      --dataset-arg "action_norm_starts=\"$STARTS\"" \
      --dataset-arg history_size=3 \
      --dataset-arg num_preds=1 \
      --dataset-arg frameskip=5 \
      --encoder-factory backends.lewm.encoding:make_encoder \
      --encoder-arg img_size=224 \
      --encoder-arg frameskip=5 \
      --pretrained-model "$CHECKPOINT" \
      --output "$LATENT_CACHE" \
      --device cuda \
      --transition-batch-size 128 \
      --frame-batch-size 512 \
      --batch-shape-mode exact \
      --num-workers 4 \
      --cpu-threads 4
  fi
}

partition() {
  local global_dir="$WORK_ROOT/partitions/global/seed0"
  if [[ ! -f "$global_dir/manifest.json" ]]; then
    "$PYTHON" experiments/control_matrix/fit_partition.py \
      --method global --dataset-name "$DATASET_NAME" \
      --latent-cache "$LATENT_CACHE" --frameskip 5 \
      --out-dir "$global_dir"
  fi
  for method in "${methods[@]}"; do
    for pseed in "${partition_seeds[@]}"; do
      local out="$WORK_ROOT/partitions/$method/seed$pseed"
      [[ -f "$out/manifest.json" ]] && continue
      "$PYTHON" experiments/control_matrix/fit_partition.py \
        --method "$method" --dataset-name "$DATASET_NAME" \
        --data-file "$DATA_FILE" --latent-cache "$LATENT_CACHE" \
        --frameskip 5 --num-clusters 3 --seed "$pseed" \
        --gpu-id 0 --out-dir "$out"
    done
  done
}

train() {
  for tseed in "${train_seeds[@]}"; do
    local joint="$WORK_ROOT/training/joint/train$tseed"
    if [[ ! -f "$joint/manifest.json" ]]; then
      "$PYTHON" experiments/tworoom/joint_continue_tworoom.py \
        --dataset-name "$DATASET_NAME" --data-file "$DATA_FILE" \
        --checkpoint "$CHECKPOINT" --out-dir "$joint" \
        --seed "$tseed" --epochs 3 --precision fp32 \
        --frameskip 5 --num-workers 4 --cpu-threads 4
    fi
    local global="$WORK_ROOT/training/global/train$tseed"
    if [[ ! -f "$global/manifest.json" ]]; then
      "$PYTHON" experiments/control_matrix/train_predictors.py \
        --dataset-name "$DATASET_NAME" --latent-cache "$LATENT_CACHE" \
        --pretrained-model "$CHECKPOINT" \
        --partition-dir "$WORK_ROOT/partitions/global/seed0" \
        --out-dir "$global" --train-seed "$tseed" --epochs 50
    fi
    for method in "${methods[@]}"; do
      for pseed in "${partition_seeds[@]}"; do
        local out="$WORK_ROOT/training/$method/partition${pseed}_train$tseed"
        [[ -f "$out/manifest.json" ]] && continue
        "$PYTHON" experiments/control_matrix/train_predictors.py \
          --dataset-name "$DATASET_NAME" --latent-cache "$LATENT_CACHE" \
          --pretrained-model "$CHECKPOINT" \
          --partition-dir "$WORK_ROOT/partitions/$method/seed$pseed" \
          --out-dir "$out" --train-seed "$tseed" --epochs 50
      done
    done
  done
}

eval_one() {
  local mode=$1 checkpoint=$2 lap_run=$3 out=$4 eval_seed=$5 starts=$6
  local args=(
    --mode "$mode" --seed "$eval_seed" --checkpoint "$checkpoint"
    --config-name "$EVAL_CONFIG" --dataset-tag "$DATASET_NAME"
    --cache-dir "$CACHE_DIR" --out-dir "$out" --num-eval 50
  )
  [[ -n "$lap_run" ]] && args+=(--lap-run-dir "$lap_run" --latent-routing mpc)
  [[ -n "$starts" ]] && args+=(--eval-start-indices "$starts")
  [[ -z "$starts" ]] && args+=(--sample-eval-starts)
  "$PYTHON" experiments/tworoom/tworoom_success_rate_eval.py "${args[@]}"
}

evaluate() {
  for eseed in "${eval_seeds[@]}"; do
    local baseline="$WORK_ROOT/eval/official/eval$eseed"
    if [[ ! -f "$baseline/results.json" ]]; then
      eval_one baseline "$CHECKPOINT" "" "$baseline" "$eseed" ""
    fi
    local starts="$baseline/results.json"
    for tseed in "${train_seeds[@]}"; do
      local joint="$WORK_ROOT/eval/joint/train$tseed/eval$eseed"
      if [[ ! -f "$joint/results.json" ]]; then
        eval_one baseline \
          "$WORK_ROOT/training/joint/train$tseed/joint_continue_object.ckpt" \
          "" "$joint" "$eseed" "$starts"
      fi
      local global="$WORK_ROOT/eval/global/train$tseed/eval$eseed"
      if [[ ! -f "$global/results.json" ]]; then
        eval_one baseline \
          "$WORK_ROOT/training/global/train$tseed/P_train_cluster0_object.ckpt" \
          "" "$global" "$eseed" "$starts"
      fi
      for method in "${methods[@]}"; do
        for pseed in "${partition_seeds[@]}"; do
          local run="$WORK_ROOT/training/$method/partition${pseed}_train$tseed"
          local out="$WORK_ROOT/eval/$method/partition${pseed}_train$tseed/eval$eseed"
          [[ -f "$out/results.json" ]] && continue
          eval_one lap "$CHECKPOINT" "$run" "$out" "$eseed" "$starts"
        done
      done
    done
  done
}

aggregate() {
  "$PYTHON" experiments/control_matrix/aggregate_matrix.py \
    --root "$WORK_ROOT" --dataset-name "$DATASET_NAME" \
    --train-seeds "$TRAIN_SEEDS" --partition-seeds "$PARTITION_SEEDS" \
    --eval-seeds "$EVAL_SEEDS"
}

case "$PHASE" in
  prepare) prepare ;;
  partition) partition ;;
  train) train ;;
  eval) evaluate ;;
  aggregate) aggregate ;;
  all) prepare; partition; train; evaluate; aggregate ;;
  *) echo "unknown PHASE=$PHASE" >&2; exit 2 ;;
esac
