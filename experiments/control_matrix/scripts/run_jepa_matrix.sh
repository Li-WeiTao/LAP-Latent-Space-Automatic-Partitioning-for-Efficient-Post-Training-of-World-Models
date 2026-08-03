#!/usr/bin/env bash
# Generic JEPA matrix driver for LeWM and Sub-JEPA. Task paths come from CLI/env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
TMP_RESOLVED="$(mktemp)"
"$PYTHON" experiments/control_matrix/resolve_jepa_matrix_config.py "$@" --output "$TMP_RESOLVED"
WORK_ROOT="$( "$PYTHON" -c "import json; print(json.load(open('$TMP_RESOLVED'))['work_root'])" )"
mkdir -p "$WORK_ROOT/manifests"
RESOLVED_JSON="$WORK_ROOT/manifests/resolved_config.json"
if [[ "${RESOLVED_CONFIG_SKIP:-0}" != "1" ]]; then
  cp "$TMP_RESOLVED" "$RESOLVED_JSON"
elif [[ -n "${RESOLVED_CONFIG_LEAF:-}" ]]; then
  mkdir -p "$(dirname "$RESOLVED_CONFIG_LEAF")"
  cp "$TMP_RESOLVED" "$RESOLVED_CONFIG_LEAF"
fi
rm -f "$TMP_RESOLVED"

eval "$("$PYTHON" experiments/control_matrix/resolve_jepa_matrix_config.py "$@" --emit-shell)"

if [[ -z "${DATA_FILE:-}" || -z "${CHECKPOINT:-}" ]]; then
  echo "resolved configuration is missing DATA_FILE or CHECKPOINT" >&2
  exit 2
fi

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
mkdir -p "$WORK_ROOT"

IFS=, read -r -a train_seeds <<< "$TRAIN_SEEDS"
IFS=, read -r -a partition_seeds <<< "$PARTITION_SEEDS"
IFS=, read -r -a eval_seeds <<< "$EVAL_SEEDS"
IFS=, read -r -a methods <<< "$METHODS"

PREP="$WORK_ROOT/preparation"
EMBEDDING_CACHE="$PREP/embedding_cache.npz"
LATENT_CACHE="$EMBEDDING_CACHE"
PREPARE_MAX_STARTS="${PREPARE_MAX_STARTS:-0}"
PREPARE_REFERENCE_CACHE="${PREPARE_REFERENCE_CACHE:-}"
PREPARE_OVERWRITE="${PREPARE_OVERWRITE:-0}"
NUM_CLUSTERS="${NUM_CLUSTERS:-3}"
GATE_DIAGNOSTIC_SEEDS="${GATE_DIAGNOSTIC_SEEDS:-0,1,2}"
GATE_DEPLOYMENT_SEED="${GATE_DEPLOYMENT_SEED:-0}"
GATE_NUM_LANDMARKS="${GATE_NUM_LANDMARKS:-20000}"
GATE_NOMINAL_KNN="${GATE_NOMINAL_KNN:-30}"
GATE_PERTURB_KNN="${GATE_PERTURB_KNN:-27,33}"
GATE_PERTURBATION_MULTIPLIER="${GATE_PERTURBATION_MULTIPLIER:-2.0}"
GATE_RETENTION_THRESHOLD="${GATE_RETENTION_THRESHOLD:-0.5}"
GATE_BACKGROUND_GAP_COUNT="${GATE_BACKGROUND_GAP_COUNT:-10}"
GATE_BACKGROUND_MAD_MULTIPLIER="${GATE_BACKGROUND_MAD_MULTIPLIER:-3.0}"
GATE_EPSILON="${GATE_EPSILON:-1e-8}"
SMOKE_EVAL_STARTS="${SMOKE_EVAL_STARTS:-5}"
SMOKE_TRAIN_EPOCHS="${SMOKE_TRAIN_EPOCHS:-1}"

log_cmd() {
  echo "+ $*" >&2
}

run_or_print() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    log_cmd "$@"
  else
    "$@"
  fi
}

probe() {
  local out="$WORK_ROOT/probe/checkpoint_probe.json"
  mkdir -p "$WORK_ROOT/probe"
  run_or_print \
    "$PYTHON" scripts/probe_jepa_checkpoint.py \
    --model-family "$MODEL_FAMILY" \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATA_FILE" \
    --frameskip "$FRAMESKIP" \
    --history-size "$HISTORY_SIZE" \
    --num-preds "$NUM_PREDS" \
    --img-size "$IMG_SIZE" \
    --max-samples 16 \
    --output "$out"
}

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
  local -a prepare_args=(
    "$PYTHON" experiments/control_matrix/prepare_lewm_cache.py
    --dataset-name "$DATASET_NAME"
    --data-file "$DATA_FILE"
    --checkpoint "$CHECKPOINT"
    --out-dir "$PREP"
    --model-family "$MODEL_FAMILY"
    --frameskip "$FRAMESKIP"
    --history-size "$HISTORY_SIZE"
    --num-preds "$NUM_PREDS"
    --img-size "$IMG_SIZE"
    --device cuda
    --transition-batch-size 128
    --frame-batch-size 512
    --num-workers "$CPU_THREADS"
    --cpu-threads "$CPU_THREADS"
    --python "$PYTHON"
  )
  [[ "$PREPARE_MAX_STARTS" -gt 0 ]] && prepare_args+=(--max-starts "$PREPARE_MAX_STARTS")
  [[ -n "$PREPARE_REFERENCE_CACHE" ]] && prepare_args+=(--reference-cache "$PREPARE_REFERENCE_CACHE")
  [[ "$PREPARE_OVERWRITE" == "1" ]] && prepare_args+=(--overwrite)
  run_or_print "${prepare_args[@]}"
}

partition_global() {
  local global_dir="$WORK_ROOT/partitions/global/seed0"
  if [[ ! -f "$global_dir/manifest.json" || "$DRY_RUN" == "1" ]]; then
    run_or_print "$PYTHON" experiments/control_matrix/fit_partition.py \
      --method global --dataset-name "$DATASET_NAME" \
      --latent-cache "$LATENT_CACHE" --frameskip "$FRAMESKIP" \
      --cpu-threads "$CPU_THREADS" --out-dir "$global_dir"
  fi
}

partition_regions() {
  for method in "${methods[@]}"; do
    for pseed in "${partition_seeds[@]}"; do
      local out="$WORK_ROOT/partitions/$method/seed$pseed"
      [[ -f "$out/manifest.json" && "$DRY_RUN" != "1" ]] && continue
      run_or_print "$PYTHON" experiments/control_matrix/fit_partition.py \
        --method "$method" --dataset-name "$DATASET_NAME" \
        --data-file "$DATA_FILE" --latent-cache "$LATENT_CACHE" \
        --frameskip "$FRAMESKIP" --num-clusters "$NUM_CLUSTERS" --seed "$pseed" \
        --gpu-id 0 --cpu-threads "$CPU_THREADS" --out-dir "$out"
    done
  done
}

partition() {
  partition_global
  partition_regions
}

partition_auto() {
  local out="$WORK_ROOT/auto/partition"
  if [[ ! -f "$out/manifest.json" || "$DRY_RUN" == "1" ]]; then
    run_or_print "$PYTHON" experiments/control_matrix/fit_partition.py \
      --method auto --dataset-name "$DATASET_NAME" \
      --data-file "$DATA_FILE" --latent-cache "$LATENT_CACHE" \
      --frameskip "$FRAMESKIP" --num-clusters "$NUM_CLUSTERS" \
      --num-landmarks "$GATE_NUM_LANDMARKS" --knn "$GATE_NOMINAL_KNN" \
      --perturb-knn "$GATE_PERTURB_KNN" \
      --diagnostic-seeds "$GATE_DIAGNOSTIC_SEEDS" \
      --deployment-seed "$GATE_DEPLOYMENT_SEED" \
      --gate-perturbation-multiplier "$GATE_PERTURBATION_MULTIPLIER" \
      --gate-retention-threshold "$GATE_RETENTION_THRESHOLD" \
      --gate-background-gap-count "$GATE_BACKGROUND_GAP_COUNT" \
      --gate-background-mad-multiplier "$GATE_BACKGROUND_MAD_MULTIPLIER" \
      --gate-epsilon "$GATE_EPSILON" \
      --gpu-id 0 --cpu-threads "$CPU_THREADS" --out-dir "$out"
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    "$PYTHON" - "$out/manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
gate = manifest["method_metadata"]["automatic_gate"]
print(
    "[auto-gate] selected_method=" + gate["selected_method"]
    + " reason=" + gate["reason"]
    + " S=" + str(gate["retained_safety_fraction"])
    + " R=" + str(gate["robust_residual_gap"])
    + " T_bg=" + str(gate["background_threshold"]),
    flush=True,
)
PY
  fi
}

train_epochs() {
  local epochs="$1"
  shift
  local -a base=( "$@" )
  base+=( --epochs "$epochs" --model-family "$MODEL_FAMILY" )
  run_or_print "${base[@]}"
}

train_joint() {
  for tseed in "${train_seeds[@]}"; do
    local joint="$WORK_ROOT/training/joint/train$tseed"
    if [[ ! -f "$joint/manifest.json" || "$DRY_RUN" == "1" ]]; then
      run_or_print "$PYTHON" experiments/tworoom/joint_continue_tworoom.py \
        --dataset-name "$DATASET_NAME" --data-file "$DATA_FILE" \
        --checkpoint "$CHECKPOINT" --out-dir "$joint" \
        --seed "$tseed" --epochs 3 --precision fp32 \
        --frameskip "$FRAMESKIP" --num-workers "$CPU_THREADS" \
        --cpu-threads "$CPU_THREADS" --model-family "$MODEL_FAMILY"
    fi
  done
}

train_global() {
  local epochs="${1:-50}"
  for tseed in "${train_seeds[@]}"; do
    local global="$WORK_ROOT/training/global/train$tseed"
    if [[ ! -f "$global/manifest.json" || "$DRY_RUN" == "1" ]]; then
      train_epochs "$epochs" \
        "$PYTHON" experiments/control_matrix/train_predictors.py \
        --dataset-name "$DATASET_NAME" --latent-cache "$LATENT_CACHE" \
        --pretrained-model "$CHECKPOINT" \
        --partition-dir "$WORK_ROOT/partitions/global/seed0" \
        --out-dir "$global" --train-seed "$tseed" \
        --cpu-threads "$CPU_THREADS"
    fi
  done
}

train_regions() {
  local epochs="${1:-50}"
  for tseed in "${train_seeds[@]}"; do
    for method in "${methods[@]}"; do
      for pseed in "${partition_seeds[@]}"; do
        local out="$WORK_ROOT/training/$method/partition${pseed}_train$tseed"
        [[ -f "$out/manifest.json" && "$DRY_RUN" != "1" ]] && continue
        train_epochs "$epochs" \
          "$PYTHON" experiments/control_matrix/train_predictors.py \
          --dataset-name "$DATASET_NAME" --latent-cache "$LATENT_CACHE" \
          --pretrained-model "$CHECKPOINT" \
          --partition-dir "$WORK_ROOT/partitions/$method/seed$pseed" \
          --out-dir "$out" --train-seed "$tseed" \
          --cpu-threads "$CPU_THREADS"
      done
    done
  done
}

train() {
  [[ "$SKIP_JOINT" != "1" ]] && train_joint
  train_global 50
  train_regions 50
}

train_auto() {
  [[ -f "$WORK_ROOT/auto/partition/manifest.json" || "$DRY_RUN" == "1" ]] || partition_auto
  for tseed in "${train_seeds[@]}"; do
    local out="$WORK_ROOT/auto/training/train$tseed"
    [[ -f "$out/manifest.json" && "$DRY_RUN" != "1" ]] && continue
    train_epochs 50 \
      "$PYTHON" experiments/control_matrix/train_predictors.py \
      --dataset-name "$DATASET_NAME" --latent-cache "$LATENT_CACHE" \
      --pretrained-model "$CHECKPOINT" \
      --partition-dir "$WORK_ROOT/auto/partition" \
      --out-dir "$out" --train-seed "$tseed" \
      --cpu-threads "$CPU_THREADS"
  done
}

resolve_eval_starts() {
  local eseed="$1"
  local horizon="$2"
  local paired_root="${PAIRED_START_ROOT:-}"
  if [[ "$horizon" == "short" && -n "${PAIRED_START_ROOT_SHORT:-}" ]]; then
    paired_root="$PAIRED_START_ROOT_SHORT"
  elif [[ "$horizon" == "long" && -n "${PAIRED_START_ROOT_LONG:-}" ]]; then
    paired_root="$PAIRED_START_ROOT_LONG"
  fi
  if [[ -n "$paired_root" ]]; then
    echo "$paired_root/eval/official/eval${eseed}/results.json"
  else
    echo "$WORK_ROOT/eval/official/eval${eseed}/results.json"
  fi
}

eval_one() {
  local mode=$1 checkpoint=$2 lap_run=$3 out=$4 eval_seed=$5 starts=$6
  local goal_offset=$7
  local -a args=(
    --mode "$mode" --seed "$eval_seed" --checkpoint "$checkpoint"
    --config-name "$EVAL_CONFIG" --dataset-tag "$DATASET_NAME"
    --cache-dir "$CACHE_DIR" --out-dir "$out" --num-eval "$NUM_EVAL"
    --model-family "$MODEL_FAMILY"
  )
  [[ -n "$goal_offset" ]] && args+=(--goal-offset "$goal_offset")
  args+=(--eval-budget "$EVAL_BUDGET")
  [[ -n "$lap_run" ]] && args+=(--lap-run-dir "$lap_run" --latent-routing mpc)
  [[ -n "$starts" ]] && args+=(--eval-start-indices "$starts")
  [[ -z "$starts" ]] && args+=(--sample-eval-starts)
  run_or_print "$PYTHON" experiments/tworoom/tworoom_success_rate_eval.py "${args[@]}"
}

evaluate_official() {
  local goal_offset="${1:-}"
  local out_root="${2:-$WORK_ROOT/eval/official}"
  local horizon="${3:-}"
  for eseed in "${eval_seeds[@]}"; do
    local starts
    starts="$(resolve_eval_starts "$eseed" "$horizon")"
    [[ -f "$starts" || "$DRY_RUN" == "1" ]] || {
      echo "missing paired starts for official eval seed $eseed: $starts" >&2
      exit 1
    }
    local baseline="$out_root/eval$eseed"
    if [[ ! -f "$baseline/results.json" || "$DRY_RUN" == "1" ]]; then
      eval_one baseline "$CHECKPOINT" "" "$baseline" "$eseed" "$starts" "$goal_offset"
    fi
  done
}

evaluate_global() {
  local goal_offset="${1:-}"
  local out_root="${2:-$WORK_ROOT/eval/global}"
  for eseed in "${eval_seeds[@]}"; do
    local starts
    starts="$(resolve_eval_starts "$eseed" "${3:-}")"
    [[ -f "$starts" || "$DRY_RUN" == "1" ]] || { echo "missing paired starts: $starts" >&2; exit 1; }
    for tseed in "${train_seeds[@]}"; do
      local global="$out_root/train$tseed/eval$eseed"
      if [[ ! -f "$global/results.json" || "$DRY_RUN" == "1" ]]; then
        eval_one baseline \
          "$WORK_ROOT/training/global/train$tseed/P_train_cluster0_object.ckpt" \
          "" "$global" "$eseed" "$starts" "$goal_offset"
      fi
    done
  done
}

evaluate_regions() {
  local goal_offset="${1:-}"
  local out_root="${2:-$WORK_ROOT/eval}"
  for eseed in "${eval_seeds[@]}"; do
    local starts
    starts="$(resolve_eval_starts "$eseed" "${3:-}")"
    [[ -f "$starts" || "$DRY_RUN" == "1" ]] || { echo "missing paired starts: $starts" >&2; exit 1; }
    for tseed in "${train_seeds[@]}"; do
      for method in "${methods[@]}"; do
        for pseed in "${partition_seeds[@]}"; do
          local run="$WORK_ROOT/training/$method/partition${pseed}_train$tseed"
          local out="$out_root/$method/partition${pseed}_train$tseed/eval$eseed"
          [[ -f "$out/results.json" && "$DRY_RUN" != "1" ]] && continue
          eval_one lap "$CHECKPOINT" "$run" "$out" "$eseed" "$starts" "$goal_offset"
        done
      done
    done
  done
}

evaluate() {
  local goal_offset="${1:-}"
  evaluate_official "$goal_offset"
  evaluate_global "$goal_offset"
  evaluate_regions "$goal_offset"
}

aggregate() {
  local -a aggregate_args=(
    "$PYTHON" experiments/control_matrix/aggregate_matrix.py
    --root "$WORK_ROOT"
    --dataset-name "$DATASET_NAME"
    --train-seeds "$TRAIN_SEEDS"
    --partition-seeds "$PARTITION_SEEDS"
    --eval-seeds "$EVAL_SEEDS"
    --methods "$METHODS"
  )
  [[ "$SKIP_JOINT" == "1" ]] && aggregate_args+=(--skip-joint)
  run_or_print "${aggregate_args[@]}"
}

smoke() {
  local _train_seeds="$TRAIN_SEEDS"
  local _partition_seeds="$PARTITION_SEEDS"
  local _eval_seeds="$EVAL_SEEDS"
  local _methods="$METHODS"
  TRAIN_SEEDS=0
  PARTITION_SEEDS=0
  EVAL_SEEDS=0
  METHODS=spectral
  IFS=, read -r -a train_seeds <<< "$TRAIN_SEEDS"
  IFS=, read -r -a partition_seeds <<< "$PARTITION_SEEDS"
  IFS=, read -r -a eval_seeds <<< "$EVAL_SEEDS"
  IFS=, read -r -a methods <<< "$METHODS"

  probe
  prepare
  run_or_print "$PYTHON" experiments/control_matrix/validate_jepa_smoke.py \
    --phase cache-equivalence \
    --model-family "$MODEL_FAMILY" \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATA_FILE" \
    --cache "$LATENT_CACHE" \
    --frameskip "$FRAMESKIP" \
    --history-size "$HISTORY_SIZE" \
    --num-preds "$NUM_PREDS" \
    --img-size "$IMG_SIZE" \
    --transition-batch-size 128 \
    --frame-batch-size 512 \
    --num-samples 16 \
    --work-root "$WORK_ROOT"

  partition_global
  partition_regions
  train_global "$SMOKE_TRAIN_EPOCHS"
  train_regions "$SMOKE_TRAIN_EPOCHS"

  run_or_print "$PYTHON" experiments/control_matrix/validate_jepa_smoke.py \
    --phase frozen-audit \
    --checkpoint "$CHECKPOINT" \
    --work-root "$WORK_ROOT"

  NUM_EVAL="$SMOKE_EVAL_STARTS"
  evaluate_official "$SHORT_GOAL_OFFSET"
  evaluate_global "$SHORT_GOAL_OFFSET"
  evaluate_regions "$SHORT_GOAL_OFFSET"

  run_or_print "$PYTHON" experiments/control_matrix/validate_jepa_smoke.py \
    --phase route-equivalence \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATA_FILE" \
    --work-root "$WORK_ROOT"

  aggregate

  TRAIN_SEEDS="$_train_seeds"
  PARTITION_SEEDS="$_partition_seeds"
  EVAL_SEEDS="$_eval_seeds"
  METHODS="$_methods"
}

echo "[jepa-matrix] model_family=$MODEL_FAMILY phase=$PHASE work_root=$WORK_ROOT" >&2
echo "[jepa-matrix] resolved_config=$RESOLVED_JSON" >&2

case "$PHASE" in
  probe) probe ;;
  prepare) prepare ;;
  gate) partition_auto ;;
  partition_global) partition_global ;;
  partition_regions) partition_regions ;;
  partition) partition ;;
  partition_auto) partition_auto ;;
  train_joint) train_joint ;;
  train_global) train_global 50 ;;
  train_regions) train_regions 50 ;;
  train) train ;;
  train_auto) train_auto ;;
  eval-short)
    evaluate "$SHORT_GOAL_OFFSET"
    ;;
  eval-long)
    evaluate "$LONG_GOAL_OFFSET"
    ;;
  eval_official) evaluate_official "${GOAL_OFFSET:-}" ;;
  eval_global) evaluate_global "${GOAL_OFFSET:-}" ;;
  eval_regions) evaluate_regions "${GOAL_OFFSET:-}" ;;
  eval) evaluate "${GOAL_OFFSET:-}" ;;
  aggregate) aggregate ;;
  smoke) smoke ;;
  all)
    probe
    prepare
    partition
    partition_auto
    train
    train_auto
    evaluate "$SHORT_GOAL_OFFSET"
    aggregate
    ;;
  *)
    echo "unknown PHASE=$PHASE" >&2
    exit 2
    ;;
esac
