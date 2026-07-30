#!/usr/bin/env bash
# GPU-pool dispatcher: launch the next job as soon as a GPU frees up.
# Safe to restart mid-run — skips finished / already-running jobs, detects busy GPUs.
#
# Usage:
#   bash dispatch_geometry_retrain.sh
#   GPUS="0 1 2 3 4 5 6 7" bash dispatch_geometry_retrain.sh
set -euo pipefail

ROOT="experiments/tworoom/scripts"
MAIN_DIR="experiments/tworoom/results/tworoom_geometry_train_region_predictors"
GLOBAL_DIR="experiments/tworoom/results/tworoom_geometry_train_global_ft_65ep"
DOORWAY80_DIR="experiments/tworoom/results/tworoom_geometry_train_region_predictors_doorway80ep"
DISPATCH_LOG="experiments/tworoom/results/retrain_dispatch.log"
SEED="${SEED:-42}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"

log() { echo "$*" >> "${DISPATCH_LOG}"; }

gpu_memory_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v g="${gpu}" '$1 == g {print $2; exit}'
}

gpu_is_busy() {
  local gpu="$1"
  local mib
  mib="$(gpu_memory_mib "${gpu}")"
  [[ -n "${mib}" && "${mib}" -gt 500 ]]
}

# job_id|kind|...  (kind = region | global_ft | doorway80 | latent)
JOBS=(
  "region_30_common|region|common|30"
  "region_30_doorway|region|doorway_corridor|30"
  "region_30_left|region|left_room|30"
  "region_30_near_wall|region|near_wall|30"
  "region_30_right|region|right_room|30"
  "global_ft_65|global_ft|65"
  "region_50_doorway|region|doorway_corridor|50|best"
  "region_50_left|region|left_room|50|best"
  "region_50_right|region|right_room|50|best"
  "doorway80|doorway80|80"
  "latent_outer0|latent|0"
  "latent_outer1|latent|1"
  "latent_outer2|latent|2"
)

parse_job() {
  local spec="$1"
  JOB_ID="${spec%%|*}"
  local remainder="${spec#*|}"
  JOB_KIND="${remainder%%|*}"
  JOB_REST="${remainder#*|}"
  read -r -a JOB_ARGS <<< "${JOB_REST//|/ }"
}

job_done() {
  local spec="$1"
  parse_job "${spec}"
  case "${JOB_KIND}" in
    region)
      local region="${JOB_ARGS[0]}"
      local epochs="${JOB_ARGS[1]}"
      [[ -f "${MAIN_DIR}/_work/${region}_${epochs}ep_seed${SEED}/P_train_${region}_object.ckpt" ]]
      ;;
    global_ft)
      [[ -f "${GLOBAL_DIR}/P_train_global_ft_object.ckpt" ]]
      ;;
    doorway80)
      [[ -f "${DOORWAY80_DIR}/P_train_doorway_corridor_object.ckpt" ]]
      ;;
    latent)
      local outer="${JOB_ARGS[0]}"
      [[ -f "experiments/tworoom/results/tworoom_latent_kmeanspp_kmeanspp_R50_outer${outer}/manifest.json" ]]
      ;;
    *) return 1 ;;
  esac
}

job_running() {
  local spec="$1"
  parse_job "${spec}"
  case "${JOB_KIND}" in
    region)
      local region="${JOB_ARGS[0]}"
      local epochs="${JOB_ARGS[1]}"
      pgrep -af "trajectory.py" | grep -qE -- "--regions ${region} .*--epochs ${epochs}"
      ;;
    global_ft)
      pgrep -af "trajectory.py" | grep -q "train-global-predictor"
      ;;
    doorway80)
      pgrep -af "trajectory.py" | grep -q "tworoom_geometry_train_region_predictors_doorway80ep"
      ;;
    latent)
      local outer="${JOB_ARGS[0]}"
      pgrep -af "latent_cluster_train_predictors.py" | grep -q "outer${outer}"
      ;;
    *) return 1 ;;
  esac
}

should_skip_job() {
  local spec="$1"
  parse_job "${spec}"
  if job_done "${spec}"; then
    log "[skip:done] ${JOB_ID}"
    return 0
  fi
  if job_running "${spec}"; then
    log "[skip:running] ${JOB_ID}"
    return 0
  fi
  return 1
}

LAUNCH_PID=""

launch_job() {
  local gpu="$1"
  local spec="$2"
  parse_job "${spec}"

  case "${JOB_KIND}" in
    region)
      local region="${JOB_ARGS[0]}"
      local epochs="${JOB_ARGS[1]}"
      if [[ "${JOB_ARGS[2]:-}" == "best" ]]; then
        env GPU="${gpu}" REGION="${region}" EPOCHS="${epochs}" SEED="${SEED}" \
          SELECT_BEST=1 SAVE_EPOCHS=20,30,40,50 \
          bash "${ROOT}/run_geometry_train_one_region.sh" &
      else
        env GPU="${gpu}" REGION="${region}" EPOCHS="${epochs}" SEED="${SEED}" \
          bash "${ROOT}/run_geometry_train_one_region.sh" &
      fi
      ;;
    global_ft)
      env GPU="${gpu}" EPOCHS="${JOB_ARGS[0]}" SEED="${SEED}" \
        bash "${ROOT}/run_geometry_train_global_ft.sh" &
      ;;
    doorway80)
      env GPU="${gpu}" EPOCHS="${JOB_ARGS[0]}" SEED="${SEED}" \
        bash "${ROOT}/run_geometry_train_doorway_one_region.sh" &
      ;;
    latent)
      env GPU="${gpu}" OUTER_SEED="${JOB_ARGS[0]}" \
        bash "${ROOT}/run_latent_kmeanspp_train_predictors_50ep.sh" &
      ;;
    *)
      log "[fail] unknown kind for ${JOB_ID}"
      return 1
      ;;
  esac
  LAUNCH_PID=$!
  log "[launch] gpu=${gpu} job=${JOB_ID} pid=${LAUNCH_PID} at $(date)"
}

assign_next_job() {
  local gpu="$1"
  while [[ "${queue_idx}" -lt "${#JOBS[@]}" ]]; do
    local spec="${JOBS[$queue_idx]}"
    queue_idx=$((queue_idx + 1))
    if should_skip_job "${spec}"; then
      continue
    fi
    launch_job "${gpu}" "${spec}"
    GPU_PID["${gpu}"]="${LAUNCH_PID}"
    return 0
  done
  return 1
}

declare -A GPU_PID=()
queue_idx=0
failures=0

log "==== GPU-pool dispatch start at $(date) GPUs=${GPUS[*]} ===="

# Fill idle GPUs from the queue.
for gpu in "${GPUS[@]}"; do
  if gpu_is_busy "${gpu}"; then
    log "[busy-gpu] ${gpu} ($(gpu_memory_mib "${gpu}") MiB) — external or prior job"
    continue
  fi
  assign_next_job "${gpu}" || true
done

while [[ "${#GPU_PID[@]}" -gt 0 ]] || [[ "${queue_idx}" -lt "${#JOBS[@]}" ]]; do
  if [[ "${#GPU_PID[@]}" -eq 0 ]]; then
    for gpu in "${GPUS[@]}"; do
      if gpu_is_busy "${gpu}"; then
        continue
      fi
      if assign_next_job "${gpu}"; then
        break
      fi
    done
    if [[ "${#GPU_PID[@]}" -eq 0 && "${queue_idx}" -ge "${#JOBS[@]}" ]]; then
      break
    fi
    sleep 5
    continue
  fi

  if ! wait -n; then
    failures=$((failures + 1))
  fi

  for gpu in "${!GPU_PID[@]}"; do
    pid="${GPU_PID[$gpu]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then
        log "[done] gpu=${gpu} pid=${pid}"
      else
        log "[fail] gpu=${gpu} pid=${pid}"
        failures=$((failures + 1))
      fi
      unset 'GPU_PID[$gpu]'
      assign_next_job "${gpu}" || true
    fi
  done
done

if [[ "${failures}" -ne 0 ]]; then
  log "==== GPU-pool dispatch finished with ${failures} failure(s) at $(date) ===="
  exit 1
fi

log "==== GPU-pool dispatch done at $(date) ===="
