#!/usr/bin/env bash
# Wait until a GPU is sufficiently idle for efficiency benchmarks.
#
# After the initial idle threshold is met, require a stabilization window so the
# GPU is not launched immediately after a heavy job ends (thermals, clocks, and
# allocator state). Joint/LAP and LeWM/LAP pairs run back-to-back within one
# session; stabilization applies only once at session start.
set -euo pipefail

MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
GPU_INDEX="${GPU_INDEX:-}"
POLL_SEC="${POLL_SEC:-300}"
# 4 checks × 60 s apart → 3 minute stabilization (t, t+60, t+120, t+180).
IDLE_STABILIZE_CHECKS="${IDLE_STABILIZE_CHECKS:-4}"
IDLE_STABILIZE_INTERVAL_SEC="${IDLE_STABILIZE_INTERVAL_SEC:-60}"
LOG="${LOG:-/dev/stderr}"

gpu_wait_log() {
  # Keep stdout clean: callers capture only the selected GPU index.
  echo "$@" | tee -a "${LOG}" >&2
}

strip_unit() {
  echo "${1// /}" | sed -E 's/(MiB|%)$//'
}

gpu_snapshot() {
  local idx="$1"
  strip_unit "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${idx}")"
  strip_unit "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${idx}")"
}

gpu_ready() {
  local idx="$1"
  local free util
  read -r free util < <(gpu_snapshot "${idx}")
  [[ "${free}" -ge "${MIN_FREE_MIB}" && "${util}" -le "${MAX_UTIL_PCT}" ]]
}

log_gpu_status() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx free util; do
    gpu_wait_log "[gpu-wait] GPU$(strip_unit "${idx}") free=$(strip_unit "${free}") MiB util=$(strip_unit "${util}")%"
  done
}

pick_gpu() {
  local idx free util
  if [[ -n "${GPU_INDEX}" ]]; then
    if gpu_ready "${GPU_INDEX}"; then
      echo "${GPU_INDEX}"
      return 0
    fi
    return 1
  fi
  local best_idx="" best_free=-1
  while IFS=',' read -r idx free util; do
    idx="$(strip_unit "${idx}")"
    free="$(strip_unit "${free}")"
    util="$(strip_unit "${util}")"
    if [[ "${free}" -ge "${MIN_FREE_MIB}" && "${util}" -le "${MAX_UTIL_PCT}" && "${free}" -gt "${best_free}" ]]; then
      best_idx="${idx}"
      best_free="${free}"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
  if [[ -n "${best_idx}" ]]; then
    echo "${best_idx}"
    return 0
  fi
  return 1
}

stabilize_gpu() {
  local idx="$1"
  local check
  if [[ "${IDLE_STABILIZE_CHECKS}" -le 0 ]]; then
    return 0
  fi
  local total_sec=$(( (IDLE_STABILIZE_CHECKS - 1) * IDLE_STABILIZE_INTERVAL_SEC ))
  gpu_wait_log "[gpu-wait] GPU ${idx} passed initial idle check; stabilizing for ${total_sec}s (${IDLE_STABILIZE_CHECKS} checks every ${IDLE_STABILIZE_INTERVAL_SEC}s)"
  for ((check = 1; check <= IDLE_STABILIZE_CHECKS; check++)); do
    if ! gpu_ready "${idx}"; then
      local free util
      read -r free util < <(gpu_snapshot "${idx}")
      gpu_wait_log "[gpu-wait] GPU ${idx} left idle during stabilization (check ${check}/${IDLE_STABILIZE_CHECKS}, free=${free} MiB util=${util}%)"
      return 1
    fi
    local free util
    read -r free util < <(gpu_snapshot "${idx}")
    gpu_wait_log "[gpu-wait] stabilize ${check}/${IDLE_STABILIZE_CHECKS}: GPU ${idx} free=${free} MiB util=${util}%"
    if (( check < IDLE_STABILIZE_CHECKS )); then
      sleep "${IDLE_STABILIZE_INTERVAL_SEC}"
    fi
  done
  return 0
}

gpu_wait_log "[gpu-wait] need free>=${MIN_FREE_MIB} MiB and util<=${MAX_UTIL_PCT}%${GPU_INDEX:+ on GPU ${GPU_INDEX}}; stabilization=${IDLE_STABILIZE_CHECKS}x${IDLE_STABILIZE_INTERVAL_SEC}s"
while true; do
  if selected="$(pick_gpu)"; then
    if stabilize_gpu "${selected}"; then
      gpu_wait_log "[gpu-wait] selected GPU ${selected} after stabilization"
      echo "${selected}"
      exit 0
    fi
    gpu_wait_log "[gpu-wait] stabilization failed for GPU ${selected}; resuming poll"
  fi
  log_gpu_status
  sleep "${POLL_SEC}"
done
