#!/usr/bin/env bash
# Wait until a GPU is sufficiently idle for efficiency benchmarks.
set -euo pipefail

MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
GPU_INDEX="${GPU_INDEX:-}"
POLL_SEC="${POLL_SEC:-300}"
LOG="${LOG:-/dev/stdout}"

strip_unit() {
  echo "${1// /}" | sed -E 's/(MiB|%)$//'
}

gpu_ready() {
  local idx="$1"
  local free util
  free="$(strip_unit "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${idx}")")"
  util="$(strip_unit "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${idx}")")"
  [[ "${free}" -ge "${MIN_FREE_MIB}" && "${util}" -le "${MAX_UTIL_PCT}" ]]
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

echo "[gpu-wait] need free>=${MIN_FREE_MIB} MiB and util<=${MAX_UTIL_PCT}%${GPU_INDEX:+ on GPU ${GPU_INDEX}}" | tee -a "${LOG}"
while true; do
  if selected="$(pick_gpu)"; then
    echo "[gpu-wait] selected GPU ${selected}" | tee -a "${LOG}"
    echo "${selected}"
    exit 0
  fi
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx free util; do
    echo "[gpu-wait] GPU$(strip_unit "${idx}") free=$(strip_unit "${free}") MiB util=$(strip_unit "${util}")%" | tee -a "${LOG}"
  done
  sleep "${POLL_SEC}"
done
