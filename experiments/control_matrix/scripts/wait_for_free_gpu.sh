#!/usr/bin/env bash
set -euo pipefail

# Cooperative GPU queue for the comparison matrix.  A GPU is selected only
# after it stays below both thresholds for several consecutive polls.  The
# per-GPU flock prevents two LAP jobs launched through this wrapper from
# choosing the same device.
GPU_CANDIDATES=${GPU_CANDIDATES:-0,1,2,3,4,5,6,7}
MAX_USED_MIB=${MAX_USED_MIB:-2048}
MAX_UTIL_PERCENT=${MAX_UTIL_PERCENT:-10}
STABLE_POLLS=${STABLE_POLLS:-3}
POLL_SECONDS=${POLL_SECONDS:-60}
LOCK_ROOT=${LOCK_ROOT:-/tmp/lap-control-matrix-gpu-locks}

if [[ ${1:-} == -- ]]; then
  shift
fi
if [[ $# -eq 0 ]]; then
  echo "usage: wait_for_free_gpu.sh -- COMMAND [ARG ...]" >&2
  exit 2
fi

mkdir -p "$LOCK_ROOT"
IFS=, read -r -a candidates <<< "$GPU_CANDIDATES"
declare -A stable=()
poll=0

while true; do
  poll=$((poll + 1))
  for gpu in "${candidates[@]}"; do
    gpu=${gpu//[[:space:]]/}
    [[ -n "$gpu" ]] || continue
    exec {lock_fd}>"$LOCK_ROOT/gpu${gpu}.lock"
    if ! flock -n "$lock_fd"; then
      stable[$gpu]=0
      exec {lock_fd}>&-
      continue
    fi
    IFS=, read -r used util < <(
      nvidia-smi --id="$gpu" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )
    used=${used//[[:space:]]/}
    util=${util//[[:space:]]/}
    if (( used <= MAX_USED_MIB && util <= MAX_UTIL_PERCENT )); then
      stable[$gpu]=$(( ${stable[$gpu]:-0} + 1 ))
    else
      stable[$gpu]=0
    fi
    if (( ${stable[$gpu]} >= STABLE_POLLS )); then
      echo "[gpu-queue] selected physical GPU $gpu after ${stable[$gpu]} stable polls" >&2
      export CUDA_VISIBLE_DEVICES=$gpu
      export LAP_ASSIGNED_PHYSICAL_GPU=$gpu
      "$@"
      exit $?
    fi
    exec {lock_fd}>&-
  done
  if (( poll == 1 || poll % 10 == 0 )); then
    echo "[gpu-queue] no free GPU at poll $poll; sleeping ${POLL_SECONDS}s" >&2
  fi
  sleep "$POLL_SECONDS"
done
