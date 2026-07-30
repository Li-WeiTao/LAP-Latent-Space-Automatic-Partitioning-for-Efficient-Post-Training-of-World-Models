#!/usr/bin/env bash
# Wait without occupying GPUs, run a one-episode real-checkpoint smoke test, then launch 3 outers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
ROOT="experiments/tworoom"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
read -r -a CANDIDATE_GPUS <<< "${GPU_CANDIDATES:-0 1 2 3 4 5 6 7}"
GPUS=()

gpu_used_mib() {
  nvidia-smi \
    --query-compute-apps=used_memory \
    --format=csv,noheader,nounits \
    --id="$1" \
    | awk '{sum += $1} END {print sum + 0}'
}

while true; do
  GPUS=()
  status_line=""
  for gpu in "${CANDIDATE_GPUS[@]}"; do
    used="$(gpu_used_mib "${gpu}")"
    status_line+=" gpu${gpu}=${used}MiB"
    if (( used <= MAX_USED_MIB )); then
      GPUS+=("${gpu}")
    fi
  done
  echo "[$(date)] waiting:${status_line} free=${#GPUS[@]}/3 threshold=${MAX_USED_MIB}MiB"
  if (( ${#GPUS[@]} >= 3 )); then
    GPUS=("${GPUS[@]:0:3}")
    break
  fi
  sleep 60
done

echo "[$(date)] selected GPUs: ${GPUS[*]}; starting one-episode real-checkpoint smoke test"
. .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
SMOKE_OUT="${ROOT}/results/smoke_latent_kmeanspp_R50_outer0_step_routing_20260714"
mkdir -p "${SMOKE_OUT}"

if [[ ! -f "${SMOKE_OUT}/results.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python "${ROOT}/tworoom_success_rate_eval.py" \
    --mode latent_cluster3 \
    --latent-routing step \
    --checkpoint "${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}" \
    --seed 0 \
    --num-eval 1 \
    --goal-offset 50 \
    --eval-budget 25 \
    --out-dir "${SMOKE_OUT}" \
    --eval-start-indices "${ROOT}/results/tworoom_success_rate_baseline_exp6_seed0/results.json" \
    --kmeanspp-label-npz "${ROOT}/results/latent_kmeanspp_multirestart_k3/labels/kmeanspp_R50_outer0.npz" \
    --zscore-params "${ROOT}/results/latent_kmeanspp_multirestart_k3/zscore_params.npz" \
    --cluster-predictor-dir "${ROOT}/results/tworoom_latent_kmeanspp_kmeanspp_R50_outer0" \
    > "${SMOKE_OUT}/run.log" 2>&1
fi

PYTHONPATH=.:experiments/tworoom python - "${SMOKE_OUT}/results.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
assert data["latent_routing"] == "step", data.get("latent_routing")
assert data["inference_route_calls"] > 0
assert data["inference_route_assignments"] > 0
assert sum(data["inference_route_histogram"].values()) == data["inference_route_assignments"]
print(
    "Smoke test passed:",
    {
        "route_calls": data["inference_route_calls"],
        "route_assignments": data["inference_route_assignments"],
        "route_switch_rate": data["inference_route_switch_rate"],
        "route_fraction": data["inference_route_fraction"],
    },
)
PY

echo "[$(date)] smoke test passed; launching outer seeds 0/1/2"
GPU0="${GPUS[0]}" GPU1="${GPUS[1]}" GPU2="${GPUS[2]}" \
  bash "${ROOT}/scripts/run_success_rate_latent_kmeanspp_step_routing_parallel.sh"

echo "[$(date)] queued latent step-routing experiment completed"
