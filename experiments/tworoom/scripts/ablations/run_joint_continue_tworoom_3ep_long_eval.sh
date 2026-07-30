#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

GPU="${GPU:?set GPU to one available physical GPU id}"
TRAIN_SEED="${TRAIN_SEED:?set TRAIN_SEED}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT=experiments/tworoom
RUN_DIR="${ROOT}/results/tworoom_joint_continue_3ep_trainseed${TRAIN_SEED}"
CKPT="${RUN_DIR}/joint_continue_object.ckpt"
MANIFEST="${RUN_DIR}/manifest.json"
TAG="joint_continue_3ep_trainseed${TRAIN_SEED}_long"
SUMMARY="${ROOT}/results/tworoom_success_rate_${TAG}_5eval_summary.csv"
SUMMARY_JSON="${ROOT}/results/tworoom_success_rate_${TAG}_5eval_summary.json"

python - "${MANIFEST}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
assert manifest["formal_full_epoch"] is True
assert manifest["config"]["epochs"] == 3
assert manifest["config"]["precision"] == "fp32"
PY

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 2
fi
if [[ -e "${SUMMARY}" || -e "${SUMMARY_JSON}" ]]; then
  echo "Refusing to overwrite ${SUMMARY} or ${SUMMARY_JSON}" >&2
  exit 2
fi

echo "seed,mode,success_rate,successes,num_eval,out_dir" > "${SUMMARY}"
for eval_seed in 0 1 2 3 4; do
  starts="${ROOT}/results/tworoom_success_rate_baseline_exp6_seed${eval_seed}/results.json"
  out_dir="${ROOT}/results/tworoom_success_rate_${TAG}_evalseed${eval_seed}"
  if [[ -e "${out_dir}" ]]; then
    echo "Refusing to overwrite ${out_dir}" >&2
    exit 2
  fi
  mkdir -p "${out_dir}"
  /usr/bin/time -p nice -n 10 python "${ROOT}/tworoom_success_rate_eval.py" \
    --mode baseline \
    --checkpoint "${CKPT}" \
    --seed "${eval_seed}" \
    --out-dir "${out_dir}" \
    --goal-offset 50 \
    --eval-start-indices "${starts}" \
    > "${out_dir}/run.log" 2>&1
  python - "${out_dir}/results.json" "${SUMMARY}" "${eval_seed}" "${out_dir}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
metrics = result["metrics"]
with open(sys.argv[2], "a") as handle:
    handle.write(
        f"{sys.argv[3]},joint_continue_3ep,{metrics['success_rate']},"
        f"{sum(metrics['episode_successes'])},{result['num_eval']},{sys.argv[4]}\n"
    )
print(f"eval_seed={sys.argv[3]} success_rate={metrics['success_rate']}%")
PY
done

python "${ROOT}/aggregate_success_rate_5seed.py" \
  --summary "${SUMMARY}" \
  --out-json "${SUMMARY_JSON}"

