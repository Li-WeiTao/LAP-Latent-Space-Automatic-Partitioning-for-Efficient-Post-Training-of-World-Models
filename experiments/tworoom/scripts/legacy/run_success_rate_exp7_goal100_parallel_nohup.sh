#!/usr/bin/env bash
# Launch 4-model parallel eval: goal_offset=100, eval_budget=50, 5 seeds each.
# GPU 0=baseline, 1=rooms3_50ep, 2=global_ft_50ep, 3=latent_kmeanspp (3 outer seq).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${SCRIPT_DIR}/../ablations/run_success_rate_exp7_goal100_model.sh"
LOG="experiments/tworoom/results/success_rate_exp7_goal100_parallel.log"
PIDFILE="experiments/tworoom/results/success_rate_exp7_goal100_parallel.pid"

chmod +x "${RUN}"

if [[ -f "${PIDFILE}" ]]; then
  old="$(cat "${PIDFILE}")"
  if kill -0 "${old}" 2>/dev/null; then
    echo "Already running pid=${old}. Log: ${LOG}"
    exit 0
  fi
fi

GPUS=(0 1 2 3)
MODELS=(baseline rooms3_50ep global_ft_50ep latent_kmeanspp)

echo "==== exp7 goal_offset=100 eval_budget=50 parallel dispatch started at $(date) ====" >> "${LOG}"

pids=()
for i in "${!MODELS[@]}"; do
  gpu="${GPUS[$i]}"
  model="${MODELS[$i]}"
  setsid nohup env GPU="${gpu}" MODEL="${model}" bash "${RUN}" >> "${LOG}" 2>&1 < /dev/null &
  pids+=($!)
  echo "launched MODEL=${model} GPU=${gpu} pid=${pids[-1]}" >> "${LOG}"
  disown "${pids[-1]}" 2>/dev/null || true
done

echo "${pids[*]}" > "${PIDFILE}"
echo "Detached pids: ${pids[*]}"
echo "  GPU0 baseline  GPU1 rooms3_50ep  GPU2 global_ft_50ep  GPU3 latent_kmeanspp"
echo "  Protocol: goal_offset=100, eval_budget=50"
echo "Log: ${LOG}"
