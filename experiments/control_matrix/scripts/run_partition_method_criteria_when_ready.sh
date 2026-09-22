#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 PARTITION_METHOD [NUM_CLUSTERS] [WAIT_HOURS]" >&2
  exit 2
fi

METHOD=$1
NUM_CLUSTERS=${2:-4}
WAIT_HOURS=${3:-96}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ASSETS="$REPO/experiments/control_matrix/assets/lewm_k${NUM_CLUSTERS}_${METHOD}"
OUT="$ASSETS/criteria22"
STATUS="$OUT/status.txt"
LOG="$OUT/run.log"
TASKS=(tworoom pusht reacher cube)
SEEDS=(0 1 2)
mkdir -p "$OUT"

ready() {
  [[ -s "$ASSETS/long_comparison.csv" ]] || return 1
  local task seed root
  for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      root="$REPO/experiments/$task/matrix_k${NUM_CLUSTERS}/partitions/$METHOD/seed$seed"
      [[ -s "$root/manifest.json" && -s "$root/cluster_labels.npz" ]] || return 1
    done
  done
}

deadline=$(( $(date +%s) + WAIT_HOURS * 3600 ))
printf 'WAITING %s method=%s K=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$METHOD" "$NUM_CLUSTERS" > "$STATUS"
while ! ready; do
  if (( $(date +%s) >= deadline )); then
    printf 'FAILED wait_timeout %s method=%s K=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$METHOD" "$NUM_CLUSTERS" > "$STATUS"
    exit 124
  fi
  sleep 30
done

printf 'READY %s method=%s K=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$METHOD" "$NUM_CLUSTERS" > "$STATUS"
exec 9>"$REPO/experiments/control_matrix/assets/lewm_partition_criteria22.lock"
if ! flock -w 86400 9; then
  printf 'FAILED analysis_lock_timeout %s method=%s K=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$METHOD" "$NUM_CLUSTERS" > "$STATUS"
  exit 125
fi

printf 'RUNNING %s method=%s K=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$METHOD" "$NUM_CLUSTERS" > "$STATUS"
set +e
timeout --signal=TERM --kill-after=60s 8h \
  env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "$REPO/.venv/bin/python" -u \
  "$REPO/experiments/control_matrix/analyze_partition_method_criteria.py" \
  --repo "$REPO" \
  --partition-method "$METHOD" \
  --num-clusters "$NUM_CLUSTERS" \
  --cpu-threads 4 \
  --output-dir "$OUT" > "$LOG" 2>&1
rc=$?
set -e
if (( rc == 0 )); then
  printf 'COMPLETE %s method=%s K=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$METHOD" "$NUM_CLUSTERS" > "$STATUS"
else
  printf 'FAILED rc=%s %s method=%s K=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$METHOD" "$NUM_CLUSTERS" > "$STATUS"
fi
exit "$rc"
