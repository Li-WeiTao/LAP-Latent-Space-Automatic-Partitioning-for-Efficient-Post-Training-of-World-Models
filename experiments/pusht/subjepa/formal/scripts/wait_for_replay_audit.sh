#!/usr/bin/env bash
# Wait until formal replay audit summary exists and all_passed=true.
# Do NOT use pgrep on audit command strings — the waiter shell can match itself.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/experiments/pusht/subjepa/env.sh"

SUMMARY="$FORMAL/manifests/replay_audit_summary.json"
INTERVAL="${WAIT_INTERVAL_SEC:-10}"

while true; do
  if [[ -f "$SUMMARY" ]] && "$PYTHON" -c "import json,sys; r=json.load(open('$SUMMARY')); sys.exit(0 if r.get('all_passed') else 1)"; then
    echo "[gate-wait] replay audit passed: $SUMMARY"
    exit 0
  fi
  echo "[gate-wait] waiting for replay audit ($SUMMARY)..."
  sleep "$INTERVAL"
done
