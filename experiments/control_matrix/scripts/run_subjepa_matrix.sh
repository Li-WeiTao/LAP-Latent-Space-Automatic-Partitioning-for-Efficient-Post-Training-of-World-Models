#!/usr/bin/env bash
# Sub-JEPA entry point. Forwards all parameters to the generic JEPA matrix driver.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! MODEL_FAMILY="${MODEL_FAMILY:-subjepa}"; then
  :
fi

exec env MODEL_FAMILY="$MODEL_FAMILY" \
  bash "$SCRIPT_DIR/run_jepa_matrix.sh" \
  --model-family "$MODEL_FAMILY" \
  "$@"
