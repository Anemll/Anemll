#!/usr/bin/env bash
set -eEuo pipefail

# Wrapper for multi-context multifunction combine:
# - infer_ctx{N} for all contexts
# - prefill_ctx{N} for all contexts
# - infer/prefill aliases from max context
#
# Extra options are forwarded to base script, e.g.:
# - --split-infer-prefill
# - --no-alias-functions

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/combine_vibethinker_infer_contexts.sh"

if [[ ! -x "${BASE_SCRIPT}" ]]; then
  echo "Missing executable script: ${BASE_SCRIPT}" >&2
  exit 1
fi

exec "${BASE_SCRIPT}" --prefill-all-contexts --infer-kind FFN_PF --copy-source-chunks "$@"
