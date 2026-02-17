#!/usr/bin/env bash
set -eEuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[info] scripts/combine_vibethinker_all_context_functions.sh is now a compatibility wrapper."
echo "[info] Forwarding to scripts/combine_state_transition_contexts.sh"

exec "${SCRIPT_DIR}/combine_state_transition_contexts.sh" "$@"

