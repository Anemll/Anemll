#!/usr/bin/env bash
set -eEuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/env-anemll/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/env-anemll/bin/activate"
fi

exec python3 "${REPO_ROOT}/anemll/utils/combine_state_transition_contexts.py" "$@"

