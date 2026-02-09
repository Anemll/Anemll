#!/usr/bin/env bash
set -eEuo pipefail

# Build per-context VibeThinker hybrid exports, then combine them into
# state-transition multifunction chunk packages in one output folder.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
CONTEXTS_RAW="512 1024 2048 3072 4096"
CONTEXT_ROOT=""
CONTEXT_ROOT_AUTO=false
CONTEXT_NAME_TEMPLATE="vibethinker_1.5b_ctx{context}_fp16_hybrid"
OUTPUT_DIR="/Volumes/Models/ANE/vibethinker_1.5b_state_transition"
OUTPUT_SUFFIX="_statex"
REBUILD_SCRIPT="${REPO_ROOT}/scripts/rebuild_vibethinker_hybrid.sh"
LUT1="none"
LUT3="none"
CONTEXT_LUT2="none"
POST_LUT2="none"

is_none_lut() {
  local v="${1:-}"
  v="$(echo "${v}" | tr '[:upper:]' '[:lower:]')"
  case "${v}" in
    ""|none|no|false) return 0 ;;
    *) return 1 ;;
  esac
}

NO_BUILD=false
REBUILD_EXISTING=false
SKIP_SMOKE_BUILD=true
COMPILE=true
FORCE_OUTPUT=false
NO_COPY_SHARED=false
VERIFY_INFER=true
VERIFY_ONCE=true
VERIFY_PROMPT="2+2="
VERIFY_MAX_TOKENS=20
VERIFY_NO_THINK=true

REBUILD_EXTRA_ARGS=""
MAX_CONTEXT=""
OUTPUT_BASE=""
INFER_FN="infer"
PREFILL_FN="prefill"
VERIFY_ERROR_REGEX="Error in chat loop|RuntimeError: Error compiling model|Failed to create a working directory|couldn.t be copied to .weights. because there isn.t enough space"

usage() {
  cat <<'EOF'
Usage:
  scripts/build_vibethinker_state_transition.sh [options]

Build flow:
  1) Create/refresh context exports (ctx512/ctx1024/...)
  2) Verify per-context inference smoke (unless disabled/skipped)
  3) Combine into one state-transition folder with infer_ctx* + prefill

Options:
  --contexts "LIST"            Context sizes (default: "512 1024 2048 3072 4096")
                                Also accepts comma list: "512,1024,2048,3072,4096"
  --context-root DIR            Where per-context folders live
                                (default: <output>/_contexts)
  --context-name-template STR   Folder name template with {context}
                                (default: vibethinker_1.5b_ctx{context}_fp16_hybrid)
  --output DIR                  Final combined output folder
                                (default: /Volumes/Models/ANE/vibethinker_1.5b_state_transition)
  --output-suffix STR           Suffix for FFN base in combined chunks (default: _statex)
  --output-base STR             Override output FFN base
  --max-context N               Context used for prefill source (default: max listed)
  --lut1 V                      Embeddings LUT for initial static export (default: none)
  --lut3 V                      LM head LUT for initial static export (default: none)
  --lut2 V                      Alias for --post-lut2 (quantize once per combined chunk)
  --context-lut2 V              LUT used during per-context rebuild (default: none)
                                Keep this as 'none' to avoid repeated quantization.
  --post-lut2 V                 LUT applied once after per-chunk multi-context combine
                                Format: bits or bits,per_channel (default: none)

  --no-build                    Skip rebuild stage; only combine/export
  --rebuild-existing            Rebuild even if context folder already has meta.yaml
  --no-skip-smoke-build         Do not pass --skip-smoke to rebuild script
  --rebuild-script PATH         Override rebuild script path
  --rebuild-extra-args "ARGS"  Extra args appended to each rebuild call
                                Example: "--model-path /path --lut2 6,4"

  --no-compile                  Do not compile combined output chunks
  --force-output                Allow overwrite/reuse of output folder
                                Auto-enabled when using default <output>/_contexts layout
  --no-copy-shared              Do not copy tokenizer/embedding/lm_head/meta assets
  --no-verify-infer             Skip per-context inference smoke check before combine
  --verify-always               Run inference smoke even if prior PASS marker exists
  --verify-prompt TEXT          Prompt for per-context inference smoke (default: "2+2=")
  --verify-max-tokens N         Max tokens for per-context inference smoke (default: 20)
  --verify-think                Do not pass --no-think during verify smoke

  --infer-fn NAME               Source infer function name (default: infer)
  --prefill-fn NAME             Source prefill function name (default: prefill)

  -h, --help                    Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --contexts) CONTEXTS_RAW="$2"; shift 2 ;;
    --context-root) CONTEXT_ROOT="$2"; shift 2 ;;
    --context-name-template) CONTEXT_NAME_TEMPLATE="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --output-suffix) OUTPUT_SUFFIX="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --max-context) MAX_CONTEXT="$2"; shift 2 ;;
    --lut1) LUT1="$2"; shift 2 ;;
    --lut3) LUT3="$2"; shift 2 ;;
    --lut2) POST_LUT2="$2"; shift 2 ;;
    --context-lut2) CONTEXT_LUT2="$2"; shift 2 ;;
    --post-lut2) POST_LUT2="$2"; shift 2 ;;

    --no-build) NO_BUILD=true; shift 1 ;;
    --rebuild-existing) REBUILD_EXISTING=true; shift 1 ;;
    --no-skip-smoke-build) SKIP_SMOKE_BUILD=false; shift 1 ;;
    --rebuild-script) REBUILD_SCRIPT="$2"; shift 2 ;;
    --rebuild-extra-args) REBUILD_EXTRA_ARGS="$2"; shift 2 ;;

    --no-compile) COMPILE=false; shift 1 ;;
    --force-output) FORCE_OUTPUT=true; shift 1 ;;
    --no-copy-shared) NO_COPY_SHARED=true; shift 1 ;;
    --no-verify-infer) VERIFY_INFER=false; shift 1 ;;
    --verify-always) VERIFY_ONCE=false; shift 1 ;;
    --verify-prompt) VERIFY_PROMPT="$2"; shift 2 ;;
    --verify-max-tokens) VERIFY_MAX_TOKENS="$2"; shift 2 ;;
    --verify-think) VERIFY_NO_THINK=false; shift 1 ;;

    --infer-fn) INFER_FN="$2"; shift 2 ;;
    --prefill-fn) PREFILL_FN="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -x "${REBUILD_SCRIPT}" ]]; then
  echo "Rebuild script is missing or not executable: ${REBUILD_SCRIPT}" >&2
  exit 1
fi
if [[ ! "${VERIFY_MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --verify-max-tokens: ${VERIFY_MAX_TOKENS}" >&2
  exit 1
fi

# Guard against repeated FFN quantization when combining multiple contexts.
if ! is_none_lut "${POST_LUT2}" && ! is_none_lut "${CONTEXT_LUT2}"; then
  echo "Invalid LUT setup: both context-lut2 (${CONTEXT_LUT2}) and post-lut2 (${POST_LUT2}) are set." >&2
  echo "Use context-lut2=none and post-lut2=<bits[,per_channel]> to quantize once per combined chunk." >&2
  exit 1
fi

if [[ -f "${REPO_ROOT}/env-anemll/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/env-anemll/bin/activate"
fi

# Default context root under output if not explicitly set.
if [[ -z "${CONTEXT_ROOT}" ]]; then
  CONTEXT_ROOT="${OUTPUT_DIR}/_contexts"
  CONTEXT_ROOT_AUTO=true
fi

# Parse contexts from comma/space separated list
CONTEXTS=()
# shellcheck disable=SC2206
CONTEXTS=( $(echo "${CONTEXTS_RAW}" | tr ',' ' ') )
if [[ ${#CONTEXTS[@]} -eq 0 ]]; then
  echo "No contexts specified" >&2
  exit 1
fi
for c in "${CONTEXTS[@]}"; do
  if [[ ! "${c}" =~ ^[0-9]+$ ]]; then
    echo "Invalid context value: ${c}" >&2
    exit 1
  fi
done

mkdir -p "${CONTEXT_ROOT}"

context_entries=()
static_source_dir=""

if [[ "${VERIFY_INFER}" == "true" ]]; then
  if [[ "${VERIFY_ONCE}" == "true" ]]; then
    echo "[verify] mode=once (skips contexts with .infer_smoke_ok marker)"
  else
    echo "[verify] mode=always"
  fi
  echo "[verify] prompt=\"${VERIFY_PROMPT}\" max_tokens=${VERIFY_MAX_TOKENS} no_think=${VERIFY_NO_THINK}"
else
  echo "[verify] disabled (--no-verify-infer)"
fi

run_context_infer_smoke() {
  local ctx="$1"
  local ctx_dir="$2"
  local marker_file="${ctx_dir}/.infer_smoke_ok"
  local smoke_log="${ctx_dir}/infer_smoke_ctx${ctx}.log"
  local cmd=(
    python3 "${REPO_ROOT}/tests/chat.py"
    --meta "${ctx_dir}/meta.yaml"
    --prompt "${VERIFY_PROMPT}"
    --max-tokens "${VERIFY_MAX_TOKENS}"
  )

  if [[ "${VERIFY_NO_THINK}" == "true" ]]; then
    cmd+=(--no-think)
  fi

  if [[ "${VERIFY_ONCE}" == "true" && -f "${marker_file}" ]]; then
    echo "[verify] Context ${ctx}: already verified, skipping (${marker_file})"
    return 0
  fi

  echo "[verify] Context ${ctx}: inference smoke"
  echo "         ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "${smoke_log}"

  if grep -Eq "${VERIFY_ERROR_REGEX}" "${smoke_log}"; then
    echo "[verify] Context ${ctx}: FAILED (runtime errors). See ${smoke_log}" >&2
    return 1
  fi

  {
    echo "verified_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "context=${ctx}"
    echo "prompt=${VERIFY_PROMPT}"
    echo "max_tokens=${VERIFY_MAX_TOKENS}"
    if [[ "${VERIFY_NO_THINK}" == "true" ]]; then
      echo "no_think=true"
    else
      echo "no_think=false"
    fi
  } > "${marker_file}"

  echo "[verify] Context ${ctx}: PASS"
}

for ctx in "${CONTEXTS[@]}"; do
  ctx_name="${CONTEXT_NAME_TEMPLATE//\{context\}/${ctx}}"
  ctx_dir="${CONTEXT_ROOT}/${ctx_name}"

  if [[ "${NO_BUILD}" == "false" ]]; then
    if [[ -f "${ctx_dir}/meta.yaml" && "${REBUILD_EXISTING}" == "false" ]]; then
      echo "[build] Reusing existing context ${ctx}: ${ctx_dir}"
      if [[ -z "${static_source_dir}" ]]; then
        static_source_dir="${ctx_dir}"
      fi
    else
      cmd=(
        bash "${REBUILD_SCRIPT}"
        --context "${ctx}"
        --output "${ctx_dir}"
        --lut1 "${LUT1}"
        --lut2 "${CONTEXT_LUT2}"
        --lut3 "${LUT3}"
      )
      if [[ -n "${static_source_dir}" ]]; then
        cmd+=(--reuse-static-from "${static_source_dir}")
      fi
      if [[ "${SKIP_SMOKE_BUILD}" == "true" ]]; then
        cmd+=(--skip-smoke)
      fi
      if [[ "${REBUILD_EXISTING}" == "true" ]]; then
        cmd+=(--force-clean)
      fi
      if [[ -n "${REBUILD_EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        extra=( ${REBUILD_EXTRA_ARGS} )
        cmd+=("${extra[@]}")
      fi

      echo "[build] Context ${ctx}: ${ctx_dir}"
      echo "        ${cmd[*]}"
      "${cmd[@]}"

      if [[ -z "${static_source_dir}" ]]; then
        static_source_dir="${ctx_dir}"
      fi
    fi
  fi

  if [[ ! -f "${ctx_dir}/meta.yaml" ]]; then
    echo "Missing context export for ${ctx}: ${ctx_dir}/meta.yaml not found" >&2
    echo "Either run without --no-build or ensure that folder exists." >&2
    exit 1
  fi

  if [[ "${VERIFY_INFER}" == "true" ]]; then
    run_context_infer_smoke "${ctx}" "${ctx_dir}"
  fi

  context_entries+=("${ctx}=${ctx_dir}")
done

export_cmd=(
  python3 "${REPO_ROOT}/tests/dev/export_state_transition_chunks.py"
  --contexts
  "${context_entries[@]}"
  --output-dir "${OUTPUT_DIR}"
  --output-suffix "${OUTPUT_SUFFIX}"
  --infer-fn "${INFER_FN}"
  --prefill-fn "${PREFILL_FN}"
)

if [[ -n "${MAX_CONTEXT}" ]]; then
  export_cmd+=(--max-context "${MAX_CONTEXT}")
fi
if [[ -n "${OUTPUT_BASE}" ]]; then
  export_cmd+=(--output-base "${OUTPUT_BASE}")
fi
if ! is_none_lut "${POST_LUT2}"; then
  export_cmd+=(--lut2 "${POST_LUT2}")
fi
if [[ "${COMPILE}" == "true" ]]; then
  export_cmd+=(--compile)
fi
if [[ "${FORCE_OUTPUT}" == "false" && "${CONTEXT_ROOT_AUTO}" == "true" ]]; then
  # Default layout creates <output> before combine; force is required for exporter.
  FORCE_OUTPUT=true
fi
if [[ "${FORCE_OUTPUT}" == "true" ]]; then
  export_cmd+=(--force)
fi
if [[ "${NO_COPY_SHARED}" == "true" ]]; then
  export_cmd+=(--no-copy-shared)
fi

echo "[combine] Building state-transition chunks in one folder: ${OUTPUT_DIR}"
echo "          ${export_cmd[*]}"
"${export_cmd[@]}"

echo
echo "Done."
echo "Context folders root: ${CONTEXT_ROOT}"
echo "Combined output dir:  ${OUTPUT_DIR}"
