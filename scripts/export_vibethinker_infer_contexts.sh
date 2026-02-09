#!/usr/bin/env bash
set -eEuo pipefail

# Export context folders for state-transition work without combine/compile.
#
# Behavior:
# 1) Max context (default: largest in list) -> run steps 1..4:
#    embeddings + lm_head + FFN(infer) + prefill
# 2) Remaining contexts -> run step 3 only:
#    FFN(infer) chunks only
#
# No combine (step 5), no compile (step 6), no tokenizer/meta generation (step 7),
# and no chat test (step 8).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONVERT_SCRIPT="${REPO_ROOT}/anemll/utils/convert_model.sh"

MODEL_ID="WeiboAI/VibeThinker-1.5B"
CONTEXTS_RAW="512 1024 2048 3072 4096"
MAX_CONTEXT=""
CONTEXT_ROOT="/Volumes/Models/ANE/vibethinker_1.5b_state_transition/_contexts"
CONTEXT_NAME_TEMPLATE="vibethinker_1.5b_ctx{context}_fp16_hybrid"

PREFIX="qwen25"
BATCH=32
CHUNKS=3
LUT1="none"
LUT2="none"
LUT3="none"

SKIP_CHECK=true
REBUILD_EXISTING=false
FORCE_CLEAN=false
EXTRA_ARGS=""

usage() {
  cat <<'EOF'
Usage:
  scripts/export_vibethinker_infer_contexts.sh [options]

Flow:
  - Max context first: steps 1,2,3,4 (no step 5/6/7/8)
  - Remaining contexts: step 3 only (FFN infer chunks only)

Options:
  --model ID                    Model id/path for convert_model.sh
                                (default: WeiboAI/VibeThinker-1.5B)
  --contexts "LIST"             Context list, comma or space separated
                                (default: "512 1024 2048 3072 4096")
  --max-context N               Context to build as full export first
                                (default: max value from --contexts)
  --context-root DIR            Root folder for per-context outputs
                                (default: /Volumes/Models/ANE/vibethinker_1.5b_state_transition/_contexts)
  --context-name-template STR   Folder template containing {context}
                                (default: vibethinker_1.5b_ctx{context}_fp16_hybrid)

  --prefix STR                  Model prefix passed to converter (default: qwen25)
  --batch N                     Batch size (default: 32)
  --chunks N                    FFN chunk count (default: 3)
  --lut1 V                      LUT for embeddings (default: none)
  --lut2 V                      LUT for FFN/prefill (default: none)
  --lut3 V                      LUT for lm_head (default: none)

  --rebuild-existing            Rebuild even if expected artifacts already exist
  --force-clean                 Delete output dir before rebuilding each context
  --no-skip-check               Do not pass --skip-check to convert_model.sh
  --extra-args "ARGS"          Extra args appended to each convert_model.sh call

  -h, --help                    Show help
EOF
}

convert_step_part_name() {
  case "${1}" in
    1) echo "embeddings" ;;
    2) echo "lm_head" ;;
    3) echo "ffn_infer" ;;
    4) echo "prefill" ;;
    5) echo "combine" ;;
    6) echo "compile" ;;
    7) echo "meta_tokenizer" ;;
    8) echo "smoke_test" ;;
    *) echo "unknown" ;;
  esac
}

lut_bits_or_empty() {
  local raw="${1:-}"
  raw="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
  case "${raw}" in
    ""|none|no|false) echo "" ;;
    *,*) echo "${raw%%,*}" ;;
    *) echo "${raw}" ;;
  esac
}

ctx_dir_for() {
  local ctx="$1"
  local name="${CONTEXT_NAME_TEMPLATE//\{context\}/${ctx}}"
  echo "${CONTEXT_ROOT}/${name}"
}

run_convert_step() {
  local step="$1"
  local ctx="$2"
  local out_dir="$3"
  local part_name
  part_name="$(convert_step_part_name "${step}")"

  local cmd=(
    bash "${CONVERT_SCRIPT}"
    --model "${MODEL_ID}"
    --output "${out_dir}"
    --context "${ctx}"
    --batch "${BATCH}"
    --chunk "${CHUNKS}"
    --prefix "${PREFIX}"
    --lut1 "${LUT1}"
    --lut2 "${LUT2}"
    --lut3 "${LUT3}"
    --restart "${step}"
    --only "${step}"
  )
  if [[ "${SKIP_CHECK}" == "true" ]]; then
    cmd+=(--skip-check)
  fi
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${EXTRA_ARGS} )
    cmd+=("${extra[@]}")
  fi

  echo "  step ${step} (${part_name}): ${cmd[*]}"
  echo "    convert_model.sh --only ${step}  # part ${step}: ${part_name}"
  "${cmd[@]}"
}

have_max_context_artifacts() {
  local out_dir="$1"
  local lut1_bits lut2_bits lut3_bits emb_stem ffn_stem pf_stem lmh_stem
  lut1_bits="$(lut_bits_or_empty "${LUT1}")"
  lut2_bits="$(lut_bits_or_empty "${LUT2}")"
  lut3_bits="$(lut_bits_or_empty "${LUT3}")"

  emb_stem="${PREFIX}_embeddings${lut1_bits:+_lut${lut1_bits}}"
  lmh_stem="${PREFIX}_lm_head${lut3_bits:+_lut${lut3_bits}}"
  ffn_stem="${PREFIX}_FFN${lut2_bits:+_lut${lut2_bits}}"
  pf_stem="${PREFIX}_prefill${lut2_bits:+_lut${lut2_bits}}"

  [[ -d "${out_dir}/${emb_stem}.mlpackage" ]] || return 1
  [[ -d "${out_dir}/${lmh_stem}.mlpackage" ]] || return 1
  local i
  for ((i=1; i<=CHUNKS; i++)); do
    local suf
    suf="$(printf "_chunk_%02dof%02d.mlpackage" "${i}" "${CHUNKS}")"
    [[ -d "${out_dir}/${ffn_stem}${suf}" ]] || return 1
    [[ -d "${out_dir}/${pf_stem}${suf}" ]] || return 1
  done
  return 0
}

have_infer_only_artifacts() {
  local out_dir="$1"
  local lut2_bits ffn_stem
  lut2_bits="$(lut_bits_or_empty "${LUT2}")"
  ffn_stem="${PREFIX}_FFN${lut2_bits:+_lut${lut2_bits}}"

  local i
  for ((i=1; i<=CHUNKS; i++)); do
    local suf
    suf="$(printf "_chunk_%02dof%02d.mlpackage" "${i}" "${CHUNKS}")"
    [[ -d "${out_dir}/${ffn_stem}${suf}" ]] || return 1
  done
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ID="$2"; shift 2 ;;
    --contexts) CONTEXTS_RAW="$2"; shift 2 ;;
    --max-context) MAX_CONTEXT="$2"; shift 2 ;;
    --context-root) CONTEXT_ROOT="$2"; shift 2 ;;
    --context-name-template) CONTEXT_NAME_TEMPLATE="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --lut1) LUT1="$2"; shift 2 ;;
    --lut2) LUT2="$2"; shift 2 ;;
    --lut3) LUT3="$2"; shift 2 ;;
    --rebuild-existing) REBUILD_EXISTING=true; shift 1 ;;
    --force-clean) FORCE_CLEAN=true; shift 1 ;;
    --no-skip-check) SKIP_CHECK=false; shift 1 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -x "${CONVERT_SCRIPT}" ]]; then
  echo "convert_model.sh not found or not executable: ${CONVERT_SCRIPT}" >&2
  exit 1
fi
if [[ ! "${BATCH}" =~ ^[0-9]+$ ]] || [[ ! "${CHUNKS}" =~ ^[0-9]+$ ]]; then
  echo "--batch and --chunks must be integers" >&2
  exit 1
fi

# shellcheck disable=SC2206
CONTEXTS=( $(echo "${CONTEXTS_RAW}" | tr ',' ' ') )
if [[ ${#CONTEXTS[@]} -eq 0 ]]; then
  echo "No contexts provided" >&2
  exit 1
fi
for c in "${CONTEXTS[@]}"; do
  if [[ ! "${c}" =~ ^[0-9]+$ ]]; then
    echo "Invalid context value: ${c}" >&2
    exit 1
  fi
done

if [[ -z "${MAX_CONTEXT}" ]]; then
  MAX_CONTEXT="${CONTEXTS[0]}"
  for c in "${CONTEXTS[@]}"; do
    if (( c > MAX_CONTEXT )); then
      MAX_CONTEXT="${c}"
    fi
  done
fi
if [[ ! "${MAX_CONTEXT}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --max-context: ${MAX_CONTEXT}" >&2
  exit 1
fi

has_max=false
for c in "${CONTEXTS[@]}"; do
  if [[ "${c}" == "${MAX_CONTEXT}" ]]; then
    has_max=true
    break
  fi
done
if [[ "${has_max}" != "true" ]]; then
  echo "--max-context (${MAX_CONTEXT}) must be included in --contexts" >&2
  exit 1
fi

mkdir -p "${CONTEXT_ROOT}"

echo "Export plan:"
echo "  Model: ${MODEL_ID}"
echo "  Context root: ${CONTEXT_ROOT}"
echo "  Contexts: ${CONTEXTS[*]}"
echo "  Max context (full export first): ${MAX_CONTEXT}"
echo "  Prefix: ${PREFIX}, Batch: ${BATCH}, Chunks: ${CHUNKS}"
echo "  LUT: ${LUT1}/${LUT2}/${LUT3}"
echo "  No combine/compile/test steps are executed."

# 1) Build max context first: steps 1..4.
max_dir="$(ctx_dir_for "${MAX_CONTEXT}")"
if [[ -d "${max_dir}" && "${FORCE_CLEAN}" == "true" ]]; then
  echo "[max:${MAX_CONTEXT}] Cleaning ${max_dir}"
  rm -rf "${max_dir}"
fi
mkdir -p "${max_dir}"

if [[ "${REBUILD_EXISTING}" == "false" ]] && have_max_context_artifacts "${max_dir}"; then
  echo "[max:${MAX_CONTEXT}] Reusing existing full export: ${max_dir}"
else
  echo "[max:${MAX_CONTEXT}] Export full artifacts (steps 1..4 only)"
  run_convert_step 1 "${MAX_CONTEXT}" "${max_dir}"
  run_convert_step 2 "${MAX_CONTEXT}" "${max_dir}"
  run_convert_step 3 "${MAX_CONTEXT}" "${max_dir}"
  run_convert_step 4 "${MAX_CONTEXT}" "${max_dir}"
fi

if ! have_max_context_artifacts "${max_dir}"; then
  echo "[max:${MAX_CONTEXT}] Missing required artifacts after export: ${max_dir}" >&2
  exit 1
fi

# 2) Remaining contexts: step 3 only (infer chunks).
for ctx in "${CONTEXTS[@]}"; do
  if [[ "${ctx}" == "${MAX_CONTEXT}" ]]; then
    continue
  fi

  ctx_dir="$(ctx_dir_for "${ctx}")"
  if [[ -d "${ctx_dir}" && "${FORCE_CLEAN}" == "true" ]]; then
    echo "[ctx:${ctx}] Cleaning ${ctx_dir}"
    rm -rf "${ctx_dir}"
  fi
  mkdir -p "${ctx_dir}"

  if [[ "${REBUILD_EXISTING}" == "false" ]] && have_infer_only_artifacts "${ctx_dir}"; then
    echo "[ctx:${ctx}] Reusing existing infer-only chunks: ${ctx_dir}"
    continue
  fi

  echo "[ctx:${ctx}] Export infer-only chunks (step 3 only)"
  run_convert_step 3 "${ctx}" "${ctx_dir}"

  if ! have_infer_only_artifacts "${ctx_dir}"; then
    echo "[ctx:${ctx}] Missing infer-only chunk artifacts after step 3: ${ctx_dir}" >&2
    exit 1
  fi
done

echo
echo "Done."
echo "Max context full export dir: ${max_dir}"
echo "Other context dirs contain FFN infer-only chunks."
echo "No combine/compile artifacts were created."
