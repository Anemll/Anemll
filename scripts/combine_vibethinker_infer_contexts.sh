#!/usr/bin/env bash
set -eEuo pipefail

# Combine infer-only context exports into multi-function FFN_PF state-transition chunks.
#
# Inputs are expected from scripts/export_vibethinker_infer_contexts.sh:
# - max context has FFN + prefill + embeddings + lm_head
# - other contexts have FFN infer chunks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTEXTS_RAW="512 1024 2048 3072 4096"
MAX_CONTEXT=4096
CONTEXT_ROOT="/Volumes/Models/ANE/vibethinker_1.5b_state_transition/_contexts"
CONTEXT_NAME_TEMPLATE="vibethinker_1.5b_ctx{context}_fp16_hybrid"

OUTPUT_DIR="/Volumes/Models/ANE/vibethinker_1.5b_state_transition"
PREFIX="qwen25"
NUM_CHUNKS=3
OUTPUT_BASE=""
INFER_FN="infer"
PREFILL_FN="prefill"
INFER_KIND="FFN_PF"
INFER_CHUNK1_KIND="auto"
PREFILL_KIND="auto"
PREFILL_ALL_CONTEXTS=false
SPLIT_INFER_PREFILL=false
NO_ALIAS_FUNCTIONS=false
OUTPUT_BASE_INFER=""
OUTPUT_BASE_PREFILL=""
BATCH_SIZE=""
SPLIT_LM_HEAD=""
ARCHITECTURE="qwen2"
MODEL_NAME="anemll-vibethinker-1.5b-state-transition"
TEMP_DIR="/Volumes/Models/ANE/tmp_coreml_compile"
TOKENIZER_PATH=""
HF_MODEL_ID="WeiboAI/VibeThinker-1.5B"

FORCE=false
NO_COPY_SHARED=false
NO_META=false
CLEAN_OUTPUT=false
COMPILE=true
COPY_SOURCE_CHUNKS=false
COPY_SOURCE_CHUNKS_CONTEXT=""
DEDUP_DIAGNOSTICS=false
SKIP_ANEMLL_DEDUP=false

usage() {
  cat <<'EOF'
Usage:
  scripts/combine_vibethinker_infer_contexts.sh [options]

Options:
  --contexts "LIST"             Context list (default: "512 1024 2048 3072 4096")
  --max-context N               Prefill source context (default: 4096)
  --context-root DIR            Context export root
                                (default: /Volumes/Models/ANE/vibethinker_1.5b_state_transition/_contexts)
  --context-name-template STR   Folder template with {context}
                                (default: vibethinker_1.5b_ctx{context}_fp16_hybrid)

  --output DIR                  Output dir for combined chunks
                                (default: /Volumes/Models/ANE/vibethinker_1.5b_state_transition)
  --prefix STR                  Model prefix (default: qwen25)
  --num-chunks N                Number of chunks (default: 3)
  --output-base STR             Output chunk base (default: <prefix>_FFN_PF_statex)
  --infer-fn STR                Source infer function name (default: infer)
  --prefill-fn STR              Source prefill function name (default: prefill)
  --infer-kind STR              Input infer chunk kind (default: FFN_PF)
  --infer-chunk1-kind STR       Input infer chunk kind for chunk_01 only
                                (default: auto; follows infer-kind resolution.
                                Use FFN_attn_fp32 explicitly to force FP32 chunk_01)
  --prefill-kind STR            Input prefill chunk kind (default: auto; FFN_PF preferred, else prefill)
  --prefill-all-contexts        Add prefill_ctx{N} functions for every context
                                (keeps alias 'prefill' from --max-context)
  --split-infer-prefill         Emit separate infer/prefill multifunction packages per chunk
  --no-alias-functions          Do not emit compatibility aliases (infer/prefill)
  --output-base-infer STR       Infer output base (split mode only)
  --output-base-prefill STR     Prefill output base (split mode only)
  --batch-size N                Batch size written to meta.yaml (default: from max-context meta.yaml)
  --split-lm-head N             split_lm_head in meta.yaml (default: from max-context meta.yaml)
  --architecture STR            architecture field for meta.yaml (default: qwen2)
  --model-name STR              model_info.name for meta.yaml
  --temp-dir DIR                Temp scratch for CoreMLTools/work dirs
                                (default: /Volumes/Models/ANE/tmp_coreml_compile)
  --tokenizer-path DIR          Optional tokenizer directory fallback.
                                If omitted and max-context exports do not contain tokenizer files,
                                auto-detect from HF cache for --hf-model-id.
  --hf-model-id ID              HF model id for tokenizer auto-detect
                                (default: WeiboAI/VibeThinker-1.5B)

  --force                       Overwrite existing output chunk packages
  --clean-output                Remove prior combined chunk artifacts for output-base
                                before combine (does not touch context-root folders)
  --no-compile                  Do not compile output chunks/shared assets to .mlmodelc
  --no-copy-shared              Do not copy tokenizer/embeddings/lm_head
  --dedup-diagnostics           Enable CoreMLTools dedup diagnostics while saving
                                multifunction packages
  --copy-source-chunks          Copy source chunk artifacts into output folder
                                (mirrors per-context layout; includes chunk1 FP32
                                and regular infer chunk1 when they differ)
  --copy-source-chunks-context N
                                Context to copy source chunks from (default: --max-context)
  --skip-anemll-dedup           Disable anemll-dedup surgical weight dedup
                                (default: enabled)
  --no-meta                     Do not write meta.yaml
  -h, --help                    Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --contexts) CONTEXTS_RAW="$2"; shift 2 ;;
    --max-context) MAX_CONTEXT="$2"; shift 2 ;;
    --context-root) CONTEXT_ROOT="$2"; shift 2 ;;
    --context-name-template) CONTEXT_NAME_TEMPLATE="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --num-chunks) NUM_CHUNKS="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --infer-fn) INFER_FN="$2"; shift 2 ;;
    --prefill-fn) PREFILL_FN="$2"; shift 2 ;;
    --infer-kind) INFER_KIND="$2"; shift 2 ;;
    --infer-chunk1-kind) INFER_CHUNK1_KIND="$2"; shift 2 ;;
    --prefill-kind) PREFILL_KIND="$2"; shift 2 ;;
    --prefill-all-contexts) PREFILL_ALL_CONTEXTS=true; shift 1 ;;
    --split-infer-prefill) SPLIT_INFER_PREFILL=true; shift 1 ;;
    --no-alias-functions) NO_ALIAS_FUNCTIONS=true; shift 1 ;;
    --output-base-infer) OUTPUT_BASE_INFER="$2"; shift 2 ;;
    --output-base-prefill) OUTPUT_BASE_PREFILL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --split-lm-head) SPLIT_LM_HEAD="$2"; shift 2 ;;
    --architecture) ARCHITECTURE="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --temp-dir) TEMP_DIR="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --hf-model-id) HF_MODEL_ID="$2"; shift 2 ;;
    --force) FORCE=true; shift 1 ;;
    --clean-output) CLEAN_OUTPUT=true; shift 1 ;;
    --no-compile) COMPILE=false; shift 1 ;;
    --no-copy-shared) NO_COPY_SHARED=true; shift 1 ;;
    --dedup-diagnostics) DEDUP_DIAGNOSTICS=true; shift 1 ;;
    --skip-anemll-dedup) SKIP_ANEMLL_DEDUP=true; shift 1 ;;
    --copy-source-chunks) COPY_SOURCE_CHUNKS=true; shift 1 ;;
    --copy-source-chunks-context) COPY_SOURCE_CHUNKS_CONTEXT="$2"; shift 2 ;;
    --no-meta) NO_META=true; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! "${MAX_CONTEXT}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --max-context: ${MAX_CONTEXT}" >&2
  exit 1
fi
if [[ ! "${NUM_CHUNKS}" =~ ^[0-9]+$ ]]; then
  echo "--num-chunks must be an integer" >&2
  exit 1
fi
if [[ -n "${BATCH_SIZE}" ]] && [[ ! "${BATCH_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "--batch-size must be an integer" >&2
  exit 1
fi
if [[ -n "${SPLIT_LM_HEAD}" ]] && [[ ! "${SPLIT_LM_HEAD}" =~ ^[0-9]+$ ]]; then
  echo "--split-lm-head must be an integer" >&2
  exit 1
fi
if [[ -n "${COPY_SOURCE_CHUNKS_CONTEXT}" ]] && [[ ! "${COPY_SOURCE_CHUNKS_CONTEXT}" =~ ^[0-9]+$ ]]; then
  echo "--copy-source-chunks-context must be an integer" >&2
  exit 1
fi

if [[ -f "${REPO_ROOT}/env-anemll/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/env-anemll/bin/activate"
fi

# shellcheck disable=SC2206
CONTEXTS=( $(echo "${CONTEXTS_RAW}" | tr ',' ' ') )
if [[ ${#CONTEXTS[@]} -eq 0 ]]; then
  echo "No contexts provided" >&2
  exit 1
fi

context_entries=()
max_ctx_dir=""
for ctx in "${CONTEXTS[@]}"; do
  if [[ ! "${ctx}" =~ ^[0-9]+$ ]]; then
    echo "Invalid context value: ${ctx}" >&2
    exit 1
  fi
  ctx_name="${CONTEXT_NAME_TEMPLATE//\{context\}/${ctx}}"
  ctx_dir="${CONTEXT_ROOT}/${ctx_name}"
  if [[ ! -d "${ctx_dir}" ]]; then
    echo "Missing context dir: ${ctx_dir}" >&2
    exit 1
  fi
  context_entries+=("${ctx}=${ctx_dir}")
  if [[ "${ctx}" == "${MAX_CONTEXT}" ]]; then
    max_ctx_dir="${ctx_dir}"
  fi
done

if [[ -z "${OUTPUT_BASE}" ]]; then
  OUTPUT_BASE="${PREFIX}_FFN_PF_statex"
fi

mkdir -p "${TEMP_DIR}"
export TMPDIR="${TEMP_DIR}"
export TMP="${TEMP_DIR}"
export TEMP="${TEMP_DIR}"

if [[ -z "${TOKENIZER_PATH}" ]]; then
  if [[ -n "${max_ctx_dir}" ]] && [[ ! -f "${max_ctx_dir}/tokenizer.json" || ! -f "${max_ctx_dir}/tokenizer_config.json" ]]; then
    hf_cache_root="${HOME}/.cache/huggingface/hub/models--${HF_MODEL_ID//\//--}/snapshots"
    if [[ -d "${hf_cache_root}" ]]; then
      for snap in "${hf_cache_root}"/*; do
        if [[ -f "${snap}/tokenizer.json" && -f "${snap}/tokenizer_config.json" ]]; then
          TOKENIZER_PATH="${snap}"
          echo "Tokenizer fallback auto-detected from HF cache: ${TOKENIZER_PATH}"
          break
        fi
      done
    fi
  fi
fi

echo "Requested chunk kinds: infer=${INFER_KIND}, chunk1=${INFER_CHUNK1_KIND}, prefill=${PREFILL_KIND}"
echo "Include prefill_ctx per context: ${PREFILL_ALL_CONTEXTS}"
echo "Split infer/prefill outputs: ${SPLIT_INFER_PREFILL}"
echo "Alias functions enabled: $([[ \"${NO_ALIAS_FUNCTIONS}\" == \"true\" ]] && echo false || echo true)"
echo "Compile output artifacts: ${COMPILE}"
echo "Copy source chunks: ${COPY_SOURCE_CHUNKS}"

if [[ "${CLEAN_OUTPUT}" == "true" ]]; then
  echo "Cleaning previous combined artifacts in: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}/${OUTPUT_BASE}"_chunk_*.mlpackage
  rm -rf "${OUTPUT_DIR}/${OUTPUT_BASE}"_chunk_*.mlmodelc
  rm -rf "${OUTPUT_DIR}/${OUTPUT_BASE}"_infer_chunk_*.mlpackage
  rm -rf "${OUTPUT_DIR}/${OUTPUT_BASE}"_infer_chunk_*.mlmodelc
  rm -rf "${OUTPUT_DIR}/${OUTPUT_BASE}"_prefill_chunk_*.mlpackage
  rm -rf "${OUTPUT_DIR}/${OUTPUT_BASE}"_prefill_chunk_*.mlmodelc
  if [[ -n "${OUTPUT_BASE_INFER}" ]]; then
    rm -rf "${OUTPUT_DIR}/${OUTPUT_BASE_INFER}"_chunk_*.mlpackage
    rm -rf "${OUTPUT_DIR}/${OUTPUT_BASE_INFER}"_chunk_*.mlmodelc
  fi
  if [[ -n "${OUTPUT_BASE_PREFILL}" ]]; then
    rm -rf "${OUTPUT_DIR}/${OUTPUT_BASE_PREFILL}"_chunk_*.mlpackage
    rm -rf "${OUTPUT_DIR}/${OUTPUT_BASE_PREFILL}"_chunk_*.mlmodelc
  fi
  rm -f "${OUTPUT_DIR}/state_transition_manifest.yaml"
  rm -f "${OUTPUT_DIR}/meta.yaml"
fi

cmd=(
  python3 "${REPO_ROOT}/tests/dev/combine_infer_context_exports.py"
  --contexts
  "${context_entries[@]}"
  --output-dir "${OUTPUT_DIR}"
  --max-context "${MAX_CONTEXT}"
  --prefix "${PREFIX}"
  --num-chunks "${NUM_CHUNKS}"
  --output-base "${OUTPUT_BASE}"
  --infer-fn "${INFER_FN}"
  --prefill-fn "${PREFILL_FN}"
  --infer-kind "${INFER_KIND}"
  --infer-chunk1-kind "${INFER_CHUNK1_KIND}"
  --prefill-kind "${PREFILL_KIND}"
  --architecture "${ARCHITECTURE}"
  --model-name "${MODEL_NAME}"
)
if [[ "${PREFILL_ALL_CONTEXTS}" == "true" ]]; then
  cmd+=(--prefill-all-contexts)
fi
if [[ "${SPLIT_INFER_PREFILL}" == "true" ]]; then
  cmd+=(--split-infer-prefill)
fi
if [[ "${NO_ALIAS_FUNCTIONS}" == "true" ]]; then
  cmd+=(--no-alias-functions)
fi
if [[ -n "${OUTPUT_BASE_INFER}" ]]; then
  cmd+=(--output-base-infer "${OUTPUT_BASE_INFER}")
fi
if [[ -n "${OUTPUT_BASE_PREFILL}" ]]; then
  cmd+=(--output-base-prefill "${OUTPUT_BASE_PREFILL}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  cmd+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${SPLIT_LM_HEAD}" ]]; then
  cmd+=(--split-lm-head "${SPLIT_LM_HEAD}")
fi
if [[ "${FORCE}" == "true" ]]; then
  cmd+=(--force)
fi
if [[ "${NO_COPY_SHARED}" == "true" ]]; then
  cmd+=(--no-copy-shared)
fi
if [[ "${DEDUP_DIAGNOSTICS}" == "true" ]]; then
  cmd+=(--dedup-diagnostics)
fi
if [[ "${SKIP_ANEMLL_DEDUP}" == "true" ]]; then
  cmd+=(--skip-anemll-dedup)
fi
if [[ "${COPY_SOURCE_CHUNKS}" == "true" ]]; then
  cmd+=(--copy-source-chunks)
fi
if [[ -n "${COPY_SOURCE_CHUNKS_CONTEXT}" ]]; then
  cmd+=(--copy-source-chunks-context "${COPY_SOURCE_CHUNKS_CONTEXT}")
fi
if [[ "${NO_META}" == "true" ]]; then
  cmd+=(--no-meta)
fi
if [[ "${COMPILE}" != "true" ]]; then
  cmd+=(--no-compile)
fi
if [[ -n "${TOKENIZER_PATH}" ]]; then
  cmd+=(--tokenizer-path "${TOKENIZER_PATH}")
fi

echo "${cmd[*]}"
"${cmd[@]}"
