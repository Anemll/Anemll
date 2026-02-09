#!/usr/bin/env bash
set -eEuo pipefail

# Step-driven hybrid rebuild wrapper for VibeThinker.
#
# Supports running convert_model.sh stages explicitly plus custom stages:
#   - 1..8 : convert_model.sh --only <step>
#   - fp32 : apply chunk_01 FP32 attention patch
#   - smoke: run tests/chat.py smoke test
#
# Example:
#   bash scripts/rebuild_vibethinker_hybrid_steps.sh \
#     --steps "1,2,3,4,fp32" \
#     --context 4096 \
#     --output /Volumes/Models/ANE/vibethinker_1.5b_ctx4096_fp16_hybrid \
#     --force-clean

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONVERT_SCRIPT="${REPO_ROOT}/anemll/utils/convert_model.sh"
FP32_PATCH_SCRIPT="${REPO_ROOT}/tests/dev/proto_qwen25_chunk1_fp32.py"
CHAT_SCRIPT="${REPO_ROOT}/tests/chat.py"

MODEL_ID="WeiboAI/VibeThinker-1.5B"
MODEL_PATH="${HOME}/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B"
OUTPUT_DIR="/Volumes/Models/ANE/vibethinker_1.5b_ctx4096_fp16_hybrid"
CONTEXT=4096
BATCH=32
CHUNKS=3
PREFIX="qwen25"
LUT1="none"
LUT2="none"
LUT3="none"
STEPS_RAW="1,2,3,4"
TEMP_DIR="/Volumes/Models/ANE/tmp_coreml_compile"
PROMPT="2+2="
MAX_TOKENS=20
NO_THINK=true
FORCE_CLEAN=false
SKIP_CHECK=true
REUSE_STATIC_FROM=""
EXTRA_CONVERT_ARGS=""

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

usage() {
  cat <<'EOF'
Usage:
  scripts/rebuild_vibethinker_hybrid_steps.sh [options]

Options:
  --steps LIST                Comma/space separated steps.
                              Supported tokens:
                                1..8   convert_model.sh steps
                                fp32   apply hybrid FP32 patch
                                smoke  run chat.py smoke test
                              Default: "1,2,3,4"

  --output DIR                Output directory
                              (default: /Volumes/Models/ANE/vibethinker_1.5b_ctx4096_fp16_hybrid)
  --model ID                  HF model id/path for convert_model.sh
                              (default: WeiboAI/VibeThinker-1.5B)
  --model-path DIR            HF snapshot/cache path for FP32 patch model load
  --context N                 Context length (default: 4096)
  --batch N                   Batch size (default: 32)
  --chunks N                  Chunk count (default: 3)
  --prefix STR                Model prefix (default: qwen25)
  --lut1 V                    LUT embeddings (default: none)
  --lut2 V                    LUT FFN/prefill (default: none)
  --lut3 V                    LUT lm_head (default: none)
  --temp-dir DIR              TMPDIR for smoke (default: /Volumes/Models/ANE/tmp_coreml_compile)
  --prompt TEXT               Smoke prompt (default: "2+2=")
  --max-tokens N              Smoke max tokens (default: 20)
  --no-no-think               Do not pass --no-think to smoke
  --force-clean               Remove output dir first
  --no-skip-check             Do not pass --skip-check to convert_model.sh
  --reuse-static-from DIR     Copy embeddings/lm_head/tokenizer/meta from DIR before steps
  --extra-convert-args "ARGS" Extra args appended to each convert_model.sh call
  -h, --help                  Show this help

Examples:
  # Export-only flow for one context (no combine/compile/meta copy)
  bash scripts/rebuild_vibethinker_hybrid_steps.sh --steps "1,2,3,4" --output /tmp/ctx4096 --force-clean

  # Full convert steps, then FP32 patch
  bash scripts/rebuild_vibethinker_hybrid_steps.sh --steps "1,2,3,4,5,6,7,fp32" --output /tmp/ctx4096 --force-clean

  # Patch + smoke only (expects existing converted output with meta.yaml)
  bash scripts/rebuild_vibethinker_hybrid_steps.sh --steps "fp32,smoke" --output /tmp/ctx4096
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) STEPS_RAW="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --model) MODEL_ID="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --context) CONTEXT="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --lut1) LUT1="$2"; shift 2 ;;
    --lut2) LUT2="$2"; shift 2 ;;
    --lut3) LUT3="$2"; shift 2 ;;
    --temp-dir) TEMP_DIR="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --no-no-think) NO_THINK=false; shift 1 ;;
    --force-clean) FORCE_CLEAN=true; shift 1 ;;
    --no-skip-check) SKIP_CHECK=false; shift 1 ;;
    --reuse-static-from) REUSE_STATIC_FROM="$2"; shift 2 ;;
    --extra-convert-args) EXTRA_CONVERT_ARGS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! "${CONTEXT}" =~ ^[0-9]+$ ]] || [[ ! "${BATCH}" =~ ^[0-9]+$ ]] || [[ ! "${CHUNKS}" =~ ^[0-9]+$ ]] || [[ ! "${MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "--context, --batch, --chunks, --max-tokens must be integers" >&2
  exit 1
fi

if [[ ! -x "${CONVERT_SCRIPT}" ]]; then
  echo "Missing convert script: ${CONVERT_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${FP32_PATCH_SCRIPT}" ]]; then
  echo "Missing FP32 patch script: ${FP32_PATCH_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${CHAT_SCRIPT}" ]]; then
  echo "Missing chat script: ${CHAT_SCRIPT}" >&2
  exit 1
fi

if [[ -f "${REPO_ROOT}/env-anemll/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/env-anemll/bin/activate"
fi

mkdir -p "${TEMP_DIR}"
export TMPDIR="${TEMP_DIR}"
export TMP="${TEMP_DIR}"
export TEMP="${TEMP_DIR}"

if [[ -d "${OUTPUT_DIR}" && "${FORCE_CLEAN}" == "true" ]]; then
  echo "Cleaning output dir: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

copy_reused_static() {
  local src_dir="$1"
  if [[ ! -d "${src_dir}" ]]; then
    echo "--reuse-static-from does not exist: ${src_dir}" >&2
    exit 1
  fi

  echo "Reusing static artifacts from: ${src_dir}"

  shopt -s nullglob
  for src in \
    "${src_dir}"/qwen25_embeddings*.mlpackage \
    "${src_dir}"/qwen25_embeddings*.mlmodelc \
    "${src_dir}"/qwen25_lm_head*.mlpackage \
    "${src_dir}"/qwen25_lm_head*.mlmodelc \
    "${src_dir}"/tokenizer.json \
    "${src_dir}"/tokenizer_config.json \
    "${src_dir}"/tokenizer.model \
    "${src_dir}"/special_tokens_map.json \
    "${src_dir}"/generation_config.json \
    "${src_dir}"/chat_template.jinja \
    "${src_dir}"/config.json \
    "${src_dir}"/meta.yaml; do
    dst="${OUTPUT_DIR}/$(basename "${src}")"
    rm -rf "${dst}"
    cp -R "${src}" "${dst}"
  done
  shopt -u nullglob
}

run_convert_step() {
  local step="$1"
  local part_name
  part_name="$(convert_step_part_name "${step}")"
  local cmd=(
    bash "${CONVERT_SCRIPT}"
    --model "${MODEL_ID}"
    --output "${OUTPUT_DIR}"
    --context "${CONTEXT}"
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
  if [[ -n "${EXTRA_CONVERT_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${EXTRA_CONVERT_ARGS} )
    cmd+=("${extra[@]}")
  fi
  echo "Step ${step} (${part_name}): ${cmd[*]}"
  echo "  convert_model.sh --only ${step}  # part ${step}: ${part_name}"
  "${cmd[@]}"
}

run_fp32_patch() {
  if [[ ! -f "${OUTPUT_DIR}/meta.yaml" ]]; then
    echo "FP32 patch requires ${OUTPUT_DIR}/meta.yaml. Run convert step 7 first or provide reused meta." >&2
    exit 1
  fi

  local patch_cmd=(
    python3 "${FP32_PATCH_SCRIPT}"
    --model-path "${MODEL_PATH}"
    --source-dir "${OUTPUT_DIR}"
    --out-dir "${OUTPUT_DIR}"
    --reuse-out-dir
    --context-length "${CONTEXT}"
    --batch-size "${BATCH}"
    --lut1 "${LUT1}"
    --lut2 "${LUT2}"
    --lut3 "${LUT3}"
    --argmax-in-model false
    --recommended-do-sample true
    --recommended-temperature 0.6
    --recommended-top-p 0.95
    --recommended-top-k 0
  )
  if [[ -n "${REUSE_STATIC_FROM}" ]]; then
    patch_cmd+=(--reuse-lm-head)
  fi
  echo "Step fp32: ${patch_cmd[*]}"
  "${patch_cmd[@]}"
}

run_smoke() {
  if [[ ! -f "${OUTPUT_DIR}/meta.yaml" ]]; then
    echo "Smoke test requires ${OUTPUT_DIR}/meta.yaml" >&2
    exit 1
  fi

  local smoke_cmd=(
    python3 "${CHAT_SCRIPT}"
    --meta "${OUTPUT_DIR}/meta.yaml"
    --prompt "${PROMPT}"
    --max-tokens "${MAX_TOKENS}"
  )
  if [[ "${NO_THINK}" == "true" ]]; then
    smoke_cmd+=(--no-think)
  fi
  echo "Step smoke: ${smoke_cmd[*]}"
  "${smoke_cmd[@]}"
}

if [[ -n "${REUSE_STATIC_FROM}" ]]; then
  copy_reused_static "${REUSE_STATIC_FROM}"
fi

# shellcheck disable=SC2206
STEPS=( $(echo "${STEPS_RAW}" | tr ',' ' ') )
if [[ ${#STEPS[@]} -eq 0 ]]; then
  echo "No steps specified." >&2
  exit 1
fi

echo "Plan:"
echo "  Output: ${OUTPUT_DIR}"
echo "  Context: ${CONTEXT}  Batch: ${BATCH}  Chunks: ${CHUNKS}"
echo "  LUT: ${LUT1}/${LUT2}/${LUT3}"
echo "  Steps: ${STEPS[*]}"

for token in "${STEPS[@]}"; do
  case "${token}" in
    [1-8]) run_convert_step "${token}" ;;
    fp32) run_fp32_patch ;;
    smoke) run_smoke ;;
    *)
      echo "Unsupported step token: ${token}" >&2
      echo "Allowed: 1..8, fp32, smoke" >&2
      exit 1
      ;;
  esac
done

echo "Done."
echo "Output: ${OUTPUT_DIR}"
