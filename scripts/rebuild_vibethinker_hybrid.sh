#!/usr/bin/env bash
set -eEuo pipefail

# End-to-end rebuild for VibeThinker hybrid export:
# 1) Fresh base conversion (chunked, no LUT by default)
# 2) In-place hybrid patch (chunk_01of03 = FP32 attention-only)
# 3) Metadata patch for no-argmax + recommended sampling
# 4) Smoke test with tests/chat.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORIG_ARGS=("$@")

MODEL_ID="WeiboAI/VibeThinker-1.5B"
MODEL_PATH="${HOME}/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B"
OUTPUT_DIR="/Volumes/Models/ANE/vibethinker_1.5b_ctx4096_fp16_hybrid"
CONTEXT=4096
BATCH=32
CHUNKS=3
REMAINING_CHUNKS=3
LUT1="none"
LUT2="none"
LUT3="none"
TEMP_DIR="/Volumes/Models/ANE/tmp_coreml_compile"
LOG_FILE="/Volumes/Models/ANE/logs/rebuild_vibethinker_hybrid_$(date -u +%Y%m%dT%H%M%SZ).log"
PROMPT="2+2="
MAX_TOKENS=20
NO_THINK=true
FORCE_CLEAN=false
SKIP_SMOKE=false
WAIT_TIMEOUT=180
REUSE_STATIC_FROM=""
REUSE_INFER_ONLY=false
EXPORT_STANDALONE_FP32=false
PURGE_LEGACY_BASE_CHUNKS=true

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

usage() {
  cat <<'EOF'
Usage:
  scripts/rebuild_vibethinker_hybrid.sh [options]

Options:
  --output DIR              Output directory (default: /Volumes/Models/ANE/vibethinker_1.5b_ctx4096_fp16_hybrid)
  --model ID                HF model id (default: WeiboAI/VibeThinker-1.5B)
  --model-path DIR          Local HF cache/snapshot path root
  --context N               Context length (default: 4096)
  --batch N                 Batch size (default: 32)
  --chunks N                Number of chunks for base conversion (default: 3)
  --remaining-chunks N      Number of post-attention chunks for hybrid patch.
                            Final hybrid chunk count = 1 + remaining-chunks.
                            Default: 3 (=> chunk_01of04..chunk_04of04)
  --lut1 V                  LUT for embeddings (default: none)
  --lut2 V                  LUT for FFN (default: none)
  --lut3 V                  LUT for LM head (default: none)
  --temp-dir DIR            Temp directory for CoreML compilation scratch
  --log-file FILE           Write full execution log to FILE
  --prompt TEXT             Smoke-test prompt (default: "2+2=")
  --max-tokens N            Smoke-test max tokens (default: 20)
  --no-no-think             Do not pass --no-think during smoke test
  --skip-smoke              Skip smoke test
  --wait-timeout SEC        Wait timeout for conversion stabilization (default: 180)
  --force-clean             Delete existing output dir before rebuild
  --reuse-static-from DIR   Reuse embeddings/lm_head from DIR and only rebuild
                            FFN/prefill chunks for this context (faster multi-context build)
  --reuse-infer-only        With --reuse-static-from, run only convert part 3 (ffn_infer)
                            and export standalone FP32-first-attn chunk_01 infer artifact
                            (full regular chunk_01 contract, not attention-only split).
                            Skips converter parts 4..7.
  --export-standalone-fp32 Force standalone FP32 chunk_01 artifact export in full-hybrid mode
                            (default: disabled in full-hybrid mode; always enabled for --reuse-infer-only)
  --keep-legacy-base-chunks Keep converter base artifacts (qwen25_FFN_chunk_*, qwen25_prefill_chunk_*)
                            after hybrid patch. Default purges them in full-hybrid mode to avoid confusion.
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --model) MODEL_ID="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --context) CONTEXT="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --remaining-chunks) REMAINING_CHUNKS="$2"; shift 2 ;;
    --lut1) LUT1="$2"; shift 2 ;;
    --lut2) LUT2="$2"; shift 2 ;;
    --lut3) LUT3="$2"; shift 2 ;;
    --temp-dir) TEMP_DIR="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --no-no-think) NO_THINK=false; shift 1 ;;
    --skip-smoke) SKIP_SMOKE=true; shift 1 ;;
    --wait-timeout) WAIT_TIMEOUT="$2"; shift 2 ;;
    --force-clean) FORCE_CLEAN=true; shift 1 ;;
    --reuse-static-from) REUSE_STATIC_FROM="$2"; shift 2 ;;
    --reuse-infer-only) REUSE_INFER_ONLY=true; shift 1 ;;
    --export-standalone-fp32) EXPORT_STANDALONE_FP32=true; shift 1 ;;
    --keep-legacy-base-chunks) PURGE_LEGACY_BASE_CHUNKS=false; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

trap 'echo "ERROR: command failed at line ${LINENO}. See log: ${LOG_FILE}"' ERR

cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/env-anemll/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/env-anemll/bin/activate"
fi

mkdir -p "${TEMP_DIR}"
export TMPDIR="${TEMP_DIR}"
export TMP="${TEMP_DIR}"
export TEMP="${TEMP_DIR}"

echo "[1/5] Preflight"
echo "  Log file: ${LOG_FILE}"
echo "  Command: ${0} ${ORIG_ARGS[*]}"
echo "  Repo: ${REPO_ROOT}"
echo "  Model: ${MODEL_ID}"
echo "  Model path: ${MODEL_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Context: ${CONTEXT}, Batch: ${BATCH}, Base chunks: ${CHUNKS}, Hybrid remaining chunks: ${REMAINING_CHUNKS}"
echo "  LUT: ${LUT1}/${LUT2}/${LUT3}"

active_pids="$(pgrep -f "convert_model.sh.*--output[[:space:]]+${OUTPUT_DIR}|combine_models.py.*${OUTPUT_DIR}|compile_models.py.*${OUTPUT_DIR}" || true)"
if [[ -n "${active_pids}" ]]; then
  echo "Another conversion appears to be running for this output dir: ${OUTPUT_DIR}" >&2
  echo "Active PID(s): ${active_pids}" >&2
  echo "Wait for that process to finish, or use a different --output." >&2
  exit 1
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
  if [[ "${FORCE_CLEAN}" == "true" ]]; then
    echo "  Cleaning existing output dir: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
  else
    echo "Output dir already exists. Use --force-clean to rebuild from scratch." >&2
    exit 1
  fi
fi

if [[ -n "${REUSE_STATIC_FROM}" ]]; then
  REUSE_STATIC_FROM="$(cd "$(dirname "${REUSE_STATIC_FROM}")" && pwd)/$(basename "${REUSE_STATIC_FROM}")"
  if [[ ! -d "${REUSE_STATIC_FROM}" ]]; then
    echo "--reuse-static-from directory does not exist: ${REUSE_STATIC_FROM}" >&2
    exit 1
  fi

  echo "[2/5] Reuse static artifacts and rebuild context-specific parts"
  echo "  Static source: ${REUSE_STATIC_FROM}"
  mkdir -p "${OUTPUT_DIR}"

  if [[ "${REUSE_INFER_ONLY}" == "true" ]]; then
    echo "  Reuse mode: infer-only (convert part 3 + standalone FP32-first-attn full chunk_01 export)"
    echo "  Skipping static artifact copy for infer-only mode."
    step_list=(3)
  else
    copied_static=false
    copied_static_core=false
    shopt -s nullglob
    for src in \
      "${REUSE_STATIC_FROM}"/qwen25_embeddings*.mlpackage \
      "${REUSE_STATIC_FROM}"/qwen25_embeddings*.mlmodelc \
      "${REUSE_STATIC_FROM}"/qwen25_lm_head*.mlpackage \
      "${REUSE_STATIC_FROM}"/qwen25_lm_head*.mlmodelc \
      "${REUSE_STATIC_FROM}"/tokenizer.json \
      "${REUSE_STATIC_FROM}"/tokenizer_config.json \
      "${REUSE_STATIC_FROM}"/tokenizer.model \
      "${REUSE_STATIC_FROM}"/special_tokens_map.json \
      "${REUSE_STATIC_FROM}"/generation_config.json \
      "${REUSE_STATIC_FROM}"/chat_template.jinja \
      "${REUSE_STATIC_FROM}"/config.json \
      "${REUSE_STATIC_FROM}"/meta.yaml; do
      if [[ ! -e "${src}" ]]; then
        continue
      fi
      dst="${OUTPUT_DIR}/$(basename "${src}")"
      if [[ -e "${dst}" ]]; then
        rm -rf "${dst}"
      fi
      cp -R "${src}" "${dst}"
      copied_static=true
      case "$(basename "${src}")" in
        qwen25_embeddings*|qwen25_lm_head*) copied_static_core=true ;;
      esac
    done
    shopt -u nullglob

    if [[ "${copied_static_core}" != "true" ]]; then
      echo "No static embeddings/lm_head artifacts found in: ${REUSE_STATIC_FROM}" >&2
      exit 1
    fi

    # 3=FFN, 4=prefill, 5=combine, 6=compile, 7=meta/tokenizer.
    echo "  Reuse mode: standard (convert parts 3..7 + FP32 patch)"
    step_list=(3 4 5 6 7)
  fi

  for step in "${step_list[@]}"; do
    local_step_cmd=()
    part_name="$(convert_step_part_name "${step}")"
    echo "  convert_model.sh --only ${step}  # part ${step}: ${part_name}"
    local_step_cmd=(
      "${REPO_ROOT}/anemll/utils/convert_model.sh"
      --model "${MODEL_ID}"
      --output "${OUTPUT_DIR}"
      --context "${CONTEXT}"
      --batch "${BATCH}"
      --lut1 "${LUT1}" --lut2 "${LUT2}" --lut3 "${LUT3}"
      --chunk "${CHUNKS}"
      --skip-check
      --restart "${step}"
      --only "${step}"
    )
    "${local_step_cmd[@]}"
  done
else
  echo "[2/5] Base conversion"
  base_cmd=(
    "${REPO_ROOT}/anemll/utils/convert_model.sh"
    --model "${MODEL_ID}"
    --output "${OUTPUT_DIR}"
    --context "${CONTEXT}"
    --batch "${BATCH}"
    --lut1 "${LUT1}" --lut2 "${LUT2}" --lut3 "${LUT3}"
    --chunk "${CHUNKS}"
    --skip-check
  )
  "${base_cmd[@]}"
fi

echo "[2.5/5] Wait for conversion processes to settle"
elapsed=0
while true; do
  active_pids="$(pgrep -f "convert_model.sh.*--output[[:space:]]+${OUTPUT_DIR}|combine_models.py.*${OUTPUT_DIR}|compile_models.py.*${OUTPUT_DIR}" || true)"
  if [[ -z "${active_pids}" ]]; then
    break
  fi
  if (( elapsed >= WAIT_TIMEOUT )); then
    echo "Timed out waiting for conversion processes to finish for ${OUTPUT_DIR}" >&2
    echo "Still active PID(s): ${active_pids}" >&2
    exit 1
  fi
  echo "  Waiting... active PID(s): ${active_pids}"
  sleep 5
  elapsed=$((elapsed + 5))
done

if [[ ! "${REMAINING_CHUNKS}" =~ ^[0-9]+$ ]] || [[ "${REMAINING_CHUNKS}" -le 0 ]]; then
  echo "--remaining-chunks must be a positive integer" >&2
  exit 1
fi

if [[ -n "${REUSE_STATIC_FROM}" && "${REUSE_INFER_ONLY}" == "true" ]]; then
  echo "[3/5] Verify minimal artifacts (infer-only reuse mode)"
  LUT2_BITS="$(lut_bits_or_empty "${LUT2}")"
  FFN_BASE="qwen25_FFN${LUT2_BITS:+_lut${LUT2_BITS}}"
  for idx in 01 02 03; do
    if [[ ! -d "${OUTPUT_DIR}/${FFN_BASE}_chunk_${idx}of03.mlpackage" && ! -d "${OUTPUT_DIR}/${FFN_BASE}_chunk_${idx}of03.mlmodelc" ]]; then
      echo "Missing infer chunk after part 3: ${OUTPUT_DIR}/${FFN_BASE}_chunk_${idx}of03(.mlpackage/.mlmodelc)" >&2
      exit 1
    fi
  done

  echo "[4/5] Export standalone FP32-first-attn chunk_01 (infer-only mode, full chunk contract)"
  FP32_INFER_BASE="qwen25_FFN_attn_fp32"
  fp32_args=(
    python3 "${REPO_ROOT}/tests/dev/proto_qwen25_chunk1_fp32.py"
    --model-path "${MODEL_PATH}"
    --source-dir "${OUTPUT_DIR}"
    --out-dir "${OUTPUT_DIR}"
    --reuse-out-dir
    --context-length "${CONTEXT}"
    --batch-size "${BATCH}"
    --lut1 "${LUT1}" --lut2 "${LUT2}" --lut3 "${LUT3}"
    --infer-only
    --num-chunks "${CHUNKS}"
    --infer-only-out-base "${FP32_INFER_BASE}"
    --no-compile
  )
  "${fp32_args[@]}"

  echo "[5/5] Verify infer-only outputs"
  if [[ ! -d "${OUTPUT_DIR}/${FP32_INFER_BASE}_chunk_01of03.mlpackage" ]]; then
    echo "Missing standalone FP32-first-attn package: ${OUTPUT_DIR}/${FP32_INFER_BASE}_chunk_01of03.mlpackage" >&2
    exit 1
  fi
  for idx in 01 02 03; do
    if [[ ! -d "${OUTPUT_DIR}/${FFN_BASE}_chunk_${idx}of03.mlpackage" && ! -d "${OUTPUT_DIR}/${FFN_BASE}_chunk_${idx}of03.mlmodelc" ]]; then
      echo "Missing base infer chunk after FP32 export: ${OUTPUT_DIR}/${FFN_BASE}_chunk_${idx}of03(.mlpackage/.mlmodelc)" >&2
      exit 1
    fi
  done

  echo "[5/5] Done (infer-only)"
  echo "Model: ${OUTPUT_DIR}"
  echo "Log: ${LOG_FILE}"
  exit 0
else
  echo "[3/5] Verify base artifacts"
  LUT1_BITS="$(lut_bits_or_empty "${LUT1}")"
  LUT2_BITS="$(lut_bits_or_empty "${LUT2}")"
  LUT3_BITS="$(lut_bits_or_empty "${LUT3}")"
  EMB_STEM="qwen25_embeddings${LUT1_BITS:+_lut${LUT1_BITS}}"
  LMH_STEM="qwen25_lm_head${LUT3_BITS:+_lut${LUT3_BITS}}"
  FFN_BASE="qwen25_FFN_PF${LUT2_BITS:+_lut${LUT2_BITS}}"
  required_files=(
    "${OUTPUT_DIR}/meta.yaml"
    "${OUTPUT_DIR}/tokenizer.json"
    "${OUTPUT_DIR}/tokenizer_config.json"
    "${OUTPUT_DIR}/${EMB_STEM}.mlmodelc"
    "${OUTPUT_DIR}/${LMH_STEM}.mlmodelc"
    "${OUTPUT_DIR}/${FFN_BASE}_chunk_01of03.mlmodelc"
    "${OUTPUT_DIR}/${FFN_BASE}_chunk_02of03.mlmodelc"
    "${OUTPUT_DIR}/${FFN_BASE}_chunk_03of03.mlmodelc"
  )
  for f in "${required_files[@]}"; do
    if [[ ! -e "${f}" ]]; then
      echo "Missing required file after base conversion: ${f}" >&2
      exit 1
    fi
  done
fi

echo "[4/5] Apply hybrid FP32 chunk_01 patch (in-place)"
patch_args=(
  python3 "${REPO_ROOT}/tests/dev/proto_qwen25_chunk1_fp32.py"
  --model-path "${MODEL_PATH}"
  --source-dir "${OUTPUT_DIR}"
  --out-dir "${OUTPUT_DIR}"
  --reuse-out-dir
  --context-length "${CONTEXT}"
  --batch-size "${BATCH}"
  --remaining-chunks "${REMAINING_CHUNKS}"
  --lut1 "${LUT1}" --lut2 "${LUT2}" --lut3 "${LUT3}"
  --argmax-in-model false
  --recommended-do-sample true
  --recommended-temperature 0.6
  --recommended-top-p 0.95
  --recommended-top-k 0
)
if [[ -n "${REUSE_STATIC_FROM}" ]]; then
  patch_args+=(--reuse-lm-head)
fi
"${patch_args[@]}"

if [[ "${PURGE_LEGACY_BASE_CHUNKS}" == "true" ]]; then
  echo "[4.2/5] Purge legacy base chunk artifacts"
  shopt -s nullglob
  for p in \
    "${OUTPUT_DIR}"/qwen25_FFN_chunk_01of03.mlpackage \
    "${OUTPUT_DIR}"/qwen25_FFN_chunk_02of03.mlpackage \
    "${OUTPUT_DIR}"/qwen25_FFN_chunk_03of03.mlpackage \
    "${OUTPUT_DIR}"/qwen25_FFN_chunk_01of03.mlmodelc \
    "${OUTPUT_DIR}"/qwen25_FFN_chunk_02of03.mlmodelc \
    "${OUTPUT_DIR}"/qwen25_FFN_chunk_03of03.mlmodelc \
    "${OUTPUT_DIR}"/qwen25_prefill_chunk_01of03.mlpackage \
    "${OUTPUT_DIR}"/qwen25_prefill_chunk_02of03.mlpackage \
    "${OUTPUT_DIR}"/qwen25_prefill_chunk_03of03.mlpackage \
    "${OUTPUT_DIR}"/qwen25_prefill_chunk_01of03.mlmodelc \
    "${OUTPUT_DIR}"/qwen25_prefill_chunk_02of03.mlmodelc \
    "${OUTPUT_DIR}"/qwen25_prefill_chunk_03of03.mlmodelc; do
    [[ -e "${p}" ]] || continue
    rm -rf "${p}"
  done
  shopt -u nullglob
fi

if [[ "${EXPORT_STANDALONE_FP32}" == "true" ]]; then
  echo "[4.5/5] Export standalone FP32 attention chunk_01 artifact"
  FP32_INFER_BASE="qwen25_FFN_attn_fp32"
  fp32_standalone_args=(
    python3 "${REPO_ROOT}/tests/dev/proto_qwen25_chunk1_fp32.py"
    --model-path "${MODEL_PATH}"
    --source-dir "${OUTPUT_DIR}"
    --out-dir "${OUTPUT_DIR}"
    --reuse-out-dir
    --context-length "${CONTEXT}"
    --batch-size "${BATCH}"
    --lut1 "${LUT1}" --lut2 "${LUT2}" --lut3 "${LUT3}"
    --infer-only
    --num-chunks "${CHUNKS}"
    --infer-only-out-base "${FP32_INFER_BASE}"
    --no-compile
  )
  "${fp32_standalone_args[@]}"
  if [[ ! -d "${OUTPUT_DIR}/${FP32_INFER_BASE}_chunk_01of03.mlpackage" ]]; then
    echo "Missing standalone FP32 attention package: ${OUTPUT_DIR}/${FP32_INFER_BASE}_chunk_01of03.mlpackage" >&2
    exit 1
  fi
fi

echo "[5/5] Verify hybrid metadata"
if ! grep -q "argmax_in_model: false" "${OUTPUT_DIR}/meta.yaml"; then
  echo "meta.yaml does not contain argmax_in_model: false" >&2
  exit 1
fi
if ! grep -q "recommended_sampling:" "${OUTPUT_DIR}/meta.yaml"; then
  echo "meta.yaml does not contain recommended_sampling block" >&2
  exit 1
fi

if [[ "${SKIP_SMOKE}" == "false" ]]; then
  echo "[smoke] tests/chat.py"
  smoke_log="${OUTPUT_DIR}/smoke_chat.log"
  smoke_args=(
    python3 "${REPO_ROOT}/tests/chat.py"
    --meta "${OUTPUT_DIR}/meta.yaml"
    --prompt "${PROMPT}"
    --max-tokens "${MAX_TOKENS}"
  )
  if [[ "${NO_THINK}" == "true" ]]; then
    smoke_args+=(--no-think)
  fi
  TMPDIR="${TEMP_DIR}" TMP="${TEMP_DIR}" TEMP="${TEMP_DIR}" "${smoke_args[@]}" 2>&1 | tee "${smoke_log}"

  # chat.py can swallow runtime exceptions and still return 0.
  # Treat known runtime failures as hard failures for this orchestrator.
  if grep -Eq "Error in chat loop|RuntimeError: Error compiling model|Failed to create a working directory|couldn.t be copied to .weights. because there isn.t enough space" "${smoke_log}"; then
    echo "Smoke test detected runtime errors. See: ${smoke_log}" >&2
    exit 1
  fi
fi

echo "Done."
echo "Model: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Chat command:"
echo "  python3 tests/chat.py --meta \"${OUTPUT_DIR}/meta.yaml\" --prompt \"Who are you ?\" --max-tokens 40 --no-think"
