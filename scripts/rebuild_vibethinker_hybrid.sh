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
echo "  Context: ${CONTEXT}, Batch: ${BATCH}, Chunks: ${CHUNKS}"
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

echo "[2/5] Base conversion"
"${REPO_ROOT}/anemll/utils/convert_model.sh" \
  --model "${MODEL_ID}" \
  --output "${OUTPUT_DIR}" \
  --context "${CONTEXT}" \
  --batch "${BATCH}" \
  --lut1 "${LUT1}" --lut2 "${LUT2}" --lut3 "${LUT3}" \
  --chunk "${CHUNKS}" \
  --skip-check

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

echo "[3/5] Verify base artifacts"
required_files=(
  "${OUTPUT_DIR}/meta.yaml"
  "${OUTPUT_DIR}/tokenizer.json"
  "${OUTPUT_DIR}/tokenizer_config.json"
  "${OUTPUT_DIR}/qwen25_embeddings.mlmodelc"
  "${OUTPUT_DIR}/qwen25_lm_head.mlmodelc"
  "${OUTPUT_DIR}/qwen25_FFN_PF_chunk_01of03.mlmodelc"
  "${OUTPUT_DIR}/qwen25_FFN_PF_chunk_02of03.mlmodelc"
  "${OUTPUT_DIR}/qwen25_FFN_PF_chunk_03of03.mlmodelc"
)
for f in "${required_files[@]}"; do
  if [[ ! -e "${f}" ]]; then
    echo "Missing required file after base conversion: ${f}" >&2
    exit 1
  fi
done

echo "[4/5] Apply hybrid FP32 chunk_01 patch (in-place)"
python3 "${REPO_ROOT}/tests/dev/proto_qwen25_chunk1_fp32.py" \
  --model-path "${MODEL_PATH}" \
  --source-dir "${OUTPUT_DIR}" \
  --out-dir "${OUTPUT_DIR}" \
  --reuse-out-dir \
  --context-length "${CONTEXT}" \
  --batch-size "${BATCH}" \
  --lut1 "${LUT1}" --lut2 "${LUT2}" --lut3 "${LUT3}" \
  --argmax-in-model false \
  --recommended-do-sample true \
  --recommended-temperature 0.6 \
  --recommended-top-p 0.95 \
  --recommended-top-k 0

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
