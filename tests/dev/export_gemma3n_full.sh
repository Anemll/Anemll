#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: export_gemma3n_full.sh [--model PATH] [--output DIR] [--context-length N] [--chunk N] [--lut N] [--lut-per-channel N] [--lut-workers N] [--lut-scope all|linear|conv|none] [--lut-include REGEX] [--lut-exclude REGEX] [--lut-report] [--flat]

Defaults:
  --model: latest HF snapshot for google/gemma-3n-E2B-it
  --output: ~/Models/ANE/gemma3n
  --context-length: 512
  --chunk: 4
  --lut: disabled
  --lut-per-channel: 8
  --lut-workers: 1
  --lut-scope: all
  --lut-include: none
  --lut-exclude: none
  --lut-report: disabled
  --flat: disabled (write parts into subfolders)

Example:
  tests/dev/export_gemma3n_full.sh --output ~/Models/ANE/gemma3n
EOF
}

MODEL_PATH=""
OUTPUT_DIR="${HOME}/Models/ANE/gemma3n"
CONTEXT_LENGTH="512"
CHUNK="4"
LUT=""
LUT_PER_CHANNEL="8"
LUT_WORKERS="1"
LUT_SCOPE="all"
LUT_INCLUDE=""
LUT_EXCLUDE=""
LUT_REPORT="0"
FLAT="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --context-length)
      CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --chunk)
      CHUNK="$2"
      shift 2
      ;;
    --lut)
      LUT="$2"
      shift 2
      ;;
    --lut-per-channel)
      LUT_PER_CHANNEL="$2"
      shift 2
      ;;
    --lut-workers)
      LUT_WORKERS="$2"
      shift 2
      ;;
    --lut-scope)
      LUT_SCOPE="$2"
      shift 2
      ;;
    --lut-include)
      LUT_INCLUDE="$2"
      shift 2
      ;;
    --lut-exclude)
      LUT_EXCLUDE="$2"
      shift 2
      ;;
    --lut-report)
      LUT_REPORT="1"
      shift 1
      ;;
    --flat)
      FLAT="1"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH=$(ls -td ~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/* | head -1)
fi

echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Context length: ${CONTEXT_LENGTH}"
echo "Chunk: ${CHUNK}"
if [[ -n "${LUT}" ]]; then
  echo "LUT: ${LUT}"
  echo "LUT per-channel group size: ${LUT_PER_CHANNEL}"
  echo "LUT workers: ${LUT_WORKERS}"
  echo "LUT scope: ${LUT_SCOPE}"
  if [[ -n "${LUT_INCLUDE}" ]]; then
    echo "LUT include: ${LUT_INCLUDE}"
  fi
  if [[ -n "${LUT_EXCLUDE}" ]]; then
    echo "LUT exclude: ${LUT_EXCLUDE}"
  fi
  if [[ "${LUT_REPORT}" == "1" ]]; then
    echo "LUT report: enabled"
  fi
fi
if [[ "${FLAT}" == "1" ]]; then
  echo "Flat layout: enabled"
fi

NO_SUBDIR_FLAG=""
if [[ "${FLAT}" == "1" ]]; then
  NO_SUBDIR_FLAG="--no-subdir"
fi

source env-anemll/bin/activate

python tests/dev/export_gemma3n.py \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --part infer \
  --context-length "${CONTEXT_LENGTH}" \
  --chunk "${CHUNK}" \
  ${LUT:+--lut "${LUT}"} \
  --lut-per-channel "${LUT_PER_CHANNEL}" \
  --lut-workers "${LUT_WORKERS}" \
  --lut-scope "${LUT_SCOPE}" \
  ${LUT_INCLUDE:+--lut-include "${LUT_INCLUDE}"} \
  ${LUT_EXCLUDE:+--lut-exclude "${LUT_EXCLUDE}"} \
  $([[ "${LUT_REPORT}" == "1" ]] && echo --lut-report) \
  ${NO_SUBDIR_FLAG}

python tests/dev/export_gemma3n.py \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --part infer_rotate \
  --context-length "${CONTEXT_LENGTH}" \
  --chunk "${CHUNK}" \
  ${LUT:+--lut "${LUT}"} \
  --lut-per-channel "${LUT_PER_CHANNEL}" \
  --lut-workers "${LUT_WORKERS}" \
  --lut-scope "${LUT_SCOPE}" \
  ${LUT_INCLUDE:+--lut-include "${LUT_INCLUDE}"} \
  ${LUT_EXCLUDE:+--lut-exclude "${LUT_EXCLUDE}"} \
  $([[ "${LUT_REPORT}" == "1" ]] && echo --lut-report) \
  ${NO_SUBDIR_FLAG}

python tests/dev/export_gemma3n.py \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --part prefill \
  --context-length "${CONTEXT_LENGTH}" \
  --chunk "${CHUNK}" \
  ${LUT:+--lut "${LUT}"} \
  --lut-per-channel "${LUT_PER_CHANNEL}" \
  --lut-workers "${LUT_WORKERS}" \
  --lut-scope "${LUT_SCOPE}" \
  ${LUT_INCLUDE:+--lut-include "${LUT_INCLUDE}"} \
  ${LUT_EXCLUDE:+--lut-exclude "${LUT_EXCLUDE}"} \
  $([[ "${LUT_REPORT}" == "1" ]] && echo --lut-report) \
  ${NO_SUBDIR_FLAG}

python tests/dev/export_gemma3n.py \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --part prefill_rotate \
  --context-length "${CONTEXT_LENGTH}" \
  --chunk "${CHUNK}" \
  ${LUT:+--lut "${LUT}"} \
  --lut-per-channel "${LUT_PER_CHANNEL}" \
  --lut-workers "${LUT_WORKERS}" \
  --lut-scope "${LUT_SCOPE}" \
  ${LUT_INCLUDE:+--lut-include "${LUT_INCLUDE}"} \
  ${LUT_EXCLUDE:+--lut-exclude "${LUT_EXCLUDE}"} \
  $([[ "${LUT_REPORT}" == "1" ]] && echo --lut-report) \
  ${NO_SUBDIR_FLAG}

python tests/dev/export_gemma3n.py \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --part lm_head \
  ${LUT:+--lut "${LUT}"} \
  --lut-per-channel "${LUT_PER_CHANNEL}" \
  --lut-workers "${LUT_WORKERS}" \
  --lut-scope "${LUT_SCOPE}" \
  ${LUT_INCLUDE:+--lut-include "${LUT_INCLUDE}"} \
  ${LUT_EXCLUDE:+--lut-exclude "${LUT_EXCLUDE}"} \
  $([[ "${LUT_REPORT}" == "1" ]] && echo --lut-report) \
  ${NO_SUBDIR_FLAG}

python tests/dev/export_gemma3n.py \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --part tokenizer \
  ${NO_SUBDIR_FLAG}

python tests/dev/export_gemma3n.py \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --part combine_streams \
  ${NO_SUBDIR_FLAG}

if [[ "${FLAT}" != "1" ]]; then
  cp -r "${OUTPUT_DIR}/lm_head/gemma3n_lm_head.mlpackage" "${OUTPUT_DIR}/infer/"
  cp -r "${OUTPUT_DIR}/combine_streams/gemma3n_combine_streams.mlpackage" "${OUTPUT_DIR}/infer/"
  cp -r "${OUTPUT_DIR}/infer_rotate/"*.mlpackage "${OUTPUT_DIR}/infer/" 2>/dev/null || true
  cp -r "${OUTPUT_DIR}/prefill/"*.mlpackage "${OUTPUT_DIR}/infer/" 2>/dev/null || true
  cp -r "${OUTPUT_DIR}/prefill_rotate/"*.mlpackage "${OUTPUT_DIR}/infer/" 2>/dev/null || true
  cp "${OUTPUT_DIR}/tokenizer/"*.json "${OUTPUT_DIR}/infer/"
  cp "${OUTPUT_DIR}/tokenizer/tokenizer.model" "${OUTPUT_DIR}/infer/"
fi

if [[ "${FLAT}" == "1" ]]; then
  echo "Done. Bundle ready at: ${OUTPUT_DIR}"
else
  echo "Done. Bundle ready at: ${OUTPUT_DIR}/infer/"
fi
