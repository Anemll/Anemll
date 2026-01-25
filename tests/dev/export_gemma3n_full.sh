#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: export_gemma3n_full.sh [--model PATH] [--output DIR] [--context-length N] [--chunk N]

Defaults:
  --model: latest HF snapshot for google/gemma-3n-E2B-it
  --output: ~/Models/ANE/gemma3n
  --context-length: 512
  --chunk: 4

Example:
  tests/dev/export_gemma3n_full.sh --output ~/Models/ANE/gemma3n
EOF
}

MODEL_PATH=""
OUTPUT_DIR="${HOME}/Models/ANE/gemma3n"
CONTEXT_LENGTH="512"
CHUNK="4"

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

source env-anemll/bin/activate

python tests/dev/export_gemma3n.py \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --part infer \
  --context-length "${CONTEXT_LENGTH}" \
  --chunk "${CHUNK}"

python tests/dev/export_gemma3n.py \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --part lm_head

python tests/dev/export_gemma3n.py \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --part tokenizer

python tests/dev/export_gemma3n.py \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --part combine_streams

cp -r "${OUTPUT_DIR}/lm_head/gemma3n_lm_head.mlpackage" "${OUTPUT_DIR}/infer/"
cp -r "${OUTPUT_DIR}/combine_streams/gemma3n_combine_streams.mlpackage" "${OUTPUT_DIR}/infer/"
cp "${OUTPUT_DIR}/tokenizer/"*.json "${OUTPUT_DIR}/infer/"
cp "${OUTPUT_DIR}/tokenizer/tokenizer.model" "${OUTPUT_DIR}/infer/"

echo "Done. Bundle ready at: ${OUTPUT_DIR}/infer/"
