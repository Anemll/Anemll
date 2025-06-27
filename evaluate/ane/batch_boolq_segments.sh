#!/usr/bin/env bash
# Batch-run BoolQ evaluation in fixed-size window segments and report the worst-performing ones.

set -euo pipefail

# Default parameters (adjust as needed)
MODEL_PATH=""
OUTPUT_DIR="results"
STEP=100           # Number of examples per segment
NUM_SHOTS=0        # Few-shot setting (0 for zero-shot)
BATCH_SIZE=1       # Batch size for strictly serial execution
WORST_COUNT=5      # Number of worst segments to report
TASK="boolq"
TOTAL=""         # Total number of examples (skip auto-detect via 'datasets')

# Ensure required tools are available
command -v jq >/dev/null 2>&1 || { echo "Error: 'jq' is required but not installed." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Error: 'python3' is required but not installed." >&2; exit 1; }
# Ensure numpy is available in python3 environment
if ! python3 - << 'EOF'
import numpy  # noqa: F401
EOF
then
  echo "Error: python3 cannot import numpy. Please activate the project virtualenv or install 'numpy' in this environment." >&2
  exit 1
fi

usage() {
  echo "Usage: $0 --model <model_dir> [--output-dir <dir>] [--step <n>] [--worst <k>] [--total <n>]" >&2
  echo "  --total:    total number of examples (skip auto-detection via 'datasets' pkg)" >&2
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_PATH="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --step)
      STEP="$2"; shift 2;;
    --worst)
      WORST_COUNT="$2"; shift 2;;
    --total)
      TOTAL="$2"; shift 2;;
    *)
      usage;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  usage
fi

# Create output dir and initialize a combined JSON file
mkdir -p "$OUTPUT_DIR"
COMBINED_JSON="$OUTPUT_DIR/eval_$(basename "$MODEL_PATH")_${NUM_SHOTS}shot_${TASK}.json"
echo "{}" > "$COMBINED_JSON"

# Determine total examples in BoolQ validation split (auto-detect unless overridden)
if [[ -z "${TOTAL:-}" ]]; then
  # Try auto-detect via HuggingFace 'datasets'; handle missing pkg gracefully
  PY_RET=0
  TOTAL=$(python3 - << 'EOF'
import sys
try:
    from datasets import load_dataset
except ModuleNotFoundError:
    sys.exit(2)
try:
    print(len(load_dataset("boolq", split="validation")))
except Exception:
    sys.exit(1)
EOF
  )
  PY_RET=$?
  if [[ $PY_RET -eq 2 ]]; then
    echo "Error: Python package 'datasets' not found. Install with 'pip install datasets' or pass --total <n>." >&2
    exit 1
  elif [[ $PY_RET -ne 0 ]]; then
    echo "Error: Could not auto-detect total examples. Pass --total <n>." >&2
    exit 1
  fi
fi
echo "Total BoolQ validation examples: $TOTAL"
echo "Segment size: $STEP examples; reporting $WORST_COUNT worst segments"

# Summary file header
SUMMARY_CSV="$OUTPUT_DIR/boolq_segments_summary.tsv"
echo -e "start	end	acc" > "$SUMMARY_CSV"

# Loop over segments
for (( skip=0; skip<TOTAL; skip+=STEP )); do
  limit=$STEP
  if (( skip + STEP > TOTAL )); then
    limit=$(( TOTAL - skip ))
  fi
  echo "Processing samples [${skip} .. $((skip + limit - 1))]"
  # Run harness and capture into a temporary segment file
  SEG_TMP="$OUTPUT_DIR/window_${skip}_${skip+limit-1}.json"
  python3 ./evaluate/ane/evaluate_with_harness.py \
    --model "$MODEL_PATH" \
    --tasks $TASK \
    --batch-size $BATCH_SIZE \
    --limit $limit \
    --skip $skip \
    --output-dir "$OUTPUT_DIR" \
    --output-path "$SEG_TMP"
  if [[ ! -f "$SEG_TMP" ]]; then
    echo "Error: expected output file not found: $SEG_TMP" >&2
    exit 1
  fi
  # Extract accuracy (first key matching 'acc' or 'acc,')
  acc=$(jq -r --arg task "$TASK" '
    .[$task]
    | to_entries
    | map(select(.key | test("^acc($|,)") ))
    | first.value
  ' "$SEG_TMP")
  echo -e "${skip}	$((skip + limit - 1))	${acc}" >> "$SUMMARY_CSV"
  # Append this segment's JSON under its key to the combined results
  segment_key="${skip}_to_$((skip+limit-1))"
  jq --arg key "$segment_key" --slurpfile seg "$SEG_TMP" \
     '.[$key] = $seg[0]' \
     "$COMBINED_JSON" > "$COMBINED_JSON.tmp" && mv "$COMBINED_JSON.tmp" "$COMBINED_JSON"
done

echo
echo "Worst $WORST_COUNT segments by accuracy (lowest first):"
sort -k3,3n "$SUMMARY_CSV" | head -n "$WORST_COUNT"