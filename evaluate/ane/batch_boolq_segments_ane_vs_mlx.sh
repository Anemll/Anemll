#!/usr/bin/env bash
# Batch-run BoolQ evaluation in fixed-size window segments for both ANE and MLX models and compare results.

set -euo pipefail

# Default parameters (adjust as needed)
MODEL_PATH=""
MLX_MODEL_PATH=""
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
  echo "Usage: $0 --model <ane_model_dir> --mlx-model <mlx_model> [--output-dir <dir>] [--step <n>] [--worst <k>] [--total <n>]" >&2
  echo "  --model:    ANE model directory" >&2
  echo "  --mlx-model: MLX model path (e.g., Qwen/Qwen2.5-0.5B)" >&2
  echo "  --total:    total number of examples (skip auto-detection via 'datasets' pkg)" >&2
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_PATH="$2"; shift 2;;
    --mlx-model)
      MLX_MODEL_PATH="$2"; shift 2;;
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

if [[ -z "$MODEL_PATH" ]] || [[ -z "$MLX_MODEL_PATH" ]]; then
  usage
fi

# Create output dir and initialize combined JSON files
mkdir -p "$OUTPUT_DIR"
ANE_COMBINED_JSON="$OUTPUT_DIR/eval_ane_$(basename "$MODEL_PATH")_${NUM_SHOTS}shot_${TASK}.json"
MLX_COMBINED_JSON="$OUTPUT_DIR/eval_mlx_$(basename "$MLX_MODEL_PATH" | tr '/' '_')_${NUM_SHOTS}shot_${TASK}.json"
echo "{}" > "$ANE_COMBINED_JSON"
echo "{}" > "$MLX_COMBINED_JSON"

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

# Summary file headers
ANE_SUMMARY_CSV="$OUTPUT_DIR/ane_boolq_segments_summary.tsv"
MLX_SUMMARY_CSV="$OUTPUT_DIR/mlx_boolq_segments_summary.tsv"
COMPARISON_CSV="$OUTPUT_DIR/ane_vs_mlx_comparison.tsv"
echo -e "start\tend\tacc" > "$ANE_SUMMARY_CSV"
echo -e "start\tend\tacc" > "$MLX_SUMMARY_CSV"
echo -e "start\tend\tane_acc\tmlx_acc\tdiff" > "$COMPARISON_CSV"

# Loop over segments
for (( skip=0; skip<TOTAL; skip+=STEP )); do
  limit=$STEP
  if (( skip + STEP > TOTAL )); then
    limit=$(( TOTAL - skip ))
  fi
  echo "Processing samples [${skip} .. $((skip + limit - 1))]"
  
  # Run ANE evaluation
  ANE_SEG_TMP="$OUTPUT_DIR/ane_window_${skip}_$((skip+limit-1)).json"
  echo "  Running ANE evaluation..."
  python3 ./evaluate/ane/evaluate_with_harness.py \
    --model "$MODEL_PATH" \
    --tasks $TASK \
    --batch-size $BATCH_SIZE \
    --limit $limit \
    --skip $skip \
    --output-dir "$OUTPUT_DIR" \
    --output-path "$ANE_SEG_TMP"
  if [[ ! -f "$ANE_SEG_TMP" ]]; then
    echo "Error: expected ANE output file not found: $ANE_SEG_TMP" >&2
    exit 1
  fi
  
  # Run MLX evaluation using simple script (note: different scoring method than ANE)
  MLX_SEG_TMP="$OUTPUT_DIR/mlx_window_${skip}_$((skip+limit-1)).json"
  echo "  Running MLX evaluation..."
  python3 ./tests/dev/mlx_simple_eval.py \
    --model "$MLX_MODEL_PATH" \
    --skip $skip \
    --limit $limit \
    --output "$MLX_SEG_TMP"
  if [[ ! -f "$MLX_SEG_TMP" ]]; then
    echo "Error: expected MLX output file not found: $MLX_SEG_TMP" >&2
    exit 1
  fi
  
  # Extract accuracies
  ane_acc=$(jq -r --arg task "$TASK" '
    .[$task]
    | to_entries
    | map(select(.key | test("^acc($|,)") ))
    | first.value
  ' "$ANE_SEG_TMP")
  
  mlx_acc=$(jq -r --arg task "$TASK" '
    .[$task]
    | to_entries
    | map(select(.key | test("^acc($|,)") ))
    | first.value
  ' "$MLX_SEG_TMP")
  
  # Calculate difference (ANE - MLX)
  diff=$(python3 -c "print(f'{float(\"$ane_acc\") - float(\"$mlx_acc\"):.4f}')")
  
  echo -e "${skip}\t$((skip + limit - 1))\t${ane_acc}" >> "$ANE_SUMMARY_CSV"
  echo -e "${skip}\t$((skip + limit - 1))\t${mlx_acc}" >> "$MLX_SUMMARY_CSV"
  echo -e "${skip}\t$((skip + limit - 1))\t${ane_acc}\t${mlx_acc}\t${diff}" >> "$COMPARISON_CSV"
  
  echo "  ANE: ${ane_acc}, MLX: ${mlx_acc}, Diff: ${diff}"
  
  # Append segments to combined JSON files
  segment_key="${skip}_to_$((skip+limit-1))"
  jq --arg key "$segment_key" --slurpfile seg "$ANE_SEG_TMP" \
     '.[$key] = $seg[0]' \
     "$ANE_COMBINED_JSON" > "$ANE_COMBINED_JSON.tmp" && mv "$ANE_COMBINED_JSON.tmp" "$ANE_COMBINED_JSON"
  jq --arg key "$segment_key" --slurpfile seg "$MLX_SEG_TMP" \
     '.[$key] = $seg[0]' \
     "$MLX_COMBINED_JSON" > "$MLX_COMBINED_JSON.tmp" && mv "$MLX_COMBINED_JSON.tmp" "$MLX_COMBINED_JSON"
done

echo
echo "Worst $WORST_COUNT ANE segments by accuracy (lowest first):"
sort -k3,3n "$ANE_SUMMARY_CSV" | head -n "$WORST_COUNT"

echo
echo "Worst $WORST_COUNT MLX segments by accuracy (lowest first):"
sort -k3,3n "$MLX_SUMMARY_CSV" | head -n "$WORST_COUNT"

echo
echo "Biggest divergences (ANE - MLX, most negative first):"
sort -k5,5n "$COMPARISON_CSV" | head -n "$WORST_COUNT"

echo
echo "Biggest divergences (ANE - MLX, most positive first):"
sort -k5,5nr "$COMPARISON_CSV" | head -n "$WORST_COUNT"