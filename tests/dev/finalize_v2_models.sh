#!/bin/bash
# Finalize V2 models: combine, compile, and prepare for testing
#
# Usage:
#   ./tests/dev/finalize_v2_models.sh /tmp/v2_test
#   ./tests/dev/finalize_v2_models.sh /tmp/v2_test --lut 4
#   ./tests/dev/finalize_v2_models.sh /tmp/v2_test --lut 4 --test

set -e

# Default values
OUTPUT_DIR="${1:-.}"
LUT_BITS=""
RUN_TEST=false
PREFIX="qwen"

# Parse arguments
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --lut)
            LUT_BITS="$2"
            shift 2
            ;;
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --test)
            RUN_TEST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate environment
source env-anemll/bin/activate

echo "=============================================="
echo "Finalizing V2 Models"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "LUT bits: ${LUT_BITS:-auto-detect}"
echo "Prefix: $PREFIX"
echo ""

# Auto-detect LUT bits if not specified
if [[ -z "$LUT_BITS" ]]; then
    if ls "$OUTPUT_DIR"/${PREFIX}_FFN_lut*_chunk_*.mlpackage 1>/dev/null 2>&1; then
        LUT_BITS=$(ls "$OUTPUT_DIR"/${PREFIX}_FFN_lut*_chunk_*.mlpackage | head -1 | sed 's/.*lut\([0-9]*\).*/\1/')
        echo "Auto-detected LUT bits: $LUT_BITS"
    else
        echo "Error: Could not auto-detect LUT bits. Use --lut option."
        exit 1
    fi
fi

# Step 1: Combine FFN + Prefill
echo ""
echo "Step 1: Combining FFN + Prefill..."
python anemll/utils/combine_models.py \
    --input "$OUTPUT_DIR" \
    --prefix "$PREFIX" \
    --output "$OUTPUT_DIR" \
    --chunk 1 \
    --lut "$LUT_BITS"

# Step 2: Compile combined model
echo ""
echo "Step 2: Compiling combined model..."
COMBINED_MODEL="${PREFIX}_FFN_PF_lut${LUT_BITS}_chunk_01of01.mlpackage"
if [[ -d "$OUTPUT_DIR/$COMBINED_MODEL" ]]; then
    xcrun coremlcompiler compile "$OUTPUT_DIR/$COMBINED_MODEL" "$OUTPUT_DIR/"
    echo "Compiled: $COMBINED_MODEL"
else
    echo "Error: Combined model not found: $OUTPUT_DIR/$COMBINED_MODEL"
    exit 1
fi

# Step 3: Update meta.yaml
echo ""
echo "Step 3: Checking meta.yaml..."
META_FILE="$OUTPUT_DIR/meta.yaml"
if [[ -f "$META_FILE" ]]; then
    # Check if ffn entry is correct
    EXPECTED_FFN="${PREFIX}_FFN_PF_lut${LUT_BITS}_chunk_01of01.mlmodelc"
    if grep -q "ffn:.*$EXPECTED_FFN" "$META_FILE"; then
        echo "meta.yaml already has correct ffn entry"
    else
        echo "Updating ffn entry in meta.yaml..."
        sed -i '' "s|ffn:.*|ffn: $EXPECTED_FFN|" "$META_FILE"
    fi
else
    echo "Warning: meta.yaml not found"
fi

# Step 4: List final models
echo ""
echo "Step 4: Final models in $OUTPUT_DIR:"
echo ""
ls -la "$OUTPUT_DIR"/*.mlmodelc 2>/dev/null | awk '{print "  " $NF ": " $5/1024/1024 " MB"}' || echo "  No compiled models found"

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="

# Step 5: Test if requested
if $RUN_TEST; then
    echo ""
    echo "Step 5: Running test..."
    echo "What is the capital of France?" | python tests/chat.py --meta "$META_FILE" 2>&1 | tail -20
fi

echo ""
echo "To test manually:"
echo "  echo 'Hello' | python tests/chat.py --meta $META_FILE"
