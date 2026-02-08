#!/bin/bash
#
# Convert Qwen2.5-0.5B-Instruct to CoreML (Monolithic, FP16)
#
# Model: Qwen/Qwen2.5-0.5B-Instruct
# Context: 2048
# Quantization: FP16 (no LUT)
# Export: Monolithic (single file with embeddings + FFN + LM head)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR="/Volumes/Models/ANE/qwen25_0.5b_instruct_ctx2048_LUT6_mono"
CONTEXT_LENGTH=2048
BATCH_SIZE=64

echo "=========================================="
echo "Qwen2.5-0.5B-Instruct Monolithic Conversion"
echo "=========================================="
echo "Model:    $MODEL"
echo "Output:   $OUTPUT_DIR"
echo "Context:  $CONTEXT_LENGTH"
echo "Batch:    $BATCH_SIZE"
echo "LUT:      6,4 (6-bit, per_channel=4)"
echo "Export:   Monolithic"
echo "=========================================="
echo ""

# Run conversion using convert_monolith.sh
# LUT 6,4 = 6-bit with per_channel group size 4
# Embeddings will use FP16 (no LUT) - handled by converter
"$PROJECT_ROOT/anemll/utils/convert_monolith.sh" \
    --model "$MODEL" \
    --output "$OUTPUT_DIR" \
    --context $CONTEXT_LENGTH \
    --batch $BATCH_SIZE \
    --lut 6,4 \
    "$@"

echo ""
echo "Conversion complete!"
echo ""
echo "To test the model:"
echo "  python tests/chat.py --meta $OUTPUT_DIR/meta.yaml"
