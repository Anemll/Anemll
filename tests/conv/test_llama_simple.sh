#!/bin/bash

# Simple test script for LLaMA model conversion
# Tests conversion and inference for Meta-Llama-3.2-1B

set -e

echo "=== LLaMA Model Conversion Test ==="

# Use /tmp for output
OUTPUT_DIR="/tmp/test-llama-simple-$(date +%s)"

echo "Model: ~/Models/HF/Meta-Llama-3.2-1B"
echo "Output directory: $OUTPUT_DIR"

# Check if model exists
MODEL_PATH="$HOME/Models/HF/Meta-Llama-3.2-1B"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Run the conversion
echo -e "\nConverting LLaMA model..."
./anemll/utils/convert_model.sh \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_DIR" \
    --chunk 1 \
    --lut1 "" \
    --lut2 "" \
    --lut3 "" \
    --context 512

echo -e "\nConversion complete!"
echo "Output in: $OUTPUT_DIR"

# Test with Python chat
echo -e "\nTesting with Python chat..."
python3 tests/chat.py --meta "$OUTPUT_DIR/meta.yaml" --prompt "What is machine learning?" --max-tokens 50

# Test with Swift CLI
echo -e "\nTesting with Swift CLI..."
cd anemll-swift-cli && swift run anemllcli \
    --meta "$OUTPUT_DIR/meta.yaml" \
    --prompt "Explain the Apple Neural Engine (ANE) in depth: define its purpose, architecture, and integration within Apple silicon; trace its development timeline from the first A11 Bionic release to the present, highlighting key milestones; describe practical workflows for developers to leverage ANE via Core ML, Metal Performance Shaders, and third-party frameworks, noting tooling prerequisites and deployment tips; compile a numbered list of significant United States patents explicitly covering ANE design, optimization, or usage, citing patent numbers, titles, filing years, and inventors; spotlight principal Apple engineers, executives, and researchers who drove ANE conception, implementation, and ongoing innovation across multiple hardware generations globally." \
    --max-tokens 50

echo -e "\nTest complete!"
echo "To clean up: rm -rf $OUTPUT_DIR"