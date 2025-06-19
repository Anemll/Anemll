#!/bin/bash

# Simple test script for Qwen model conversion
# Tests conversion and inference for Qwen3-0.6B

set -e

echo "=== Qwen Model Conversion Test ==="

# Use /tmp for output
OUTPUT_DIR="/tmp/test-qwen-simple-$(date +%s)"

# Model paths
MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/"

echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please download the model first using:"
    echo "python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')\""
    exit 1
fi

# Run the conversion
echo -e "\nConverting Qwen model..."
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