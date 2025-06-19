#!/bin/bash

# Generic test script for HuggingFace model conversion
# Tests conversion and inference for any HuggingFace model
# Usage: ./tests/conv/test_hf_model.sh [model_name] [output_dir] [num_chunks]
# Example: ./tests/conv/test_hf_model.sh meta-llama/Llama-3.2-1B-Instruct
# Example: ./tests/conv/test_hf_model.sh Qwen/Qwen2.5-0.5B-Instruct /tmp/my-output
# Example: ./tests/conv/test_hf_model.sh meta-llama/Llama-3.2-8B-Instruct /tmp/llama8b 4

set -e

# Get model name from argument or use default
if [ -n "$1" ]; then
    MODEL_NAME="$1"
else
    MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
fi

# Extract model basename for output directory
# Convert model name to safe directory name (replace / with -)
MODEL_BASENAME=$(echo "$MODEL_NAME" | sed 's/\//-/g')

# Use provided output directory or construct from model name
if [ -n "$2" ]; then
    OUTPUT_DIR="$2"
else
    OUTPUT_DIR="/tmp/anemll-${MODEL_BASENAME}"
fi

# Get number of chunks from argument or use default
if [ -n "$3" ]; then
    NUM_CHUNKS="$3"
else
    NUM_CHUNKS="1"
fi

echo "=== HuggingFace Model Conversion Test ==="
echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Number of chunks: $NUM_CHUNKS"

# Activate virtual environment
if [ -f "env-anemll/bin/activate" ]; then
    echo "Activating env-anemll virtual environment..."
    source env-anemll/bin/activate
elif [ -f "anemll-env/bin/activate" ]; then
    echo "Activating anemll-env virtual environment..."
    source anemll-env/bin/activate
else
    echo "Warning: No virtual environment found. Proceeding with system Python..."
fi

# Download model using HuggingFace transformers
echo -e "\nDownloading model from HuggingFace..."
MODEL_PATH=$(python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import glob
import sys

model_name = '$MODEL_NAME'

try:
    # Try to get HuggingFace token from environment or cache
    hf_token = None
    token_file = os.path.expanduser('~/.cache/huggingface/token')
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            hf_token = f.read().strip()
    
    # Download both model and tokenizer
    print(f'Downloading model {model_name}...', file=sys.stderr)
    if hf_token:
        print(f'Using HuggingFace token for authentication', file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f'Downloading tokenizer {model_name}...', file=sys.stderr)
    if hf_token:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get the cache directory
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')

    # Convert model name to cache format (replace / with --)
    cache_model_name = 'models--' + model_name.replace('/', '--')
    
    # Find the model directory in cache
    model_dirs = glob.glob(os.path.join(cache_dir, cache_model_name, 'snapshots', '*'))
    if model_dirs:
        model_path = model_dirs[0]
        print(f'Model and tokenizer downloaded to: {model_path}', file=sys.stderr)
        print(model_path)  # This is the only stdout output
    else:
        print(f'Model {model_name} not found in cache', file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f'Error downloading model {model_name}: {e}', file=sys.stderr)
    sys.exit(1)
")

if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Failed to download or locate model"
    exit 1
fi

echo "Model downloaded to: $MODEL_PATH"

# Check if output directory exists and warn user
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "\nWarning: Output directory already exists: $OUTPUT_DIR"
    echo "Press Ctrl+C to cancel or Enter to continue and overwrite..."
    read -r
fi

# Run the conversion
echo -e "\nConverting $MODEL_NAME..."
./anemll/utils/convert_model.sh \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_DIR" \
    --chunk "$NUM_CHUNKS" \
    --lut1 "" \
    --lut2 "" \
    --lut3 "" \
    --context 512

echo -e "\nConversion complete!"
echo "Output in: $OUTPUT_DIR"

# Test with Python chat
echo -e "\nTesting with Python chat..."
python3 tests/chat.py --meta "$OUTPUT_DIR/meta.yaml" --prompt "What is machine learning?" --max-tokens 50

# Test with Swift CLI (if available)
echo -e "\nTesting with Swift CLI..."
if [ -d "anemll-swift-cli" ]; then
    cd anemll-swift-cli && swift run anemllcli \
        --meta "$OUTPUT_DIR/meta.yaml" \
        --prompt "Explain the Apple Neural Engine (ANE) in one sentence." \
        --max-tokens 50
    cd ..
else
    echo "Swift CLI not available, skipping Swift test"
fi

echo -e "\nTest complete!"
echo "Output directory: $OUTPUT_DIR"
echo "To clean up: rm -rf $OUTPUT_DIR"