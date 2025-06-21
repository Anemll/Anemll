#!/usr/bin/env python3
#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

"""
Test script for Qwen model conversion and inference with 4K context.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_qwen_tests():
    """Run Qwen model tests with 4K context size"""
    
    # Get the project root directory
    script_dir = Path(__file__).parent.parent
    project_root = script_dir.parent
    test_script = project_root / "tests" / "conv" / "test_hf_model_custom_context.sh"
    
    # Create custom test script with 4K context
    custom_script_content = """#!/bin/bash
set -e

MODEL_NAME="$1"
OUTPUT_DIR="$2"
NUM_CHUNKS="$3"
CONTEXT_SIZE="$4"

echo "=== HuggingFace Model Conversion Test (Custom Context) ==="
echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Number of chunks: $NUM_CHUNKS"
echo "Context size: $CONTEXT_SIZE"

# Activate virtual environment
if [ -f "env-anemll/bin/activate" ]; then
    echo "Activating env-anemll virtual environment..."
    source env-anemll/bin/activate
elif [ -f "anemll-env/bin/activate" ]; then
    echo "Activating anemll-env virtual environment..."
    source anemll-env/bin/activate
fi

# Download model
echo -e "\\nDownloading model from HuggingFace..."
MODEL_PATH=$(python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import glob
import sys

model_name = '$MODEL_NAME'

try:
    hf_token = None
    token_file = os.path.expanduser('~/.cache/huggingface/token')
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            hf_token = f.read().strip()
    
    print(f'Downloading model {model_name}...', file=sys.stderr)
    if hf_token:
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
    cache_model_name = 'models--' + model_name.replace('/', '--')
    model_dirs = glob.glob(os.path.join(cache_dir, cache_model_name, 'snapshots', '*'))
    
    if model_dirs:
        model_path = model_dirs[0]
        print(f'Model downloaded to: {model_path}', file=sys.stderr)
        print(model_path)
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

# Run the conversion with custom context size
echo -e "\\nConverting $MODEL_NAME with context size $CONTEXT_SIZE..."
./anemll/utils/convert_model.sh \\
    --model "$MODEL_PATH" \\
    --output "$OUTPUT_DIR" \\
    --chunk "$NUM_CHUNKS" \\
    --lut1 "" \\
    --lut2 "" \\
    --lut3 "" \\
    --context "$CONTEXT_SIZE"

echo -e "\\nConversion complete!"
echo "Output in: $OUTPUT_DIR"

# Test with Python chat
echo -e "\\nTesting with Python chat..."
python3 tests/chat.py --meta "$OUTPUT_DIR/meta.yaml" --prompt "What is machine learning?" --max-tokens 50

echo -e "\\nTest complete!"
"""
    
    test_script.write_text(custom_script_content)
    test_script.chmod(0o755)
    
    print("=== Qwen Model Test Suite (4K Context) ===")
    
    # Test different Qwen models with 4K context
    test_cases = [
        {
            "name": "Qwen3 0.6B (4K context)",
            "model": "Qwen/Qwen3-0.6B",
            "output": "/tmp/test-qwen-0.6b-4k",
            "chunks": "1"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing {test_case['name']} ---")
        
        try:
            # Run the test script with 4K context
            cmd = [
                str(test_script),
                test_case["model"],
                test_case["output"],
                test_case["chunks"],
                "4000"  # 4K context size
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, cwd=project_root)
            
            if result.returncode == 0:
                print(f"✓ {test_case['name']} test passed")
            else:
                print(f"✗ {test_case['name']} test failed")
                return 1
                
        except subprocess.CalledProcessError as e:
            print(f"✗ {test_case['name']} test failed with error: {e}")
            return 1
        except KeyboardInterrupt:
            print(f"\n✗ {test_case['name']} test interrupted by user")
            return 1
    
    print("\n=== All Qwen tests completed successfully! ===")
    return 0

if __name__ == "__main__":
    sys.exit(run_qwen_tests())