#!/usr/bin/env python3
#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

"""
Test script for Qwen 2.5 model conversion and inference.
This script uses the generic test_hf_model.sh script for testing.
"""

import subprocess
import sys
import os
import json
import shutil
from pathlib import Path

def clean_quantization_config(model_id):
    """Remove quantization_config from config.json if it exists"""
    try:
        from huggingface_hub import snapshot_download
        
        # Download the model if not cached
        model_path = snapshot_download(repo_id=model_id)
        config_file = os.path.join(model_path, "config.json")
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if 'quantization_config' in config:
                print(f"Removing quantization_config from {config_file}")
                del config['quantization_config']
                
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                print("✓ quantization_config removed")
            else:
                print("No quantization_config found in config.json")
        
        return model_path
    except Exception as e:
        print(f"Error cleaning quantization config: {e}")
        return None

def run_qwen25_tests():
    """Run Qwen 2.5 model tests using test_hf_model.sh"""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    test_script = project_root / "tests" / "conv" / "test_hf_model.sh"
    
    if not test_script.exists():
        print(f"Error: Test script not found at {test_script}")
        return 1
    
    print("=== Qwen 2.5 Model Test Suite ===")
    print("Using test_hf_model.sh for model conversion and testing")
    
    # Test different Qwen 2.5 models
    test_cases = [
        {
            "name": "Qwen2.5 0.5B 4bit-PerTensor",
            "model": "smpanaro/Qwen2.5-0.5B-4bit-PerTensor",
            "output": "/tmp/test-qwen25-sp-quant-0.5b",
            "chunks": "1"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing {test_case['name']} ---")
        
        try:
            # Remove destination folder if it exists
            output_path = Path(test_case["output"])
            if output_path.exists():
                print(f"Removing existing output directory: {output_path}")
                shutil.rmtree(output_path)
                print("✓ Output directory removed")
            
            # Clean quantization_config from model
            print("Cleaning quantization config...")
            model_path = clean_quantization_config(test_case["model"])
            if not model_path:
                print(f"✗ Failed to clean quantization config for {test_case['model']}")
                return 1
            
            # Set environment variable for per-tensor quantization
            env = os.environ.copy()
            env['ENABLE_SP_QUANT'] = 'true'
            
            # Run the test script (LUT quantization already disabled in test_hf_model.sh)
            cmd = [
                str(test_script),
                test_case["model"],
                test_case["output"],
                test_case["chunks"]
            ]
            
            print(f"Running: {' '.join(cmd)}")
            print(f"  with ENABLE_SP_QUANT=true")
            print(f"  with LUT quantization disabled (--lut1=\"\" --lut2=\"\" --lut3=\"\")")
            result = subprocess.run(cmd, check=True, cwd=project_root, env=env)
            
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
    
    print("\n=== All Qwen 2.5 tests completed successfully! ===")
    return 0

if __name__ == "__main__":
    sys.exit(run_qwen25_tests()) 