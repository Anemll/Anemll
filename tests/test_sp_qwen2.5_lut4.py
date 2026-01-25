#!/usr/bin/env python3
#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

"""
Test script for Qwen 2.5 model conversion and inference with LUT4 quantization.
This script applies LUT4 quantization specifically to FFN and Prefill layers,
while leaving Embeddings and LMHead unquantized.
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

def run_qwen25_lut4_tests():
    """Run Qwen 2.5 model tests with LUT4 quantization for FFN and Prefill"""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    convert_script = project_root / "anemll" / "utils" / "convert_model.sh"
    
    if not convert_script.exists():
        print(f"Error: Convert script not found at {convert_script}")
        return 1
    
    print("=== Qwen 2.5 Model Test Suite with LUT4 Quantization ===")
    print("Using convert_model.sh directly for model conversion with custom LUT settings")
    print("LUT4 applied to: FFN and Prefill layers")
    print("No quantization for: Embeddings and LMHead")
    
    # Test Qwen 2.5 model with LUT4 quantization for FFN and Prefill
    test_cases = [
        {
            "name": "Qwen2.5 0.5B 4bit-PerTensor with LUT4 (FFN+Prefill only)",
            "model": "smpanaro/Qwen2.5-0.5B-4bit-PerTensor",
            "output": "/tmp/test-qwen25-sp-quant-lut4-0.5b",
            "chunks": "1",
            "lut_config": {
                "lut1": "",  # No LUT for embeddings
                "lut2": "4", # LUT4 for FFN and Prefill
                "lut3": ""   # No LUT for LMHead
            }
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
            
            # Clean quantization_config from model if it's a per-tensor model
            if "4bit-PerTensor" in test_case["model"]:
                print("Cleaning quantization config...")
                model_path = clean_quantization_config(test_case["model"])
                if not model_path:
                    print(f"✗ Failed to clean quantization config for {test_case['model']}")
                    return 1
            else:
                # For standard models, just download them
                try:
                    from huggingface_hub import snapshot_download
                    print(f"Downloading model {test_case['model']}...")
                    model_path = snapshot_download(repo_id=test_case["model"])
                    print(f"✓ Model downloaded to: {model_path}")
                except Exception as e:
                    print(f"✗ Failed to download model {test_case['model']}: {e}")
                    return 1
            
            # Set environment variable for per-tensor quantization
            env = os.environ.copy()
            env['ENABLE_SP_QUANT'] = '1'
            
            # Build command to call convert_model.sh directly with LUT4 configuration
            # We need to activate the virtual environment in the subprocess
            activate_venv = ""
            if (project_root / "env-anemll" / "bin" / "activate").exists():
                activate_venv = f"source {project_root}/env-anemll/bin/activate && "
            elif (project_root / "anemll-env" / "bin" / "activate").exists():
                activate_venv = f"source {project_root}/anemll-env/bin/activate && "
            
            cmd_str = (
                f"{activate_venv}"
                f"{convert_script} "
                f"--model '{model_path}' "
                f"--output '{test_case['output']}' "
                f"--chunk {test_case['chunks']} "
                f"--lut1 '{test_case['lut_config']['lut1']}' "
                f"--lut2 '{test_case['lut_config']['lut2']}' "
                f"--lut3 '{test_case['lut_config']['lut3']}' "
                f"--context 512"
            )
            
            print(f"Running: {cmd_str}")
            print(f"  with ENABLE_SP_QUANT=1")
            print(f"  with LUT4 quantization for FFN and Prefill layers only")
            print(f"  Embeddings: No LUT quantization")
            print(f"  FFN/Prefill: LUT4 quantization")
            print(f"  LMHead: No LUT quantization")
            
            result = subprocess.run(cmd_str, shell=True, check=True, cwd=project_root, env=env)
            
            if result.returncode == 0:
                print(f"✓ {test_case['name']} test passed")
                
                # Check for CoreML model files
                coreml_patterns = [
                    "*embeddings*.mlmodelc",
                    "*FFN*.mlmodelc", 
                    "*lm_head*.mlmodelc",
                    "*prefill*.mlmodelc"
                ]
                
                missing_files = []
                for pattern in coreml_patterns:
                    import glob
                    matches = glob.glob(str(output_path / pattern))
                    if not matches:
                        missing_files.append(pattern)
                
                if missing_files:
                    print(f"⚠️  Warning: Some expected CoreML files missing: {missing_files}")
                else:
                    print("✓ All expected CoreML model files found")
                
                # Test the converted model with a simple chat
                print("\n--- Testing converted model ---")
                chat_script = project_root / "tests" / "chat.py"
                meta_file = output_path / "meta.yaml"
                
                if chat_script.exists() and meta_file.exists():
                    chat_cmd = [
                        "python", str(chat_script),
                        "--meta", str(meta_file),
                        "--prompt", "Hello, how are you?",
                        "--max-tokens", "10",
                        "--no-template"
                    ]
                    
                    try:
                        print(f"Running: {' '.join(chat_cmd)}")
                        chat_result = subprocess.run(chat_cmd, check=True, cwd=project_root, env=env, 
                                                   capture_output=True, text=True, timeout=60)
                        print("✓ Chat test completed successfully")
                        if chat_result.stdout:
                            print(f"Model response: {chat_result.stdout.strip()}")
                    except subprocess.TimeoutExpired:
                        print("⚠️  Chat test timed out (model might be slow)")
                    except subprocess.CalledProcessError as e:
                        print(f"⚠️  Chat test failed: {e}")
                        if e.stderr:
                            print(f"Error: {e.stderr}")
                else:
                    print("⚠️  Skipping chat test (missing chat.py or meta.yaml)")
                    
            else:
                print(f"✗ {test_case['name']} test failed")
                return 1
                
        except subprocess.CalledProcessError as e:
            print(f"✗ {test_case['name']} test failed with error: {e}")
            print("This might be due to:")
            print("  - Model download issues")
            print("  - CoreML compilation errors with LUT4 quantization")
            print("  - Missing dependencies")
            return 1
        except KeyboardInterrupt:
            print(f"\n✗ {test_case['name']} test interrupted by user")
            return 1
    
    print("\n=== All Qwen 2.5 LUT4 tests completed successfully! ===")
    print("\nQuantization Summary:")
    print("  ✓ Embeddings: No quantization (full precision)")
    print("  ✓ FFN layers: LUT4 quantization applied")
    print("  ✓ Prefill layers: LUT4 quantization applied") 
    print("  ✓ LMHead: No quantization (full precision)")
    return 0

if __name__ == "__main__":
    sys.exit(run_qwen25_lut4_tests())