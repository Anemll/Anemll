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
from pathlib import Path

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
            "name": "Qwen2.5 0.5B Instruct",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "output": "/tmp/test-qwen25-0.5b",
            "chunks": "1"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing {test_case['name']} ---")
        
        try:
            # Run the test script
            cmd = [
                str(test_script),
                test_case["model"],
                test_case["output"],
                test_case["chunks"]
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
    
    print("\n=== All Qwen 2.5 tests completed successfully! ===")
    return 0

if __name__ == "__main__":
    sys.exit(run_qwen25_tests()) 