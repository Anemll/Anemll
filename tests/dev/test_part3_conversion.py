#!/usr/bin/env python3
"""Test part 3 (LM head) conversion without printing huge tensors."""

import os
import sys

# Redirect stdout to prevent huge tensor printing
class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Test the conversion
try:
    print("Starting part 3 conversion test...")
    print("This will suppress tensor printing to avoid log overflow")
    
    # Import after setting up suppression
    from anemll.ane_converter.qwen_converter import test_conversion
    
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    output_dir = "/tmp/qwen-part3-test/"
    
    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    
    # Run conversion with output suppression
    print("Running conversion (output suppressed)...")
    with SuppressPrint():
        result = test_conversion(
            model_path=model_path,
            prefix="test_qwen",
            part="3",
            output_dir=output_dir,
            context_length=256,
            batch_size=64,
        )
    
    print("✅ Conversion completed successfully!")
    print(f"Result type: {type(result)}")
    
    # Check if output directory exists (.mlpackage is a directory)
    expected_dir = os.path.join(output_dir, "test_qwen_lm_head.mlpackage")
    if os.path.isdir(expected_dir):
        print(f"✅ Output directory created: {expected_dir}")
    else:
        print(f"❌ Expected output directory not found: {expected_dir}")
        
except Exception as e:
    print(f"❌ Error during conversion: {str(e)}")
    print("Error type:", type(e).__name__)
    # Don't print full traceback to avoid tensor printing