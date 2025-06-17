#!/usr/bin/env python3
"""Test part 2 (FFN) conversion."""

import os
import sys

try:
    print("Starting part 2 (FFN) conversion test...")
    
    # Import after setting up
    from anemll.ane_converter.qwen_converter import test_conversion
    
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    output_dir = "/tmp/qwen-part2-test/"
    
    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    
    # Run conversion
    print("Running conversion...")
    result = test_conversion(
        model_path=model_path,
        prefix="test_qwen",
        part="2",
        output_dir=output_dir,
        context_length=256,
        batch_size=64,
        num_chunks=2,  # Test with 2 chunks
    )
    
    print("✅ Conversion completed successfully!")
    print(f"Result type: {type(result)}")
    
    # Check if output directories exist (.mlpackage are directories)
    for i in range(2):
        expected_dir = os.path.join(output_dir, f"test_qwen_FFN_PF_chunk_{i+1:02d}of02.mlpackage")
        if os.path.isdir(expected_dir):
            print(f"✅ Output directory created: {expected_dir}")
        else:
            print(f"❌ Expected output directory not found: {expected_dir}")
        
except Exception as e:
    print(f"❌ Error during conversion: {str(e)}")
    print("Error type:", type(e).__name__)
    import traceback
    traceback.print_exc()