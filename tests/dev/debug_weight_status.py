#!/usr/bin/env python3

import os
import torch
import sys

# Set environment variable BEFORE importing the module
os.environ['ENABLE_SP_QUANT'] = '1'

# Add project root to path
sys.path.append('/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def debug_weight_status():
    print("Debugging weight loading status...")
    
    model_path = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    
    # Create ANEMLL model
    config = Qwen25Config()
    anemll_model = Qwen25ForCausalLM(config)
    
    # Load weights
    print("Loading weights...")
    result = anemll_model.load_pretrained_weights(model_path)
    
    print(f"Weight loading result: {result}")
    print("Done.")

if __name__ == "__main__":
    debug_weight_status()