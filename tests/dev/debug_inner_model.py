#!/usr/bin/env python3

import os
import torch
import sys

# Set environment variable BEFORE importing the module
os.environ['ENABLE_SP_QUANT'] = '1'

# Add project root to path
sys.path.append('/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def debug_inner_model():
    print("Debugging inner model parameter names...")
    
    # Create ANEMLL model
    config = Qwen25Config()
    anemll_model = Qwen25ForCausalLM(config)
    
    print("=== OUTER MODEL (Qwen25ForCausalLM) ===")
    outer_param_names = list(anemll_model.named_parameters())
    print("First 10 parameter names:")
    for name, param in outer_param_names[:10]:
        print(f"  {name}: {param.shape}")
    
    print(f"\nTotal outer parameters: {len(outer_param_names)}")
    
    print("\n=== INNER MODEL (self.model) ===")
    inner_param_names = list(anemll_model.model.named_parameters())
    print("First 10 parameter names:")
    for name, param in inner_param_names[:10]:
        print(f"  {name}: {param.shape}")
    
    print(f"\nTotal inner parameters: {len(inner_param_names)}")

if __name__ == "__main__":
    debug_inner_model()