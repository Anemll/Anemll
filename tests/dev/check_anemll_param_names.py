#!/usr/bin/env python3

import os
import torch
import sys

# Set environment variable BEFORE importing the module
os.environ['ENABLE_SP_QUANT'] = '1'

# Add project root to path
sys.path.append('/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def check_param_names():
    print("Checking ANEMLL parameter names...")
    
    config = Qwen25Config()
    anemll_model = Qwen25ForCausalLM(config)
    
    print("LayerNorm parameter names in ANEMLL model:")
    for name, param in anemll_model.named_parameters():
        if 'layernorm' in name or 'norm' in name:
            print(f"  {name}")
            if 'layers.0' in name:
                break  # Just show first layer examples

if __name__ == "__main__":
    check_param_names()