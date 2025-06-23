#!/usr/bin/env python3
"""
Check if biases are causing overflow in SP quantized model.
"""

import os
os.environ['ENABLE_SP_QUANT'] = '0'

import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
from huggingface_hub import snapshot_download

def check_biases():
    """Check bias values in the model."""
    print("="*80)
    print("CHECKING BIASES IN SP QUANTIZED MODEL")
    print("="*80)
    
    #model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    model_name = "Qwen/Qwen2.5-0.5B"

    model_path = snapshot_download(model_name)
    
    # Create and load model
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    
    print("\nChecking bias values...")
    
    # Check all biases in the model
    for name, param in model.named_parameters():
        if 'bias' in name:
            if param is not None:
                bias_max = param.max().item()
                bias_min = param.min().item()
                bias_mean = param.mean().item()
                bias_std = param.std().item()
                
                # Flag large biases
                if abs(bias_max) > 10 or abs(bias_min) > 10:
                    print(f"\n⚠️  LARGE BIAS: {name}")
                    print(f"   Shape: {param.shape}")
                    print(f"   Max: {bias_max:.6f}")
                    print(f"   Min: {bias_min:.6f}")
                    print(f"   Mean: {bias_mean:.6f}")
                    print(f"   Std: {bias_std:.6f}")
                    
                    # Show first few values
                    print(f"   First 5 values: {param.flatten()[:5].tolist()}")
                    
                    # Check for specific problematic values
                    if abs(bias_max) > 100:
                        print(f"   ❌ EXTREMELY LARGE BIAS VALUE!")

if __name__ == "__main__":
    check_biases()