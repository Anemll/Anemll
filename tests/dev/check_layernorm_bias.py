#!/usr/bin/env python3
"""
Check if transformers LayerNorm actually has bias parameters.
"""

import os
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

def check_layernorm_bias():
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    
    # Load transformers model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Check first layer input layernorm
    hf_ln = hf_model.model.layers[0].input_layernorm
    
    print("Transformers LayerNorm details:")
    print(f"Type: {type(hf_ln)}")
    print(f"Has bias attribute: {hasattr(hf_ln, 'bias')}")
    
    if hasattr(hf_ln, 'bias'):
        bias = hf_ln.bias
        print(f"Bias is None: {bias is None}")
        if bias is not None:
            print(f"Bias shape: {bias.shape}")
            print(f"Bias dtype: {bias.dtype}")
            print(f"Bias mean: {bias.mean().item():.8f}")
            print(f"Bias std: {bias.std().item():.8f}")
            print(f"Bias first 10 values: {bias[:10].tolist()}")
            print(f"All bias values are zero: {torch.allclose(bias, torch.zeros_like(bias))}")
    
    # Check weight too
    weight = hf_ln.weight
    print(f"\nWeight shape: {weight.shape}")
    print(f"Weight dtype: {weight.dtype}")
    print(f"Weight mean: {weight.mean().item():.8f}")
    print(f"Weight std: {weight.std().item():.8f}")
    print(f"Weight first 10 values: {weight[:10].tolist()}")

if __name__ == "__main__":
    check_layernorm_bias()