#!/usr/bin/env python3
"""
Debug why quantization scales aren't being loaded properly.
"""

import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
from huggingface_hub import snapshot_download
from safetensors import safe_open

def debug_scale_loading():
    """Debug the scale loading process step by step."""
    
    # Enable SP quantization
    os.environ['ENABLE_SP_QUANT'] = '1'
    if 'SKIP_SP_FORWARD' in os.environ:
        del os.environ['SKIP_SP_FORWARD']
    
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    model_path = snapshot_download(model_name)
    
    print("="*80)
    print("DEBUG: Scale Loading Process")
    print("="*80)
    
    # Load state dict manually to check
    state_dict = {}
    with safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    
    print(f"Total keys in state dict: {len(state_dict)}")
    
    # Collect scales like the model does
    input_scales = {}
    output_scales = {}
    codebooks = {}
    
    print("\n--- First pass: collecting scales and codebooks ---")
    for k, v in state_dict.items():
        if "input_scales" in k:
            input_scales[k] = v
            print(f"Found input scale: {k}, shape: {v.shape}")
        elif "output_scales" in k:
            output_scales[k] = v
            print(f"Found output scale: {k}, shape: {v.shape}")
        elif "codebook" in k:
            codebooks[k] = v
            print(f"Found codebook: {k}, shape: {v.shape}")
    
    print(f"\nCollected {len(input_scales)} input scales")
    print(f"Collected {len(output_scales)} output scales")
    print(f"Collected {len(codebooks)} codebooks")
    
    # Check a specific weight key
    test_weight_key = "model.layers.0.self_attn.q_proj.weight"
    print(f"\n--- Testing scale lookup for {test_weight_key} ---")
    
    # This is what the model code does
    output_scale_key = test_weight_key.replace(".weight", ".output_scales")
    input_scale_key = test_weight_key.replace(".weight", ".input_scales")
    codebook_key = test_weight_key.replace(".weight", ".codebook")
    
    print(f"Looking for output_scale_key: {output_scale_key}")
    print(f"Looking for input_scale_key: {input_scale_key}")
    print(f"Looking for codebook_key: {codebook_key}")
    
    print(f"output_scale_key in output_scales: {output_scale_key in output_scales}")
    print(f"input_scale_key in input_scales: {input_scale_key in input_scales}")
    print(f"codebook_key in codebooks: {codebook_key in codebooks}")
    
    if output_scale_key in output_scales and input_scale_key in input_scales:
        print("✓ Both scales found - defusion should happen")
        output_scale = output_scales[output_scale_key]
        input_scale = input_scales[input_scale_key]
        print(f"Output scale shape: {output_scale.shape}, values: {output_scale.flatten()[:5].tolist()}")
        print(f"Input scale shape: {input_scale.shape}, values: {input_scale.flatten()[:5].tolist()}")
    else:
        print("✗ Scales not found - defusion will NOT happen")
    
    # Now create model and see what happens
    print(f"\n--- Creating model and loading weights ---")
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    
    # Check if ENABLE_SP_QUANT is working
    from anemll.models.qwen2_5_model import ENABLE_SP_QUANT
    print(f"ENABLE_SP_QUANT value: {ENABLE_SP_QUANT}")
    
    # Check if model has scale buffers
    layer0_attn = model.model.layers[0].self_attn
    has_q_input_scale = hasattr(layer0_attn, 'q_proj_input_scale')
    has_q_output_scale = hasattr(layer0_attn, 'q_proj_output_scale')
    
    print(f"Model has q_proj_input_scale buffer: {has_q_input_scale}")
    print(f"Model has q_proj_output_scale buffer: {has_q_output_scale}")
    
    if has_q_input_scale:
        print(f"Initial q_proj_input_scale: {layer0_attn.q_proj_input_scale.flatten()[:5].tolist()}")
    if has_q_output_scale:
        print(f"Initial q_proj_output_scale: {layer0_attn.q_proj_output_scale.flatten()[:5].tolist()}")
    
    # Load weights and see what changes
    print("\nLoading pretrained weights...")
    model.load_pretrained_weights(model_path)
    
    if has_q_input_scale:
        print(f"After loading q_proj_input_scale: {layer0_attn.q_proj_input_scale.flatten()[:5].tolist()}")
    if has_q_output_scale:
        print(f"After loading q_proj_output_scale: {layer0_attn.q_proj_output_scale.flatten()[:5].tolist()}")

if __name__ == "__main__":
    debug_scale_loading()