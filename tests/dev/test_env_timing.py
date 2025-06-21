#!/usr/bin/env python3
"""
Test to verify environment variable timing for ENABLE_SP_QUANT.
"""

import os
import sys

# CRITICAL: Set environment variable BEFORE any imports
os.environ['ENABLE_SP_QUANT'] = '1'
if 'SKIP_SP_FORWARD' in os.environ:
    del os.environ['SKIP_SP_FORWARD']

print(f"Environment variable set: ENABLE_SP_QUANT = {os.environ.get('ENABLE_SP_QUANT')}")

# Now import the model module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config, ENABLE_SP_QUANT

print(f"Module ENABLE_SP_QUANT value: {ENABLE_SP_QUANT}")

# Quick test to create model and check if it has scale buffers
from huggingface_hub import snapshot_download

model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
model_path = snapshot_download(model_name)

config = Qwen25Config.from_json(f'{model_path}/config.json')
model = Qwen25ForCausalLM(config, disable_kv_cache=False)

# Check if model has quantization scale buffers
layer0_attn = model.model.layers[0].self_attn
has_q_input_scale = hasattr(layer0_attn, 'q_proj_input_scale')
has_q_output_scale = hasattr(layer0_attn, 'q_proj_output_scale')

print(f"Model has q_proj_input_scale buffer: {has_q_input_scale}")
print(f"Model has q_proj_output_scale buffer: {has_q_output_scale}")

if has_q_input_scale and has_q_output_scale:
    print("✅ SUCCESS: SP quantization is properly enabled!")
    
    # Load weights to test scale loading
    print("\nLoading weights to test scale loading...")
    model.load_pretrained_weights(model_path)
    
    # Check if scales were actually loaded
    scale_count = 0
    for name, module in model.named_modules():
        for attr in ['gate_proj_input_scale', 'up_proj_input_scale', 'down_proj_input_scale', 
                     'q_proj_input_scale', 'k_proj_input_scale', 'v_proj_input_scale', 'o_proj_input_scale']:
            if hasattr(module, attr):
                scale = getattr(module, attr)
                scale_count += 1
                if scale_count <= 3:  # Show first 3
                    print(f"  {name}.{attr}: shape={scale.shape}, mean={scale.mean().item():.6f}")
    
    print(f"Total quantization scales loaded: {scale_count}")
    
    if scale_count > 0:
        print("✅ Quantization scales loaded successfully!")
    else:
        print("❌ No quantization scales loaded - there's still an issue")
        
else:
    print("❌ FAILED: SP quantization buffers not created")