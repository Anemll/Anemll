#!/usr/bin/env python3
"""Quick script to check quantization scale values"""

import os
os.environ['ENABLE_SP_QUANT'] = '1'

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
from huggingface_hub import snapshot_download

# Load model
model_path = snapshot_download('smpanaro/Qwen2.5-0.5B-4bit-PerTensor')
config = Qwen25Config.from_json(f'{model_path}/config.json')
model = Qwen25ForCausalLM(config, disable_kv_cache=False)
model.load_pretrained_weights(model_path)

# Check specific scale values
layer0_attn = model.model.layers[0].self_attn
layer0_mlp = model.model.layers[0].mlp

print('Layer 0 Attention Scale Values:')
print(f'q_proj_input_scale: {layer0_attn.q_proj_input_scale.flatten()[:5].tolist()}')
print(f'q_proj_output_scale: {layer0_attn.q_proj_output_scale.flatten()[:5].tolist()}')
print(f'k_proj_input_scale: {layer0_attn.k_proj_input_scale.flatten()[:5].tolist()}')
print(f'k_proj_output_scale: {layer0_attn.k_proj_output_scale.flatten()[:5].tolist()}')

print('\nLayer 0 MLP Scale Values:')
print(f'gate_proj_input_scale: {layer0_mlp.gate_proj_input_scale.flatten()[:5].tolist()}')
print(f'gate_proj_output_scale: {layer0_mlp.gate_proj_output_scale.flatten()[:5].tolist()}')
print(f'up_proj_input_scale: {layer0_mlp.up_proj_input_scale.flatten()[:5].tolist()}')
print(f'up_proj_output_scale: {layer0_mlp.up_proj_output_scale.flatten()[:5].tolist()}')