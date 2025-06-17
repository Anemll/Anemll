#!/usr/bin/env python3
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname('.'), 'anemll'))
from anemll.models.qwen_model import *

model_path = '~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/'
model_dirs = glob.glob(os.path.expanduser(model_path + '*'))
model_dir = model_dirs[0]

config_path = os.path.join(model_dir, 'config.json')
config = QwenConfig.from_json(config_path)

print(f'Config values:')
print(f'  hidden_size: {config.hidden_size}')  
print(f'  num_attention_heads: {config.num_attention_heads}')
print(f'  num_key_value_heads: {config.num_key_value_heads}')
print(f'  head_dim (calculated): {config.hidden_size // config.num_attention_heads}')
print(f'  head_dim (from config): {getattr(config, "head_dim", "not set")}')
print(f'  Expected total: num_heads * head_dim = {config.num_attention_heads} * {config.hidden_size // config.num_attention_heads} = {config.num_attention_heads * (config.hidden_size // config.num_attention_heads)}')
print(f'  Actual problem: 10240 / 1 / 1 = 10240, expected 2048')
print(f'  Ratio: 10240 / 2048 = {10240 / 2048}')

# Check what the real config says
import json
with open(config_path, 'r') as f:
    raw_config = json.load(f)
    
print(f'\nRaw config.json:')
for key in ['hidden_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim']:
    if key in raw_config:
        print(f'  {key}: {raw_config[key]}')
    else:
        print(f'  {key}: not present') 