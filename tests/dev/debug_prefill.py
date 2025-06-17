#!/usr/bin/env python3
"""Debug script for KV cache prefill."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob

# Load model quickly
model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
model = QwenForCausalLM(config, enable_coreml=False)
model.load_pretrained_weights(model_path)

# Test prefill with debug
test_tokens = [100, 200, 300, 400, 500]
prefill_input_ids = torch.tensor([test_tokens], dtype=torch.long, device=TEST_DEVICE)
prefill_position_ids = torch.arange(len(test_tokens), dtype=torch.long, device=TEST_DEVICE)
prefill_causal_mask = torch.zeros((1, 1, len(test_tokens), config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)

print('Testing prefill...')
try:
    model.prefill_kv_cache(
        input_ids=prefill_input_ids,
        position_ids=prefill_position_ids,
        start_pos=0,
        causal_mask=prefill_causal_mask
    )
    print('Success!')
except Exception as e:
    print(f'Error: {e}') 