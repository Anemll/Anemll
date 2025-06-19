#!/usr/bin/env python3
"""Debug KV cache shapes to find the issue."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob

# Load model quickly
model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))

print(f"Config values:")
print(f"  context_length: {getattr(config, 'context_length', 'NOT SET')}")
print(f"  state_length: {getattr(config, 'state_length', 'NOT SET')}")
print(f"  num_attention_heads: {config.num_attention_heads}")
print(f"  num_key_value_heads: {config.num_key_value_heads}")
print(f"  head_dim: {getattr(config, 'head_dim', 'NOT SET')}")
print(f"  hidden_size: {config.hidden_size}")

model = QwenForCausalLM(config, enable_coreml=False)
model.load_pretrained_weights(model_path)

# Reset cache and do single token
model.model.kv_cache_0.zero_()

# Single token forward to populate cache
test_token = 785  # "The"
input_ids = torch.tensor([[test_token]], dtype=torch.long, device=TEST_DEVICE)
position_ids = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)
current_pos = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)
update_mask = torch.ones((1, 1, getattr(config, 'context_length', 512), 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
causal_mask = torch.zeros((1, 1, 1, getattr(config, 'context_length', 512)), dtype=MODEL_DTYPE, device=TEST_DEVICE)

print(f"\nFirst forward pass to populate cache...")
with torch.no_grad():
    _ = model(
        input_ids=input_ids,
        update_mask=update_mask,
        position_ids=position_ids,
        causal_mask=causal_mask,
        current_pos=current_pos,
        IN_PREFILL=False
    )

print(f"Cache populated. Now test retrieval...")

# Now debug the cache retrieval in forward_regular
layer = model.model.layers[0]  # First layer
print(f"\nLayer 0 attention config:")
print(f"  num_heads: {layer.self_attn.num_heads}")
print(f"  num_kv_heads: {layer.self_attn.num_kv_heads}")
print(f"  head_dim: {layer.self_attn.head_dim}")

# Get KV cache
key_idx = 0
value_idx = 0 + config.num_hidden_layers
kv_cache = model.model.kv_cache_0

print(f"\nKV cache shapes:")
print(f"  Full cache shape: {kv_cache.shape}")
print(f"  Key cache [layer 0] shape: {kv_cache[key_idx].shape}")
print(f"  Value cache [layer 0] shape: {kv_cache[value_idx].shape}")

# Test the slicing that causes issues
context_length = getattr(config, 'context_length', 512)
print(f"\nTesting context_length slicing (context_length={context_length}):")
K_layer_cache = kv_cache[key_idx]
V_layer_cache = kv_cache[value_idx]
print(f"  Before slicing - K: {K_layer_cache.shape}, V: {V_layer_cache.shape}")

K_layer_cache_sliced = K_layer_cache[..., :context_length, :]
V_layer_cache_sliced = V_layer_cache[..., :context_length, :]
print(f"  After slicing - K: {K_layer_cache_sliced.shape}, V: {V_layer_cache_sliced.shape}")

# Test repeat_kv
n_rep = layer.self_attn.num_heads // layer.self_attn.num_kv_heads
print(f"\nTesting repeat_kv (n_rep={n_rep}):")
key_states = layer.self_attn.repeat_kv(K_layer_cache_sliced, n_rep)
value_states = layer.self_attn.repeat_kv(V_layer_cache_sliced, n_rep)
print(f"  After repeat_kv - K: {key_states.shape}, V: {value_states.shape}")

print(f"\nThis should give us the total size for reshaping:")
expected_elements = key_states.shape[0] * key_states.shape[1] * key_states.shape[2]
print(f"  Expected elements for attention output: {expected_elements}")
print(f"  Expected after reshape: [1, 1, {layer.self_attn.num_heads * layer.self_attn.head_dim}] = [1, 1, {layer.self_attn.num_heads * layer.self_attn.head_dim}]") 