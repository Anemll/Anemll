#!/usr/bin/env python3
"""Layer-by-layer comparison of ANEMLL vs HuggingFace Gemma3.

This test traces through each layer to identify exactly where divergence occurs.
"""

import sys
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from anemll.models.gemma3_model import (
    Gemma3ForCausalLM,
    Gemma3Config,
    MODEL_DTYPE,
)
from anemll.ane_converter.gemma3_converter import Gemma3Converter


def load_models(model_path, context_length=64):
    """Load both HF and ANEMLL models."""
    # Load HF model (bfloat16 for stability)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True
    )
    hf_model.eval()

    # Load ANEMLL model
    config = Gemma3Config.from_json(f'{model_path}/config.json')
    config.context_length = context_length
    config.state_length = context_length

    anemll_model = Gemma3ForCausalLM(config)
    converter = Gemma3Converter(anemll_model, context_length=context_length, batch_size=context_length)
    converter.load_weights_from_hf(model_path)
    anemll_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return hf_model, anemll_model, tokenizer


def compare_hidden_states(name, hf_hidden, anemll_hidden, threshold=1.0):
    """Compare hidden states and report differences."""
    # Convert to comparable format
    hf_h = hf_hidden.float().detach()
    anemll_h = anemll_hidden.float().detach()

    # Handle shape differences
    if hf_h.shape != anemll_h.shape:
        print(f"  {name}: SHAPE MISMATCH - HF={hf_h.shape}, ANEMLL={anemll_h.shape}")
        return False

    diff = (hf_h - anemll_h).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    status = "OK" if max_diff < threshold else "DIVERGED"
    print(f"  {name}: max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f} [{status}]")
    print(f"    HF:     mean={hf_h.mean():.4f}, std={hf_h.std():.4f}, max={hf_h.abs().max():.4f}")
    print(f"    ANEMLL: mean={anemll_h.mean():.4f}, std={anemll_h.std():.4f}, max={anemll_h.abs().max():.4f}")

    return max_diff < threshold


def compare_single_layer(hf_model, anemll_model, hidden_states_hf, hidden_states_anemll, layer_idx, position_ids):
    """Compare a single layer's computation step by step."""
    hf_layer = hf_model.model.layers[layer_idx]
    anemll_layer = anemll_model.model.layers[layer_idx]

    print(f"\n=== Layer {layer_idx} Detailed Comparison ===")

    # 1. Input layernorm
    hf_input_normed = hf_layer.input_layernorm(hidden_states_hf.to(torch.bfloat16))
    anemll_input_normed = anemll_layer.input_layernorm(hidden_states_anemll.to(MODEL_DTYPE))
    compare_hidden_states("input_layernorm", hf_input_normed, anemll_input_normed, threshold=0.1)

    # 2. Q/K/V projections
    hf_attn = hf_layer.self_attn
    anemll_attn = anemll_layer.self_attn

    # HF Q projection
    hf_q = hf_attn.q_proj(hf_input_normed)
    # ANEMLL Q projection (need to reshape for Conv2d)
    anemll_input_conv = anemll_input_normed.permute(0, 2, 1).unsqueeze(2)
    anemll_q = anemll_attn.q_proj(anemll_input_conv).squeeze(2).permute(0, 2, 1)
    compare_hidden_states("q_proj", hf_q, anemll_q, threshold=0.5)

    # HF K projection
    hf_k = hf_attn.k_proj(hf_input_normed)
    anemll_k = anemll_attn.k_proj(anemll_input_conv).squeeze(2).permute(0, 2, 1)
    compare_hidden_states("k_proj", hf_k, anemll_k, threshold=0.5)

    # HF V projection
    hf_v = hf_attn.v_proj(hf_input_normed)
    anemll_v = anemll_attn.v_proj(anemll_input_conv).squeeze(2).permute(0, 2, 1)
    compare_hidden_states("v_proj", hf_v, anemll_v, threshold=0.5)

    # 3. Reshape Q/K/V for attention
    batch_size, seq_len, _ = hf_q.shape
    num_heads = hf_model.config.num_attention_heads
    num_kv_heads = hf_model.config.num_key_value_heads
    head_dim = hf_model.config.head_dim

    # Reshape HF
    hf_q_reshaped = hf_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    hf_k_reshaped = hf_k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    hf_v_reshaped = hf_v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Reshape ANEMLL
    anemll_q_reshaped = anemll_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    anemll_k_reshaped = anemll_k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    anemll_v_reshaped = anemll_v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # 4. Q/K normalization
    hf_q_normed = hf_attn.q_norm(hf_q_reshaped)
    hf_k_normed = hf_attn.k_norm(hf_k_reshaped)

    anemll_q_normed = anemll_attn.q_norm(anemll_q_reshaped.to(MODEL_DTYPE))
    anemll_k_normed = anemll_attn.k_norm(anemll_k_reshaped.to(MODEL_DTYPE))

    compare_hidden_states("q_norm", hf_q_normed, anemll_q_normed, threshold=0.5)
    compare_hidden_states("k_norm", hf_k_normed, anemll_k_normed, threshold=0.5)

    # Return the normalized hidden states for next layer
    return hf_input_normed, anemll_input_normed


def main():
    model_path = '/Users/anemll/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3'
    context_length = 64

    print("Loading models...")
    hf_model, anemll_model, tokenizer = load_models(model_path, context_length)

    # Prepare input
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    seq_len = input_ids.shape[1]

    print(f"\nInput sequence length: {seq_len}")
    print(f"Tokens: {input_ids[0].tolist()}")

    # Get embeddings
    with torch.no_grad():
        # HF embeddings
        hf_embed = hf_model.model.embed_tokens(input_ids)
        hf_hidden = hf_embed * (hf_model.config.hidden_size ** 0.5)

        # ANEMLL embeddings
        anemll_embed = anemll_model.model.embed_tokens(input_ids)
        anemll_hidden = anemll_embed * anemll_model.model.embedding_scale
        anemll_hidden = anemll_hidden.to(MODEL_DTYPE)

    # Compare embeddings
    print("\n=== Embedding Comparison ===")
    compare_hidden_states("embeddings_raw", hf_embed, anemll_embed, threshold=0.001)
    compare_hidden_states("embeddings_scaled", hf_hidden, anemll_hidden.to(hf_hidden.dtype), threshold=0.01)

    # Position IDs
    position_ids = torch.arange(seq_len).unsqueeze(0)

    # Compare layer by layer
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 70)

    current_hf_hidden = hf_hidden
    current_anemll_hidden = anemll_hidden

    for layer_idx in range(min(5, len(hf_model.model.layers))):  # First 5 layers
        compare_single_layer(
            hf_model, anemll_model,
            current_hf_hidden, current_anemll_hidden,
            layer_idx, position_ids
        )


if __name__ == "__main__":
    main()
