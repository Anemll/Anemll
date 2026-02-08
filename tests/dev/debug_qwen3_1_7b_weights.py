#!/usr/bin/env python3
"""Debug script to check Qwen3-1.7B weight loading and dimensions."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import safetensors.torch
from transformers import AutoConfig
import torch

MODEL_ID = "Qwen/Qwen3-1.7B"

def main():
    print("=" * 70)
    print("Qwen3-1.7B Weight Loading Diagnostic")
    print("=" * 70)

    # Load config
    config = AutoConfig.from_pretrained(MODEL_ID)
    print(f"\nConfig from HuggingFace:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {getattr(config, 'head_dim', 'NOT SET')}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  vocab_size: {config.vocab_size}")

    # Expected dimensions
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    q_proj_dim = config.num_attention_heads * head_dim
    kv_proj_dim = config.num_key_value_heads * head_dim

    print(f"\n  Calculated head_dim: {head_dim}")
    print(f"  Q projection: {config.hidden_size} -> {q_proj_dim}")
    print(f"  K/V projection: {config.hidden_size} -> {kv_proj_dim}")
    print(f"  O projection: {q_proj_dim} -> {config.hidden_size}")

    # Find model path in cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots")
    if not os.path.exists(cache_dir):
        print(f"\nError: Model not found in cache: {cache_dir}")
        return

    snapshot_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    if not snapshot_dirs:
        print(f"\nError: No snapshots found in {cache_dir}")
        return

    model_path = os.path.join(cache_dir, snapshot_dirs[0])
    print(f"\nModel path: {model_path}")

    # Load weights
    state_dict = {}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            print(f"  Loading: {file}")
            state_dict.update(safetensors.torch.load_file(os.path.join(model_path, file)))

    print(f"\nTotal keys loaded: {len(state_dict)}")

    # Check attention weights
    print("\n" + "=" * 70)
    print("Attention Weight Dimensions (Layer 0):")
    print("=" * 70)

    attention_keys = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ]

    for key in attention_keys:
        if key in state_dict:
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
            # Verify dimensions
            proj_name = key.split('.')[-2]
            if proj_name == 'q_proj':
                expected_out = q_proj_dim
                expected_in = config.hidden_size
            elif proj_name in ('k_proj', 'v_proj'):
                expected_out = kv_proj_dim
                expected_in = config.hidden_size
            elif proj_name == 'o_proj':
                expected_out = config.hidden_size
                expected_in = q_proj_dim

            if shape[0] != expected_out or shape[1] != expected_in:
                print(f"    WARNING: Expected [{expected_out}, {expected_in}]!")
        else:
            print(f"  {key}: NOT FOUND")

    # Check QK norm weights (Qwen3 specific)
    print("\n" + "=" * 70)
    print("QK Norm Weights (Layer 0):")
    print("=" * 70)

    qk_norm_keys = [
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
    ]

    for key in qk_norm_keys:
        if key in state_dict:
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
            # QK norm should be head_dim sized
            if shape[0] != head_dim:
                print(f"    WARNING: Expected [{head_dim}]!")
        else:
            print(f"  {key}: NOT FOUND")

    # Check MLP weights
    print("\n" + "=" * 70)
    print("MLP Weight Dimensions (Layer 0):")
    print("=" * 70)

    mlp_keys = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]

    for key in mlp_keys:
        if key in state_dict:
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
        else:
            print(f"  {key}: NOT FOUND")

    # Check embeddings and LM head
    print("\n" + "=" * 70)
    print("Embeddings and LM Head:")
    print("=" * 70)

    embed_keys = [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]

    for key in embed_keys:
        if key in state_dict:
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
        else:
            print(f"  {key}: NOT FOUND (may be tied)")

    # Check if tie_word_embeddings is true
    if config.tie_word_embeddings:
        print(f"\n  NOTE: tie_word_embeddings=True, lm_head uses embed_tokens weights")

    # Test loading with our model
    print("\n" + "=" * 70)
    print("Testing Model Weight Loading:")
    print("=" * 70)

    try:
        from anemll.models.qwen_model import QwenConfig, QwenForCausalLM

        # Create config
        qwen_config = QwenConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            context_length=256,
            state_length=256,
        )

        print(f"  Creating QwenForCausalLM with config...")
        model = QwenForCausalLM(qwen_config, enable_coreml=True)

        print(f"  Loading weights...")
        success = model.load_pretrained_weights(model_path)

        if success:
            print("  SUCCESS: All weights loaded correctly!")
        else:
            print("  FAILED: Weight loading failed!")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
