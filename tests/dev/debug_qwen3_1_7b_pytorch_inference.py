#!/usr/bin/env python3
"""Debug script to test Qwen3-1.7B inference with our PyTorch model implementation."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Disable network access to use local cache only
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoConfig, AutoTokenizer
import numpy as np

MODEL_ID = "Qwen/Qwen3-1.7B"
CONTEXT_LENGTH = 256
# Use converted model's tokenizer as fallback
TOKENIZER_PATH = "/Volumes/Models/ANE/qwen3_1.7b_ctx2048_FP16"

def main():
    print("=" * 70)
    print("Qwen3-1.7B PyTorch Inference Test (Our Implementation)")
    print("=" * 70)

    # Load config from local cache
    try:
        config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception as e:
        print(f"Could not load config from HF cache, using manual config")
        # Manual config for Qwen3-1.7B
        from transformers import PretrainedConfig
        config = type('Config', (), {
            'hidden_size': 2048,
            'num_attention_heads': 16,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'num_hidden_layers': 28,
            'intermediate_size': 6144,
            'vocab_size': 151936,
            'rms_norm_eps': 1e-06,
            'rope_theta': 1000000,
        })()
    print(f"\nConfig: hidden_size={config.hidden_size}, heads={config.num_attention_heads}, layers={config.num_hidden_layers}")

    # Load tokenizer from local path
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    except Exception as e:
        print(f"Error loading tokenizer from {TOKENIZER_PATH}: {e}")
        return
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")

    # Find model path
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots")
    snapshot_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    model_path = os.path.join(cache_dir, snapshot_dirs[0])
    print(f"Model path: {model_path}")

    # Create and load our model
    from anemll.models.qwen_model import QwenConfig, QwenForCausalLM, MODEL_DTYPE

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
        context_length=CONTEXT_LENGTH,
        state_length=CONTEXT_LENGTH,
    )

    print("\nCreating our QwenForCausalLM model...")
    model = QwenForCausalLM(qwen_config, enable_coreml=False, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    model.eval()

    # Test prompt
    prompt = "Who are you?"
    print(f"\nPrompt: {prompt}")

    # Tokenize
    tokens = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    print(f"Tokens: {tokens.tolist()}")
    print(f"Token text: {[tokenizer.decode([t]) for t in tokens[0].tolist()]}")

    # Create inputs
    seq_len = tokens.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.int32)

    # Create causal mask
    causal_mask = torch.zeros((1, 1, seq_len, CONTEXT_LENGTH), dtype=MODEL_DTYPE)
    for i in range(seq_len):
        for j in range(i + 1, CONTEXT_LENGTH):
            causal_mask[0, 0, i, j] = float('-inf')

    update_mask = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE)
    current_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

    print(f"\nInput shapes:")
    print(f"  tokens: {tokens.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  causal_mask: {causal_mask.shape}")
    print(f"  current_pos: {current_pos}")

    # Run prefill
    print("\n" + "=" * 70)
    print("Running Prefill...")
    print("=" * 70)

    with torch.no_grad():
        # Convert tokens to int32
        tokens_int32 = tokens.to(torch.int32)

        # Call model with prefill mode
        outputs = model(
            input_ids=tokens_int32,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=True,
        )

    # Combine logits if split
    if isinstance(outputs, tuple):
        print(f"Got {len(outputs)} logit tensors")
        logits = torch.cat(outputs, dim=-1)
    else:
        logits = outputs

    print(f"Logits shape: {logits.shape}")
    print(f"Logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")

    # Get top tokens
    probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
    top_probs, top_indices = torch.topk(probs, 10)

    print(f"\nTop 10 predictions for next token:")
    for i, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
        token_str = tokenizer.decode([idx])
        print(f"  {i+1}. '{token_str}' (id={idx}, prob={prob:.4f})")

    # Generate a few tokens
    print("\n" + "=" * 70)
    print("Generating tokens...")
    print("=" * 70)

    generated_tokens = tokens_int32.clone()
    current_position = seq_len

    for step in range(20):
        # Get next token
        next_token = top_indices[0].unsqueeze(0).unsqueeze(0).to(torch.int32)
        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        # Update inputs for next token
        new_position_ids = torch.tensor([current_position], dtype=torch.int32)
        new_causal_mask = torch.zeros((1, 1, 1, CONTEXT_LENGTH), dtype=MODEL_DTYPE)
        for j in range(current_position + 1, CONTEXT_LENGTH):
            new_causal_mask[0, 0, 0, j] = float('-inf')
        new_current_pos = torch.tensor([current_position], dtype=torch.int32)

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                update_mask=update_mask,
                position_ids=new_position_ids,
                causal_mask=new_causal_mask,
                current_pos=new_current_pos,
                IN_PREFILL=False,
            )

        if isinstance(outputs, tuple):
            logits = torch.cat(outputs, dim=-1)
        else:
            logits = outputs

        probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)

        next_token_id = top_indices[0].item()
        next_token_str = tokenizer.decode([next_token_id])
        print(f"  Step {step+1}: '{next_token_str}' (id={next_token_id}, prob={top_probs[0].item():.4f})")

        current_position += 1

        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            print("  [EOS reached]")
            break

    # Print full generated text
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    print(f"\nFull generated text:")
    print(f"  {generated_text}")

    # Compare with HF model (skip if offline)
    print("\n" + "=" * 70)
    print("Comparing with HuggingFace Model...")
    print("=" * 70)

    try:
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True)
        hf_model.eval()

        with torch.no_grad():
            hf_outputs = hf_model(tokens, use_cache=False)
            hf_logits = hf_outputs.logits

        print(f"HF Logits shape: {hf_logits.shape}")

        hf_probs = torch.softmax(hf_logits[0, -1, :].float(), dim=-1)
        hf_top_probs, hf_top_indices = torch.topk(hf_probs, 10)

        print(f"\nHF Top 10 predictions for next token:")
        for i, (prob, idx) in enumerate(zip(hf_top_probs.tolist(), hf_top_indices.tolist())):
            token_str = tokenizer.decode([idx])
            print(f"  {i+1}. '{token_str}' (id={idx}, prob={prob:.4f})")

        # Compare top tokens
        our_top5 = set(top_indices[:5].tolist())
        hf_top5 = set(hf_top_indices[:5].tolist())
        overlap = our_top5 & hf_top5
        print(f"\nOverlap in top-5: {len(overlap)}/5 tokens match")
    except Exception as e:
        print(f"Skipping HF comparison (offline mode): {e}")

if __name__ == "__main__":
    main()
