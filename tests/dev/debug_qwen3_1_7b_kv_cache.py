#!/usr/bin/env python3
"""Debug KV cache handling in Qwen3-1.7B inference."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

MODEL_ID = "Qwen/Qwen3-1.7B"
TOKENIZER_PATH = "/Volumes/Models/ANE/qwen3_1.7b_ctx2048_FP16"
CONTEXT_LENGTH = 256

def main():
    print("=" * 70)
    print("Qwen3-1.7B KV Cache Debugging")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True)
    hf_model.eval()

    # Find model path
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots")
    snapshot_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    model_path = os.path.join(cache_dir, snapshot_dirs[0])

    # Create our model WITH KV cache enabled
    from anemll.models.qwen_model import QwenConfig, QwenForCausalLM, MODEL_DTYPE

    hf_config = hf_model.config
    qwen_config = QwenConfig(
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads),
        num_hidden_layers=hf_config.num_hidden_layers,
        intermediate_size=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        context_length=CONTEXT_LENGTH,
        state_length=CONTEXT_LENGTH,
    )

    # Test with KV cache DISABLED (reference - should work)
    print("\n" + "=" * 70)
    print("Test 1: KV Cache DISABLED (should work)")
    print("=" * 70)

    our_model_no_cache = QwenForCausalLM(qwen_config, enable_coreml=False, disable_kv_cache=True)
    our_model_no_cache.load_pretrained_weights(model_path)
    our_model_no_cache.eval()

    prompt = "Hello"
    tokens = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    tokens_int32 = tokens.to(torch.int32)
    seq_len = tokens.shape[1]
    print(f"Tokens: {tokens.tolist()}")

    # Create inputs
    position_ids = torch.arange(seq_len, dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, seq_len, CONTEXT_LENGTH), dtype=MODEL_DTYPE)
    for i in range(seq_len):
        for j in range(i + 1, CONTEXT_LENGTH):
            causal_mask[0, 0, i, j] = float('-inf')
    update_mask = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE)
    current_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

    with torch.no_grad():
        outputs_no_cache = our_model_no_cache(
            input_ids=tokens_int32,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=True,
        )

    if isinstance(outputs_no_cache, tuple):
        logits_no_cache = torch.cat(outputs_no_cache, dim=-1)
    else:
        logits_no_cache = outputs_no_cache

    probs_no_cache = torch.softmax(logits_no_cache[0, -1, :].float(), dim=-1)
    _, top5_no_cache = torch.topk(probs_no_cache, 5)
    print(f"NO_CACHE top-5: {[tokenizer.decode([t.item()]) for t in top5_no_cache]}")

    # Test with KV cache ENABLED
    print("\n" + "=" * 70)
    print("Test 2: KV Cache ENABLED (may have issues)")
    print("=" * 70)

    our_model_cache = QwenForCausalLM(qwen_config, enable_coreml=False, disable_kv_cache=False)
    our_model_cache.load_pretrained_weights(model_path)
    our_model_cache.eval()

    with torch.no_grad():
        outputs_cache = our_model_cache(
            input_ids=tokens_int32,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=True,
        )

    if isinstance(outputs_cache, tuple):
        logits_cache = torch.cat(outputs_cache, dim=-1)
    else:
        logits_cache = outputs_cache

    probs_cache = torch.softmax(logits_cache[0, -1, :].float(), dim=-1)
    _, top5_cache = torch.topk(probs_cache, 5)
    print(f"WITH_CACHE top-5: {[tokenizer.decode([t.item()]) for t in top5_cache]}")

    # Compare
    logits_diff = (logits_no_cache.float() - logits_cache.float()).abs()
    print(f"\nLogits diff (no_cache vs cache):")
    print(f"  max: {logits_diff.max().item():.6f}")
    print(f"  mean: {logits_diff.mean().item():.6f}")

    if logits_diff.max() > 0.1:
        print("\nWARNING: KV cache version differs from no-cache version!")
        print("This indicates a bug in the KV cache path.")
    else:
        print("\nOK: KV cache version matches no-cache version")

    # Test multi-token prefill with KV cache
    print("\n" + "=" * 70)
    print("Test 3: Multi-token prefill with KV cache")
    print("=" * 70)

    prompt = "Who are you?"
    tokens = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    tokens_int32 = tokens.to(torch.int32)
    seq_len = tokens.shape[1]
    print(f"Tokens: {tokens.tolist()} ({seq_len} tokens)")

    # Reset model state by recreating it
    our_model_cache2 = QwenForCausalLM(qwen_config, enable_coreml=False, disable_kv_cache=False)
    our_model_cache2.load_pretrained_weights(model_path)
    our_model_cache2.eval()

    position_ids = torch.arange(seq_len, dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, seq_len, CONTEXT_LENGTH), dtype=MODEL_DTYPE)
    for i in range(seq_len):
        for j in range(i + 1, CONTEXT_LENGTH):
            causal_mask[0, 0, i, j] = float('-inf')
    current_pos = torch.tensor([0], dtype=torch.int32)  # Start from 0 for prefill

    with torch.no_grad():
        outputs_prefill = our_model_cache2(
            input_ids=tokens_int32,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=True,
        )

    if isinstance(outputs_prefill, tuple):
        logits_prefill = torch.cat(outputs_prefill, dim=-1)
    else:
        logits_prefill = outputs_prefill

    print(f"Prefill logits shape: {logits_prefill.shape}")
    probs_prefill = torch.softmax(logits_prefill[0, -1, :].float(), dim=-1)
    _, top5_prefill = torch.topk(probs_prefill, 5)
    print(f"PREFILL top-5: {[tokenizer.decode([t.item()]) for t in top5_prefill]}")

    # Compare with HF
    with torch.no_grad():
        hf_outputs = hf_model(tokens, use_cache=False)
        hf_logits = hf_outputs.logits

    hf_probs = torch.softmax(hf_logits[0, -1, :].float(), dim=-1)
    _, hf_top5 = torch.topk(hf_probs, 5)
    print(f"HF top-5: {[tokenizer.decode([t.item()]) for t in hf_top5]}")

    # Check overlap
    our_set = set(top5_prefill.tolist())
    hf_set = set(hf_top5.tolist())
    overlap = our_set & hf_set
    print(f"\nOverlap: {len(overlap)}/5")

    if len(overlap) < 3:
        print("WARNING: Poor overlap with HF - investigating prefill path...")

        # Check KV cache state after prefill
        kv_cache = our_model_cache2.model.kv_cache_0
        print(f"\nKV cache shape: {kv_cache.shape}")
        print(f"KV cache stats: min={kv_cache.min():.4f}, max={kv_cache.max():.4f}, mean={kv_cache.mean():.4f}")

        # Check if cache was actually populated
        cache_slice = kv_cache[:, :, :seq_len, :]
        print(f"Cache slice (0:seq_len) stats: min={cache_slice.min():.4f}, max={cache_slice.max():.4f}, mean={cache_slice.mean():.4f}")

        # Check outside cache slice
        cache_outside = kv_cache[:, :, seq_len:seq_len+5, :]
        print(f"Cache outside slice (seq_len:seq_len+5) stats: min={cache_outside.min():.4f}, max={cache_outside.max():.4f}")

if __name__ == "__main__":
    main()
