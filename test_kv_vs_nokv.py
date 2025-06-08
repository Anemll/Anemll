#!/usr/bin/env python3
"""Direct comparison of KV cache vs non-KV cache outputs."""

import torch
import sys
import os

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *
from transformers import AutoTokenizer

def create_causal_mask(context_length):
    """Create causal attention mask."""
    mask = torch.full((1, 1, context_length, context_length), float('-inf'), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    for i in range(context_length):
        for j in range(i + 1):
            mask[0, 0, i, j] = 0.0
    return mask

def test_kv_vs_nokv():
    """Compare outputs with and without KV cache."""
    print("🔄 Comparing KV Cache vs No KV Cache")
    print("=" * 60)
    
    # Setup
    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    
    from huggingface_hub import snapshot_download
    cached_dir = snapshot_download(model_id, allow_patterns=["config.json", "*.safetensors"])
    config = QwenConfig.from_json(os.path.join(cached_dir, 'config.json'))
    
    # Test prompt
    prompt = "What is Apple Neural Engine?"
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0].tolist()
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {input_ids}")
    
    # Model with KV cache
    print(f"\n1️⃣ Model WITH KV Cache...")
    model_kv = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model_kv.load_pretrained_weights(str(cached_dir))
    model_kv.eval()
    
    # Model without KV cache
    print(f"\n2️⃣ Model WITHOUT KV Cache...")
    model_nokv = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
    model_nokv.load_pretrained_weights(str(cached_dir))
    model_nokv.eval()
    
    # Run both models on the same sequence
    causal_mask = create_causal_mask(config.state_length)
    outputs_kv = []
    outputs_nokv = []
    
    print(f"\n🔥 Processing tokens sequentially...")
    
    # Process each token
    for i in range(len(input_ids)):
        current_tokens = input_ids[:i+1]
        current_sequence = torch.tensor([current_tokens], dtype=torch.long)
        seq_len = len(current_tokens)
        
        print(f"\nStep {i+1}: Token {input_ids[i]} at position {i}")
        
        # KV cache version
        position_ids = torch.tensor([i], dtype=torch.long)
        update_mask = torch.ones((1, 1), dtype=torch.float32)
        single_causal_mask = causal_mask[:, :, i:i+1, :]
        
        with torch.no_grad():
            output_kv = model_kv(
                input_ids=torch.tensor([[input_ids[i]]], dtype=torch.long),
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=single_causal_mask,
                current_pos=torch.tensor(i, dtype=torch.long),
                IN_PREFILL=False
            )
            logits_kv = output_kv[0, 0, :]
            next_kv = torch.argmax(logits_kv).item()
            prob_kv = torch.softmax(logits_kv, dim=-1)[next_kv].item()
            outputs_kv.append(next_kv)
        
        # Non-KV cache version
        position_ids_nokv = torch.arange(seq_len, dtype=torch.long)
        update_mask_nokv = torch.zeros((1, 1, 256, 1), dtype=torch.float16)
        causal_mask_nokv = torch.full((1, 1, seq_len, 256), -torch.inf, dtype=torch.float16)
        for j in range(seq_len):
            causal_mask_nokv[:, :, j, :j+1] = 0
        
        with torch.no_grad():
            output_nokv = model_nokv(
                input_ids=current_sequence,
                update_mask=update_mask_nokv,
                position_ids=position_ids_nokv,
                causal_mask=causal_mask_nokv,
                current_pos=torch.tensor(i, dtype=torch.long),
                IN_PREFILL=False
            )
            logits_nokv = output_nokv[0, 0, :]
            next_nokv = torch.argmax(logits_nokv).item()
            prob_nokv = torch.softmax(logits_nokv, dim=-1)[next_nokv].item()
            outputs_nokv.append(next_nokv)
        
        print(f"  KV cache predicts: {next_kv} ('{tokenizer.decode([next_kv])}', prob: {prob_kv:.4f})")
        print(f"  No KV cache predicts: {next_nokv} ('{tokenizer.decode([next_nokv])}', prob: {prob_nokv:.4f})")
        print(f"  Match: {'✅' if next_kv == next_nokv else '❌'}")
    
    # Compare results
    print(f"\n📊 FINAL COMPARISON")
    print(f"=" * 50)
    all_match = all(kv == nokv for kv, nokv in zip(outputs_kv, outputs_nokv))
    print(f"All predictions match: {'✅ YES' if all_match else '❌ NO'}")
    
    if not all_match:
        print(f"\nMismatches:")
        for i, (kv, nokv) in enumerate(zip(outputs_kv, outputs_nokv)):
            if kv != nokv:
                print(f"  Position {i}: KV={kv} ('{tokenizer.decode([kv])}') vs NoKV={nokv} ('{tokenizer.decode([nokv])}')")
    
    # Generate a few more tokens to see the pattern
    print(f"\n🚀 Generating 10 more tokens...")
    last_kv = outputs_kv[-1]
    last_nokv = outputs_nokv[-1]
    
    for gen_i in range(10):
        pos = len(input_ids) + gen_i
        
        # KV version
        with torch.no_grad():
            output_kv = model_kv(
                input_ids=torch.tensor([[last_kv]], dtype=torch.long),
                update_mask=torch.ones((1, 1), dtype=torch.float32),
                position_ids=torch.tensor([pos], dtype=torch.long),
                causal_mask=causal_mask[:, :, pos:pos+1, :],
                current_pos=torch.tensor(pos, dtype=torch.long),
                IN_PREFILL=False
            )
            last_kv = torch.argmax(output_kv[0, 0, :]).item()
        
        # NoKV version
        current_tokens = input_ids + outputs_nokv[:gen_i+1]
        with torch.no_grad():
            output_nokv = model_nokv(
                input_ids=torch.tensor([current_tokens], dtype=torch.long),
                update_mask=torch.zeros((1, 1, 256, 1), dtype=torch.float16),
                position_ids=torch.arange(len(current_tokens), dtype=torch.long),
                causal_mask=torch.full((1, 1, len(current_tokens), 256), -torch.inf, dtype=torch.float16),
                current_pos=torch.tensor(len(current_tokens)-1, dtype=torch.long),
                IN_PREFILL=False
            )
            last_nokv = torch.argmax(output_nokv[0, 0, :]).item()
        
        print(f"Gen {gen_i+1}: KV={tokenizer.decode([last_kv])} vs NoKV={tokenizer.decode([last_nokv])}")
    
    return all_match

if __name__ == "__main__":
    test_kv_vs_nokv() 