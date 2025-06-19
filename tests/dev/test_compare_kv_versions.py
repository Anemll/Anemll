#!/usr/bin/env python3
"""Compare working version vs KV cache version to isolate the bug."""

import numpy as np
import torch
import os
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def test_compare_versions():
    print("🔍 Comparing Working Version vs KV Cache Version")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    
    # Load model path
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    
    # Test with single meaningful token
    test_token = 3838  # "What"
    print(f"Testing with single token: {test_token} ('{tokenizer.decode([test_token])}')")
    
    # Create minimal inputs
    CONTEXT_LENGTH = 256
    PAD_TOKEN = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    input_ids = torch.tensor([[test_token] + [PAD_TOKEN] * (CONTEXT_LENGTH - 1)], dtype=torch.long)
    position_ids = torch.arange(CONTEXT_LENGTH, dtype=torch.long)
    
    # Causal mask
    causal_mask = torch.full((1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH), -float('inf'), dtype=torch.float16)
    for i in range(CONTEXT_LENGTH):
        for j in range(i + 1):
            causal_mask[0, 0, i, j] = 0
    
    current_pos_input = torch.tensor([0], dtype=torch.long)
    update_mask = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=torch.float16)
    
    # Test working version
    print(f"\n✅ Testing Working Version (qwen_model.py)")
    print("-" * 40)
    
    from anemll.models.qwen_model import QwenForCausalLM, QwenConfig
    
    config = QwenConfig.from_json(os.path.join(model_path, "config.json"))
    config.context_length = CONTEXT_LENGTH
    
    model_working = QwenForCausalLM(config, enable_coreml=True)
    success = model_working.load_pretrained_weights(model_path)
    if not success:
        print("❌ Failed to load pretrained weights for working version")
        return
    
    model_working.eval()
    
    with torch.no_grad():
        outputs_working = model_working(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos_input,
            IN_PREFILL=False
        )
        
        if isinstance(outputs_working, tuple):
            logits_working = torch.cat(outputs_working, dim=-1)
        else:
            logits_working = outputs_working
        
        logits_1d_working = logits_working[0, 0, :]
    
    # Get top predictions for working version
    top_indices_working = torch.argsort(logits_1d_working, descending=True)[:5]
    print(f"Working version top 5:")
    for i, idx in enumerate(top_indices_working):
        token_text = tokenizer.decode([int(idx)])
        print(f"  {i+1}. Token {int(idx):5d}: '{token_text}' (logit: {logits_1d_working[idx]:.3f})")
    
    # Test KV cache version  
    print(f"\n⚠️  Testing KV Cache Version (qwen_model_kv_cache.py)")
    print("-" * 40)
    
    # Import KV cache version
    import sys
    sys.path.insert(0, 'anemll/models')
    import qwen_model_kv_cache
    
    config_kv = qwen_model_kv_cache.QwenConfig.from_json(os.path.join(model_path, "config.json"))
    config_kv.context_length = CONTEXT_LENGTH
    
    model_kv = qwen_model_kv_cache.QwenForCausalLM(config_kv, enable_coreml=True, disable_kv_cache=True)
    success = model_kv.load_pretrained_weights(model_path)
    if not success:
        print("❌ Failed to load pretrained weights for KV cache version")
        return
    
    model_kv.eval()
    
    with torch.no_grad():
        outputs_kv = model_kv(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos_input,
            IN_PREFILL=False
        )
        
        if isinstance(outputs_kv, tuple):
            logits_kv = torch.cat(outputs_kv, dim=-1)
        else:
            logits_kv = outputs_kv
        
        logits_1d_kv = logits_kv[0, 0, :]
    
    # Get top predictions for KV cache version
    top_indices_kv = torch.argsort(logits_1d_kv, descending=True)[:5]
    print(f"KV cache version top 5:")
    for i, idx in enumerate(top_indices_kv):
        token_text = tokenizer.decode([int(idx)])
        print(f"  {i+1}. Token {int(idx):5d}: '{token_text}' (logit: {logits_1d_kv[idx]:.3f})")
    
    # Compare outputs
    print(f"\n📊 COMPARISON")
    print("=" * 30)
    
    # Check if logits are identical
    logits_diff = torch.abs(logits_1d_working - logits_1d_kv).max()
    print(f"Max logits difference: {logits_diff:.6f}")
    
    if logits_diff < 1e-5:
        print("✅ Logits are essentially identical")
    else:
        print("❌ Logits are different")
        
        # Show which tokens have the biggest differences
        diff_tensor = torch.abs(logits_1d_working - logits_1d_kv)
        top_diff_indices = torch.argsort(diff_tensor, descending=True)[:10]
        print(f"\nTop 10 tokens with biggest logit differences:")
        for i, idx in enumerate(top_diff_indices):
            token_text = tokenizer.decode([int(idx)])
            working_logit = logits_1d_working[idx].item()
            kv_logit = logits_1d_kv[idx].item()
            diff = diff_tensor[idx].item()
            print(f"  {i+1:2d}. Token {int(idx):5d} '{token_text}': Working={working_logit:.3f}, KV={kv_logit:.3f}, Diff={diff:.3f}")
    
    # Compare top tokens
    print(f"\nTop token comparison:")
    working_top = int(top_indices_working[0])
    kv_top = int(top_indices_kv[0])
    print(f"  Working: {working_top} ('{tokenizer.decode([working_top])}')")
    print(f"  KV Cache: {kv_top} ('{tokenizer.decode([kv_top])}')")
    
    if working_top == kv_top:
        print("✅ Top predictions match")
        return True
    else:
        print("❌ Top predictions differ")
        return False

if __name__ == "__main__":
    match = test_compare_versions()
    print(f"\n{'✅ Versions produce identical results' if match else '❌ Versions produce different results'}") 