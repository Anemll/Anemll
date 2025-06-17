#!/usr/bin/env python3
"""Debug what the no-cache version actually does."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def debug_no_cache_behavior():
    """Debug what the no-cache version actually does when processing the sequence."""
    
    print("🔍 No-Cache Behavior Debug: Understanding the Reference Implementation")
    print("=" * 70)
    
    # Setup
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Create no-cache model
    model = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
    model.load_pretrained_weights(model_path)
    model.eval()
    
    # Test sequence
    tokens = [785, 6722, 151643]  # ['The', ' capital', '<|endoftext|>']
    token_names = [tokenizer.decode([t]) for t in tokens]
    print(f"Processing sequence: {tokens} → {token_names}")
    
    # Process the full sequence
    input_ids = torch.tensor([tokens], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.tensor([0, 1, 2], dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([2], dtype=torch.long, device=TEST_DEVICE)
    
    print(f"\nInput tensor shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  current_pos: {current_pos.shape}")
    
    # Process through model
    causal_mask = torch.zeros((1, 1, 3, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    print(f"\n=== PROCESSING THROUGH NO-CACHE MODEL ===")
    
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            update_mask=torch.ones((1, 1, 512, 3), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            position_ids=position_ids.unsqueeze(0),  # Add batch dimension
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False,
        )
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Output logits norm: {torch.norm(logits).item():.6f}")
    
    # Get predictions for each position
    for i in range(3):
        pos_logits = logits[0, i, :]
        top_token = torch.argmax(pos_logits).item()
        top_logit = pos_logits[top_token].item()
        print(f"Position {i} (token '{token_names[i]}'): predicts {top_token} ('{tokenizer.decode([top_token])}') with logit {top_logit:.3f}")
    
    # The key insight: In no-cache mode, we're getting the prediction for position 2
    # But what position does position_ids=[2] actually correspond to?
    print(f"\n=== UNDERSTANDING POSITION EXTRACTION ===")
    
    # The current_pos=2 means we want prediction for the token that comes AFTER position 2
    # So logits[0, 2, :] should be the prediction for the next token after '<|endoftext|>'
    final_position_logits = logits[0, 2, :]
    final_token = torch.argmax(final_position_logits).item()
    final_logit = final_position_logits[final_token].item()
    
    print(f"Final prediction (after '<|endoftext|>' at position 2):")
    print(f"  Token: {final_token} ('{tokenizer.decode([final_token])}')")
    print(f"  Logit: {final_logit:.3f}")
    
    # But wait - let's check what the fair comparison was actually doing
    print(f"\n=== WHAT SHOULD THE FAIR COMPARISON BE? ===")
    
    # In the fair comparison, both methods process ['The', ' capital', '<|endoftext|>']
    # The no-cache method returns token 785 ('The')
    # But that seems wrong - why would it predict 'The' after processing that sequence?
    
    # Let me trace through the logic more carefully
    print(f"Traced through the fair comparison:")
    print(f"1. Both methods process the sequence: {token_names}")
    print(f"2. No-cache method predicts: {final_token} ('{tokenizer.decode([final_token])}')")
    print(f"3. But in fair comparison, no-cache returned: 785 ('The')")
    print(f"4. This suggests there might be a position extraction issue")
    
    # Let's check what position extraction does
    print(f"\n=== POSITION EXTRACTION LOGIC ===")
    
    # In the forward method, there's this logic:
    # if seq_len == 1:
    #     pos_tensor = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    # But our sequence has length 3, so it would use current_pos
    
    print(f"Sequence length: {input_ids.shape[1]}")
    print(f"Current pos: {current_pos.item()}")
    
    # The position extraction should use torch.index_select with current_pos
    # So it should extract logits[0, 2, :] which we calculated above
    
    print(f"\nConclusion:")
    print(f"- No-cache method should predict: {final_token} ('{tokenizer.decode([final_token])}')")
    print(f"- If fair comparison shows different result, there's a bug in position extraction")

if __name__ == "__main__":
    from anemll.models.qwen_model import TEST_DEVICE, MODEL_DTYPE
    debug_no_cache_behavior() 