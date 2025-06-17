#!/usr/bin/env python3
"""Minimal test with PyTorch qwen_model.py to verify base model correctness."""

import numpy as np
import torch
import os
from transformers import AutoTokenizer
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig
import warnings
warnings.filterwarnings('ignore')

def test_minimal_pytorch_qwen():
    print("🔬 Minimal PyTorch Qwen Test - Single Token (DISABLE_KV_CACHE=True)")
    print("=" * 70)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    
    # Load PyTorch Qwen model
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    print(f"Loading PyTorch model from: {model_path}")
    
    # Load config and create model
    config = QwenConfig.from_json(os.path.join(model_path, "config.json"))
    config.context_length = 256  # Match our test setup
    
    # Create model with CoreML features enabled AND KV cache disabled
    model = QwenForCausalLM(config, enable_coreml=True, disable_kv_cache=True)
    
    # Load pretrained weights
    success = model.load_pretrained_weights(model_path)
    if not success:
        print("❌ Failed to load pretrained weights")
        return False
    
    model.eval()
    print("✅ PyTorch model loaded successfully with KV cache DISABLED")
    
    # Test with single meaningful token
    test_token = 3838  # "What"
    print(f"Testing with single token: {test_token} ('{tokenizer.decode([test_token])}')")
    
    # Create minimal inputs (context length 256, but only first position used)
    CONTEXT_LENGTH = 256
    PAD_TOKEN = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Input: single token at position 0, rest padding
    input_ids = torch.tensor([[test_token] + [PAD_TOKEN] * (CONTEXT_LENGTH - 1)], dtype=torch.long)
    position_ids = torch.arange(CONTEXT_LENGTH, dtype=torch.long)
    
    # Proper causal mask (each position can see previous positions)
    causal_mask = torch.full((1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH), -float('inf'), dtype=torch.float16)
    for i in range(CONTEXT_LENGTH):
        for j in range(i + 1):  # Position i can attend to positions 0 through i
            causal_mask[0, 0, i, j] = 0
    
    # Extract logits from position 0 (like CoreML test)
    current_pos_input = torch.tensor([0], dtype=torch.long)
    update_mask = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=torch.float16)
    
    print(f"\n📊 Inputs:")
    print(f"  input_ids[0][:10]: {input_ids[0][:10]}")
    print(f"  position_ids[:10]: {position_ids[:10]}")
    print(f"  current_pos: {current_pos_input} (extract logits from here)")
    print(f"  extracting from position: {current_pos_input[0]} (to predict what comes at position {current_pos_input[0]+1})")
    print(f"  ℹ️  In sequence generation:")
    print(f"     Step 1: extract from pos 0 → predict token for pos 1") 
    print(f"     Step 2: extract from pos 1 → predict token for pos 2")
    print(f"     Step 3: extract from pos 2 → predict token for pos 3")
    print(f"     etc.")
    print(f"  🚫 KV cache is DISABLED for this test")
    
    # Debug causal mask
    print(f"\n🎭 Causal Mask (first 5x5):")
    mask_slice = causal_mask[0, 0, :5, :5]
    for i in range(5):
        row_vals = [f"{mask_slice[i,j]:6.1f}" for j in range(5)]
        print(f"    pos {i}: [{', '.join(row_vals)}]")
    print(f"  (0.0 = can attend, -inf = masked)")
    
    print(f"\n🔢 Position IDs pattern:")
    print(f"  First 10: {position_ids[:10]}")
    print(f"  Should be: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")
    
    # Run inference
    print(f"predicting for current_pos_input: {current_pos_input[0]}")
    
    with torch.no_grad():
        # The model returns 16 logit parts when enable_coreml=True
        outputs = model(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos_input,
            IN_PREFILL=False
        )
        
        # Check if we got 16 separate outputs (like CoreML) or concatenated
        if isinstance(outputs, tuple) and len(outputs) == 16:
            print("✅ Got 16 separate logit outputs (CoreML mode)")
            # Concatenate the 16 parts
            logits = torch.cat(outputs, dim=-1)  # [batch, seq_len, vocab_size]
        else:
            print("✅ Got concatenated logits output")
            logits = outputs
        
        print(f"  output logits shape: {logits.shape}")
        
        # Extract logits for the single position
        if logits.shape[1] == 1:
            # Model already extracted single position
            logits_1d = logits[0, 0, :]  # [vocab_size]
            print("✅ Model extracted single position internally")
        else:
            # Extract from position 0
            logits_1d = logits[0, 0, :]  # [vocab_size]
            print(f"✅ Extracted logits from position 0")
    
    # Get top 10 predictions
    top_indices = torch.argsort(logits_1d, descending=True)[:10]
    print(f"\n🔍 Top 10 predictions after 'What':")
    for i, idx in enumerate(top_indices):
        token_text = tokenizer.decode([int(idx)])
        print(f"  {i+1:2d}. Token {int(idx):5d}: '{token_text}' (logit: {logits_1d[idx]:.3f})")
    
    print(f"\n🤔 Expected reasonable predictions after 'What':")
    print(f"  Common completions: ' is', ' are', ' can', ' will', ' do', etc.")
    
    # Check if any top predictions make sense
    reasonable_tokens = [' is', ' are', ' can', ' will', ' do', ' does', ' would', ' should', ' question',' Question']
    found_reasonable = False
    for i, idx in enumerate(top_indices[:5]):
        token_text = tokenizer.decode([int(idx)])
        if token_text.strip().lower() in [t.strip().lower() for t in reasonable_tokens]:
            print(f"  ✅ Found reasonable prediction at rank {i+1}: '{token_text}'")
            found_reasonable = True
    
    if not found_reasonable:
        print(f"  ❌ No reasonable predictions in top 5 - base model may have issues")
        print(f"  🔍 Let's check what the model actually thinks 'What' means...")
        
        # Additional debugging - check if the embeddings are reasonable
        with torch.no_grad():
            embedding = model.model.embed_tokens(torch.tensor([test_token]))
            print(f"  Token embedding shape: {embedding.shape}")
            print(f"  Embedding norm: {embedding.norm().item():.3f}")
    
    return found_reasonable

if __name__ == "__main__":
    success = test_minimal_pytorch_qwen()
    print(f"\n{'✅ PyTorch model with DISABLED KV cache seems OK' if success else '❌ PyTorch model with DISABLED KV cache has issues'}")
    print(f"This will help us determine if the issue is in:")
    print(f"  - Base model implementation {'❌' if not success else '✅'}")
    print(f"  - KV cache implementation {'(likely)' if not success else '(very likely)'}") 