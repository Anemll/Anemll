#!/usr/bin/env python3
"""Minimal test with single token to isolate CoreML issues."""

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def test_minimal_coreml():
    print("🔬 Minimal CoreML Test - Single Token")
    print("=" * 50)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

    coreml_model = ct.models.MLModel("/tmp/qwen-test/float32/test_qwen.mlpackage",compute_units=ct.ComputeUnit.CPU_ONLY)
    
    # Test with single meaningful token
    test_token = 3838  # "What"
    print(f"Testing with single token: {test_token} ('{tokenizer.decode([test_token])}')")
    
    # Create minimal inputs (context length 256, but only first position used)
    CONTEXT_LENGTH = 256
    PAD_TOKEN = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Input: single token at position 0, rest padding
    input_ids = np.array([[test_token] + [PAD_TOKEN] * (CONTEXT_LENGTH - 1)], dtype=np.int32)
    position_ids = np.arange(CONTEXT_LENGTH, dtype=np.int32)
    
    # Proper causal mask (each position can see previous positions)
    causal_mask = np.full((1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH), -np.inf, dtype=np.float16)
    for i in range(CONTEXT_LENGTH):
        for j in range(i + 1):  # Position i can attend to positions 0 through i
            causal_mask[0, 0, i, j] = 0
    
    # Extract logits from position 0
    current_pos_input = np.array([0], dtype=np.int32)
    update_mask = np.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=np.float16)
    
    print(f"\n📊 Inputs:")
    print(f"  input_ids[0][:10]: {input_ids[0][:10]}")
    print(f"  position_ids[:10]: {position_ids[:10]}")
    print(f"  current_pos: {current_pos_input} (extract logits from here)")
    print(f"  extracting from position: 0 (to predict what comes at position 1)")
    print(f"  ℹ️  In sequence generation:")
    print(f"     Step 1: extract from pos 1 → predict token for pos 1") 
    print(f"     Step 2: extract from pos 2 → predict token for pos 2")
    print(f"     Step 3: extract from pos 3 → predict token for pos 3")
    print(f"     etc.")
    
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
    coreml_inputs = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'causal_mask': causal_mask,
        'current_pos': current_pos_input,
        'update_mask': update_mask
    }
    print(f"predicy for current_pos_input: {current_pos_input[0]}")
    coreml_outputs = coreml_model.predict(coreml_inputs)
    
    # Concatenate logits
    logits_parts = []
    for i in range(1, 17):
        key = f'logits{i}'
        if key in coreml_outputs:
            logits_parts.append(coreml_outputs[key])
    
    logits = np.concatenate(logits_parts, axis=-1)[0, 0, :]
    print(f"  output logits shape: {logits.shape}")
    
    # Get top 10 predictions
    top_indices = np.argsort(logits)[::-1][:10]
    print(f"\n🔍 Top 10 predictions after 'What':")
    for i, idx in enumerate(top_indices):
        token_text = tokenizer.decode([int(idx)])
        print(f"  {i+1:2d}. Token {int(idx):5d}: '{token_text}' (logit: {logits[idx]:.3f})")
    
    print(f"\n🤔 Expected reasonable predictions after 'What':")
    print(f"  Common completions: ' is', ' are', ' can', ' will', ' do', etc.")
    
    # Check if any top predictions make sense
    reasonable_tokens = [' is', ' are', ' can', ' will', ' do', ' does', ' would', ' should']
    found_reasonable = False
    for i, idx in enumerate(top_indices[:5]):
        token_text = tokenizer.decode([int(idx)])
        if token_text.strip().lower() in [t.strip().lower() for t in reasonable_tokens]:
            print(f"  ✅ Found reasonable prediction at rank {i+1}: '{token_text}'")
            found_reasonable = True
    
    if not found_reasonable:
        print(f"  ❌ No reasonable predictions in top 5 - model may have conversion issues")
    
    return found_reasonable

if __name__ == "__main__":
    success = test_minimal_coreml()
    print(f"\n{'✅ Model seems OK' if success else '❌ Model has issues'}") 