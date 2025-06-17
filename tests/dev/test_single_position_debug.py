#!/usr/bin/env python3
"""Debug single position extraction in CoreML model."""

import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def test_position_extraction():
    print("🔍 Testing Position Extraction in CoreML Model")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    
    # Load CoreML model
    coreml_model = ct.models.MLModel("/tmp/qwen-test/test_qwen_after_conv2d.mlpackage")
    
    # Same prompt as our test
    prompt = "What is Apple Neural Engine?"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_tokens = inputs.input_ids[0].tolist()
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {prompt_tokens}")
    print(f"Length: {len(prompt_tokens)}")
    
    # Create fixed window
    CONTEXT_LENGTH = 256
    PAD_TOKEN = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    window = prompt_tokens + [PAD_TOKEN] * (CONTEXT_LENGTH - len(prompt_tokens))
    
    # Test extracting from different positions
    test_positions = [0, 1, 2, 3, 4, 5]  # positions 0-5 are the real tokens
    
    for pos in test_positions:
        print(f"\n--- Testing extraction from position {pos} ---")
        print(f"Token at position {pos}: {window[pos]} ('{tokenizer.decode([window[pos]])}')")
        
        # Create inputs
        input_ids = np.array([window], dtype=np.int32)
        position_ids = np.arange(CONTEXT_LENGTH, dtype=np.int32)
        
        # Proper causal mask
        causal_mask = np.full((1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH), -np.inf, dtype=np.float16)
        for i in range(CONTEXT_LENGTH):
            for j in range(i + 1):
                causal_mask[0, 0, i, j] = 0
        
        # Extract from this position
        current_pos_input = np.array([pos], dtype=np.int32)
        update_mask = np.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=np.float16)
        
        # Run inference
        coreml_inputs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'causal_mask': causal_mask,
            'current_pos': current_pos_input,
            'update_mask': update_mask
        }
        
        coreml_outputs = coreml_model.predict(coreml_inputs)
        
        # Concatenate logits
        logits_parts = []
        for i in range(1, 17):
            key = f'logits{i}'
            if key in coreml_outputs:
                logits_parts.append(coreml_outputs[key])
        
        logits = np.concatenate(logits_parts, axis=-1)[0, 0, :]
        
        # Get top 3 predictions
        top_indices = np.argsort(logits)[::-1][:3]
        print(f"Top 3 predictions when extracting from position {pos}:")
        for i, idx in enumerate(top_indices):
            token_text = tokenizer.decode([int(idx)])
            print(f"  {i+1}. Token {int(idx)}: '{token_text}' (logit: {logits[idx]:.3f})")
    
    print(f"\n📊 ANALYSIS:")
    print(f"If the model is working correctly:")
    print(f"- Positions 0-4 should predict the next token in the sequence")
    print(f"- Position 5 should predict what comes after the '?' token")
    print(f"- All positions should give different predictions (showing position-dependent behavior)")

if __name__ == "__main__":
    test_position_extraction() 