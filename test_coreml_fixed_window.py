#!/usr/bin/env python3
"""Test CoreML Qwen model using fixed window approach."""

import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def safe_decode(tokenizer, tokens, skip_special_tokens=False):
    """Safely decode tokens avoiding MLX import issues."""
    try:
        if isinstance(tokens, list) and len(tokens) == 1:
            return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        elif isinstance(tokens, int):
            return tokenizer.decode([tokens], skip_special_tokens=skip_special_tokens)
        else:
            return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    except Exception as e:
        if isinstance(tokens, int):
            return f"<token_{tokens}>"
        elif isinstance(tokens, list):
            if len(tokens) == 1:
                return f"<token_{tokens[0]}>"
            else:
                return f"<tokens_{len(tokens)}>"
        else:
            return f"<decode_error>"

def test_coreml_fixed_window():
    print("🪟 Testing CoreML Model with Fixed Window Approach")
    print("=" * 70)
    
    # Load tokenizer
    print("Loading tokenizer...")
    model_name = "Qwen/Qwen2.5-0.5B"  # Use similar model for tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load CoreML model
    print("Loading CoreML model...")
    coreml_model = ct.models.MLModel("/tmp/qwen-test/test_qwen_after_conv2d.mlpackage")
    print("CoreML model loaded successfully!")
    
    print("Model inputs:", list(coreml_model.get_spec().description.input))
    print("Model outputs:", list(coreml_model.get_spec().description.output))
    
    # Use the same prompt as PyTorch tests
    prompt = "What is Apple Neural Engine?"
    print(f"\nTesting with prompt: '{prompt}'")
    
    # Tokenize the full prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids
    prompt_tokens = input_ids[0].tolist()
    print(f"Tokenized to {len(prompt_tokens)} tokens: {prompt_tokens}")
    
    # Decode tokens safely to avoid MLX import issues
    token_meanings = [safe_decode(tokenizer, t) for t in prompt_tokens]
    print(f"Token meanings: {token_meanings}")
    
    # Fixed window parameters
    CONTEXT_LENGTH = 256
    PAD_TOKEN = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    print(f"\n🪟 Fixed Window Setup:")
    print(f"  Context length: {CONTEXT_LENGTH}")
    print(f"  Pad token: {PAD_TOKEN}")
    print(f"  Prompt length: {len(prompt_tokens)}")
    
    # Create fixed window with prompt + padding
    window = prompt_tokens + [PAD_TOKEN] * (CONTEXT_LENGTH - len(prompt_tokens))
    current_pos = len(prompt_tokens)  # Position to predict (first position after prompt)
    
    print(f"  Initial current_pos: {current_pos} (position for NEXT token)")
    print(f"  Last real token at position: {current_pos-1}")
    print(f"  Window[0:10]: {window[:10]}")
    print(f"  Window[{current_pos-2}:{current_pos+3}]: {window[current_pos-2:current_pos+3]}")
    print(f"  Real tokens: positions 0-{current_pos-1}, padding: positions {current_pos}-{CONTEXT_LENGTH-1}")
    
    # Generate tokens using fixed window approach
    max_new_tokens = 5
    generated_tokens = []
    
    print(f"\n🚀 Generating {max_new_tokens} tokens with Fixed Window...")
    print(f"\n🔍 COMPARISON TARGET:")
    print(f"  PyTorch generates: [8162, 60477, 8200, 374, 264] (from original model test)")
    print(f"  Expected first token: 8162")
    
    for gen_step in range(max_new_tokens):
        print(f"\n--- Generation Step {gen_step + 1} ---")
        print(f"Current pos: {current_pos}")
        print(f"Token at current_pos-1: {window[current_pos-1]} ('{safe_decode(tokenizer, window[current_pos-1])}')")
        
        # Create inputs for fixed window
        input_ids = np.array([window], dtype=np.int32)  # [1, CONTEXT_LENGTH]
        position_ids = np.arange(CONTEXT_LENGTH, dtype=np.int32)  # [0, 1, 2, ..., 255]
        
        # Create proper causal mask - each position can attend to all previous positions (use float16 to match converter)
        causal_mask = np.full((1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH), -np.inf, dtype=np.float16)
        for i in range(CONTEXT_LENGTH):
            for j in range(i + 1):  # Position i can attend to positions 0 through i
                causal_mask[0, 0, i, j] = 0
        
        # Current position to extract logits from (current_pos - 1, which is the last valid token position)
        extract_position = current_pos - 1
        current_pos_input = np.array([extract_position], dtype=np.int32)  # Position to extract logits from
        
        # Update mask (not used in fixed window but keep for compatibility, use float16)
        update_mask = np.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=np.float16)
        
        print(f"📊 CoreML inputs:")
        print(f"  input_ids: shape {input_ids.shape}")
        print(f"  position_ids: shape {position_ids.shape}, values [0..{CONTEXT_LENGTH-1}]")
        print(f"  causal_mask: shape {causal_mask.shape}")
        print(f"  current_pos: {current_pos_input} (extract logits from position {extract_position})")
        print(f"  update_mask: shape {update_mask.shape}")
        print(f"  Window content around pos {current_pos}: {window[current_pos-3:current_pos+2]}")
        print(f"  Real tokens: {window[:current_pos]}")
        print(f"  Padding tokens: {window[current_pos:current_pos+3]}...")
        
        # Run CoreML inference
        try:
            coreml_inputs = {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'causal_mask': causal_mask,
                'current_pos': current_pos_input,
                'update_mask': update_mask
            }
            
            coreml_outputs = coreml_model.predict(coreml_inputs)
            
            # Concatenate all logits parts (16 parts)
            num_logits = 16
            logits_parts = []
            for i in range(1, num_logits + 1):
                key = f'logits{i}'
                if key in coreml_outputs:
                    logits_parts.append(coreml_outputs[key])
            
            coreml_logits = np.concatenate(logits_parts, axis=-1)  # [1, 1, 151936] - model extracts position internally
            
            print(f"  CoreML logits shape: {coreml_logits.shape}")
            print(f"  ✅ Model correctly extracted single position (current_pos={current_pos-1}) internally!")
            
            # Model already extracted the correct position - just get the logits
            logits_1d = coreml_logits[0, 0, :]  # [151936] - always index 0 since model returns single position
            
            # Get next token
            next_token = int(np.argmax(logits_1d))
            
            # Show top 5 predictions for debugging
            top_indices = np.argsort(logits_1d)[::-1][:5]
            print(f"🔍 Top 5 predictions:")
            for i, idx in enumerate(top_indices):
                idx_int = int(idx)
                token_text = safe_decode(tokenizer, idx_int)
                print(f"  {i+1}. Token {idx_int}: '{token_text}' (logit: {logits_1d[idx]:.4f})")
            
            if gen_step == 0:
                print(f"\n🎯 CRITICAL FIRST STEP COMPARISON:")
                print(f"  Expected (PyTorch): 8162")
                print(f"  Got (CoreML): {next_token} ('{safe_decode(tokenizer, next_token)}')")
                print(f"  Match: {next_token == 8162}")
                if next_token != 8162:
                    print(f"  ❌ MISMATCH! CoreML still not reproducing PyTorch results")
                else:
                    print(f"  ✅ MATCH! CoreML now reproduces PyTorch results with fixed window!")
            
            # Place predicted token in window at current_pos
            window[current_pos] = next_token
            generated_tokens.append(next_token)
            current_pos += 1
            
            print(f"\n🎯 Prediction:")
            print(f"  Predicted token: {next_token} ('{safe_decode(tokenizer, next_token)}')")
            print(f"  Updated window[{current_pos-3}:{current_pos+2}]: {window[current_pos-3:current_pos+2]}")
            
            # Break if we reach context limit
            if current_pos >= CONTEXT_LENGTH:
                print("⚠️  Reached context length limit")
                break
                
        except Exception as e:
            print(f"❌ CoreML inference failed at step {gen_step + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Show final results
    print(f"\n📝 GENERATION RESULTS")
    print(f"=" * 50)
    print(f"Original prompt: '{prompt}'")
    print(f"Generated tokens: {generated_tokens}")
    
    # Decode the new tokens
    if generated_tokens:
        generated_text = safe_decode(tokenizer, generated_tokens, skip_special_tokens=True)
        print(f"Generated text: '{generated_text}'")
        
        # Decode full sequence up to current_pos
        full_sequence = window[:current_pos]
        full_text = safe_decode(tokenizer, full_sequence, skip_special_tokens=True)
        print(f"Full text: '{full_text}'")
        
        # Compare with PyTorch target
        pytorch_tokens = [8162, 60477, 8200, 374, 264]
        
        print(f"\n🔍 COMPARISON WITH PYTORCH:")
        print(f"  PyTorch tokens: {pytorch_tokens}")
        print(f"  CoreML tokens:  {generated_tokens}")
        
        matches = sum(1 for i, (p, c) in enumerate(zip(pytorch_tokens, generated_tokens)) if p == c)
        print(f"  Matching tokens: {matches}/{min(len(pytorch_tokens), len(generated_tokens))}")
        
        if matches == len(generated_tokens) and len(generated_tokens) >= 3:
            print(f"  ✅ SUCCESS! CoreML matches PyTorch with fixed window approach")
            return True
        else:
            print(f"  ❌ Partial match - need to debug further")
            return False
    else:
        print(f"  ❌ No tokens generated")
        return False

if __name__ == "__main__":
    success = test_coreml_fixed_window()
    print(f"\n{'🎉 BREAKTHROUGH' if success else '🔧 DEBUGGING NEEDED'}: Fixed window CoreML {'WORKS!' if success else 'needs refinement'}") 