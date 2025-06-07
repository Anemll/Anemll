#!/usr/bin/env python3

import torch
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
import sys
import os
import glob

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

def test_single_token_generation():
    """Test single-token generation with our fixed CoreML model."""
    
    print("🚀 Testing Single-Token Generation Model")
    print("=" * 60)
    
    # Configuration
    #prompt = "What is Apple Neural Engine?"
    prompt = "What is capital of France?"

    max_tokens = 5
    context_length = 256
    
    # Load tokenizer
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    if not model_dirs:
        print("❌ Error: Qwen model not found in cache")
        return False
    
    tokenizer_path = model_dirs[0]
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    
    # Load CoreML model
    coreml_path = "../qwen-test/qwen_mlp_fixed.mlpackage"
    if not os.path.exists(coreml_path):
        print(f"❌ Error: CoreML model not found at {coreml_path}")
        return False
    
    print(f"Loading CoreML model from: {coreml_path}")
    coreml_model = ct.models.MLModel(coreml_path)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_tokens = inputs.input_ids[0].tolist()
    
    print(f"\\nPrompt: '{prompt}'")
    print(f"Tokenized to {len(prompt_tokens)} tokens: {prompt_tokens}")
    print(f"Token meanings: {[repr(tokenizer.decode([t])) for t in prompt_tokens]}")
    
    # Generate tokens one by one
    generated_tokens = prompt_tokens[:]
    generated_text = prompt
    
    print(f"\\n🎯 Starting single-token generation...")
    
    for step in range(max_tokens):
        current_pos = len(generated_tokens) - 1
        
        # Take the last token as input
        input_token = generated_tokens[-1]
        
        print(f"\\nStep {step + 1}: Generating token at position {current_pos}")
        print(f"  Input token: {input_token} ('{tokenizer.decode([input_token])}')")
        
        # Prepare inputs for single-token model
        input_ids = np.array([[input_token]], dtype=np.int32)  # [1, 1]
        position_ids = np.array([current_pos], dtype=np.int32)  # [1]
        current_pos_tensor = np.array([current_pos], dtype=np.int32)  # [1]
        
        # Create causal mask for current position
        causal_mask = np.full((1, 1, 1, context_length), -np.inf, dtype=np.float16)
        causal_mask[:, :, 0, :current_pos + 1] = 0  # Allow attention to all previous tokens
        
        # Create inputs dict
        coreml_inputs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'causal_mask': causal_mask,
            'current_pos': current_pos_tensor
        }
        
        print(f"  Input shapes:")
        print(f"    input_ids: {input_ids.shape}")
        print(f"    position_ids: {position_ids.shape}")
        print(f"    causal_mask: {causal_mask.shape}")
        print(f"    current_pos: {current_pos_tensor.shape}")
        
        # Run CoreML model
        try:
            outputs = coreml_model.predict(coreml_inputs)
            
            # Combine logits from all 16 parts
            all_logits = []
            for i in range(1, 17):
                logits_key = f"logits{i}"
                if logits_key in outputs:
                    all_logits.append(outputs[logits_key])
                else:
                    print(f"❌ Missing logits{i} in outputs")
                    return False
            
            # Concatenate all logits parts
            combined_logits = np.concatenate(all_logits, axis=-1)
            print(f"  Combined logits shape: {combined_logits.shape}")
            
            # Get the logits for the current position (should be [1, 1, 151936])
            if combined_logits.shape[1] == 1:
                final_logits = combined_logits[0, 0, :]  # [151936]
            else:
                print(f"❌ Unexpected logits shape: {combined_logits.shape}")
                return False
            
            # Sample next token (greedy for now)
            next_token = np.argmax(final_logits)
            next_token_text = tokenizer.decode([next_token])
            
            print(f"  🎯 Generated token {next_token}: '{next_token_text}'")
            
            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                print("  🏁 EOS token generated, stopping")
                break
            
            # Add to generated sequence
            generated_tokens.append(next_token)
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"  Generated so far: '{generated_text}'")
            
        except Exception as e:
            print(f"❌ CoreML inference failed at step {step + 1}: {e}")
            return False
    
    print(f"\\n🎉 GENERATION COMPLETE!")
    print(f"📝 Final text: '{generated_text}'")
    print(f"📊 Generated {len(generated_tokens) - len(prompt_tokens)} new tokens")
    
    return True

if __name__ == "__main__":
    success = test_single_token_generation()
    if success:
        print("\\n✅ Single-token generation test passed!")
    else:
        print("\\n❌ Single-token generation test failed!")
        sys.exit(1) 