#!/usr/bin/env python3

import torch
import numpy as np
import sys
import os

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *

def test_fixed_coreml_output():
    """Test the fixed CoreML model output vs PyTorch."""
    
    print("=== Testing Fixed CoreML Model Output ===")
    
    # Model paths
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/"
    model_path = os.path.expanduser(model_path)
    coreml_path = "/tmp/qwen-test/float32/test_qwen.mlpackage"
    
    # Check if CoreML model exists
    if not os.path.exists(coreml_path):
        print(f"❌ CoreML model not found: {coreml_path}")
        return
    
    print(f"✅ Found CoreML model: {coreml_path}")
    
    # Load PyTorch model
    print(f"\n🔄 Loading PyTorch model...")
    config = QwenConfig.from_json(f"{model_path}/config.json")
    pytorch_model = QwenForCausalLM(config, enable_coreml=True)
    pytorch_model.load_pretrained_weights(model_path)
    pytorch_model.eval()
    
    # Create test input
    prompt = "What is Apple Neural Engine?"
    print(f"🔤 Test prompt: '{prompt}'")
    
    # Use simple token IDs that should work
    raw_tokens = [9195, 374, 8868, 61688]  # "What is Apple Neural"
    CONTEXT_LENGTH = 256  # Match the CoreML model's fixed window size
    PAD_TOKEN = 151643  # Use model's pad token
    
    # Pad to full context length (fixed window approach)
    padded_tokens = raw_tokens + [PAD_TOKEN] * (CONTEXT_LENGTH - len(raw_tokens))
    input_ids = torch.tensor([padded_tokens], dtype=torch.int32)  # [1, 256]
    
    position_ids = torch.arange(CONTEXT_LENGTH, dtype=torch.int32)  # [256] - [0, 1, 2, ..., 255]
    causal_mask = torch.zeros((1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH), dtype=torch.float16)  # [1, 1, 256, 256]
    
    # Set up proper causal mask (each position can attend to previous positions)
    for i in range(CONTEXT_LENGTH):
        for j in range(i + 1):
            causal_mask[0, 0, i, j] = 0  # Can attend
        for j in range(i + 1, CONTEXT_LENGTH):
            causal_mask[0, 0, i, j] = -float('inf')  # Cannot attend
    
    # Extract from the last real token position (position 3, since we have 4 tokens 0,1,2,3)
    current_pos = torch.tensor([len(raw_tokens) - 1], dtype=torch.int32)  # [3]
    
    print(f"📏 Input shapes:")
    print(f"  input_ids: {input_ids.shape} (first 10: {input_ids[0][:10].tolist()})")
    print(f"  position_ids: {position_ids.shape} (first 10: {position_ids[:10].tolist()})")
    print(f"  causal_mask: {causal_mask.shape}")
    print(f"  current_pos: {current_pos.shape} = {current_pos.tolist()} (extracting from position {current_pos[0]} to predict next)")
    print(f"  Raw tokens: {raw_tokens} ({len(raw_tokens)} tokens)")
    print(f"  Context length: {CONTEXT_LENGTH}")
    
    # Test PyTorch model
    print(f"\n🔥 Running PyTorch model...")
    with torch.no_grad():
        pytorch_output = pytorch_model(
            input_ids=input_ids,
            update_mask=torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=torch.float16),
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False,
        )
    
    if isinstance(pytorch_output, tuple):
        print(f"📊 PyTorch output: {len(pytorch_output)} tensors")
        for i, tensor in enumerate(pytorch_output):
            print(f"  logits{i+1}: {tensor.shape}")
        
        # Concatenate for comparison
        pytorch_logits = torch.cat(pytorch_output, dim=2)
        print(f"📐 Concatenated PyTorch logits: {pytorch_logits.shape}")
        
        # Get top token (extract from current_pos position, which should be 0 due to our position extraction logic)
        extract_pos = 0  # PyTorch model internally extracts from current_pos, returns [1, 1, vocab_size]
        pytorch_top_token = torch.argmax(pytorch_logits[0, extract_pos, :]).item()
        pytorch_top_logit = torch.max(pytorch_logits[0, extract_pos, :]).item()
        print(f"🔸 PyTorch top token: {pytorch_top_token}, logit: {pytorch_top_logit:.3f} (from position {extract_pos})")
        
    else:
        pytorch_logits = pytorch_output
        print(f"📐 PyTorch single output: {pytorch_logits.shape}")
    
    # Test CoreML model
    print(f"\n🤖 Loading CoreML model...")
    try:
        import coremltools as ct
        mlmodel = ct.models.MLModel(coreml_path)
        
        # Prepare CoreML inputs (must include update_mask)
        update_mask = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=torch.float16)
        
        coreml_input = {
            "input_ids": input_ids.numpy().astype(np.int32),
            "position_ids": position_ids.numpy().astype(np.int32),
            "causal_mask": causal_mask.numpy().astype(np.float16),
            "current_pos": current_pos.numpy().astype(np.int32),
            "update_mask": update_mask.numpy().astype(np.float16),
        }
        
        print(f"🔥 Running CoreML model...")
        coreml_output = mlmodel.predict(coreml_input)
        
        print(f"📊 CoreML output keys: {list(coreml_output.keys())}")
        
        # Concatenate CoreML outputs
        coreml_tensors = []
        for i in range(1, 17):  # 16-way split
            key = f"logits{i}"
            if key in coreml_output:
                tensor = coreml_output[key]
                print(f"  {key}: {tensor.shape}")
                coreml_tensors.append(tensor)
        
        if coreml_tensors:
            # Concatenate along vocab dimension (should be last dimension)
            coreml_logits = np.concatenate(coreml_tensors, axis=-1)
            print(f"📐 Concatenated CoreML logits: {coreml_logits.shape}")
            
            # Get top token (should also extract from position 0 for consistency)
            extract_pos = 0  # CoreML should also extract from same position as PyTorch
            coreml_top_token = np.argmax(coreml_logits[0, extract_pos, :])
            coreml_top_logit = np.max(coreml_logits[0, extract_pos, :])
            print(f"🔹 CoreML top token: {coreml_top_token}, logit: {coreml_top_logit:.3f} (from position {extract_pos})")
            
            # Compare outputs
            print(f"\n📊 Comparison:")
            print(f"  PyTorch shape: {pytorch_logits.shape}")
            print(f"  CoreML shape:  {coreml_logits.shape}")
            
            if pytorch_logits.shape == tuple(coreml_logits.shape):
                print(f"  ✅ SHAPES MATCH!")
                
                # Compare values
                pytorch_np = pytorch_logits.numpy()
                diff = np.abs(pytorch_np - coreml_logits)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                print(f"  📈 Max difference: {max_diff:.6f}")
                print(f"  📈 Mean difference: {mean_diff:.6f}")
                
                if max_diff < 0.1:
                    print(f"  ✅ EXCELLENT: Outputs are nearly identical!")
                elif max_diff < 1.0:
                    print(f"  ✅ GOOD: Outputs are very close!")
                elif max_diff < 10.0:
                    print(f"  ⚠️  ACCEPTABLE: Some differences but much better than before")
                else:
                    print(f"  ❌ LARGE DIFFERENCES: Still significant mismatch")
                
                # Compare top tokens
                if pytorch_top_token == coreml_top_token:
                    print(f"  ✅ TOP TOKENS MATCH: Both predict token {pytorch_top_token}")
                else:
                    print(f"  ❌ TOP TOKENS DIFFER: PyTorch {pytorch_top_token} vs CoreML {coreml_top_token}")
                    
            else:
                print(f"  ❌ SHAPES DIFFER: Still have dimension mismatch")
        else:
            print(f"❌ No valid CoreML outputs found")
            
    except Exception as e:
        print(f"❌ CoreML test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_coreml_output() 