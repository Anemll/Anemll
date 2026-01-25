#!/usr/bin/env python3
"""
Debug the first divergence point between ANEMLL and HuggingFace Qwen3.
Focus on understanding exactly what happens at position 1 where models start diverging.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

# Add paths for imports
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from anemll.models import qwen_model
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig

# Override device to CPU for fair comparison
qwen_model.TEST_DEVICE = 'cpu'

def main():
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455"
    model_path = os.path.expanduser(model_path)
    
    # Simple test sequence
    test_text = "Question:"
    
    print("=== Debugging First Divergence Point ===")
    print(f"Test text: '{test_text}'")
    
    # Load tokenizer and get tokens
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer.encode(test_text, return_tensors='pt')
    print(f"Tokens: {input_ids[0].tolist()}")
    print(f"Token texts: {[tokenizer.decode([tid]) for tid in input_ids[0]]}")
    
    # Load HuggingFace model
    print("\nLoading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='cpu')
    hf_model.eval()
    
    # Load ANEMLL model  
    print("Loading ANEMLL model...")
    config = QwenConfig.from_json(f'{model_path}/config.json')
    config.context_length = 512
    config.state_length = 512
    
    anemll_model = QwenForCausalLM(config, disable_kv_cache=True)
    success = anemll_model.load_pretrained_weights(model_path)
    if not success:
        print("Failed to load ANEMLL weights!")
        return
    anemll_model.eval()
    
    print("\n=== Position 0 Analysis ===")
    # Test position 0 (single token "Question")
    single_token = input_ids[:, :1]  # Just "Question"
    
    with torch.no_grad():
        # HuggingFace forward pass
        hf_outputs = hf_model(single_token, output_hidden_states=True)
        hf_logits_0 = hf_outputs.logits[0, -1, :]
        hf_hidden_0 = hf_outputs.hidden_states[-1][0, -1, :]
        
        # ANEMLL forward pass for position 0
        anemll_model.model.kv_cache_0.zero_()  # Clear cache
        
        context_length = 512
        causal_mask = torch.full((1, 1, context_length, context_length), float('-inf'))
        for i in range(context_length):
            causal_mask[0, 0, i, :i+1] = 0
            
        # Feed token at position 0
        input_token = torch.tensor([[input_ids[0, 0].item()]], dtype=torch.long)
        position_ids = torch.tensor([0], dtype=torch.long)
        update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
        update_mask[0, 0, 0, 0] = 1.0
        single_causal_mask = causal_mask[:, :, 0:1, :]
        
        anemll_outputs = anemll_model(
            input_token,
            update_mask, 
            position_ids,
            single_causal_mask,
            torch.tensor(0, dtype=torch.long),
            IN_PREFILL=False
        )
        anemll_logits_0 = anemll_outputs[0, -1, :]
        
    # Compare position 0
    logits_diff_0 = (hf_logits_0.float() - anemll_logits_0.float()).abs()
    print(f"Position 0 logits diff: max={logits_diff_0.max().item():.6f}, mean={logits_diff_0.mean().item():.6f}")
    
    hf_top5_0 = torch.topk(torch.softmax(hf_logits_0, dim=-1), 5)
    anemll_top5_0 = torch.topk(torch.softmax(anemll_logits_0, dim=-1), 5)
    
    print("Position 0 top 5 predictions:")
    print("  HF:     ", end="")
    for i in range(5):
        token = tokenizer.decode([hf_top5_0.indices[i]])
        print(f"'{token}' ({hf_top5_0.values[i]:.3f})", end=" ")
    print()
    print("  ANEMLL: ", end="")
    for i in range(5):
        token = tokenizer.decode([anemll_top5_0.indices[i]])
        print(f"'{token}' ({anemll_top5_0.values[i]:.3f})", end=" ")
    print()
    
    print("\n=== Position 1 Analysis ===")
    # Test position 1 (both tokens "Question:")
    two_tokens = input_ids[:, :2]  # "Question:"
    
    with torch.no_grad():
        # HuggingFace forward pass (full sequence)
        hf_outputs = hf_model(two_tokens, output_hidden_states=True)
        hf_logits_1 = hf_outputs.logits[0, -1, :]
        hf_hidden_1 = hf_outputs.hidden_states[-1][0, -1, :]
        
        # ANEMLL forward pass building KV cache sequentially
        anemll_model.model.kv_cache_0.zero_()  # Clear cache
        
        # Process token 0 first
        input_token_0 = torch.tensor([[input_ids[0, 0].item()]], dtype=torch.long)
        position_ids_0 = torch.tensor([0], dtype=torch.long) 
        update_mask_0 = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
        update_mask_0[0, 0, 0, 0] = 1.0
        single_causal_mask_0 = causal_mask[:, :, 0:1, :]
        
        anemll_outputs_0 = anemll_model(
            input_token_0,
            update_mask_0,
            position_ids_0, 
            single_causal_mask_0,
            torch.tensor(0, dtype=torch.long),
            IN_PREFILL=False
        )
        
        # Process token 1 
        input_token_1 = torch.tensor([[input_ids[0, 1].item()]], dtype=torch.long)
        position_ids_1 = torch.tensor([1], dtype=torch.long)
        update_mask_1 = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
        update_mask_1[0, 0, 1, 0] = 1.0
        single_causal_mask_1 = causal_mask[:, :, 1:2, :]
        
        anemll_outputs_1 = anemll_model(
            input_token_1,
            update_mask_1,
            position_ids_1,
            single_causal_mask_1, 
            torch.tensor(1, dtype=torch.long),
            IN_PREFILL=False
        )
        anemll_logits_1 = anemll_outputs_1[0, -1, :]
        
    # Compare position 1
    logits_diff_1 = (hf_logits_1.float() - anemll_logits_1.float()).abs()
    print(f"Position 1 logits diff: max={logits_diff_1.max().item():.6f}, mean={logits_diff_1.mean().item():.6f}")
    
    hf_top5_1 = torch.topk(torch.softmax(hf_logits_1, dim=-1), 5)
    anemll_top5_1 = torch.topk(torch.softmax(anemll_logits_1, dim=-1), 5)
    
    print("Position 1 top 5 predictions:")
    print("  HF:     ", end="")
    for i in range(5):
        token = tokenizer.decode([hf_top5_1.indices[i]])
        print(f"'{token}' ({hf_top5_1.values[i]:.3f})", end=" ")
    print()
    print("  ANEMLL: ", end="")
    for i in range(5):
        token = tokenizer.decode([anemll_top5_1.indices[i]])
        print(f"'{token}' ({anemll_top5_1.values[i]:.3f})", end=" ")
    print()
    
    # Show largest differences
    if logits_diff_1.max().item() > 1.0:
        print("\nLargest logit differences at position 1:")
        top_diff_vals, top_diff_idx = torch.topk(logits_diff_1, 5)
        for i in range(5):
            tid = top_diff_idx[i].item()
            token = tokenizer.decode([tid])
            print(f"  '{token}' (id={tid}): HF={hf_logits_1[tid]:.4f}, ANEMLL={anemll_logits_1[tid]:.4f}, diff={top_diff_vals[i]:.4f}")
            
    print("\n=== Analysis Summary ===")
    print(f"Position 0 works well (max diff: {logits_diff_0.max().item():.6f})")
    print(f"Position 1 shows major divergence (max diff: {logits_diff_1.max().item():.6f})")
    print("This indicates an issue with KV cache accumulation or sequential processing.")

if __name__ == "__main__":
    main()