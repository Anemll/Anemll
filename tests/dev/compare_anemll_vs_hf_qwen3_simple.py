#!/usr/bin/env python3
"""
Simple token-by-token comparison between ANEMLL Qwen3 and HuggingFace Qwen3 models
Process one token at a time to find where implementations diverge
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
def make_causal_mask(length, start):
    """Create causal attention mask."""
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return torch.tensor(mask)
def compare_single_token(anemll_model, hf_model, tokenizer, input_ids, position):
    """Compare models on a single token prediction.
    Both models predict what comes AFTER the sequence up to 'position'."""
    
    # Get the input token(s) up to current position for HuggingFace
    current_input = input_ids[:, :position+1]
    
    # HuggingFace forward pass (with full sequence)
    # HF processes tokens 0...position and predicts token at position+1
    with torch.no_grad():
        hf_outputs = hf_model(current_input, output_hidden_states=True)
        hf_logits = hf_outputs.logits[0, -1, :]  # Last position logits (predicts position+1)
        hf_hidden = hf_outputs.hidden_states[-1][0, -1, :]  # Last hidden state
    
    # ANEMLL forward pass - feed tokens incrementally to build KV cache
    context_length = 512  # Fixed context size
    
    # Create fixed-size causal mask
    causal_mask = make_causal_mask(context_length, 0)
    
    with torch.no_grad():
        # Clear KV cache before starting
        if hasattr(anemll_model.model, 'kv_cache_0'):
            anemll_model.model.kv_cache_0.zero_()
        
        # Process all tokens up to and including 'position'
        # This builds up the KV cache incrementally
        for pos in range(position + 1):
            # For autoregressive generation:
            # At position 0: input token 0 to predict token 1
            # At position 1: input token 1 to predict token 2
            # etc.
            input_token_id = input_ids[0, pos].item()
            input_token = torch.tensor([[input_token_id]], dtype=torch.long)
            
            # Position ID for this token
            position_ids = torch.tensor([pos], dtype=torch.long)
            
            # Update mask is 4D: (1, 1, context_length, 1)
            update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
            update_mask[0, 0, pos, 0] = 1.0
            
            # Causal mask slice for this position
            single_causal_mask = causal_mask[:, :, pos:pos+1, :]
            
            # Call model to process this token and update KV cache
            anemll_outputs = anemll_model(
                input_token,
                update_mask,
                position_ids,
                single_causal_mask,
                torch.tensor(pos, dtype=torch.long),
                IN_PREFILL=False
            )
            
            # Debug: show what the model predicts at each step
            predicted_token_id = torch.argmax(anemll_outputs[0, -1, :]).item()
            predicted_token_text = tokenizer.decode([predicted_token_id])
            
            if pos < len(input_ids[0]) - 1:
                actual_next_id = input_ids[0, pos + 1].item()
                actual_next_text = tokenizer.decode([actual_next_id])
                print(f"  Step {pos}: Input[{pos}]='{tokenizer.decode([input_token_id])}' -> Predicts '{predicted_token_text}' (actual[{pos+1}]='{actual_next_text}')")
            else:
                print(f"  Step {pos}: Input[{pos}]='{tokenizer.decode([input_token_id])}' -> Predicts '{predicted_token_text}'")
        
        # After processing all tokens up to 'position', we have the final prediction
        # This prediction is for the token that would come at position+1
        anemll_logits = anemll_outputs[0, -1, :]  # Last position logits
        
        # For hidden states, we'll use a placeholder for now
        anemll_hidden = torch.zeros_like(hf_hidden)  # Placeholder
    
    return {
        'hf_logits': hf_logits,
        'anemll_logits': anemll_logits,
        'hf_hidden': hf_hidden,
        'anemll_hidden': anemll_hidden
    }

def main():
    # Model path
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455"
    model_path = os.path.expanduser(model_path)
    
    # Simple test text
    #test_text = "Question: A group of engineers wanted to know how"
    test_text = "Question:"
 
    print("Token-by-token comparison: ANEMLL vs HuggingFace Qwen3")
    print("="*80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer.encode(test_text, return_tensors='pt')
    
    print(f"Test text: '{test_text}'")
    print(f"Tokens: {input_ids[0].tolist()}")
    tokens_text = [tokenizer.decode([tid]) for tid in input_ids[0]]
    print(f"Token text: {tokens_text}")
    print("="*80)
    
    # Load models
    print("\nLoading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='cpu')
    hf_model.eval()
    
    print("Loading ANEMLL model...")
    config = QwenConfig.from_json(f'{model_path}/config.json')
    config.context_length = 512
    config.state_length = 512
    
    anemll_model = QwenForCausalLM(config, disable_kv_cache=True)
    
    # Load pretrained weights - this is crucial!
    print("Loading pretrained weights...")
    success = anemll_model.load_pretrained_weights(model_path)
    if not success:
        print("Failed to load ANEMLL model weights!")
        return
    else:
        print("Successfully loaded ANEMLL weights")
    
    anemll_model.eval()
    
    print("\nComparing token by token...")
    print("="*80)
    
    # First, let's check embeddings
    print("\nChecking embeddings first:")
    with torch.no_grad():
        hf_embeddings = hf_model.model.embed_tokens(input_ids)
        anemll_embeddings = anemll_model.model.embed_tokens(input_ids)
        
        emb_diff = (hf_embeddings - anemll_embeddings).abs()
        print(f"Embedding difference: max={emb_diff.max().item():.6f}, mean={emb_diff.mean().item():.6f}")
        
        if emb_diff.max().item() > 0.01:
            print("⚠️  WARNING: Large embedding differences detected!")
            print(f"HF embeddings sample: {hf_embeddings[0, 0, :5].tolist()}")
            print(f"ANEMLL embeddings sample: {anemll_embeddings[0, 0, :5].tolist()}")
    
    # Process more tokens to see how differences accumulate
    for pos in range(len(input_ids[0])):
        print(f"\n{'='*60}")
        print(f"Position {pos}: Comparing predictions after processing tokens 0-{pos}")
        print(f"Sequence so far: '{tokenizer.decode(input_ids[0][:pos+1])}'")
        if pos < len(input_ids[0]) - 1:
            print(f"Predicting next token (position {pos+1}): '{tokenizer.decode([input_ids[0][pos+1]])}'")
        else:
            print(f"Predicting next token (beyond input)")
        
        results = compare_single_token(anemll_model, hf_model, tokenizer, input_ids, pos)
        
        # Compare hidden states
        hf_hidden = results['hf_hidden']
        anemll_hidden = results['anemll_hidden']
        hidden_diff = (hf_hidden.float() - anemll_hidden.float()).abs()
        
        print(f"  Hidden state diff: max={hidden_diff.max().item():.6f}, mean={hidden_diff.mean().item():.6f}")
        
        # Compare logits
        hf_logits = results['hf_logits']
        anemll_logits = results['anemll_logits']
        logits_diff = (hf_logits.float() - anemll_logits.float()).abs()
        
        print(f"  Logits diff: max={logits_diff.max().item():.6f}, mean={logits_diff.mean().item():.6f}")
        
        # Compare top 5 predictions
        hf_probs = torch.softmax(hf_logits, dim=-1)
        anemll_probs = torch.softmax(anemll_logits, dim=-1)
        
        hf_top5_probs, hf_top5_idx = torch.topk(hf_probs, 5)
        anemll_top5_probs, anemll_top5_idx = torch.topk(anemll_probs, 5)
        
        print("  Top 5 predictions:")
        print("    HF:     ", end="")
        for i in range(5):
            token = tokenizer.decode([hf_top5_idx[i]])
            print(f"'{token}' ({hf_top5_probs[i]:.3f})", end=" ")
        print()
        
        print("    ANEMLL: ", end="")
        for i in range(5):
            token = tokenizer.decode([anemll_top5_idx[i]])
            print(f"'{token}' ({anemll_top5_probs[i]:.3f})", end=" ")
        print()
        
        # If differences are large, show more details
        if logits_diff.max().item() > 1.0:
            print("  ⚠️  Large logits difference detected!")
            # Find tokens with largest differences
            top_diff_vals, top_diff_idx = torch.topk(logits_diff, 10)
            print("  Tokens with largest logit differences:")
            for i in range(5):
                tid = top_diff_idx[i].item()
                token = tokenizer.decode([tid])
                print(f"    '{token}' (id={tid}): HF={hf_logits[tid]:.4f}, ANEMLL={anemll_logits[tid]:.4f}, diff={top_diff_vals[i]:.4f}")
    
    print("\n" + "="*80)
    print("Token-by-token comparison complete!")

if __name__ == "__main__":
    main()