#!/usr/bin/env python3
"""
Debug single sample PyTorch Qwen2.5 evaluation to see what's going wrong
"""

import os
import sys
import torch
import numpy as np
from datasets import load_dataset

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# Add paths for imports
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoTokenizer
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def debug_single_sample():
    """Debug the first BoolQ sample to see what the model predicts"""
    
    # Load model
    model_path = "Qwen/Qwen2.5-0.5B"
    print(f"Loading model from: {model_path}")
    
    # Resolve model path
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        try:
            local_path = snapshot_download(repo_id=model_path, local_files_only=True)
            model_path = local_path
        except Exception:
            local_path = snapshot_download(repo_id=model_path)
            model_path = local_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get actual vocab size
    import json
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        hf_config = json.load(f)
    
    actual_vocab_size = hf_config['vocab_size']
    print(f"Model vocab size: {actual_vocab_size}")
    
    # Create config
    config = Qwen25Config(
        vocab_size=actual_vocab_size,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        state_length=512,
        rms_norm_eps=1e-6,
    )
    
    # Load model
    model = Qwen25ForCausalLM(config, disable_kv_cache=True)
    success = model.load_pretrained_weights(model_path)
    if not success:
        raise RuntimeError("Failed to load weights")
    
    print("Model loaded successfully")
    
    # Load first BoolQ sample
    dataset = load_dataset('boolq', split='validation')
    sample = dataset[0]
    
    print(f"\nFirst BoolQ sample:")
    print(f"Question: {sample['question']}")
    print(f"Passage: {sample['passage'][:200]}...")
    print(f"Answer: {sample['answer']}")
    
    # Create the context like lm-eval does
    context = f"Passage: {sample['passage']}\nQuestion: {sample['question']}\nAnswer:"
    
    print(f"\nContext: {context[:200]}...")
    
    # Tokenize context and choices
    context_tokens = tokenizer.encode(context, add_special_tokens=True)
    yes_tokens = tokenizer.encode(" yes", add_special_tokens=False)
    no_tokens = tokenizer.encode(" no", add_special_tokens=False)
    
    print(f"\nTokenization:")
    print(f"Context tokens: {len(context_tokens)} tokens")
    print(f"Yes tokens: {yes_tokens} = '{tokenizer.decode(yes_tokens)}'")
    print(f"No tokens: {no_tokens} = '{tokenizer.decode(no_tokens)}'")
    
    # Score both choices
    def score_choice(context_tokens, choice_tokens):
        """Score a choice using the model"""
        total_logprob = 0.0
        context_length = model.config.context_length
        
        # Check length
        if len(context_tokens) + len(choice_tokens) > context_length:
            available_context = context_length - len(choice_tokens)
            if available_context <= 0:
                return float('-inf')
            context_tokens = context_tokens[-available_context:]
        
        if len(context_tokens) == 0:
            return float('-inf')
        
        # Stage A: Setup for golden token workflow
        sequence_so_far = context_tokens[:-1]  # All but last
        prev = context_tokens[-1]  # Last context token
        pos = len(sequence_so_far)  # Position to process
        
        print(f"\nScoring choice: {tokenizer.decode(choice_tokens)}")
        print(f"Context length: {len(context_tokens)}")
        print(f"Choice length: {len(choice_tokens)}")
        print(f"Starting pos: {pos}, prev token: {prev}")
        
        # Stage B: Walk through choice tokens
        for i, target_token in enumerate(choice_tokens):
            print(f"\nStep {i+1}: Processing target token {target_token} = '{tokenizer.decode([target_token])}'")
            
            # Check bounds
            if pos >= context_length:
                print(f"Position {pos} exceeds context length {context_length}")
                break
            
            # Create current sequence
            current_sequence = sequence_so_far + [prev]
            print(f"Current sequence length: {len(current_sequence)}, last few tokens: {current_sequence[-5:]}")
            
            # Convert to tensor
            last_token = torch.tensor([[current_sequence[-1]]], dtype=torch.long)
            
            # Create masks
            def make_causal_mask(length, start):
                mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
                row_indices = np.arange(length).reshape(length, 1)
                col_indices = np.arange(length).reshape(1, length)
                mask[:, :, col_indices <= (row_indices + start)] = 0
                return mask
            
            causal_mask_data = make_causal_mask(context_length, 0)
            causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
            
            update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
            if pos < context_length:
                update_mask[0, 0, pos, 0] = 1.0
            
            try:
                with torch.no_grad():
                    logits = model(
                        last_token,
                        update_mask,
                        torch.tensor([pos], dtype=torch.long),
                        causal_mask[:, :, pos:pos+1, :],
                        torch.tensor(pos, dtype=torch.long),
                        IN_PREFILL=False
                    )
                    
                    next_token_logits = logits[0, -1, :]
                    print(f"Logits shape: {next_token_logits.shape}")
                    
                    # Get top predictions
                    top_logits, top_indices = torch.topk(next_token_logits, 5)
                    print(f"Top 5 predictions:")
                    for j, (logit, idx) in enumerate(zip(top_logits, top_indices)):
                        token_text = tokenizer.decode([idx.item()])
                        print(f"  {j+1}. Token {idx.item()} = '{token_text}' (logit: {logit.item():.3f})")
                    
                    # Score target token
                    log_probs = torch.log_softmax(next_token_logits.float(), dim=-1)
                    token_logprob = log_probs[target_token].item()
                    print(f"Target token {target_token} log prob: {token_logprob:.6f}")
                    
                    total_logprob += token_logprob
                    
            except Exception as e:
                print(f"Error at pos {pos}: {e}")
                return float('-inf')
            
            # Golden token update
            sequence_so_far = current_sequence
            prev = target_token
            pos += 1
        
        print(f"Total log probability: {total_logprob:.6f}")
        return total_logprob
    
    # Score both choices
    yes_score = score_choice(context_tokens, yes_tokens)
    no_score = score_choice(context_tokens, no_tokens)
    
    print(f"\n" + "="*50)
    print(f"FINAL RESULTS:")
    print(f"Yes score: {yes_score:.6f}")
    print(f"No score: {no_score:.6f}")
    print(f"Predicted: {'Yes' if yes_score > no_score else 'No'}")
    print(f"Correct: {'Yes' if sample['answer'] else 'No'}")
    print(f"Match: {(yes_score > no_score) == sample['answer']}")

if __name__ == "__main__":
    debug_single_sample()