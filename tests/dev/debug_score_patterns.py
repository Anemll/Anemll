#!/usr/bin/env python3
"""
Debug score patterns to see if there's a systematic bias
"""

import os
import sys
import torch
import numpy as np
from datasets import load_dataset

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add paths
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoTokenizer
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def debug_score_patterns():
    """Debug score patterns for first 10 samples"""
    
    # Load model
    model_path = "Qwen/Qwen2.5-0.5B"
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        try:
            model_path = snapshot_download(repo_id=model_path, local_files_only=True)
        except:
            model_path = snapshot_download(repo_id=model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    import json
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        hf_config = json.load(f)
    
    config = Qwen25Config(
        vocab_size=hf_config['vocab_size'],
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        state_length=512,
        rms_norm_eps=1e-6,
    )
    
    model = Qwen25ForCausalLM(config, disable_kv_cache=True)
    if not model.load_pretrained_weights(model_path):
        raise RuntimeError("Failed to load weights")
    
    print("Model loaded for pattern analysis")
    
    # Load first 10 BoolQ samples
    dataset = load_dataset('boolq', split='validation')
    
    def score_single_token(context, continuation):
        """Score single token using our approach"""
        context_tokens = tokenizer.encode(context, add_special_tokens=True)
        continuation_tokens = tokenizer.encode(continuation, add_special_tokens=False)
        
        if len(continuation_tokens) != 1:
            return float('-inf')
        
        pos = len(context_tokens) - 1
        last_token = torch.tensor([[context_tokens[-1]]], dtype=torch.long)
        
        def make_causal_mask(length, start):
            mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
            row_indices = np.arange(length).reshape(length, 1)
            col_indices = np.arange(length).reshape(1, length)
            mask[:, :, col_indices <= (row_indices + start)] = 0
            return mask
        
        ctx_len = model.config.context_length
        causal_mask_data = make_causal_mask(ctx_len, 0)
        causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
        
        update_mask = torch.zeros((1, 1, ctx_len, 1), dtype=torch.float16)
        if pos < ctx_len:
            update_mask[0, 0, pos, 0] = 1.0
        
        with torch.no_grad():
            logits = model(
                last_token,
                update_mask,
                torch.tensor([pos], dtype=torch.long),
                causal_mask[:, :, pos:pos+1, :],
                torch.tensor(pos, dtype=torch.long),
                IN_PREFILL=False
            )
            
            if logits.dim() == 3:
                logits = logits[0, -1]
            elif logits.dim() == 2:
                logits = logits[-1]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            return log_probs[continuation_tokens[0]].item()
    
    print("Sample Analysis:")
    print("Idx | Question | Ground Truth | No Score | Yes Score | Predicted | Correct")
    print("-" * 80)
    
    correct_count = 0
    
    for i in range(10):
        sample = dataset[i]
        context = f"{sample['passage']}\nQuestion: {sample['question']}?\nAnswer:"
        
        no_score = score_single_token(context, " no")
        yes_score = score_single_token(context, " yes")
        
        predicted = 0 if no_score > yes_score else 1
        ground_truth = int(sample['answer'])
        is_correct = predicted == ground_truth
        
        if is_correct:
            correct_count += 1
        
        print(f"{i:3d} | {sample['question'][:30]:30s} | {ground_truth:11d} | {no_score:8.3f} | {yes_score:9.3f} | {predicted:9d} | {is_correct}")
    
    accuracy = correct_count / 10
    print("-" * 80)
    print(f"Overall accuracy: {accuracy:.1%} ({correct_count}/10)")
    
    # Check if there's a systematic bias
    print(f"\nScore Analysis:")
    scores = []
    for i in range(10):
        context = f"{dataset[i]['passage']}\nQuestion: {dataset[i]['question']}?\nAnswer:"
        no_score = score_single_token(context, ' no')
        yes_score = score_single_token(context, ' yes')
        scores.append(no_score - yes_score)
    avg_diff = sum(scores) / 10
    print(f"- Average no vs yes score difference: {avg_diff:.3f}")

if __name__ == "__main__":
    debug_score_patterns()