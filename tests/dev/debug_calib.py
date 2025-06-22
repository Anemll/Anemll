#!/usr/bin/env python3
"""Debug calibration data generation"""

from datasets import load_dataset
from transformers import AutoTokenizer

def debug_wikitext():
    print("Loading WikiText-2 dataset...")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Analyzing first 20 samples:")
    for i, sample in enumerate(traindata):
        if i >= 20:
            break
            
        text = sample['text']
        if not text or len(text.strip()) < 50:
            print(f"Sample {i}: SKIPPED - too short or empty")
            continue
            
        trainenc = tokenizer(text, return_tensors='pt', max_length=2048, truncation=True)
        token_count = trainenc.input_ids.shape[1]
        
        print(f"Sample {i}: {token_count} tokens, text preview: {text[:100]!r}")
        
        if token_count >= 512:  # Try 512 instead of 2048
            print(f"  ✓ Would use this sample (>= 512 tokens)")
        else:
            print(f"  ✗ Too short for 512 token requirement")

if __name__ == '__main__':
    debug_wikitext()