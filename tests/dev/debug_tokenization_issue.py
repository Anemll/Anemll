#!/usr/bin/env python3
"""
Debug tokenization boundary issues that might cause 0% accuracy
"""

import os
import sys
from datasets import load_dataset

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoTokenizer

def debug_tokenization():
    """Debug tokenization boundary effects"""
    
    model_path = "Qwen/Qwen2.5-0.5B"
    
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        try:
            model_path = snapshot_download(repo_id=model_path, local_files_only=True)
        except:
            model_path = snapshot_download(repo_id=model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get BoolQ sample
    dataset = load_dataset('boolq', split='validation')
    sample = dataset[0]
    
    # Create context exactly like BoolQ
    context = f"{sample['passage']}\nQuestion: {sample['question']}?\nAnswer:"
    continuation_no = " no"
    continuation_yes = " yes"
    
    print("Testing tokenization boundary effects:")
    print(f"Context ends with: ...{context[-30:]}")
    
    # Method 1: Current PyTorch approach (problematic)
    print("\n=== Method 1: Current PyTorch approach ===")
    context_tokens_1 = tokenizer.encode(context, add_special_tokens=True)
    full_tokens_no_1 = tokenizer.encode(context + continuation_no, add_special_tokens=True)
    full_tokens_yes_1 = tokenizer.encode(context + continuation_yes, add_special_tokens=True)
    
    continuation_no_1 = full_tokens_no_1[len(context_tokens_1):]
    continuation_yes_1 = full_tokens_yes_1[len(context_tokens_1):]
    
    print(f"Context tokens: {len(context_tokens_1)}")
    print(f"Full 'no' tokens: {len(full_tokens_no_1)}")
    print(f"Full 'yes' tokens: {len(full_tokens_yes_1)}")
    print(f"Extracted 'no' continuation: {continuation_no_1}")
    print(f"Extracted 'yes' continuation: {continuation_yes_1}")
    
    # Method 2: Correct approach (lm-eval style)
    print("\n=== Method 2: Correct approach (lm-eval style) ===")
    context_tokens_2 = tokenizer.encode(context, add_special_tokens=True)
    continuation_no_2 = tokenizer.encode(continuation_no, add_special_tokens=False)
    continuation_yes_2 = tokenizer.encode(continuation_yes, add_special_tokens=False)
    
    print(f"Context tokens: {len(context_tokens_2)}")
    print(f"'no' continuation: {continuation_no_2}")
    print(f"'yes' continuation: {continuation_yes_2}")
    
    # Method 3: Check what concatenation actually produces
    print("\n=== Method 3: Manual concatenation check ===")
    manual_no = context_tokens_2 + continuation_no_2
    manual_yes = context_tokens_2 + continuation_yes_2
    
    print(f"Manual 'no' concatenation: {len(manual_no)} tokens")
    print(f"Manual 'yes' concatenation: {len(manual_yes)} tokens")
    
    # Compare all methods
    print("\n=== Comparison ===")
    print(f"Method 1 'no' tokens: {continuation_no_1}")
    print(f"Method 2 'no' tokens: {continuation_no_2}")
    print(f"Are they equal? {continuation_no_1 == continuation_no_2}")
    
    print(f"Method 1 'yes' tokens: {continuation_yes_1}")
    print(f"Method 2 'yes' tokens: {continuation_yes_2}")
    print(f"Are they equal? {continuation_yes_1 == continuation_yes_2}")
    
    # Check if the full sequences match
    print(f"\nFull sequence comparison:")
    print(f"full_tokens_no_1 == manual_no? {full_tokens_no_1 == manual_no}")
    print(f"full_tokens_yes_1 == manual_yes? {full_tokens_yes_1 == manual_yes}")
    
    # Show the actual token differences
    if continuation_no_1 != continuation_no_2:
        print(f"\n*** TOKENIZATION BOUNDARY ISSUE DETECTED ***")
        print(f"Context ends with token: {context_tokens_2[-5:]}")
        print(f"When tokenizing context+' no', got: {full_tokens_no_1[-10:]}")
        print(f"When tokenizing ' no' separately, got: {continuation_no_2}")
        print(f"Expected vs actual: {context_tokens_2 + continuation_no_2} vs {full_tokens_no_1}")
        
        # Decode to see the difference
        print(f"Separate decode: '{tokenizer.decode(context_tokens_2)}' + '{tokenizer.decode(continuation_no_2)}'")
        print(f"Joint decode: '{tokenizer.decode(full_tokens_no_1)}'")

if __name__ == "__main__":
    debug_tokenization()