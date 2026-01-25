#!/usr/bin/env python3
"""
Debug ground truth values for the first few BoolQ samples
"""

import os
from datasets import load_dataset

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def debug_ground_truth():
    """Check ground truth for first few samples"""
    
    dataset = load_dataset('boolq', split='validation')
    
    print("First few BoolQ samples:")
    for i in range(5):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer']} ({'yes' if sample['answer'] else 'no'})")
        print(f"  Expected choice: {int(sample['answer'])} (0=no, 1=yes)")
        print(f"  Passage preview: {sample['passage'][:100]}...")

if __name__ == "__main__":
    debug_ground_truth()