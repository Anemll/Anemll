#!/usr/bin/env python3
"""
Debug the actual BoolQ prompt format being generated
"""

import os
import sys
from datasets import load_dataset

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add paths
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoTokenizer
from lm_eval.tasks import get_task_dict

def debug_boolq_prompt():
    """Debug the actual BoolQ prompt being generated"""
    
    # Get BoolQ task
    task_dict = get_task_dict(['boolq'])
    boolq_task = task_dict['boolq']
    
    print("BoolQ Task Configuration:")
    print(f"doc_to_text: {boolq_task.config.doc_to_text}")
    print(f"doc_to_choice: {boolq_task.config.doc_to_choice}")
    print(f"target_delimiter: {repr(boolq_task.config.target_delimiter)}")
    print(f"output_type: {boolq_task.config.output_type}")
    
    # Load first sample
    dataset = load_dataset('boolq', split='validation')
    sample = dataset[0]
    
    print(f"\nFirst BoolQ sample:")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Passage (first 200 chars): {sample['passage'][:200]}...")
    
    # Generate the actual prompt using the task's doc_to_text
    doc_to_text = boolq_task.config.doc_to_text
    
    # The template uses {{passage}} and {{question}} 
    prompt = doc_to_text.replace('{{passage}}', sample['passage']).replace('{{question}}', sample['question'])
    
    print(f"\n=== ACTUAL BOOLQ PROMPT ===")
    print(f"Full prompt:")
    print(repr(prompt))
    print(f"\nPrompt ends with:")
    print(repr(prompt[-100:]))
    
    # Check the choices
    choices = boolq_task.config.doc_to_choice
    target_delimiter = boolq_task.config.target_delimiter
    
    print(f"\n=== CHOICES ===")
    for i, choice in enumerate(choices):
        full_choice = target_delimiter + choice
        print(f"Choice {i}: {repr(full_choice)}")
    
    # Show what we were using vs what we should use
    print(f"\n=== COMPARISON ===")
    my_context = f"Passage: {sample['passage']}\nQuestion: {sample['question']}\nAnswer:"
    print(f"What I was using:")
    print(repr(my_context[-100:]))
    
    print(f"What BoolQ actually uses:")
    print(repr(prompt[-100:]))
    
    print(f"Are they the same? {my_context == prompt}")
    
    # Tokenize both to see the difference
    model_path = "Qwen/Qwen2.5-0.5B"
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        try:
            model_path = snapshot_download(repo_id=model_path, local_files_only=True)
        except:
            model_path = snapshot_download(repo_id=model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    my_tokens = tokenizer.encode(my_context, add_special_tokens=True)
    correct_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    
    print(f"\n=== TOKENIZATION COMPARISON ===")
    print(f"My approach: {len(my_tokens)} tokens")
    print(f"Correct approach: {len(correct_tokens)} tokens")
    print(f"Token difference: {len(correct_tokens) - len(my_tokens)}")
    
    if my_tokens != correct_tokens:
        print(f"DIFFERENT TOKENIZATION!")
        print(f"Last 10 tokens - mine: {my_tokens[-10:]}")
        print(f"Last 10 tokens - correct: {correct_tokens[-10:]}")
        
        # Find where they diverge
        min_len = min(len(my_tokens), len(correct_tokens))
        for i in range(min_len):
            if my_tokens[i] != correct_tokens[i]:
                print(f"First difference at position {i}:")
                print(f"  Mine: {my_tokens[i]} = '{tokenizer.decode([my_tokens[i]])}'")
                print(f"  Correct: {correct_tokens[i]} = '{tokenizer.decode([correct_tokens[i]])}'")
                break

if __name__ == "__main__":
    debug_boolq_prompt()