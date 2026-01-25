#!/usr/bin/env python3
"""
Generate full BoolQ context string for a specific question ID.
"""

import argparse
from datasets import load_dataset

def generate_boolq_context(question_id):
    """Generate the full context string for a BoolQ question by ID"""
    
    # Load the BoolQ dataset
    dataset = load_dataset('boolq', split='validation')
    
    if question_id < 0 or question_id >= len(dataset):
        print(f"Error: Question ID {question_id} is out of range (0-{len(dataset)-1})")
        return None
    
    # Get the specific question
    example = dataset[question_id]
    
    # Extract components
    passage = example['passage']
    question = example['question']
    ground_truth = example['answer']  # True/False
    
    # Format like BoolQ evaluation
    context = f"{passage}\nQuestion: {question}\nAnswer:"
    
    print(f"=== BoolQ Question {question_id} ===")
    print(f"Ground Truth: {ground_truth}")
    print(f"Context Length: {len(context)} characters")
    print("\nFull Context (Python string):")
    
    # Escape quotes, backslashes, and newlines for proper Python string literal
    escaped_context = context.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
    print(f"context = '{escaped_context}'")
    
    print("\nAlternative (using triple quotes):")
    print(f'context = """{context}"""')
    
    return context

def main():
    parser = argparse.ArgumentParser(description='Generate BoolQ context for a question ID')
    parser.add_argument('question_id', type=int, help='Question ID (0-based index)')
    parser.add_argument('--show-tokens', action='store_true', help='Show tokenized length')
    
    args = parser.parse_args()
    
    context = generate_boolq_context(args.question_id)
    
    if context and args.show_tokens:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokens = tokenizer.encode(context, add_special_tokens=True)
        print(f"\nTokenized Length: {len(tokens)} tokens")
        print(f"First 10 tokens: {tokens[:10]}")
        print(f"Last 10 tokens: {tokens[-10:]}")

if __name__ == "__main__":
    main()