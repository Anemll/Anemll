#!/usr/bin/env python3
"""
Simple MLX evaluation script that produces JSON output compatible with batch script.
"""

import argparse
import json
from pathlib import Path
from mlx_lm.utils import load
from datasets import load_dataset
import mlx.core as mx

def evaluate_mlx_boolq(model_path, skip, limit, output_path):
    """Evaluate MLX model on BoolQ and save results in compatible format"""
    
    print(f"Loading MLX model: {model_path}")
    model, tokenizer = load(model_path)
    
    print(f"Loading BoolQ dataset...")
    dataset = load_dataset("google/boolq", split="validation")
    
    correct = 0
    total = 0
    
    for i in range(skip, skip + limit):
        if i >= len(dataset):
            break
            
        item = dataset[i]
        question = item['question']
        passage = item['passage']
        ground_truth = item['answer']  # True/False
        
        # Format like BoolQ evaluation
        context = f"{passage}\nQuestion: {question}\nAnswer:"
        
        # Tokenize context
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        context_mx = mx.array([context_tokens])
        
        # Get logits for next token
        model.eval()
        logits = model(context_mx)
        last_logits = logits[0, -1, :]  # Last token's logits
        
        # Get scores for " no" and " yes" 
        no_tokens = tokenizer.encode(" no", add_special_tokens=False)
        yes_tokens = tokenizer.encode(" yes", add_special_tokens=False)
        
        no_token = no_tokens[0]
        yes_token = yes_tokens[0]
        
        no_score = last_logits[no_token].item()
        yes_score = last_logits[yes_token].item()
        
        mlx_prediction = "yes" if yes_score > no_score else "no"
        ground_truth_str = "yes" if ground_truth else "no"
        
        if mlx_prediction == ground_truth_str:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    # Create results in lm-eval format
    results = {
        "boolq": {
            "alias": "boolq",
            "acc,none": accuracy,
            "acc_stderr,none": 0.0  # Placeholder
        }
    }
    
    # Write to output file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=4))
    
    print(f"MLX accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results saved to: {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="MLX model path")
    parser.add_argument("--skip", type=int, default=0, help="Skip N questions")
    parser.add_argument("--limit", type=int, default=100, help="Evaluate N questions")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    
    args = parser.parse_args()
    
    evaluate_mlx_boolq(args.model, args.skip, args.limit, args.output)

if __name__ == "__main__":
    main()