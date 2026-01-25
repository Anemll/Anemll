#!/usr/bin/env python3
"""
Get MLX predictions for specific BoolQ questions to compare with ANE.
"""

from mlx_lm.utils import load
import mlx.core as mx
from datasets import load_dataset
import torch

def get_mlx_predictions_for_questions(model_path, skip, limit):
    """Get MLX predictions for specific BoolQ questions"""
    
    print(f"Loading MLX model: {model_path}")
    model, tokenizer = load(model_path)
    
    print(f"Loading BoolQ dataset...")
    dataset = load_dataset("google/boolq", split="validation")
    
    results = []
    
    for i in range(skip, skip + limit):
        if i >= len(dataset):
            break
            
        item = dataset[i]
        question = item['question']
        passage = item['passage']
        ground_truth = item['answer']  # True/False
        
        # Format like BoolQ evaluation
        context = f"{passage}\nQuestion: {question}\nAnswer:"
        
        print(f"\nQuestion {i+1}: {question}")
        print(f"Ground truth: {'yes' if ground_truth else 'no'}")
        
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
        
        # Apply softmax to get probabilities
        import math
        no_prob = math.exp(no_score) / (math.exp(no_score) + math.exp(yes_score))
        yes_prob = math.exp(yes_score) / (math.exp(no_score) + math.exp(yes_score))
        
        mlx_prediction = "yes" if yes_score > no_score else "no"
        confidence = abs(yes_score - no_score)
        
        print(f"MLX prediction: {mlx_prediction}")
        print(f"  no score: {no_score:.4f} (prob: {no_prob:.4f})")  
        print(f"  yes score: {yes_score:.4f} (prob: {yes_prob:.4f})")
        print(f"  confidence: {confidence:.4f}")
        
        correct = (mlx_prediction == ("yes" if ground_truth else "no"))
        print(f"  correct: {correct}")
        
        results.append({
            'idx': i,
            'question': question,
            'passage': passage[:100] + "...",
            'ground_truth': "yes" if ground_truth else "no",
            'mlx_prediction': mlx_prediction,
            'mlx_no_score': no_score,
            'mlx_yes_score': yes_score,
            'mlx_confidence': confidence,
            'correct': correct
        })
    
    # Print summary
    correct_count = sum(r['correct'] for r in results)
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\nMLX SUMMARY:")
    print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.1%})")
    print(f"Questions {skip} to {skip + limit - 1}")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="MLX model path")
    parser.add_argument("--skip", type=int, default=1702, help="Skip N questions")
    parser.add_argument("--limit", type=int, default=10, help="Evaluate N questions")
    
    args = parser.parse_args()
    
    results = get_mlx_predictions_for_questions(args.model, args.skip, args.limit)
    
    print("\nMLX PREDICTIONS:")
    print("-" * 60)
    for r in results:
        status = "✓" if r['correct'] else "✗"
        print(f"{status} Q{r['idx']}: {r['question']}")
        print(f"   Truth: {r['ground_truth']}, MLX: {r['mlx_prediction']} (conf: {r['mlx_confidence']:.3f})")

if __name__ == "__main__":
    main()