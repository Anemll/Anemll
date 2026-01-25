#!/usr/bin/env python3
"""
Extract PyTorch Qwen2.5 scores for specific questions to compare with MLX and ANE.
This evaluates the same questions using the ./anemll/models/qwen2_5_model.py implementation.
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# Add paths for imports
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# We'll use the standard transformers model for comparison

def load_pytorch_qwen_model(model_path):
    """Load PyTorch Qwen2.5 model and tokenizer"""
    print(f"Loading PyTorch Qwen2.5 model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load the PyTorch model using transformers
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    
    print(f"PyTorch model loaded successfully")
    print(f"Model type: {type(model)}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    return model, tokenizer

def evaluate_single_question_pytorch(model, tokenizer, question_data):
    """Evaluate a single question with PyTorch model and return prediction details"""
    passage = question_data['passage']
    question = question_data['question']
    ground_truth = question_data['answer']  # True/False
    
    # Format like BoolQ evaluation
    context = f"{passage}\nQuestion: {question}\nAnswer:"
    
    # Tokenize context
    context_tokens = tokenizer.encode(context, add_special_tokens=True, return_tensors="pt")
    
    # Get model logits
    with torch.no_grad():
        outputs = model(context_tokens)
        logits = outputs.logits[0, -1, :]  # Last token logits
    
    # Get scores for " no" and " yes" (with space prefix)
    no_tokens = tokenizer.encode(" no", add_special_tokens=False)
    yes_tokens = tokenizer.encode(" yes", add_special_tokens=False)
    
    no_token_id = no_tokens[0]
    yes_token_id = yes_tokens[0]
    
    # Get log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    
    no_score = log_probs[no_token_id].item()
    yes_score = log_probs[yes_token_id].item()
    
    prediction = "yes" if yes_score > no_score else "no"
    ground_truth_str = "yes" if ground_truth else "no"
    is_correct = (prediction == ground_truth_str)
    
    return {
        'prediction': prediction,
        'ground_truth': ground_truth_str,
        'is_correct': is_correct,
        'no_score': no_score,
        'yes_score': yes_score,
        'confidence': abs(yes_score - no_score),
        'full_context': context
    }

def evaluate_specific_questions_pytorch(model_path, question_indices):
    """Evaluate specific questions with PyTorch model"""
    
    # Load model and tokenizer
    model, tokenizer = load_pytorch_qwen_model(model_path)
    
    # Load dataset
    dataset = load_dataset("boolq", split="validation")
    
    print(f"Evaluating {len(question_indices)} specific questions with PyTorch...")
    
    pytorch_results = []
    
    for idx in question_indices:
        print(f"  Processing Q{idx}...")
        
        item = dataset[idx]
        result = evaluate_single_question_pytorch(model, tokenizer, item)
        result['question_idx'] = idx
        result['question_text'] = item['question']
        result['passage'] = item['passage']
        
        pytorch_results.append(result)
        
        print(f"    Q{idx}: Ground truth: {result['ground_truth']}, PyTorch prediction: {result['prediction']}, Confidence: {result['confidence']:.3f}")
        print(f"    Scores: no={result['no_score']:.3f}, yes={result['yes_score']:.3f}")
    
    return pytorch_results

def load_existing_comparison_data():
    """Load existing MLX vs ANE comparison data"""
    comparison_file = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev/segment_analysis/ane_vs_mlx_detailed_scores_comparison.json")
    
    with open(comparison_file, 'r') as f:
        data = json.load(f)
    
    # Extract question indices and create lookup
    question_indices = [item['question_idx'] for item in data]
    comparison_lookup = {item['question_idx']: item for item in data}
    
    return question_indices, comparison_lookup

def compare_three_models(pytorch_results, existing_comparison):
    """Compare PyTorch vs MLX vs ANE scores for the same questions"""
    
    print(f"\n" + "=" * 100)
    print("THREE-WAY COMPARISON: PyTorch vs MLX vs ANE")
    print("=" * 100)
    
    # Create lookup for PyTorch results
    pytorch_lookup = {r['question_idx']: r for r in pytorch_results}
    
    comparison_data = []
    
    for q_idx, existing in existing_comparison.items():
        pytorch_q = pytorch_lookup.get(q_idx)
        
        if pytorch_q:
            # Determine outcomes for each model
            pytorch_correct = pytorch_q['is_correct']
            mlx_correct = existing['mlx_correct']
            ane_correct = existing['ane_correct']
            
            # Create summary
            outcomes = []
            if pytorch_correct: outcomes.append("PyTorch")
            if mlx_correct: outcomes.append("MLX")
            if ane_correct: outcomes.append("ANE")
            
            if len(outcomes) == 3:
                outcome = "ALL CORRECT"
            elif len(outcomes) == 2:
                outcome = f"{' & '.join(outcomes)} CORRECT"
            elif len(outcomes) == 1:
                outcome = f"ONLY {outcomes[0]} CORRECT"
            else:
                outcome = "ALL WRONG"
            
            comparison_data.append({
                'question_idx': q_idx,
                'question_text': pytorch_q['question_text'],
                'ground_truth': pytorch_q['ground_truth'],
                'pytorch_prediction': pytorch_q['prediction'],
                'mlx_prediction': existing['mlx_prediction'],
                'ane_prediction': existing['ane_prediction'],
                'pytorch_correct': pytorch_correct,
                'mlx_correct': mlx_correct,
                'ane_correct': ane_correct,
                'pytorch_no_score': pytorch_q['no_score'],
                'pytorch_yes_score': pytorch_q['yes_score'],
                'mlx_no_score': existing['mlx_no_score'],
                'mlx_yes_score': existing['mlx_yes_score'],
                'ane_no_score': existing['ane_no_score'],
                'ane_yes_score': existing['ane_yes_score'],
                'pytorch_confidence': pytorch_q['confidence'],
                'mlx_confidence': existing['mlx_confidence'],
                'ane_confidence': existing['ane_confidence'],
                'outcome': outcome
            })
            
            print(f"\nQ{q_idx}: {pytorch_q['question_text'][:60]}...")
            print(f"  Ground truth: {pytorch_q['ground_truth'].upper()}")
            print(f"  PyTorch: {pytorch_q['prediction'].upper()} {'✓' if pytorch_correct else '✗'} (conf: {pytorch_q['confidence']:.3f}) - no={pytorch_q['no_score']:.3f}, yes={pytorch_q['yes_score']:.3f}")
            print(f"  MLX:     {existing['mlx_prediction'].upper()} {'✓' if mlx_correct else '✗'} (conf: {existing['mlx_confidence']:.3f}) - no={existing['mlx_no_score']:.3f}, yes={existing['mlx_yes_score']:.3f}")
            print(f"  ANE:     {existing['ane_prediction'].upper()} {'✓' if ane_correct else '✗'} (conf: {existing['ane_confidence']:.3f}) - no={existing['ane_no_score']:.3f}, yes={existing['ane_yes_score']:.3f}")
            print(f"  OUTCOME: {outcome}")
    
    # Summary statistics
    outcomes = [item['outcome'] for item in comparison_data]
    all_correct = outcomes.count("ALL CORRECT")
    pytorch_mlx_correct = outcomes.count("PyTorch & MLX CORRECT")
    pytorch_ane_correct = outcomes.count("PyTorch & ANE CORRECT")
    mlx_ane_correct = outcomes.count("MLX & ANE CORRECT")
    only_pytorch = outcomes.count("ONLY PyTorch CORRECT")
    only_mlx = outcomes.count("ONLY MLX CORRECT")
    only_ane = outcomes.count("ONLY ANE CORRECT")
    all_wrong = outcomes.count("ALL WRONG")
    
    print(f"\n" + "=" * 100)
    print("THREE-WAY SUMMARY")
    print("=" * 100)
    print(f"Questions analyzed: {len(comparison_data)}")
    print(f"All three models correct: {all_correct}")
    print(f"PyTorch & MLX correct (ANE wrong): {pytorch_mlx_correct}")
    print(f"PyTorch & ANE correct (MLX wrong): {pytorch_ane_correct}")
    print(f"MLX & ANE correct (PyTorch wrong): {mlx_ane_correct}")
    print(f"Only PyTorch correct: {only_pytorch}")
    print(f"Only MLX correct: {only_mlx}")
    print(f"Only ANE correct: {only_ane}")
    print(f"All models wrong: {all_wrong}")
    
    # Save detailed comparison
    output_dir = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev/segment_analysis")
    output_dir.mkdir(exist_ok=True)
    
    comparison_file = output_dir / "pytorch_vs_mlx_vs_ane_detailed_scores_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nDetailed three-way comparison saved to: {comparison_file}")
    
    return comparison_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="HuggingFace model path")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(123)
    torch.manual_seed(123)
    
    # Load existing MLX vs ANE comparison data
    print("Loading existing MLX vs ANE comparison data...")
    question_indices, existing_comparison = load_existing_comparison_data()
    
    print(f"Will evaluate {len(question_indices)} questions: {question_indices}")
    
    # Evaluate those same questions with PyTorch
    pytorch_results = evaluate_specific_questions_pytorch(args.model, question_indices)
    
    # Compare all three models
    compare_three_models(pytorch_results, existing_comparison)

if __name__ == "__main__":
    main()