#!/usr/bin/env python3
"""
Extract ANE scores for specific questions to compare directly with MLX scores.
This evaluates the same questions that showed low MLX confidence in segment 1800-1899.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# Configure CoreML for single-threaded mode
os.environ["COREML_PARTITION_LOADER_DISABLE_MULTI_ENGINE"] = "1"

# Add paths for imports
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/evaluate/ane')

# Import the ANELM class from the ANE harness
from evaluate_with_harness import ANELM
from datasets import load_dataset
import numpy as np
import torch

def evaluate_single_question_ane(model, question_data):
    """Evaluate a single question with ANE model and return prediction details"""
    passage = question_data['passage']
    question = question_data['question']
    ground_truth = question_data['answer']  # True/False
    
    # Format like BoolQ evaluation
    context = f"{passage}\nQuestion: {question}\nAnswer:"
    
    # Create simple request objects
    class SimpleRequest:
        def __init__(self, context, continuation):
            self.args = (context, continuation)
    
    # Create requests for " no" and " yes" (with space prefix)
    requests = [
        SimpleRequest(context, " no"),
        SimpleRequest(context, " yes")
    ]
    
    # Get loglikelihood scores
    results = model.loglikelihood(requests)
    
    no_score = results[0][0]   # " no" score 
    yes_score = results[1][0]  # " yes" score
    
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

def evaluate_specific_questions_ane(model_path, question_indices):
    """Evaluate specific questions with ANE model"""
    
    print(f"Loading ANE model from: {model_path}")
    
    # Initialize ANE model
    lm = ANELM(
        model_path,
        max_tokens=2048,
        use_chat_template=False,
        verbose_output=False,
        log_incorrect_answers=False
    )
    
    # Load dataset
    dataset = load_dataset("boolq", split="validation")
    
    print(f"Evaluating {len(question_indices)} specific questions with ANE...")
    
    ane_results = []
    
    for idx in question_indices:
        print(f"  Processing Q{idx}...")
        
        item = dataset[idx]
        result = evaluate_single_question_ane(lm, item)
        result['question_idx'] = idx
        result['question_text'] = item['question']
        result['passage'] = item['passage']
        
        ane_results.append(result)
        
        print(f"    Q{idx}: Ground truth: {result['ground_truth']}, ANE prediction: {result['prediction']}, Confidence: {result['confidence']:.3f}")
        print(f"    Scores: no={result['no_score']:.3f}, yes={result['yes_score']:.3f}")
    
    return ane_results

def load_mlx_results_for_comparison(mlx_results_file):
    """Load MLX results from previous analysis"""
    with open(mlx_results_file, 'r') as f:
        data = json.load(f)
    
    # Extract questions that MLX got correct with low confidence
    mlx_correct = [q for q in data['questions'] if q['is_correct']]
    mlx_correct_sorted = sorted(mlx_correct, key=lambda x: x['confidence'])
    
    # Get the question indices for the top candidates
    top_candidates = mlx_correct_sorted[:10]  # Top 10 low-confidence correct answers
    question_indices = [q['question_idx'] for q in top_candidates]
    
    return question_indices, top_candidates

def compare_ane_vs_mlx_scores(ane_results, mlx_candidates):
    """Compare ANE vs MLX scores for the same questions"""
    
    print(f"\n" + "=" * 80)
    print("ANE vs MLX Score Comparison for Low-Confidence MLX Questions")
    print("=" * 80)
    
    # Create lookup for ANE results
    ane_lookup = {r['question_idx']: r for r in ane_results}
    
    comparison_data = []
    
    for mlx_q in mlx_candidates:
        q_idx = mlx_q['question_idx']
        ane_q = ane_lookup.get(q_idx)
        
        if ane_q:
            mlx_correct = mlx_q['is_correct']
            ane_correct = ane_q['is_correct']
            
            # Determine the outcome
            if mlx_correct and not ane_correct:
                outcome = "MLX WINS"
            elif ane_correct and not mlx_correct:
                outcome = "ANE WINS"
            elif mlx_correct and ane_correct:
                outcome = "BOTH CORRECT"
            else:
                outcome = "BOTH WRONG"
            
            comparison_data.append({
                'question_idx': q_idx,
                'question_text': mlx_q['question_text'],
                'ground_truth': mlx_q['ground_truth'],
                'mlx_prediction': mlx_q['prediction'],
                'ane_prediction': ane_q['prediction'],
                'mlx_correct': mlx_correct,
                'ane_correct': ane_correct,
                'mlx_no_score': mlx_q['no_score'],
                'mlx_yes_score': mlx_q['yes_score'],
                'ane_no_score': ane_q['no_score'],
                'ane_yes_score': ane_q['yes_score'],
                'mlx_confidence': mlx_q['confidence'],
                'ane_confidence': ane_q['confidence'],
                'outcome': outcome
            })
            
            print(f"\nQ{q_idx}: {mlx_q['question_text'][:80]}...")
            print(f"  Ground truth: {mlx_q['ground_truth']}")
            print(f"  MLX: {mlx_q['prediction']} (confidence: {mlx_q['confidence']:.3f}) - {'✓' if mlx_correct else '✗'}")
            print(f"       no={mlx_q['no_score']:.3f}, yes={mlx_q['yes_score']:.3f}")
            print(f"  ANE: {ane_q['prediction']} (confidence: {ane_q['confidence']:.3f}) - {'✓' if ane_correct else '✗'}")
            print(f"       no={ane_q['no_score']:.3f}, yes={ane_q['yes_score']:.3f}")
            print(f"  OUTCOME: {outcome}")
    
    # Summary statistics
    outcomes = [item['outcome'] for item in comparison_data]
    mlx_wins = outcomes.count("MLX WINS")
    ane_wins = outcomes.count("ANE WINS") 
    both_correct = outcomes.count("BOTH CORRECT")
    both_wrong = outcomes.count("BOTH WRONG")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Questions analyzed: {len(comparison_data)}")
    print(f"MLX wins (MLX correct, ANE wrong): {mlx_wins}")
    print(f"ANE wins (ANE correct, MLX wrong): {ane_wins}")
    print(f"Both models correct: {both_correct}")
    print(f"Both models wrong: {both_wrong}")
    
    # Save detailed comparison
    output_dir = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev/segment_analysis")
    output_dir.mkdir(exist_ok=True)
    
    comparison_file = output_dir / "ane_vs_mlx_detailed_scores_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nDetailed comparison saved to: {comparison_file}")
    
    return comparison_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ane-model", required=True, help="ANE model path")
    parser.add_argument("--mlx-results", default="./tests/dev/segment_analysis/segment_1800_mlx_detailed.json",
                       help="MLX results file from previous analysis")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(123)
    torch.manual_seed(123)
    
    # Load MLX results to get the question indices we want to test
    print("Loading MLX results for comparison...")
    question_indices, mlx_candidates = load_mlx_results_for_comparison(args.mlx_results)
    
    print(f"Will evaluate {len(question_indices)} questions: {question_indices}")
    
    # Evaluate those same questions with ANE
    ane_results = evaluate_specific_questions_ane(args.ane_model, question_indices)
    
    # Compare the results
    compare_ane_vs_mlx_scores(ane_results, mlx_candidates)

if __name__ == "__main__":
    main()