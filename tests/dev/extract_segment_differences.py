#!/usr/bin/env python3
"""
Extract specific questions where MLX and ANE differ in a given segment.
Focus on questions where MLX is correct but ANE is wrong.
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

# Add paths for imports
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')

import mlx.core as mx
from mlx_true_official_copy import MLXLM, chat_template_fn
from datasets import load_dataset

def evaluate_single_question(model, tokenizer, question_data):
    """Evaluate a single question with a model and return prediction details"""
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
        'full_context': context  # Add the complete prefill string
    }

def analyze_segment_differences(segment_start, segment_size=100):
    """Analyze differences between MLX and ANE in a specific segment"""
    
    print(f"Analyzing segment [{segment_start}..{segment_start + segment_size - 1}]")
    
    # Load dataset
    dataset = load_dataset("boolq", split="validation")
    segment_data = dataset.select(range(segment_start, segment_start + segment_size))
    
    # Initialize MLX model
    print("Loading MLX model...")
    mlx_model = MLXLM(
        "Qwen/Qwen2.5-0.5B",
        max_tokens=2048,
        use_chat_template=False,
    )
    MLXLM.apply_chat_template = chat_template_fn(**{})
    
    # For ANE, we'll need to use a different approach since ANELM requires converted models
    # For now, let's focus on getting the MLX predictions and compare with known ANE results
    
    print("Evaluating questions with MLX...")
    mlx_results = []
    
    for idx, item in enumerate(segment_data):
        question_idx = segment_start + idx
        
        result = evaluate_single_question(mlx_model, mlx_model.tokenizer, item)
        result['question_idx'] = question_idx
        result['question_text'] = item['question']
        result['passage'] = item['passage']
        
        mlx_results.append(result)
        
        if idx % 10 == 0:
            print(f"  Processed {idx + 1}/{segment_size} questions...")
    
    # Calculate MLX accuracy for this segment
    mlx_correct = sum(1 for r in mlx_results if r['is_correct'])
    mlx_accuracy = mlx_correct / len(mlx_results)
    
    print(f"\nMLX Segment Results:")
    print(f"  Correct: {mlx_correct}/{len(mlx_results)} ({mlx_accuracy:.1%})")
    
    # Based on the comparison results, ANE got 56/100 correct in segment 1800-1899
    # We can identify the likely differences
    
    # Save detailed results
    output_dir = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev/segment_analysis")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"segment_{segment_start}_mlx_detailed.json"
    
    detailed_results = {
        'segment_start': segment_start,
        'segment_size': segment_size,
        'mlx_accuracy': mlx_accuracy,
        'mlx_correct': mlx_correct,
        'questions': mlx_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed MLX results saved to: {results_file}")
    
    # Show questions where MLX was wrong (these might also be wrong for ANE)
    mlx_wrong = [r for r in mlx_results if not r['is_correct']]
    print(f"\nQuestions where MLX was wrong ({len(mlx_wrong)}):")
    for r in mlx_wrong[:5]:  # Show first 5
        print(f"  Q{r['question_idx']}: {r['question_text'][:80]}...")
        print(f"    Predicted: {r['prediction']}, Correct: {r['ground_truth']}")
    
    # Show questions where MLX was right with low confidence (might be wrong for ANE)
    mlx_right_low_conf = [r for r in mlx_results if r['is_correct'] and r['confidence'] < 1.0]
    mlx_right_low_conf.sort(key=lambda x: x['confidence'])
    
    print(f"\nQuestions where MLX was correct but with low confidence ({len(mlx_right_low_conf)}):")
    for r in mlx_right_low_conf[:10]:  # Show first 10
        print(f"  Q{r['question_idx']}: {r['question_text'][:80]}...")
        print(f"    Predicted: {r['prediction']}, Confidence: {r['confidence']:.3f}")
    
    return mlx_results

def create_ane_simulation_report(segment_start, mlx_results):
    """Create a report simulating ANE vs MLX differences based on known accuracies"""
    
    # From the comparison, we know:
    # MLX: 71/100 correct in segment 1800-1899
    # ANE: 56/100 correct in segment 1800-1899
    # Difference: 15 questions where MLX right, ANE wrong
    
    mlx_correct = [r for r in mlx_results if r['is_correct']]
    mlx_wrong = [r for r in mlx_results if not r['is_correct']]
    
    print(f"\n" + "=" * 80)
    print(f"ANE vs MLX Difference Analysis for Segment {segment_start}")
    print("=" * 80)
    
    print(f"Known results:")
    print(f"  MLX accuracy: 71/100 (71%)")
    print(f"  ANE accuracy: 56/100 (56%)")
    print(f"  Difference: 15 questions where MLX correct, ANE wrong")
    
    print(f"\nActual MLX results in this run:")
    print(f"  MLX correct: {len(mlx_correct)}/100")
    print(f"  MLX wrong: {len(mlx_wrong)}/100")
    
    # The questions most likely to be "MLX right, ANE wrong" are those where MLX was correct
    # but with lower confidence (closer to the decision boundary)
    mlx_correct_sorted = sorted(mlx_correct, key=lambda x: x['confidence'])
    
    print(f"\nQuestions most likely to be 'MLX correct, ANE wrong' (lowest MLX confidence):")
    for i, r in enumerate(mlx_correct_sorted[:10]):  # Show top 10 candidates with full context
        print(f"  {i+1:2d}. Q{r['question_idx']}: {r['question_text']}")
        print(f"      Ground truth: {r['ground_truth']}, MLX prediction: {r['prediction']}, Confidence: {r['confidence']:.3f}")
        print(f"      Scores: no={r['no_score']:.3f}, yes={r['yes_score']:.3f}")
        print(f"      FULL PREFILL CONTEXT:")
        print(f"      {r['full_context']}")
        print()
    
    # Save this analysis
    output_dir = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev/segment_analysis")
    analysis_file = output_dir / f"segment_{segment_start}_mlx_vs_ane_candidates.txt"
    
    with open(analysis_file, 'w') as f:
        f.write(f"MLX vs ANE Difference Analysis for Segment {segment_start}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Known results:\n")
        f.write(f"  MLX accuracy: 71/100 (71%)\n")
        f.write(f"  ANE accuracy: 56/100 (56%)\n")
        f.write(f"  Difference: 15 questions where MLX correct, ANE wrong\n\n")
        
        f.write(f"Questions most likely to be 'MLX correct, ANE wrong':\n")
        for i, r in enumerate(mlx_correct_sorted[:20]):
            f.write(f"{i+1:2d}. Q{r['question_idx']}: {r['question_text']}\n")
            f.write(f"    Ground truth: {r['ground_truth']}, MLX prediction: {r['prediction']}\n")
            f.write(f"    MLX confidence: {r['confidence']:.3f}\n")
            f.write(f"    Scores: no={r['no_score']:.3f}, yes={r['yes_score']:.3f}\n")
            f.write(f"    FULL PREFILL CONTEXT:\n")
            f.write(f"    {r['full_context']}\n\n")
    
    print(f"\nDetailed analysis saved to: {analysis_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-start", type=int, required=True, help="Starting index of segment to analyze")
    parser.add_argument("--segment-size", type=int, default=100, help="Size of segment")
    
    args = parser.parse_args()
    
    # Set random seed
    mx.random.seed(123)
    
    # Analyze the segment
    mlx_results = analyze_segment_differences(args.segment_start, args.segment_size)
    
    # Create difference analysis
    create_ane_simulation_report(args.segment_start, mlx_results)

if __name__ == "__main__":
    main()