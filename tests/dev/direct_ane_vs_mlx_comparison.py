#!/usr/bin/env python3
"""
Direct comparison of ANE vs MLX predictions on specific BoolQ questions.
This bypasses the evaluation harness and directly loads datasets and models.
"""

import sys
import os
sys.path.append('/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from mlx_lm.utils import load
from mlx_lm.models.qwen2 import Qwen2Model
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
import argparse

# Import ANE model components
from tests.chat import (
    load_models, run_prefill, load_metadata
)
from transformers import AutoTokenizer

def load_boolq_questions(skip, limit):
    """Load specific BoolQ questions"""
    dataset = load_dataset("google/boolq", split="validation")
    
    # Get the specific range of questions
    start_idx = skip
    end_idx = start_idx + limit
    
    questions = []
    for i in range(start_idx, min(end_idx, len(dataset))):
        item = dataset[i]
        questions.append({
            'idx': i,
            'question': item['question'],
            'passage': item['passage'],
            'answer': item['answer']  # True/False
        })
    
    return questions

def predict_with_ane(model_path, questions):
    """Get predictions using ANE model"""
    print(f"Loading ANE model from {model_path}")
    
    # Create args object for load_models
    class Args:
        def __init__(self):
            self.model = model_path
            self.d = model_path
            self.meta = os.path.join(model_path, 'meta.yaml')
            self.tokenizer = model_path
    
    args = Args()
    
    # Load metadata and models
    metadata = load_metadata(None, args)  
    embedding_model, lm_head_model, ffn_models = load_models(args, metadata)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    predictions = []
    
    for q in questions:
        # Format the question like BoolQ evaluation
        context = f"{q['passage']}\nQuestion: {q['question']}\nAnswer:"
        
        # Tokenize and get logits
        context_tokens = tokenizer.encode(context, add_special_tokens=True)
        
        # Run prefill to get the logits for next token
        logits = run_prefill(
            embedding_model, ffn_models, context_tokens,
            len(context_tokens), metadata['context_length'],
            metadata.get('batch_size', 64), None
        )
        
        # Get probabilities for " no" and " yes"
        no_token = tokenizer.encode(" no", add_special_tokens=False)[0]
        yes_token = tokenizer.encode(" yes", add_special_tokens=False)[0]
        
        no_logit = logits[no_token].item()
        yes_logit = logits[yes_token].item()
        
        prediction = "yes" if yes_logit > no_logit else "no"
        confidence = abs(yes_logit - no_logit)
        
        predictions.append({
            'idx': q['idx'],
            'question': q['question'],
            'ground_truth': "yes" if q['answer'] else "no",
            'ane_prediction': prediction,
            'ane_confidence': confidence,
            'ane_no_logit': no_logit,
            'ane_yes_logit': yes_logit
        })
        
        print(f"Q{q['idx']}: {q['question'][:50]}...")
        print(f"  ANE: {prediction} (conf: {confidence:.3f})")
    
    return predictions

def predict_with_mlx(model_path, questions):
    """Get predictions using MLX model"""
    print(f"Loading MLX model from {model_path}")
    
    model, tokenizer = load(model_path)
    
    predictions = []
    
    for q in questions:
        # Format the question like BoolQ evaluation
        context = f"{q['passage']}\nQuestion: {q['question']}\nAnswer:"
        
        # Tokenize
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        context_mx = mx.array([context_tokens])
        
        # Get logits
        with mx.no_grad():
            logits = model(context_mx)
            last_logits = logits[0, -1, :]  # Last token's logits
        
        # Get probabilities for " no" and " yes"
        no_token = tokenizer.encode(" no", add_special_tokens=False)[0]
        yes_token = tokenizer.encode(" yes", add_special_tokens=False)[0]
        
        no_logit = last_logits[no_token].item()
        yes_logit = last_logits[yes_token].item()
        
        prediction = "yes" if yes_logit > no_logit else "no"
        confidence = abs(yes_logit - no_logit)
        
        predictions.append({
            'idx': q['idx'],
            'question': q['question'],
            'ground_truth': "yes" if q['answer'] else "no",
            'mlx_prediction': prediction,
            'mlx_confidence': confidence,
            'mlx_no_logit': no_logit,
            'mlx_yes_logit': yes_logit
        })
        
        print(f"Q{q['idx']}: {q['question'][:50]}...")
        print(f"  MLX: {prediction} (conf: {confidence:.3f})")
    
    return predictions

def compare_predictions(ane_preds, mlx_preds):
    """Compare ANE vs MLX predictions"""
    print("\n" + "="*80)
    print("ANE vs MLX Prediction Comparison")
    print("="*80)
    
    disagreements = 0
    ane_correct = 0
    mlx_correct = 0
    
    for ane, mlx in zip(ane_preds, mlx_preds):
        assert ane['idx'] == mlx['idx']
        
        ane_correct += (ane['ane_prediction'] == ane['ground_truth'])
        mlx_correct += (mlx['mlx_prediction'] == mlx['ground_truth'])
        
        if ane['ane_prediction'] != mlx['mlx_prediction']:
            disagreements += 1
            print(f"\nDISAGREEMENT on Q{ane['idx']}:")
            print(f"  Question: {ane['question']}")
            print(f"  Ground truth: {ane['ground_truth']}")
            print(f"  ANE: {ane['ane_prediction']} (no:{ane['ane_no_logit']:.3f}, yes:{ane['ane_yes_logit']:.3f})")
            print(f"  MLX: {mlx['mlx_prediction']} (no:{mlx['mlx_no_logit']:.3f}, yes:{mlx['mlx_yes_logit']:.3f})")
        else:
            print(f"Q{ane['idx']}: AGREE - both predict '{ane['ane_prediction']}' (truth: {ane['ground_truth']})")
    
    total = len(ane_preds)
    print(f"\nSUMMARY:")
    print(f"Total questions: {total}")
    print(f"Disagreements: {disagreements} ({disagreements/total*100:.1f}%)")
    print(f"ANE accuracy: {ane_correct}/{total} ({ane_correct/total*100:.1f}%)")
    print(f"MLX accuracy: {mlx_correct}/{total} ({mlx_correct/total*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Direct ANE vs MLX comparison")
    parser.add_argument("--ane-model", required=True, help="Path to ANE model")
    parser.add_argument("--mlx-model", required=True, help="Path to MLX model")
    parser.add_argument("--skip", type=int, default=1702, help="Skip N questions")
    parser.add_argument("--limit", type=int, default=5, help="Evaluate N questions")
    
    args = parser.parse_args()
    
    print(f"Loading BoolQ questions {args.skip} to {args.skip + args.limit - 1}")
    questions = load_boolq_questions(args.skip, args.limit)
    
    print(f"\nEvaluating {len(questions)} questions...")
    ane_preds = predict_with_ane(args.ane_model, questions)
    mlx_preds = predict_with_mlx(args.mlx_model, questions)
    
    compare_predictions(ane_preds, mlx_preds)

if __name__ == "__main__":
    main()