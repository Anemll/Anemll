#!/usr/bin/env python3
"""
Standalone test using HuggingFace transformers to get baseline probabilities
for BoolQ prompt tokens ' no' and ' yes' with the exact same prompt from our evaluation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

def test_boolq_transformers_baseline():
    """Test with standard HuggingFace transformers for BoolQ baseline comparison"""
    
    # Sample 31 from BoolQ (Benson & Hedges) - exact same as our evaluation
    context = 'Benson & Hedges -- Benson & Hedges is a British brand of cigarettes owned by either Philip Morris International, British American Tobacco, or Japan Tobacco, depending on the region. In the UK, they are registered in Old Bond Street in London, and are manufactured in Lisnafillan, Ballymena, Northern Ireland.\nQuestion: do they still make benson & hedges cigarettes?\nAnswer:'
    
    print("=" * 80)
    print("HuggingFace Transformers BoolQ Baseline Test")
    print("=" * 80)
    
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    # Tokenize context
    context_tokens = tokenizer.encode(context, add_special_tokens=True)
    no_token_id = tokenizer.encode(" no", add_special_tokens=False)[0]
    yes_token_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
    
    print(f"\nContext: {repr(context)}")
    print(f"Context tokens length: {len(context_tokens)}")
    print(f"First 5 context tokens: {context_tokens[:5]}")
    print(f"Last 5 context tokens: {context_tokens[-5:]}")
    print(f"Token IDs: ' no'={no_token_id}, ' yes'={yes_token_id}")
    
    # Run inference
    input_ids = torch.tensor([context_tokens], dtype=torch.long)
    
    with torch.no_grad():
        print(f"\nRunning inference...")
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract specific token probabilities
        no_prob = probs[no_token_id].item()
        yes_prob = probs[yes_token_id].item()
        no_log_prob = log_probs[no_token_id].item()
        yes_log_prob = log_probs[yes_token_id].item()
        
        # Get top predicted token
        top_token_id = torch.argmax(logits).item()
        top_token_text = tokenizer.decode([top_token_id])
        top_prob = probs[top_token_id].item()
        
        print(f"\nResults:")
        print(f"  ' no' probability: {no_prob:.6f}")
        print(f"  ' yes' probability: {yes_prob:.6f}")
        print(f"  ' no' log probability: {no_log_prob:.4f}")
        print(f"  ' yes' log probability: {yes_log_prob:.4f}")
        print(f"  Predicted: '{top_token_text}' (ID: {top_token_id}, prob: {top_prob:.6f})")
        
        # Determine which is higher
        if yes_log_prob > no_log_prob:
            print(f"  Model predicts: YES (score diff: {yes_log_prob - no_log_prob:.4f})")
        else:
            print(f"  Model predicts: NO (score diff: {no_log_prob - yes_log_prob:.4f})")
    
    print("=" * 80)
    print("Comparison with our evaluation results:")
    # Show actual results from this test
    prediction_emoji = "✅" if yes_log_prob > no_log_prob else "❌"
    prediction_text = "YES" if yes_log_prob > no_log_prob else "NO"
    print(f"  HF Transformers (current): no={no_log_prob:.4f}, yes={yes_log_prob:.4f} → Predicts {prediction_text} {prediction_emoji}")
    print("  MLX:                       no=-3.2088, yes=-2.7088 → Predicts YES ✅")
    print("=" * 80)

if __name__ == "__main__":
    test_boolq_transformers_baseline()