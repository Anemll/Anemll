#!/usr/bin/env python3
"""
Standalone test using ANEMLL PyTorch Qwen2.5 model to get probabilities
for BoolQ prompt tokens ' no' and ' yes' with the exact same prompt from our evaluation
"""

import argparse
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer
import torch.nn.functional as F

# Set offline mode to prevent network calls
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# Performance optimizations
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Add paths for imports
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

# PyTorch performance optimizations
torch.set_num_threads(4)
if torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True

# Import your custom Qwen2.5 model
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def test_boolq_pytorch_baseline():
    """Test with ANEMLL PyTorch Qwen2.5 model for BoolQ baseline comparison"""
    
    # Sample 31 from BoolQ (Benson & Hedges) - exact same as our evaluation
    context = 'Benson & Hedges -- Benson & Hedges is a British brand of cigarettes owned by either Philip Morris International, British American Tobacco, or Japan Tobacco, depending on the region. In the UK, they are registered in Old Bond Street in London, and are manufactured in Lisnafillan, Ballymena, Northern Ireland.\nQuestion: do they still make benson & hedges cigarettes?\nAnswer:'
    
    print("=" * 80)
    print("ANEMLL PyTorch Qwen2.5 BoolQ Baseline Test")
    print("=" * 80)
    
    # Load model and tokenizer
    model_path = "Qwen/Qwen2.5-0.5B"
    print(f"Loading PyTorch Qwen2.5 model from: {model_path}")
    
    # Check if it's a local path or HuggingFace ID
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        try:
            print(f"Checking HuggingFace cache for {model_path}...")
            local_path = snapshot_download(repo_id=model_path, local_files_only=True)
            print(f"Found in cache: {local_path}")
            model_path = local_path
        except Exception:
            print(f"Not in cache, downloading {model_path}...")
            local_path = snapshot_download(repo_id=model_path)
            print(f"Downloaded to: {local_path}")
            model_path = local_path
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create config
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    
    # Enable KV cache
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    
    # Load pretrained weights
    print(f"Loading pretrained weights...")
    success = model.load_pretrained_weights(model_path)
    if not success:
        raise RuntimeError(f"Failed to load pretrained weights from {model_path}")

    print("\n--- Debug: Model Initialization ---")
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        print(f"lm_head.weight shape: {model.lm_head.weight.shape}")
        print(f"lm_head.weight norm: {torch.norm(model.lm_head.weight).item():.4f}")
        print(f"lm_head.weight first 5: {model.lm_head.weight.flatten()[:5].tolist()}")
    else:
        print("lm_head or its weight not found in model.")
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'weight'):
        print(f"model.embed_tokens.weight norm: {torch.norm(model.model.embed_tokens.weight).item():.4f}")
    else:
        print("model.embed_tokens.weight not found.")
    
    model.eval()
    
    print("Using CPU (ANEMLL model has MPS compatibility issues)")
    device_name = 'cpu'
    
    print(f"PyTorch Qwen2.5 model loaded with KV cache enabled")
    
    # Store context length for proper mask creation
    context_length = config.context_length
    
    # Create causal mask once
    def make_causal_mask(length, start):
        """Create causal attention mask."""
        mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
        row_indices = np.arange(length).reshape(length, 1)
        col_indices = np.arange(length).reshape(1, length)
        mask[:, :, col_indices <= (row_indices + start)] = 0
        return mask
    
    causal_mask_data = make_causal_mask(context_length, 0)
    causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
    
    print(f"Context length: {context_length}")
    
    # Tokenize context
    context_tokens = tokenizer.encode(context, add_special_tokens=True)
    no_token_id = tokenizer.encode(" no", add_special_tokens=False)[0]
    yes_token_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
    
    print(f"\nContext: {repr(context)}")
    print(f"Context tokens length: {len(context_tokens)}")
    print(f"First 5 context tokens: {context_tokens[:5]}")
    print(f"Last 5 context tokens: {context_tokens[-5:]}")
    print(f"Token IDs: ' no'={no_token_id}, ' yes'={yes_token_id}")
    
    # Clear KV cache
    if hasattr(model.model, 'clear_kv_cache'):
        model.model.clear_kv_cache()
    
    # Handle context overflow
    if len(context_tokens) > context_length - 1:
        context_tokens = context_tokens[-(context_length - 1):]
    
    prompt_length = len(context_tokens)
    
    with torch.no_grad():
        print(f"\nRunning inference...")
        device = next(model.parameters()).device
        
        # Batch prefill strategy
        input_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)
        
        # Run batched prefill (64 tokens at a time)
        batch_pos = 0
        batch_size = 64
        while batch_pos < prompt_length:
            batch_end = min(batch_pos + batch_size, prompt_length)
            current_batch_size = batch_end - batch_pos
            
            # Get current batch
            batch_input = input_ids[:, batch_pos:batch_end]
            
            # Pad to batch size if needed
            batch_input = F.pad(batch_input, (0, batch_size - current_batch_size), value=0)
            
            # Create position IDs and masks
            position_ids = torch.arange(batch_pos, batch_pos + batch_size, dtype=torch.long, device=device)
            batch_causal_mask = causal_mask[:, :, batch_pos:batch_pos + batch_size, :]
            update_mask = torch.zeros(1, batch_size, device=device)
            
            # Prefill this batch
            model(
                batch_input,
                update_mask,
                position_ids,
                batch_causal_mask,
                torch.tensor(batch_pos, dtype=torch.long, device=device),
                IN_PREFILL=True
            )
            
            batch_pos = batch_end
        
        # Single inference step to get next token logits
        current_pos = prompt_length
        last_token = torch.tensor([[context_tokens[-1]]], dtype=torch.long, device=device)
        
        # Create update mask for current position
        update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16, device=device)
        update_mask[0, 0, current_pos, 0] = 1.0
        
        # Run inference
        outputs = model(
            input_ids=last_token,
            causal_mask=causal_mask[:, :, current_pos:current_pos+1, :],
            position_ids=torch.tensor([current_pos], dtype=torch.long, device=device),
            current_pos=torch.tensor(current_pos, dtype=torch.long, device=device)
        )
        
        # Extract hidden states before lm_head
        hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else outputs
        print("\n--- Debug: Hidden States Before LM Head ---")
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Hidden states norm: {torch.norm(hidden_states).item():.4f}")
        print(f"Hidden states first 5: {hidden_states.flatten()[:5].tolist()}")

        # Extract logits and compute log probabilities
        logits = outputs[0, -1, :]
        print("\n--- Debug: Raw Logits from LM Head ---")
        print(f"Raw logits shape: {logits.shape}")
        print(f"Raw logits norm: {torch.norm(logits).item():.4f}")
        print(f"Raw logits first 5: {logits.flatten()[:5].tolist()}")
        
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        
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
    print("Comparison with baseline and evaluation results:")
    print("  HF Transformers: no=-3.3420, yes=-2.8594 → Predicts YES ✅")
    print("  MLX:             no=-3.2088, yes=-2.7088 → Predicts YES ✅")
    # Show actual results from this test
    prediction_emoji = "✅" if yes_log_prob > no_log_prob else "❌"
    prediction_text = "YES" if yes_log_prob > no_log_prob else "NO"
    print(f"  PyTorch (current): no={no_log_prob:.4f}, yes={yes_log_prob:.4f} → Predicts {prediction_text} {prediction_emoji}")
    print("=" * 80)

if __name__ == "__main__":
    test_boolq_pytorch_baseline()