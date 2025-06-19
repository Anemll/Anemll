#!/usr/bin/env python3
"""Compare PyTorch vs CoreML Qwen model outputs."""

import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

# Import our model from the proper module path

def load_pytorch_model():
    """Load our converted PyTorch model."""
    from anemll.ane_converter.qwen_converter import QwenModel, get_config
    
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    
    print("Loading PyTorch model...")
    config = get_config(model_path)
    model = QwenModel(config)
    
    # Load weights
    import torch
    state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
    
    # Split lm_head weights into 16 parts
    if 'lm_head.weight' in state_dict:
        lm_head_weight = state_dict.pop('lm_head.weight')  # Shape: [151936, 1024]
        vocab_size, hidden_size = lm_head_weight.shape
        chunk_size = vocab_size // 16
        
        for i in range(16):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk_weight = lm_head_weight[start_idx:end_idx]  # [9496, 1024]
            # Reshape to [9496, 1024, 1, 1] for Conv2d
            state_dict[f'lm_head16_{i+1}.weight'] = chunk_weight.unsqueeze(-1).unsqueeze(-1)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("PyTorch model loaded successfully!")
    return model

def test_models_comparison():
    print("=== PyTorch vs CoreML Model Comparison ===\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    model_name = "Qwen/Qwen2.5-0.5B"  # Use similar model for tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load PyTorch model
    pytorch_model = load_pytorch_model()
    
    # Load CoreML model
    print("Loading CoreML model...")
    coreml_model = ct.models.MLModel("q.mlpackage")
    print("CoreML model loaded successfully!")
    
    # Test with the same input
    text = " How does it"
    print(f"\nTesting with text: '{text}'")
    
    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"Tokens: {tokens}")
    
    # Test first token
    token_id = tokens[0]  # " How"
    print(f"\nTesting single token: {token_id} ('{tokenizer.decode([token_id])}')")
    
    # === PyTorch Inference ===
    print("\n--- PyTorch Inference ---")
    with torch.no_grad():
        input_ids = torch.tensor([[token_id]], dtype=torch.long)
        pytorch_outputs = pytorch_model(input_ids)
        pytorch_logits = pytorch_outputs.logits[0, 0]  # [vocab_size]
        
        print(f"PyTorch logits shape: {pytorch_logits.shape}")
        print(f"PyTorch logits range: [{pytorch_logits.min():.4f}, {pytorch_logits.max():.4f}]")
        
        # Get top 10 predictions
        pytorch_top_indices = torch.argsort(pytorch_logits, descending=True)[:10]
        print("PyTorch top 10 predictions:")
        for i, idx in enumerate(pytorch_top_indices):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1:2d}. Token {idx.item():5d}: '{token}' (logit: {pytorch_logits[idx]:.4f})")
    
    # === CoreML Inference ===
    print("\n--- CoreML Inference ---")
    
    # Prepare inputs for CoreML
    batch_size = 1
    context_length = 256
    
    input_ids = np.array([[token_id]], dtype=np.int32)  # [1, 1]
    position_ids = np.array([0], dtype=np.int32)        # [1]
    causal_mask = np.ones((1, 1, 1, context_length), dtype=np.float32)  # [1, 1, 1, 256]
    current_pos = np.array([0], dtype=np.int32)  # [1]
    update_mask = np.zeros((1, 1, context_length, 1), dtype=np.float32)  # [1, 1, 256, 1]
    update_mask[0, 0, 0, 0] = 1.0  # Update position 0
    
    coreml_inputs = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'causal_mask': causal_mask,
        'current_pos': current_pos,
        'update_mask': update_mask
    }
    
    coreml_outputs = coreml_model.predict(coreml_inputs)
    
    # Concatenate all logits parts (16 parts)
    num_logits = 16
    logits_parts = []
    for i in range(1, num_logits + 1):
        key = f'logits{i}'
        if key in coreml_outputs:
            logits_parts.append(coreml_outputs[key])
    
    coreml_logits = np.concatenate(logits_parts, axis=-1)  # [1, 1, 151936]
    coreml_logits_1d = coreml_logits[0, 0]  # [151936]
    
    print(f"CoreML logits shape: {coreml_logits_1d.shape}")
    print(f"CoreML logits range: [{coreml_logits_1d.min():.4f}, {coreml_logits_1d.max():.4f}]")
    
    # Get top 10 predictions
    coreml_top_indices = np.argsort(coreml_logits_1d)[::-1][:10]
    print("CoreML top 10 predictions:")
    for i, idx in enumerate(coreml_top_indices):
        idx_int = int(idx)
        token = tokenizer.decode([idx_int])
        print(f"  {i+1:2d}. Token {idx_int:5d}: '{token}' (logit: {coreml_logits_1d[idx]:.4f})")
    
    # === Comparison ===
    print("\n--- Comparison ---")
    
    # Compare top 10 tokens
    pytorch_top_10 = [idx.item() for idx in pytorch_top_indices]
    coreml_top_10 = [int(idx) for idx in coreml_top_indices]
    
    print("Token ID comparison:")
    matches = 0
    for i in range(10):
        pytorch_token = pytorch_top_10[i]
        coreml_token = coreml_top_10[i]
        match = "✅" if pytorch_token == coreml_token else "❌"
        print(f"  Rank {i+1:2d}: PyTorch {pytorch_token:5d} vs CoreML {coreml_token:5d} {match}")
        if pytorch_token == coreml_token:
            matches += 1
    
    print(f"\nMatching tokens in top 10: {matches}/10")
    
    # Compare logit values for top tokens
    print("\nLogit value comparison for top 5:")
    for i in range(5):
        pytorch_token = pytorch_top_10[i]
        coreml_token = coreml_top_10[i]
        if pytorch_token == coreml_token:
            pytorch_logit = pytorch_logits[pytorch_token].item()
            coreml_logit = coreml_logits_1d[coreml_token]
            diff = abs(pytorch_logit - coreml_logit)
            print(f"  Token {pytorch_token:5d}: PyTorch {pytorch_logit:.4f} vs CoreML {coreml_logit:.4f} (diff: {diff:.4f})")
    
    # Overall assessment
    if matches >= 8:  # At least 8/10 tokens match
        print(f"\n🎉 SUCCESS: Models are very similar! ({matches}/10 tokens match)")
    elif matches >= 5:
        print(f"\n⚠️  PARTIAL: Models are somewhat similar ({matches}/10 tokens match)")
    else:
        print(f"\n❌ FAILURE: Models produce different results ({matches}/10 tokens match)")
    
    return matches >= 8

if __name__ == "__main__":
    try:
        success = test_models_comparison()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 