#!/usr/bin/env python3
"""Compare CoreML model output directly with PyTorch model output."""

import torch
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
import sys
import glob
import os
from pathlib import Path

# Add our models to path
sys.path.append('.')
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig

def test_coreml_vs_pytorch():
    """Compare CoreML and PyTorch model outputs for identical inputs."""
    
    print("🔍 CoreML vs PyTorch Model Comparison")
    print("="*60)
    
    # Load tokenizer
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    if not model_dirs:
        print("❌ Error: Qwen model not found in cache")
        return False
    
    model_dir = model_dirs[0]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    config = QwenConfig.from_json(f"{model_dir}/config.json")
    pytorch_model = QwenForCausalLM(config)
    pytorch_model.load_pretrained_weights(model_dir)
    pytorch_model.eval()
    
    # Load CoreML model
    coreml_path = "../qwen-test/qwen.mlpackage"
    if not Path(coreml_path).exists():
        print(f"❌ Error: CoreML model not found at {coreml_path}")
        return False
    
    print("Loading CoreML model...")
    coreml_model = ct.models.MLModel(coreml_path)
    
    # Test data
    test_prompt = "What is Apple Neural Engine?"
    print(f"Test prompt: '{test_prompt}'")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    
    print(f"Tokenized to {seq_len} tokens: {input_ids.tolist()[0]}")
    
    # For this test, we'll use a 4-token window from the prompt
    # Take the last 4 tokens to match CoreML model expectations
    if seq_len >= 4:
        test_tokens = input_ids[0, -4:].tolist()
    else:
        # Pad if needed
        test_tokens = input_ids[0].tolist()
        while len(test_tokens) < 4:
            test_tokens.append(test_tokens[-1])
    
    print(f"Test tokens (4-token window): {test_tokens}")
    
    # Create inputs for both models
    batch_input_ids = torch.tensor([test_tokens], dtype=torch.long)
    position_ids = torch.arange(4, dtype=torch.long).unsqueeze(0)
    current_pos = torch.tensor([3], dtype=torch.long)  # Last position
    update_mask = torch.ones_like(batch_input_ids, dtype=torch.bool)
    
    # Create causal mask
    causal_mask = torch.full((1, 1, 4, 4), -torch.inf, dtype=torch.float16)
    for i in range(4):
        causal_mask[:, :, i, :i+1] = 0
    
    print(f"\nInput shapes:")
    print(f"  input_ids: {batch_input_ids.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  causal_mask: {causal_mask.shape}")
    print(f"  current_pos: {current_pos.shape}")
    
    # Run PyTorch model
    print(f"\n🔥 PyTorch Model Forward Pass:")
    with torch.no_grad():
        pytorch_logits = pytorch_model(
            input_ids=batch_input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
    
    print(f"PyTorch logits shape: {pytorch_logits.shape}")
    pytorch_last_logits = pytorch_logits[0, -1, :]  # Last position logits
    pytorch_top5 = torch.topk(pytorch_last_logits, 5)
    print(f"PyTorch top 5 tokens: {pytorch_top5.indices.tolist()}")
    print(f"PyTorch top 5 values: {pytorch_top5.values.tolist()}")
    print(f"PyTorch logits range: [{pytorch_last_logits.min().item():.3f}, {pytorch_last_logits.max().item():.3f}]")
    
    # Convert inputs for CoreML
    coreml_input_ids = np.array([test_tokens], dtype=np.int32)  # [1, 4]
    coreml_position_ids = np.arange(4, dtype=np.int32)  # [4]
    coreml_current_pos = np.array([3], dtype=np.int32)  # [1]
    
    # Create larger causal mask for CoreML (expects 512 kv positions)
    coreml_causal_mask = np.full((1, 1, 4, 512), -np.inf, dtype=np.float16)
    for i in range(4):
        coreml_causal_mask[:, :, i, :i+1] = 0
    
    coreml_inputs = {
        'input_ids': coreml_input_ids,
        'position_ids': coreml_position_ids,
        'causal_mask': coreml_causal_mask,
        'current_pos': coreml_current_pos
    }
    
    # Run CoreML model
    print(f"\n🍎 CoreML Model Forward Pass:")
    try:
        coreml_outputs = coreml_model.predict(coreml_inputs)
        
        # Combine 16-way split logits
        logits_parts = []
        for i in range(1, 17):
            key = f'logits{i}'
            if key in coreml_outputs:
                logits_parts.append(coreml_outputs[key])
        
        if logits_parts:
            coreml_logits = np.concatenate(logits_parts, axis=-1)  # [1, 4, vocab_size]
            print(f"CoreML logits shape: {coreml_logits.shape}")
            
            coreml_last_logits = coreml_logits[0, -1, :]  # Last position logits
            coreml_top5_indices = np.argsort(coreml_last_logits)[-5:][::-1]
            coreml_top5_values = coreml_last_logits[coreml_top5_indices]
            
            print(f"CoreML top 5 tokens: {coreml_top5_indices.tolist()}")
            print(f"CoreML top 5 values: {coreml_top5_values.tolist()}")
            print(f"CoreML logits range: [{coreml_last_logits.min():.3f}, {coreml_last_logits.max():.3f}]")
            
        else:
            print("❌ Failed to combine CoreML logits")
            return False
            
    except Exception as e:
        print(f"❌ CoreML inference failed: {e}")
        return False
    
    # Compare outputs
    print(f"\n🔍 COMPARISON:")
    print("-" * 40)
    
    # Convert PyTorch logits to numpy for comparison
    pytorch_last_logits_np = pytorch_last_logits.cpu().numpy()
    
    # Compare logits
    logits_diff = np.abs(pytorch_last_logits_np - coreml_last_logits)
    max_diff = logits_diff.max()
    mean_diff = logits_diff.mean()
    
    print(f"Max logits difference: {max_diff:.6f}")
    print(f"Mean logits difference: {mean_diff:.6f}")
    
    # Check top token alignment
    pytorch_top1 = pytorch_top5.indices[0].item()
    coreml_top1 = coreml_top5_indices[0]
    
    print(f"PyTorch top token: {pytorch_top1} ('{tokenizer.decode([pytorch_top1])}')")
    print(f"CoreML top token: {coreml_top1} ('{tokenizer.decode([coreml_top1])}')")
    print(f"Top tokens match: {pytorch_top1 == coreml_top1}")
    
    # Detailed comparison for specific tokens
    print(f"\nDetailed logits comparison:")
    for i in range(5):
        pt_idx = pytorch_top5.indices[i].item()
        pt_val = pytorch_top5.values[i].item()
        cm_val = coreml_last_logits[pt_idx]
        diff = abs(pt_val - cm_val)
        
        print(f"  Token {pt_idx} ('{tokenizer.decode([pt_idx])}'): PyTorch={pt_val:.3f}, CoreML={cm_val:.3f}, diff={diff:.3f}")
    
    # Overall assessment
    if max_diff < 0.1 and pytorch_top1 == coreml_top1:
        print(f"\n✅ PASS: Models produce very similar outputs")
        return True
    elif max_diff < 1.0:
        print(f"\n⚠️  WARN: Models have small differences (max diff: {max_diff:.3f})")
        return True
    else:
        print(f"\n❌ FAIL: Models have significant differences (max diff: {max_diff:.3f})")
        return False

if __name__ == "__main__":
    success = test_coreml_vs_pytorch()
    if success:
        print("\n🎉 Comparison completed!")
    else:
        print("\n❌ Comparison failed!")
        sys.exit(1) 