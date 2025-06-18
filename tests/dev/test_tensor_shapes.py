#!/usr/bin/env python3

import torch
import numpy as np
import sys
import os

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *
from anemll.ane_converter.qwen_converter import test_conversion

def test_tensor_transforms():
    """Test the exact tensor transformations happening in Conv2d operations."""
    
    print("=== Testing Conv2d Tensor Transformations ===")
    
    # Create a simple test
    batch_size = 1
    seq_len = 4
    hidden_size = 896
    vocab_size = 75776
    
    # Create sample hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    print(f"Original hidden_states shape: {hidden_states.shape}")
    
    # Create Conv2d layers like in our model
    vocab_split = vocab_size // 16
    print(f"vocab_split: {vocab_split}")
    
    conv_layer = torch.nn.Conv2d(hidden_size, vocab_split, 1, bias=False, dtype=torch.float16)
    
    # Apply the same transformations as in qwen_model.py
    print("\n--- PyTorch Transformations ---")
    
    # Step 1: Reshape for Conv2d
    reshaped = hidden_states.permute(0, 2, 1).unsqueeze(2)
    print(f"After permute(0,2,1).unsqueeze(2): {reshaped.shape}")
    
    # Step 2: Apply Conv2d  
    conv_output = conv_layer(reshaped)
    print(f"After Conv2d: {conv_output.shape}")
    
    # Step 3: Reshape back
    final_output = conv_output.squeeze(2).transpose(1, 2)
    print(f"After squeeze(2).transpose(1,2): {final_output.shape}")
    
    print(f"\nExpected final shape: [batch={batch_size}, seq_len={seq_len}, vocab_split={vocab_split}]")
    print(f"Actual final shape: {final_output.shape}")
    
    # Test what CoreML might be expecting
    print("\n--- Expected CoreML Behavior ---")
    print(f"If CoreML expects [batch, seq_len, vocab_split]: {(batch_size, seq_len, vocab_split)}")
    print(f"But gets transposed version: {(batch_size, vocab_split, seq_len)}")
    
    # Test if we need to transpose differently for CoreML
    print("\n--- Potential Fix ---")
    alternative_output = conv_output.squeeze(2)  # Don't transpose
    print(f"Alternative (no transpose): {alternative_output.shape}")
    
    return final_output, alternative_output

def test_coreml_shapes():
    """Test actual CoreML model shapes."""
    
    print("\n=== Testing Actual CoreML Model ===")
    
    # Use our actual model path
    model_paths = [
        "~/SourceRelease/GITHUB/ML_playground/private-anemll/Qwen3-0.6B",
        "downloaded_models/Qwen3-0.6B"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(os.path.join(path, "config.json")):
            model_path = path
            break
    
    if not model_path:
        print(f"Model not found in any of: {model_paths}")
        return
    
    print(f"Using model path: {model_path}")
    config_path = os.path.join(model_path, "config.json")
    
    config = QwenConfig.from_json(config_path)
    model = QwenForCausalLM(config, enable_coreml=True)
    model.load_pretrained_weights(model_path)
    
    # Create test input
    input_ids = torch.tensor([[9195, 374, 8868, 61688]], dtype=torch.int32)  # "What is Apple Neural"
    seq_len = input_ids.shape[1]
    
    position_ids = torch.arange(seq_len, dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, seq_len, config.state_length), dtype=torch.float16)
    current_pos = torch.zeros((1,), dtype=torch.int32)
    
    print(f"Input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  position_ids: {position_ids.shape}")  
    print(f"  causal_mask: {causal_mask.shape}")
    print(f"  current_pos: {current_pos.shape}")
    
    # Run PyTorch model
    print("\n--- PyTorch Model Output ---")
    with torch.no_grad():
        pytorch_output = model(
            input_ids=input_ids,
            update_mask=torch.zeros((1, 1, config.state_length, 1), dtype=torch.float16),
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False,
        )
    
    if isinstance(pytorch_output, tuple):
        print(f"PyTorch returns {len(pytorch_output)} tensors:")
        for i, tensor in enumerate(pytorch_output):
            print(f"  logits{i+1}: {tensor.shape}")
    else:
        print(f"PyTorch single output: {pytorch_output.shape}")
    
    # Test with existing CoreML model if available
    coreml_path = "qwen_test.mlpackage"
    if os.path.exists(coreml_path):
        print(f"\n--- Testing Existing CoreML Model: {coreml_path} ---")
        try:
            import coremltools as ct
            mlmodel = ct.models.MLModel(coreml_path)
            
            # Run CoreML prediction
            coreml_input = {
                "input_ids": input_ids.numpy().astype(np.int32),
                "position_ids": position_ids.numpy().astype(np.int32),
                "causal_mask": causal_mask.numpy().astype(np.float16),
                "current_pos": current_pos.numpy().astype(np.int32),
            }
            
            coreml_output = mlmodel.predict(coreml_input)
            print(f"CoreML output shapes:")
            for key, value in coreml_output.items():
                print(f"  {key}: {value.shape}")
                
        except Exception as e:
            print(f"Failed to test existing CoreML model: {e}")
    else:
        print(f"No existing CoreML model found at {coreml_path}")

if __name__ == "__main__":
    # Test tensor transformations
    pytorch_out, alt_out = test_tensor_transforms()
    
    # Test actual model
    test_coreml_shapes() 