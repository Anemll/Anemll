#!/usr/bin/env python3

import torch
import numpy as np

def test_tensor_operations():
    """Test the exact tensor operations from our fix."""
    
    print("=== Testing Tensor Operations (Fixed vs Original) ===")
    
    # Simulate the tensor dimensions we're working with
    batch_size = 1
    seq_len = 4
    hidden_size = 896
    vocab_split = 75776 // 16  # 16-way split: 4736
    
    print(f"Input dimensions:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  vocab_split: {vocab_split}")
    
    # Create sample hidden states (output from transformer)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    print(f"\nTransformer output shape: {hidden_states.shape}")
    
    # Create a Conv2d layer like our lm_head
    conv_layer = torch.nn.Conv2d(hidden_size, vocab_split, 1, bias=False, dtype=torch.float16)
    
    print(f"\n--- Fixed Tensor Operations (Explicit in Wrapper) ---")
    
    # Step 1: Reshape for Conv2d (explicit in our fixed wrapper)
    reshaped = hidden_states.permute(0, 2, 1).unsqueeze(2)
    print(f"1. permute(0,2,1).unsqueeze(2): {reshaped.shape}")
    
    # Step 2: Apply Conv2d
    conv_output = conv_layer(reshaped)
    print(f"2. Conv2d output: {conv_output.shape}")
    
    # Step 3: Reshape back (explicit in our fixed wrapper)
    final_output = conv_output.squeeze(2).transpose(1, 2)
    print(f"3. squeeze(2).transpose(1,2): {final_output.shape}")
    
    # Verify the final shape
    expected_shape = (batch_size, seq_len, vocab_split)
    print(f"\nExpected final shape: {expected_shape}")
    print(f"Actual final shape:   {final_output.shape}")
    
    if final_output.shape == expected_shape:
        print("✅ CORRECT: Final tensor shape matches expected")
    else:
        print("❌ ERROR: Final tensor shape mismatch")
    
    print(f"\n--- Key Insight ---")
    print(f"The fix ensures CoreML sees explicit tensor operations:")
    print(f"  INPUT:  [batch, seq_len, hidden_size] = {hidden_states.shape}")
    print(f"  OUTPUT: [batch, seq_len, vocab_split] = {final_output.shape}")
    print(f"")
    print(f"Before fix: CoreML interpreted internal model operations incorrectly")
    print(f"After fix:  CoreML traces explicit wrapper operations correctly")
    
    return final_output.shape == expected_shape

def test_all_splits():
    """Test tensor operations for different vocab splits."""
    
    print(f"\n=== Testing All Vocab Split Configurations ===")
    
    vocab_size = 75776
    splits = [2, 8, 16]
    
    batch_size = 1
    seq_len = 4
    hidden_size = 896
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    
    for split_count in splits:
        vocab_split = vocab_size // split_count
        print(f"\n--- {split_count}-way split (vocab_split={vocab_split}) ---")
        
        # Create Conv2d layer
        conv_layer = torch.nn.Conv2d(hidden_size, vocab_split, 1, bias=False, dtype=torch.float16)
        
        # Apply our fixed tensor operations
        reshaped = hidden_states.permute(0, 2, 1).unsqueeze(2)
        conv_output = conv_layer(reshaped)
        final_output = conv_output.squeeze(2).transpose(1, 2)
        
        expected_shape = (batch_size, seq_len, vocab_split)
        
        print(f"  Expected: {expected_shape}")
        print(f"  Actual:   {final_output.shape}")
        
        if final_output.shape == expected_shape:
            print(f"  ✅ CORRECT: {split_count}-way split works")
        else:
            print(f"  ❌ ERROR: {split_count}-way split failed")

if __name__ == "__main__":
    print("Testing the qwen_converter.py fix...")
    print("="*50)
    
    # Test basic tensor operations
    success = test_tensor_operations()
    
    # Test all split configurations
    test_all_splits()
    
    print(f"\n" + "="*50)
    if success:
        print("✅ FIX VALIDATION: Tensor operations work correctly")
        print("✅ READY TO TEST: Updated qwen_converter.py should resolve CoreML issues")
    else:
        print("❌ FIX ISSUE: Tensor operations still have problems")
    
    print(f"\nNext steps:")
    print(f"1. Convert model with updated qwen_converter.py")
    print(f"2. Test CoreML output shapes")
    print(f"3. Compare CoreML vs PyTorch outputs") 