#!/usr/bin/env python3
"""
Test the core evaluation functionality without dataset dependencies.
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# Add paths for imports
# Get the anemll root directory
anemll_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
evaluate_dir = os.path.join(anemll_root, "evaluate", "ane")
sys.path.append(evaluate_dir)

# Import the ANELM class from evaluate_with_harness
from evaluate_with_harness import ANELM

def test_anelm_basic(use_chat_template=None):
    """Test basic ANELM functionality without datasets."""
    print("Testing ANELM basic functionality...")
    
    model_path = "/tmp/test-qwen25-sp-quant-0.5b"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return False
        
    try:
        # Initialize the model
        print("Initializing ANELM...")
        print(f"Chat template: {'Enabled' if use_chat_template else 'Disabled' if use_chat_template is False else 'Auto-detect'}")
        lm = ANELM(model_path, use_chat_template=use_chat_template)
        print("ANELM initialized successfully")
        
        # Test tokenization
        print("\nTesting tokenization...")
        print(f"Template mode: {lm.use_chat_template}")
        texts = ["The sky is blue.", "Water boils at 100 degrees."]
        tokenized = lm._tokenize(texts)
        print(f"Tokenized {len(texts)} texts: {[len(t) for t in tokenized]} tokens each")
        
        # Show difference between template and no-template
        if lm.use_chat_template:
            print("Note: Using chat template adds system/user/assistant markers")
        else:
            print("Note: Raw tokenization without chat template")
        
        # Test _process_prompt
        print("Testing _process_prompt...")
        test_prompt = tokenized[0][:50]  # Use first 10 tokens
        print(f"Test prompt length: {len(test_prompt)}")
        # Print first 16 characters of the detokenized prompt
        preview = lm.tokenizer.decode(test_prompt[:16])
        print(f"Prompt preview: '{preview}...'")
        try:
            # Print the detokenized version of the test prompt
            detokenized = lm.tokenizer.decode(test_prompt)
            print(f"Processing prompt: '{detokenized}'")
            logprobs, cache = lm._process_prompt(test_prompt)
            print(f"_process_prompt successful, logprobs shape: {logprobs.shape}")
            # Get the token with highest probability from the last position
            last_logprobs = logprobs[-1]
            top_token_idx = torch.argmax(last_logprobs).item()
            top_token_prob = torch.softmax(last_logprobs, dim=-1)[top_token_idx].item()
            
            # Decode the token with highest probability
            top_token_str = lm.tokenizer.decode([top_token_idx])
            print(f"\nToken with highest probability:")
            print(f"Token ID: {top_token_idx}")
            print(f"Probability: {top_token_prob:.4f}")
            print(f"Decoded string: '{top_token_str}'")
            return True
        except Exception as e:
            print(f"Error in _process_prompt: {e}")
            return False
            
    except Exception as e:
        print(f"Error initializing ANELM: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ANELM core functionality")
    parser.add_argument(
        "--no-template", 
        action="store_true", 
        help="Disable chat template (raw tokenization)"
    )
    parser.add_argument(
        "--use-template", 
        action="store_true", 
        help="Force enable chat template"
    )
    args = parser.parse_args()
    
    # Determine use_chat_template value
    if args.no_template:
        use_chat_template = False
    elif args.use_template:
        use_chat_template = True
    else:
        use_chat_template = None  # Auto-detect
    
    success = test_anelm_basic(use_chat_template=use_chat_template)
    print(f"Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)