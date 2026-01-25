#!/usr/bin/env python3
"""
Test the core evaluation functionality without dataset dependencies.
Debug-safe version that handles GIL issues with CoreML.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import threading

# Add paths for imports
# Get the anemll root directory
anemll_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
evaluate_dir = os.path.join(anemll_root, "evaluate", "ane")
sys.path.append(evaluate_dir)

# Import the ANELM class from evaluate_with_harness
from evaluate_with_harness import ANELM

def test_anelm_basic():
    """Test basic ANELM functionality without datasets."""
    print("Testing ANELM basic functionality...")
    
    model_path = "/tmp/test-qwen25-sp-quant-0.5b"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return False
        
    try:
        # Initialize the model
        print("Initializing ANELM...")
        lm = ANELM(model_path)
        print("ANELM initialized successfully")
        
        # Test tokenization
        print("Testing tokenization...")
        texts = ["The sky is blue.", "Water boils at 100 degrees."]
        tokenized = lm._tokenize(texts)
        print(f"Tokenized {len(texts)} texts: {[len(t) for t in tokenized]} tokens each")
        
        # Test _process_prompt
        print("Testing _process_prompt...")
        test_prompt = tokenized[0][:50]  # Use first 50 tokens
        print(f"Test prompt length: {len(test_prompt)}")
        # Print first 16 characters of the detokenized prompt
        preview = lm.tokenizer.decode(test_prompt[:16])
        print(f"Prompt preview: '{preview}...'")
        
        try:
            # Print the detokenized version of the test prompt
            detokenized = lm.tokenizer.decode(test_prompt)
            print(f"Processing prompt: '{detokenized}'")
            
            # Wrap the CoreML call to handle GIL issues
            result = None
            error = None
            
            def run_inference():
                nonlocal result, error
                try:
                    # Ensure we have the GIL before calling CoreML
                    import _thread
                    _thread.get_ident()  # This ensures GIL is held
                    result = lm._process_prompt(test_prompt)
                except Exception as e:
                    error = e
            
            # Run in a way that's debugger-friendly
            thread = threading.Thread(target=run_inference)
            thread.daemon = True
            thread.start()
            thread.join(timeout=30)  # 30 second timeout
            
            if error:
                raise error
            if result is None:
                raise TimeoutError("Inference timed out")
                
            logprobs, cache = result
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
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"Error initializing ANELM: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Disable multi-threading in CoreML when debugging
    os.environ['COREML_USE_NEURAL_ENGINE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    success = test_anelm_basic()
    print(f"Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)