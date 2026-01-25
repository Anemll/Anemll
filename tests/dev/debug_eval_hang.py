#!/usr/bin/env python3
"""
Debug script to identify where evaluation hangs.
Run this step by step to isolate the hanging point.
"""

import os
import sys
import signal
import time
from pathlib import Path

# Timeout handler
def timeout_handler(signum, frame):
    print(f"\n[TIMEOUT] Operation timed out at: {frame.f_code.co_filename}:{frame.f_lineno}")
    print(f"[TIMEOUT] Function: {frame.f_code.co_name}")
    raise TimeoutError("Operation timed out")

def test_step(step_name, timeout_seconds=30):
    """Test a step with timeout."""
    print(f"\n{'='*50}")
    print(f"TESTING: {step_name}")
    print(f"{'='*50}")
    
    def run_test(test_func):
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time
            print(f"✅ {step_name} completed in {elapsed:.2f}s")
            return True
        except TimeoutError:
            print(f"❌ {step_name} TIMED OUT after {timeout_seconds}s")
            return False
        except Exception as e:
            print(f"❌ {step_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            signal.alarm(0)  # Cancel timeout
    
    return run_test

def main():
    model_path = "/tmp/test-qwen25-sp-quant-0.5b"
    
    # Step 1: Basic imports
    def step1():
        import torch
        import numpy as np
        print("Basic imports successful")
    
    if not test_step("Import basic modules", 10)(step1):
        return
    
    # Step 2: CoreML imports  
    def step2():
        import coremltools as ct
        print("CoreML import successful")
    
    if not test_step("Import CoreML", 10)(step2):
        return
    
    # Step 3: LM-eval imports
    def step3():
        import lm_eval
        from lm_eval.api.model import LM
        from lm_eval.api.registry import register_model
        print("LM-eval imports successful")
    
    if not test_step("Import lm-evaluation-harness", 15)(step3):
        return
    
    # Step 4: Chat imports
    def step4():
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from chat import run_prefill, generate_next_token, initialize_tokenizer, load_model
        print("Chat imports successful")
    
    if not test_step("Import chat functions", 10)(step4):
        return
    
    # Step 5: ANELM class import
    def step5():
        evaluate_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evaluate", "ane")
        sys.path.append(evaluate_dir)
        from evaluate_with_harness import ANELM
        print("ANELM import successful")
        globals()['ANELM'] = ANELM  # Make ANELM available globally
    
    if not test_step("Import ANELM class", 10)(step5):
        return
    
    # Step 6: ANELM initialization
    def step6():
        lm = globals()['ANELM'](model_path)
        print("ANELM initialization successful")
        globals()['lm'] = lm  # Make lm available globally
    
    if not test_step("Initialize ANELM", 60)(step6):
        return
    
    # Step 7: Simple tokenization test
    def step7():
        texts = ["The sky is blue."]
        tokenized = globals()['lm']._tokenize(texts)
        print(f"Tokenization successful: {len(tokenized[0])} tokens")
        globals()['tokenized'] = tokenized
    
    if not test_step("Test tokenization", 10)(step7):
        return
    
    # Step 8: Simple _process_prompt test
    def step8():
        test_prompt = globals()['tokenized'][0][:10]  # First 10 tokens
        logprobs, cache = globals()['lm']._process_prompt(test_prompt)
        print(f"_process_prompt successful: {logprobs.shape}")
    
    if not test_step("Test _process_prompt", 30)(step8):
        return
    
    # Step 9: Dataset loading test
    def step9():
        try:
            import datasets
            # Try to load just the configuration
            dataset_config = datasets.get_dataset_config_names("super_glue")
            print(f"Dataset config loaded: {dataset_config[:3]}...")  # Show first 3
        except Exception as e:
            print(f"Dataset loading issue: {e}")
        return True
    
    if not test_step("Test dataset loading", 30)(step9):
        return
    
    # Step 10: Simple evaluation test
    def step10():
        import lm_eval
        eval_args = {
            "model": globals()['lm'],
            "tasks": ["boolq"],
            "limit": 1,
            "batch_size": 1,
        }
        results = lm_eval.simple_evaluate(**eval_args)
        print(f"Evaluation successful: {list(results['results'].keys())}")
    
    if not test_step("Test simple_evaluate call", 60)(step10):
        return
    
    print(f"\n{'='*50}")
    print("🎉 ALL TESTS PASSED!")
    print(f"{'='*50}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()