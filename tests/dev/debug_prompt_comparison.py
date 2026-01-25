#!/usr/bin/env python3
"""
Debug by comparing prompts received by PyTorch vs ANE implementations
"""

import os
import sys
import json

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add paths
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/evaluate/ane')

from simple_evaluate_with_segmentation import simple_evaluate_segmented
from datasets import load_dataset
from transformers import AutoTokenizer
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

@register_model("debug_pytorch_simple")
class DebugPyTorchSimple(LM):
    """Minimal debug PyTorch model that just logs requests"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self._rank = 0
        self._world_size = 1
        self.requests_log = []
    
    @property
    def rank(self):
        return self._rank
    
    @property 
    def world_size(self):
        return self._world_size
    
    def tok_encode(self, string: str):
        # Use actual tokenizer for encoding
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        return tokenizer.encode(string, add_special_tokens=True)
    
    def tok_decode(self, tokens):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        return tokenizer.decode(tokens, skip_special_tokens=True)
    
    @property
    def max_length(self):
        return 2048
    
    @property
    def batch_size(self):
        return 1
    
    @property
    def device(self):
        return "cpu"
    
    def loglikelihood(self, requests):
        """Just log the requests and return dummy results"""
        print(f"\n=== PYTORCH DEBUG: {len(requests)} requests ===")
        
        results = []
        for i, req in enumerate(requests):
            context, continuation = req.args
            
            # Log first few requests in detail
            if i < 4:
                print(f"\nRequest {i}:")
                print(f"  Context ends with: {repr(context[-100:])}")
                print(f"  Continuation: {repr(continuation)}")
                
                # Save for later comparison
                self.requests_log.append({
                    'index': i,
                    'context': context,
                    'continuation': continuation,
                    'context_last_100': context[-100:],
                })
            
            # Return dummy results (always predict choice 0 for consistency)
            results.append((-1.0, True))
        
        return results
    
    def loglikelihood_rolling(self, requests):
        return []
    
    def generate_until(self, requests):
        return [""] * len(requests)

@register_model("debug_ane_simple")  
class DebugANESimple(LM):
    """Minimal debug ANE model that just logs requests"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self._rank = 0
        self._world_size = 1
        self.requests_log = []
    
    @property
    def rank(self):
        return self._rank
    
    @property 
    def world_size(self):
        return self._world_size
    
    def tok_encode(self, string: str):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        return tokenizer.encode(string, add_special_tokens=True)
    
    def tok_decode(self, tokens):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        return tokenizer.decode(tokens, skip_special_tokens=True)
    
    @property
    def max_length(self):
        return 2048
    
    @property
    def batch_size(self):
        return 1
    
    @property
    def device(self):
        return "cpu"
    
    def loglikelihood(self, requests):
        """Just log the requests and return dummy results"""
        print(f"\n=== ANE DEBUG: {len(requests)} requests ===")
        
        results = []
        for i, req in enumerate(requests):
            context, continuation = req.args
            
            # Log first few requests in detail
            if i < 4:
                print(f"\nRequest {i}:")
                print(f"  Context ends with: {repr(context[-100:])}")
                print(f"  Continuation: {repr(continuation)}")
                
                # Save for later comparison
                self.requests_log.append({
                    'index': i,
                    'context': context,
                    'continuation': continuation,
                    'context_last_100': context[-100:],
                })
            
            # Return dummy results (always predict choice 0 for consistency)
            results.append((-1.0, True))
        
        return results
    
    def loglikelihood_rolling(self, requests):
        return []
    
    def generate_until(self, requests):
        return [""] * len(requests)

def compare_prompt_formats():
    """Compare prompts received by PyTorch vs ANE"""
    
    print("=== TESTING PYTORCH IMPLEMENTATION ===")
    pytorch_model = DebugPyTorchSimple()
    
    pytorch_results = simple_evaluate_segmented(
        model=pytorch_model,
        tasks=["boolq"],
        segment_start=0,
        segment_size=2,  # Just 2 samples for debugging
        total_dataset_size=3270
    )
    
    print("\n=== TESTING ANE-STYLE IMPLEMENTATION ===")
    ane_model = DebugANESimple()
    
    ane_results = simple_evaluate_segmented(
        model=ane_model,
        tasks=["boolq"],
        segment_start=0,
        segment_size=2,  # Just 2 samples for debugging
        total_dataset_size=3270
    )
    
    print("\n=== COMPARISON ===")
    print(f"PyTorch requests logged: {len(pytorch_model.requests_log)}")
    print(f"ANE requests logged: {len(ane_model.requests_log)}")
    
    # Compare first few requests
    for i in range(min(len(pytorch_model.requests_log), len(ane_model.requests_log))):
        pytorch_req = pytorch_model.requests_log[i]
        ane_req = ane_model.requests_log[i]
        
        print(f"\nRequest {i} comparison:")
        print(f"PyTorch context: {repr(pytorch_req['context_last_100'])}")
        print(f"ANE context:     {repr(ane_req['context_last_100'])}")
        print(f"Same context? {pytorch_req['context'] == ane_req['context']}")
        
        print(f"PyTorch continuation: {repr(pytorch_req['continuation'])}")
        print(f"ANE continuation:     {repr(ane_req['continuation'])}")
        print(f"Same continuation? {pytorch_req['continuation'] == ane_req['continuation']}")

if __name__ == "__main__":
    compare_prompt_formats()