#!/usr/bin/env python3
"""
Direct test with lm-eval to bypass simple_evaluate_segmented
"""

import os
import sys

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add paths
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoTokenizer
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import torch
import numpy as np

@register_model("debug_direct_pytorch")
class DebugDirectPyTorchQwen25LM(LM):
    """Minimal debug PyTorch implementation"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self._rank = 0
        self._world_size = 1
        
        # Load model
        model_path = "Qwen/Qwen2.5-0.5B"
        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download
            try:
                model_path = snapshot_download(repo_id=model_path, local_files_only=True)
            except:
                model_path = snapshot_download(repo_id=model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        import json
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            hf_config = json.load(f)
        
        config = Qwen25Config(
            vocab_size=hf_config['vocab_size'],
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            state_length=512,
            rms_norm_eps=1e-6,
        )
        
        self.model = Qwen25ForCausalLM(config, disable_kv_cache=True)
        if not self.model.load_pretrained_weights(model_path):
            raise RuntimeError("Failed to load weights")
        
        print("Direct test model loaded")
    
    @property
    def rank(self):
        return self._rank
    
    @property 
    def world_size(self):
        return self._world_size
    
    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=True)
    
    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
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
        """Simple working loglikelihood"""
        print(f"\n[DIRECT] loglikelihood called with {len(requests)} requests")
        
        results = []
        
        for i, req in enumerate(requests):
            context, continuation = req.args
            
            print(f"  Request {i}: '{continuation}' -> ", end="")
            
            # Simple single-token scoring
            context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
            continuation_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)
            
            if len(continuation_tokens) != 1:
                print(f"multi-token, skipping")
                results.append((0.0, True))
                continue
            
            # Score the single token
            pos = len(context_tokens) - 1
            last_token = torch.tensor([[context_tokens[-1]]], dtype=torch.long)
            
            # Create masks
            def make_causal_mask(length, start):
                mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
                row_indices = np.arange(length).reshape(length, 1)
                col_indices = np.arange(length).reshape(1, length)
                mask[:, :, col_indices <= (row_indices + start)] = 0
                return mask
            
            ctx_len = self.model.config.context_length
            causal_mask_data = make_causal_mask(ctx_len, 0)
            causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
            
            update_mask = torch.zeros((1, 1, ctx_len, 1), dtype=torch.float16)
            if pos < ctx_len:
                update_mask[0, 0, pos, 0] = 1.0
            
            with torch.no_grad():
                logits = self.model(
                    last_token,
                    update_mask,
                    torch.tensor([pos], dtype=torch.long),
                    causal_mask[:, :, pos:pos+1, :],
                    torch.tensor(pos, dtype=torch.long),
                    IN_PREFILL=False
                )
                
                if logits.dim() == 3:
                    logits = logits[0, -1]
                elif logits.dim() == 2:
                    logits = logits[-1]
                
                log_probs = torch.log_softmax(logits, dim=-1)
                choice_score = log_probs[continuation_tokens[0]].item()
                
                print(f"score: {choice_score:.4f}")
                
                results.append((choice_score, True))
        
        # Show predictions
        if len(results) == 2:
            predicted = 0 if results[0][0] > results[1][0] else 1
            print(f"\n  Prediction: choice {predicted} ({'no' if predicted == 0 else 'yes'})")
        
        return results
    
    def loglikelihood_rolling(self, requests):
        return []
    
    def generate_until(self, requests):
        return [""] * len(requests)

def test_direct_lm_eval():
    """Test directly with lm_eval simple_evaluate"""
    
    from lm_eval import simple_evaluate
    
    model = DebugDirectPyTorchQwen25LM()
    
    print("Testing with direct lm_eval.simple_evaluate...")
    
    results = simple_evaluate(
        model=model,
        tasks=["boolq"],
        limit=2,
        batch_size=1
    )
    
    print(f"\n=== RESULTS ===")
    if "results" in results and "boolq" in results["results"]:
        boolq_results = results["results"]["boolq"]
        accuracy = boolq_results.get("acc", boolq_results.get("accuracy", 0.0))
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Full results: {boolq_results}")
    else:
        print("No BoolQ results found")
        print(f"Available: {list(results.get('results', {}).keys())}")

if __name__ == "__main__":
    test_direct_lm_eval()