#!/usr/bin/env python3
"""
Debug loglikelihood function in detail to see why accuracy is 0%
"""

import os
import sys
import torch
import numpy as np
from datasets import load_dataset

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add paths
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoTokenizer
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

@register_model("debug_pytorch_qwen25")
class DebugPyTorchQwen25LM(LM):
    """Debug version with detailed logging"""
    
    def __init__(self, model_path: str, max_tokens: int = 2048, **kwargs):
        super().__init__()
        self.model_path = model_path
        self._max_tokens = max_tokens
        self._rank = 0
        self._world_size = 1
        
        # Resolve model path
        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download
            try:
                model_path = snapshot_download(repo_id=model_path, local_files_only=True)
            except:
                model_path = snapshot_download(repo_id=model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
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
        
        print("Debug model loaded")
    
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
        return self._max_tokens
    
    @property
    def batch_size(self):
        return 1
    
    @property
    def device(self):
        return "cpu"
    
    def loglikelihood(self, requests):
        """Debug version with detailed logging"""
        print(f"\n=== DEBUG LOGLIKELIHOOD CALLED ===")
        print(f"Number of requests: {len(requests)}")
        
        results = []
        
        for req_idx, req in enumerate(requests):
            print(f"\n--- Request {req_idx} ---")
            context, continuation = req.args
            
            print(f"Context: {repr(context[-50:])}")  # Last 50 chars
            print(f"Continuation: {repr(continuation)}")
            
            # Tokenize
            context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
            continuation_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)
            
            print(f"Context tokens: {len(context_tokens)}")
            print(f"Continuation tokens: {continuation_tokens} = '{self.tokenizer.decode(continuation_tokens)}'")
            
            if len(continuation_tokens) == 0:
                print("Empty continuation - returning (0.0, True)")
                results.append((0.0, True))
                continue
            
            # Score
            total_logprob = self._score_sequence_debug(context_tokens, continuation_tokens)
            print(f"Total log probability: {total_logprob}")
            
            # Greedy check (simplified)
            is_greedy = True  # Simplified for debugging
            
            result = (total_logprob, is_greedy)
            print(f"Result: {result}")
            results.append(result)
            
            if req_idx >= 2:  # Only debug first few
                print("Skipping detailed debug for remaining requests...")
                break
        
        print(f"\n=== All Results: {results[:3]}... ===")
        return results
    
    def _score_sequence_debug(self, context_tokens, continuation_tokens):
        """Debug version of scoring"""
        print(f"  Scoring sequence with {len(context_tokens)} context + {len(continuation_tokens)} continuation")
        
        if len(continuation_tokens) == 0:
            return 0.0
        
        context_length = self.model.config.context_length
        print(f"  Model context length: {context_length}")
        
        # Check bounds
        total_length = len(context_tokens) + len(continuation_tokens)
        if total_length > context_length:
            available_context = context_length - len(continuation_tokens)
            if available_context <= 0:
                print(f"  Continuation too long, returning -inf")
                return float('-inf')
            context_tokens = context_tokens[-available_context:]
            print(f"  Truncated context to {len(context_tokens)} tokens")
        
        if len(context_tokens) == 0:
            print(f"  No context tokens, returning -inf")
            return float('-inf')
        
        # Golden token workflow
        sequence_so_far = context_tokens[:-1]
        prev = context_tokens[-1]
        pos = len(sequence_so_far)
        
        print(f"  Starting golden token workflow: pos={pos}, prev={prev}")
        
        total_logprob = 0.0
        
        for i, target_token in enumerate(continuation_tokens):
            print(f"    Step {i+1}: target_token={target_token}")
            
            if pos >= context_length:
                print(f"    Position {pos} exceeds context, stopping")
                break
            
            # Create sequence
            current_sequence = sequence_so_far + [prev]
            if len(current_sequence) > context_length:
                current_sequence = current_sequence[-context_length:]
                pos = len(current_sequence) - 1
            
            last_token = torch.tensor([[current_sequence[-1]]], dtype=torch.long)
            
            # Create masks (simplified)
            def make_causal_mask(length, start):
                mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
                row_indices = np.arange(length).reshape(length, 1)
                col_indices = np.arange(length).reshape(1, length)
                mask[:, :, col_indices <= (row_indices + start)] = 0
                return mask
            
            causal_mask_data = make_causal_mask(context_length, 0)
            causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
            
            update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
            if pos < context_length:
                update_mask[0, 0, pos, 0] = 1.0
            
            try:
                with torch.no_grad():
                    logits = self.model(
                        last_token,
                        update_mask,
                        torch.tensor([pos], dtype=torch.long),
                        causal_mask[:, :, pos:pos+1, :],
                        torch.tensor(pos, dtype=torch.long),
                        IN_PREFILL=False
                    )
                    
                    next_token_logits = logits[0, -1, :]
                    log_probs = torch.log_softmax(next_token_logits.float(), dim=-1)
                    token_logprob = log_probs[target_token].item()
                    
                    print(f"    Token log prob: {token_logprob:.6f}")
                    total_logprob += token_logprob
                    
            except Exception as e:
                print(f"    Error: {e}")
                return float('-inf')
            
            # Golden token update
            sequence_so_far = current_sequence
            prev = target_token
            pos += 1
        
        print(f"  Final total log prob: {total_logprob}")
        return total_logprob
    
    def loglikelihood_rolling(self, requests):
        return []
    
    def generate_until(self, requests):
        return [""] * len(requests)

def debug_with_lm_eval():
    """Test using lm-eval framework directly"""
    
    # Create model
    model = DebugPyTorchQwen25LM("Qwen/Qwen2.5-0.5B")
    
    # Get first few BoolQ samples
    dataset = load_dataset('boolq', split='validation')
    
    # Create manual requests like lm-eval does
    from lm_eval.api.instance import Instance
    
    requests = []
    for i in range(3):  # First 3 samples
        sample = dataset[i]
        context = f"{sample['passage']}\nQuestion: {sample['question']}?\nAnswer:"
        
        # BoolQ has two choices: 0="no", 1="yes"
        req_no = Instance(
            request_type="loglikelihood",
            doc=sample,
            arguments=(context, " no"),
            idx=i,
            metadata={}
        )
        req_yes = Instance(
            request_type="loglikelihood", 
            doc=sample,
            arguments=(context, " yes"),
            idx=i,
            metadata={}
        )
        
        requests.extend([req_no, req_yes])
    
    print(f"Created {len(requests)} requests for {len(requests)//2} samples")
    
    # Run loglikelihood
    results = model.loglikelihood(requests)
    
    # Process results
    print(f"\n=== PROCESSING RESULTS ===")
    for i in range(0, len(results), 2):
        sample_idx = i // 2
        sample = dataset[sample_idx]
        
        no_logprob, no_greedy = results[i]
        yes_logprob, yes_greedy = results[i + 1]
        
        print(f"\nSample {sample_idx}:")
        print(f"  Ground truth: {sample['answer']} ({'yes' if sample['answer'] else 'no'})")
        print(f"  No score: {no_logprob:.6f}")
        print(f"  Yes score: {yes_logprob:.6f}")
        print(f"  Predicted: {'yes' if yes_logprob > no_logprob else 'no'}")
        print(f"  Correct: {(yes_logprob > no_logprob) == sample['answer']}")

if __name__ == "__main__":
    debug_with_lm_eval()