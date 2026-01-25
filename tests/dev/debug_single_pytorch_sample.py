#!/usr/bin/env python3
"""
Debug single sample with restructured PyTorch implementation
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

@register_model("debug_single_pytorch")
class DebugSinglePyTorchQwen25LM(LM):
    """Debug single sample PyTorch implementation matching ANE structure"""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__()
        self.model_path = model_path
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
        
        print("Debug single sample model loaded")
    
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
        """Debug single request"""
        print(f"\n=== SINGLE SAMPLE DEBUG: {len(requests)} requests ===")
        
        results = []
        
        for i, req in enumerate(requests):
            context, continuation = req.args
            
            print(f"\nRequest {i}:")
            print(f"  Context ends with: {repr(context[-50:])}")
            print(f"  Continuation: {repr(continuation)}")
            
            # Tokenize 
            context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
            continuation_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)
            
            print(f"  Context tokens: {len(context_tokens)}")
            print(f"  Continuation tokens: {continuation_tokens}")
            
            if len(continuation_tokens) == 0:
                print("  Empty continuation - returning (0.0, True)")
                results.append((0.0, True))
                continue
            
            # Create full sequence and score using ANE structure
            full_sequence = torch.tensor(context_tokens + continuation_tokens, dtype=torch.long)
            print(f"  Full sequence length: {len(full_sequence)}")
            
            try:
                logp_vec, greedy_vec = self._score_sequence(full_sequence)
                print(f"  Score sequence returned: {len(logp_vec)} log probs")
                
                # Sum log probabilities for continuation tokens only
                continuation_length = len(continuation_tokens)
                if len(logp_vec) >= continuation_length:
                    total_logprob = logp_vec[-continuation_length:].sum().item()
                    print(f"  Continuation log probs: {logp_vec[-continuation_length:]}")
                else:
                    total_logprob = logp_vec.sum().item() if len(logp_vec) > 0 else float('-inf')
                    print(f"  Using all available log probs: {logp_vec}")
                
                print(f"  Total log probability: {total_logprob:.6f}")
                
                # Greedy check (simplified)
                is_greedy = True  # Simplified for debugging
                
                result = (total_logprob, is_greedy)
                print(f"  Result: {result}")
                results.append(result)
                
            except Exception as e:
                print(f"  Error in scoring: {e}")
                results.append((float('-inf'), False))
        
        return results
    
    def _score_sequence(self, tokens_1d: torch.Tensor):
        """Score sequence using ANE approach"""
        ℓ = tokens_1d.size(0)
        ctx_len = self.model.config.context_length
        
        print(f"    _score_sequence: length={ℓ}, ctx_len={ctx_len}")
        
        # Check bounds
        if ℓ > ctx_len:
            tokens_1d = tokens_1d[-ctx_len:]
            ℓ = tokens_1d.size(0)
            print(f"    Truncated to length={ℓ}")
        
        logps, greedy = [], []
        pos = 1  # position of t₁
        
        print(f"    Processing {ℓ-1} tokens starting from pos=1")
        
        for idx, gold_tok in enumerate(tokens_1d[1:]):  # t₁ … t_{ℓ-1}
            print(f"      Token {idx+1}: gold_tok={gold_tok.item()}, pos={pos}")
            
            g_id, lp_vec = self._predict_token_with_logits(
                int(gold_tok), None, pos, tokens_1d[:pos+1])
            
            print(f"        Predicted: {g_id}, Target log prob: {lp_vec[gold_tok].item():.6f}")
            
            logps.append(lp_vec[gold_tok].unsqueeze(0))
            greedy.append(torch.tensor(g_id == gold_tok))
            pos += 1
        
        if len(logps) == 0:
            return torch.tensor([]), torch.tensor([])
            
        return torch.cat(logps), torch.stack(greedy)
    
    def _predict_token_with_logits(self, gold_tok_id: int, kv, pos: int, context_tokens: torch.Tensor):
        """Predict token with logits - match ANE structure"""
        ctx_len = self.model.config.context_length
        
        # Ensure context fits
        if len(context_tokens) > ctx_len:
            context_tokens = context_tokens[-ctx_len:]
            pos = len(context_tokens) - 1
        
        # Get the token to process (should be the last token in context)
        if pos >= len(context_tokens) or pos >= ctx_len:
            vocab_size = self.model.config.vocab_size
            log_probs = torch.full((vocab_size,), float('-inf'))
            return 0, log_probs
        
        last_token = torch.tensor([[context_tokens[pos]]], dtype=torch.long)
        
        # Create masks
        def make_causal_mask(length, start):
            mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
            row_indices = np.arange(length).reshape(length, 1)
            col_indices = np.arange(length).reshape(1, length)
            mask[:, :, col_indices <= (row_indices + start)] = 0
            return mask
        
        causal_mask_data = make_causal_mask(ctx_len, 0)
        causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
        
        update_mask = torch.zeros((1, 1, ctx_len, 1), dtype=torch.float16)
        if pos < ctx_len:
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
                
                # Extract logits - match ANE structure
                if logits.dim() == 3:
                    logits = logits[0, -1]  # [vocab]
                elif logits.dim() == 2:
                    logits = logits[-1]    # [vocab]
                
                # Use exact same calculation as ANE
                log_probs = torch.log_softmax(logits, dim=-1)
                greedy_id = torch.argmax(log_probs).item()
                return greedy_id, log_probs
                
        except Exception as e:
            print(f"        Error in _predict_token_with_logits: {e}")
            vocab_size = self.model.config.vocab_size
            log_probs = torch.full((vocab_size,), float('-inf'))
            return 0, log_probs
    
    def loglikelihood_rolling(self, requests):
        return []
    
    def generate_until(self, requests):
        return [""] * len(requests)

def test_single_sample():
    """Test single BoolQ sample"""
    
    # Load first BoolQ sample manually
    dataset = load_dataset('boolq', split='validation')
    sample = dataset[0]
    
    print(f"Testing sample:")
    print(f"  Question: {sample['question']}")
    print(f"  Answer: {sample['answer']} ({'yes' if sample['answer'] else 'no'})")
    
    # Create the correct BoolQ format
    context = f"{sample['passage']}\nQuestion: {sample['question']}?\nAnswer:"
    
    # Create model
    model = DebugSinglePyTorchQwen25LM("Qwen/Qwen2.5-0.5B")
    
    # Create manual requests like lm-eval does
    from lm_eval.api.instance import Instance
    
    class SimpleRequest:
        def __init__(self, context, continuation):
            self.args = (context, continuation)
    
    requests = [
        SimpleRequest(context, " no"),   # Choice 0
        SimpleRequest(context, " yes"),  # Choice 1
    ]
    
    # Run loglikelihood
    results = model.loglikelihood(requests)
    
    print(f"\n=== FINAL RESULTS ===")
    no_score, no_greedy = results[0]
    yes_score, yes_greedy = results[1]
    
    print(f"No score: {no_score:.6f}")
    print(f"Yes score: {yes_score:.6f}")
    print(f"Predicted: {'yes' if yes_score > no_score else 'no'}")
    print(f"Correct: {'yes' if sample['answer'] else 'no'}")
    print(f"Match: {(yes_score > no_score) == sample['answer']}")
    print(f"Accuracy: {1.0 if (yes_score > no_score) == sample['answer'] else 0.0}")

if __name__ == "__main__":
    test_single_sample()