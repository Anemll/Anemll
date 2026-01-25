#!/usr/bin/env python3
"""
Test script for PyTorch Qwen2.5 segmented evaluation using anemll/models/qwen2_5_model.py
This mirrors the ANE evaluation approach but uses your custom PyTorch model.
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Set offline mode to prevent network calls
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# Add paths for imports
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from simple_evaluate_with_segmentation import simple_evaluate_segmented
from datasets import load_dataset
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import torch
import numpy as np
from transformers import AutoTokenizer

# Import your custom Qwen2.5 model
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

@register_model("pytorch_qwen25")
class PyTorchQwen25LM(LM):
    """PyTorch Qwen2.5 LM wrapper using anemll/models/qwen2_5_model.py"""
    
    def __init__(self, model_path: str, max_tokens: int = 2048, **kwargs):
        super().__init__()
        self.model_path = model_path
        self._max_tokens = max_tokens
        self._rank = 0
        self._world_size = 1
        
        print(f"Loading custom PyTorch Qwen2.5 model from: {model_path}")
        
        # Check if it's a local path or HuggingFace ID
        if not os.path.exists(model_path):
            # It's a HuggingFace ID, check cache first
            from huggingface_hub import snapshot_download
            try:
                print(f"Checking HuggingFace cache for {model_path}...")
                # Try to get from cache without downloading
                local_path = snapshot_download(repo_id=model_path, local_files_only=True)
                print(f"Found in cache: {local_path}")
                model_path = local_path
            except Exception:
                print(f"Not in cache, downloading {model_path}...")
                local_path = snapshot_download(repo_id=model_path)
                print(f"Downloaded to: {local_path}")
                model_path = local_path
        
        # Load tokenizer first to get config info
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Get actual vocab size from model config.json
        import json
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            hf_config = json.load(f)
        
        actual_vocab_size = hf_config['vocab_size']
        print(f"Model vocab size: {actual_vocab_size}, Tokenizer vocab size: {len(self.tokenizer)}")
        
        # Create config for your custom model using actual vocab size
        config = Qwen25Config(
            vocab_size=actual_vocab_size,  # Use actual model vocab size
            hidden_size=896,  # Qwen2.5-0.5B hidden size
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            state_length=512,
            rms_norm_eps=1e-6,
        )
        
        # Instantiate your custom model (use ForCausalLM which includes LM head)
        # Disable KV cache for evaluation to avoid dimension mismatches
        self.model = Qwen25ForCausalLM(config, disable_kv_cache=True)
        
        # Load pretrained weights using your method
        print(f"Loading pretrained weights...")
        success = self.model.load_pretrained_weights(model_path)
        if not success:
            raise RuntimeError(f"Failed to load pretrained weights from {model_path}")
        
        print(f"Custom PyTorch Qwen2.5 model loaded: {type(self.model)}")
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
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
        """Compute log-likelihood using your custom Qwen2.5 model"""
        print(f"[PyTorchQwen25LM] loglikelihood called with {len(requests)} requests")
        
        results = []
        
        for req_idx, req in enumerate(requests):
            # Only show progress for large batches
            if len(requests) > 50 and req_idx % 100 == 0:
                print(f"[PyTorchQwen25LM] Processing request {req_idx}/{len(requests)}")
            
            context, continuation = req.args
            
            # Debug first two requests (both choices)
            if req_idx <= 1:
                print(f"  Request {req_idx}:")
                print(f"    Context ends with: {repr(context[-50:])}")
                print(f"    Continuation: {repr(continuation)}")
            
            # Tokenize context and continuation separately 
            context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
            full_tokens = self.tokenizer.encode(context + continuation, add_special_tokens=True)
            
            # Get continuation tokens
            continuation_tokens = full_tokens[len(context_tokens):]
            
            if req_idx <= 1:
                print(f"    Context tokens: {len(context_tokens)}")
                print(f"    Continuation tokens: {continuation_tokens}")
            
            if len(continuation_tokens) == 0:
                results.append((0.0, True))
                continue
                
            # Handle context overflow with truncation - use model's actual context length
            context_length = self.model.config.context_length
            if len(context_tokens) + len(continuation_tokens) > context_length:
                # Truncate context from the beginning to fit
                available_context = context_length - len(continuation_tokens)
                if available_context > 0:
                    context_tokens = context_tokens[-available_context:]
                else:
                    # If even with maximum truncation we can't fit, skip
                    if len(requests) <= 50:  # Only show warning for small batches
                        print(f"Warning: Skipping sample - continuation too long ({len(continuation_tokens)} tokens)")
                    results.append((-float("inf"), False))
                    continue
            
            # Score using simple single-token approach like the working quick test
            if len(continuation_tokens) == 1:
                # Single token continuation - use direct scoring
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
                    
                    # Extract logits - match ANE exactly
                    if logits.dim() == 3:
                        logits = logits[0, -1]  # [vocab]
                    elif logits.dim() == 2:
                        logits = logits[-1]    # [vocab]
                    
                    # Use exact same calculation as ANE
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Score the choice token
                    choice_score = log_probs[continuation_tokens[0]].item()
                    total_logprob = choice_score
            else:
                # Multi-token continuation - use full sequence scoring (fallback)
                full_sequence = torch.tensor(context_tokens + continuation_tokens, dtype=torch.long)
                logp_vec, greedy_vec = self._score_sequence(full_sequence)
                
                # Sum log probabilities for continuation tokens only
                continuation_length = len(continuation_tokens)
                if len(logp_vec) >= continuation_length:
                    total_logprob = logp_vec[-continuation_length:].sum().item()
                else:
                    total_logprob = logp_vec.sum().item() if len(logp_vec) > 0 else float('-inf')
                
            if req_idx <= 1:
                print(f"    Scoring method: {'single-token' if len(continuation_tokens) == 1 else 'multi-token'}")
                print(f"    Continuation length: {len(continuation_tokens)}")
                print(f"    Total log prob: {total_logprob:.6f}")
            
            # For greedy check, follow the same pattern but ensure context bounds
            if len(context_tokens) > 0 and len(continuation_tokens) > 0:
                context_length = self.model.config.context_length
                pos = len(context_tokens) - 1
                
                # Ensure position is within context bounds
                if pos < context_length:
                    last_token = torch.tensor([[context_tokens[-1]]], dtype=torch.long)
                    
                    # Create masks for greedy check
                    def make_causal_mask(length, start):
                        mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
                        row_indices = np.arange(length).reshape(length, 1)
                        col_indices = np.arange(length).reshape(1, length)
                        mask[:, :, col_indices <= (row_indices + start)] = 0
                        return mask
                    
                    causal_mask_data = make_causal_mask(context_length, 0)
                    causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
                    
                    update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
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
                            greedy_token = torch.argmax(next_token_logits).item()
                            is_greedy = greedy_token == continuation_tokens[0]
                    except Exception as e:
                        print(f"Warning: Greedy check failed at pos {pos}: {e}")
                        is_greedy = False
                else:
                    # Position exceeds context, can't do greedy check
                    is_greedy = False
            else:
                is_greedy = True
            
            results.append((total_logprob, is_greedy))
        
        # Debug final results for first sample
        if len(results) == 2:  # Both choices completed
            print(f"\n  Final results for first sample:")
            print(f"    Choice 0 (no) score: {results[0][0]:.6f}")
            print(f"    Choice 1 (yes) score: {results[1][0]:.6f}")
            predicted = 0 if results[0][0] > results[1][0] else 1
            print(f"    Predicted choice: {predicted} ({'no' if predicted == 0 else 'yes'})")
            
            # Also check if we can access ground truth from request
            if hasattr(requests[0], 'doc') and requests[0].doc is not None:
                ground_truth = requests[0].doc.get('answer', 'Unknown')
                expected_choice = int(ground_truth) if isinstance(ground_truth, bool) else 'Unknown'
                print(f"    Ground truth: {ground_truth} = choice {expected_choice}")
                if expected_choice != 'Unknown':
                    is_correct = predicted == expected_choice
                    print(f"    Match: {is_correct}")
            else:
                print(f"    Ground truth: Not accessible")
        
        return results
    
    def _score_sequence(self, tokens_1d: torch.Tensor):
        """Score sequence using ANE approach - match evaluate_with_harness.py structure exactly
        
        Parameters
        ----------
        tokens_1d : 1-D LongTensor, length ℓ  (prompt + answer)

        Returns
        -------
        logp_vec   : FloatTensor [ℓ-1]   log p(t_k | t_<k)  for k = 1 … ℓ-1
        greedy_vec : BoolTensor  [ℓ-1]   top-1 == gold at every position
        """
        ℓ = tokens_1d.size(0)
        ctx_len = self.model.config.context_length
        
        # Check bounds
        if ℓ > ctx_len:
            # Truncate from beginning like ANE
            tokens_1d = tokens_1d[-ctx_len:]
            ℓ = tokens_1d.size(0)
        
        # No actual KV cache for our PyTorch model, but follow ANE structure
        # prefill the *first* token only (t₀) – no logits needed yet
        # (In our case, we don't have actual prefill, but we start from position 1)
        
        logps, greedy = [], []
        pos = 1  # position of t₁
        
        for gold_tok in tokens_1d[1:]:  # t₁ … t_{ℓ-1}
            g_id, lp_vec = self._predict_token_with_logits(
                int(gold_tok), None, pos, tokens_1d[:pos+1])
            
            logps.append(lp_vec[gold_tok].unsqueeze(0))
            greedy.append(torch.tensor(g_id == gold_tok))
            pos += 1
        
        return torch.cat(logps), torch.stack(greedy)
    
    def _predict_token_with_logits(self, gold_tok_id: int, kv, pos: int, context_tokens: torch.Tensor):
        """Predict token with logits - match ANE structure exactly
        
        Parameters
        ----------
        gold_tok_id : int              token to insert at position `pos`
        kv          : None (not used)  KV cache (not used in PyTorch version)
        pos         : int (0-based)    position of `gold_tok_id`
        context_tokens : torch.Tensor  context tokens up to current position

        Returns
        -------
        greedy_id   : int              arg-max next-token prediction
        log_probs   : FloatTensor [vocab]
        """
        ctx_len = self.model.config.context_length
        
        # Ensure context fits
        if len(context_tokens) > ctx_len:
            context_tokens = context_tokens[-ctx_len:]
            pos = len(context_tokens) - 1
        
        # Get the token to process (should be the last token in context)
        if pos >= len(context_tokens) or pos >= ctx_len:
            # Return dummy values for out-of-bounds
            vocab_size = self.model.config.vocab_size
            log_probs = torch.full((vocab_size,), float('-inf'))
            return 0, log_probs
        
        last_token = torch.tensor([[context_tokens[pos]]], dtype=torch.long)
        
        # Create masks exactly like ANE approach
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
                
                # Extract logits for vocab - match ANE structure
                if logits.dim() == 3:
                    logits = logits[0, -1]  # [vocab]
                elif logits.dim() == 2:
                    logits = logits[-1]    # [vocab]
                
                # Use exact same calculation as ANE
                log_probs = torch.log_softmax(logits, dim=-1)
                greedy_id = torch.argmax(log_probs).item()
                return greedy_id, log_probs
                
        except Exception as e:
            print(f"Error in _predict_token_with_logits at pos {pos}: {e}")
            vocab_size = self.model.config.vocab_size
            log_probs = torch.full((vocab_size,), float('-inf'))
            return 0, log_probs
    
    def loglikelihood_rolling(self, requests):
        """Compute rolling log-likelihood"""
        print(f"[PyTorchQwen25LM] loglikelihood_rolling called with {len(requests)} requests")
        return []
    
    def generate_until(self, requests):
        """Generate text until stopping condition"""
        print(f"[PyTorchQwen25LM] generate_until called with {len(requests)} requests")
        return [""] * len(requests)

def test_pytorch_qwen25_multi_segment_evaluation(model_path, segment_size, limit=None):
    """Test PyTorch Qwen2.5 multi-segment evaluation"""
    print("=" * 80)
    print(f"Testing PyTorch Qwen2.5 Multi-Segment Evaluation (segment size: {segment_size})")
    if limit:
        print(f"Limited to {limit} samples")
    print("=" * 80)
    
    # Load dataset to get total size
    dataset = load_dataset('boolq', split='validation')
    total_size = len(dataset)
    
    # Apply limit if specified
    if limit and limit < total_size:
        total_size = limit
        print(f"Limited dataset size: {total_size}")
    else:
        print(f"Total dataset size: {total_size}")
    
    # Calculate segments
    print(f"Segment size: {segment_size}")
    num_segments = (total_size + segment_size - 1) // segment_size
    print(f"Number of segments: {num_segments}")
    
    all_results = []
    
    # Load model once at the start
    print(f"\nLoading PyTorch Qwen2.5 model once for all segments...")
    lm = PyTorchQwen25LM(model_path=model_path, max_tokens=2048)
    print(f"Model loaded successfully!")
    
    for segment_idx in range(num_segments):
        start_idx = segment_idx * segment_size
        end_idx = min((segment_idx + 1) * segment_size, total_size)
        current_segment_size = end_idx - start_idx
        
        print(f"\nProcessing PyTorch Qwen2.5 segment [{start_idx}..{end_idx-1}] ({current_segment_size} samples)")
        
        # Run segmented evaluation with full lm-eval preprocessing
        results = simple_evaluate_segmented(
            model=lm,
            tasks=["boolq"],
            segment_start=start_idx,
            segment_size=current_segment_size,
            total_dataset_size=len(dataset)  # Use original dataset size for proper indexing
        )
        
        # Extract results without verbose debug output
        if "results" in results and "boolq" in results["results"]:
            boolq_results = results["results"]["boolq"]
            
            # Debug: print all available keys
            print(f"  Available result keys: {list(boolq_results.keys())}")
            
            # Extract accuracy - lm-eval uses "acc,none" key
            accuracy = boolq_results.get("acc,none", boolq_results.get("acc", boolq_results.get("accuracy", 0.0)))
            stderr = boolq_results.get("acc_stderr,none", boolq_results.get("acc_stderr", 0))
            
            # Ensure numeric values
            try:
                accuracy = float(accuracy)
            except (ValueError, TypeError):
                print(f"  Warning: Could not convert accuracy={accuracy} to float")
                accuracy = 0.0
                
            try:
                stderr = float(stderr) if stderr != "N/A" else 0.0
            except (ValueError, TypeError):
                stderr = 0.0
            
            segment_results = {
                "segment_idx": segment_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "segment_size": current_segment_size,
                "accuracy": accuracy,
                "stderr": stderr,
                "num_examples": current_segment_size
            }
            all_results.append(segment_results)
            print(f"Segment {segment_idx} accuracy: {segment_results['accuracy']:.4f} ± {segment_results['stderr']:.4f}")
        else:
            print(f"Warning: No results found for segment {segment_idx}. Available keys: {list(results.get('results', {}).keys()) if 'results' in results else 'No results key'}")
    
    # Calculate overall accuracy
    if all_results:
        total_correct = sum(r["accuracy"] * r["num_examples"] for r in all_results)
        total_examples = sum(r["num_examples"] for r in all_results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        print(f"\nOverall PyTorch Qwen2.5 Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"Total examples evaluated: {total_examples}")
        
        # Save results
        output_file = f"tests/dev/segmented_results/pytorch_qwen25_segmented_results_size_{segment_size}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        results_data = {
            "model": model_path,
            "segment_size": segment_size,
            "total_size": total_size,
            "overall_accuracy": overall_accuracy,
            "segments": all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo results collected!")

def main():
    parser = argparse.ArgumentParser(description='Test PyTorch Qwen2.5 segmented evaluation')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B',
                       help='Path to model (HuggingFace ID)')
    parser.add_argument('--segment-size', type=int, default=100,
                       help='Size of each segment')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit total number of samples to evaluate')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with single sample')
    
    args = parser.parse_args()
    
    # Start with 1 sample for debugging only if explicitly debug mode
    if args.debug and args.segment_size > 10:
        print("Starting with debug mode (1 sample)")
        args.segment_size = 1
        args.limit = 1
    elif args.debug:
        print(f"Debug mode with user-specified segment size: {args.segment_size}")
        if not args.limit:
            args.limit = args.segment_size
    
    test_pytorch_qwen25_multi_segment_evaluation(args.model, args.segment_size, args.limit)

if __name__ == "__main__":
    main()