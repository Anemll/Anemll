#!/usr/bin/env python3
"""
FIXED PyTorch Qwen2.5 segmented evaluation using proper ANEMLL inference pattern
Based on test_final_inference.py as the official/reference implementation
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

# Performance optimizations
os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # Limit MKL threads for Intel CPUs

# Add paths for imports
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from simple_evaluate_with_segmentation import simple_evaluate_segmented
from datasets import load_dataset
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

# PyTorch performance optimizations
torch.set_num_threads(4)  # Limit PyTorch threads
if torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True  # Enable MKL-DNN optimization

# Import your custom Qwen2.5 model
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

@register_model("fixed_pytorch_qwen25")
class FixedPyTorchQwen25LM(LM):
    """FIXED PyTorch Qwen2.5 LM wrapper using proper ANEMLL inference pattern from test_final_inference.py"""
    
    # Class variable to track global request count across segments
    _global_request_count = 0
    
    def __init__(self, model_path: str, max_tokens: int = 2048, use_batch_prefill: bool = True, **kwargs):
        super().__init__()
        self.model_path = model_path
        self._max_tokens = max_tokens
        self._rank = 0
        self._world_size = 1
        self.use_batch_prefill = use_batch_prefill
        
        print("Using MPS (ANEMLL model has MPS compatibility issues)")
        self.device_name = 'mps'

        # Import TEST_DEVICE from qwen2_5_model and set it to match our device
        from anemll.models.qwen2_5_model import TEST_DEVICE
        TEST_DEVICE = self.device_name


        print(f"Loading FIXED PyTorch Qwen2.5 model from: {model_path}")
        print(f"Prefill mode: {'Batch' if use_batch_prefill else 'Single-token'}")
        
        # Check if it's a local path or HuggingFace ID
        if not os.path.exists(model_path):
            # It's a HuggingFace ID, check cache first
            from huggingface_hub import snapshot_download
            try:
                print(f"Checking HuggingFace cache for {model_path}...")
                local_path = snapshot_download(repo_id=model_path, local_files_only=True)
                print(f"Found in cache: {local_path}")
                model_path = local_path
            except Exception:
                print(f"Not in cache, downloading {model_path}...")
                local_path = snapshot_download(repo_id=model_path)
                print(f"Downloaded to: {local_path}")
                model_path = local_path
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create config using test_final_inference.py pattern
        config = Qwen25Config.from_json(f'{model_path}/config.json')
        
        # CRITICAL: Enable KV cache like test_final_inference.py (line 47)
        self.model = Qwen25ForCausalLM(config, disable_kv_cache=False)
        
        # Load pretrained weights
        print(f"Loading pretrained weights...")
        success = self.model.load_pretrained_weights(model_path)
        if not success:
            raise RuntimeError(f"Failed to load pretrained weights from {model_path}")
        
        self.model.eval()
        
        # Keep on CPU for now - ANEMLL model has MPS compatibility issues with rotary embeddings
        # if torch.backends.mps.is_available():
        #     print("Moving model to MPS (Metal Performance Shaders)")
        #     self.model = self.model.to('mps')
        #     self.device_name = 'mps'
        # else:
        #     print("MPS not available, using CPU")
        #     self.device_name = 'cpu'
        
        # Try JIT compilation for faster execution
        try:
            print("Attempting JIT compilation for faster inference...")
            # Note: JIT may not work with complex models, but worth trying
            dummy_input = torch.zeros(1, 1, dtype=torch.long)
            dummy_update_mask = torch.zeros(1, 1, dtype=torch.float16)
            dummy_position_ids = torch.zeros(1, dtype=torch.long)
            dummy_causal_mask = torch.zeros(1, 1, 1, self.context_length, dtype=torch.float16)
            dummy_current_pos = torch.tensor(0, dtype=torch.long)
            
            # Trace the model (this may fail for complex models)
            # self.model = torch.jit.trace(self.model, (dummy_input, dummy_update_mask, dummy_position_ids, dummy_causal_mask, dummy_current_pos, True))
            # print("JIT compilation successful!")
            print("JIT compilation skipped (ANEMLL model too complex)")
        except Exception as e:
            print(f"JIT compilation failed: {e}, continuing with regular model")
        
        print(f"FIXED PyTorch Qwen2.5 model loaded with KV cache enabled")
        
        # Store context length for proper mask creation
        self.context_length = config.context_length
        
        # Create causal mask once (from test_final_inference.py lines 78-88)
        def make_causal_mask(length, start):
            """Create causal attention mask."""
            mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
            row_indices = np.arange(length).reshape(length, 1)
            col_indices = np.arange(length).reshape(1, length)
            mask[:, :, col_indices <= (row_indices + start)] = 0
            return mask
        
        self.make_causal_mask = make_causal_mask  # Store as instance method for reuse
        causal_mask_data = make_causal_mask(self.context_length, 0)
        self.causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
        
        # Move causal mask to same device as model
        if hasattr(self, 'device_name'):
            self.causal_mask = self.causal_mask.to(self.device_name)
            print(f"Causal mask moved to device: {self.device_name}")
        
        print(f"Context length: {self.context_length}")
    
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
        return getattr(self, 'device_name', 'cpu')
    
    def loglikelihood(self, requests):
        """Compute log-likelihood using PROPER ANEMLL inference pattern from test_final_inference.py"""
        print(f"[DEBUG] ENTERING loglikelihood with {len(requests)} requests")
        
        results = []
        
        for req_idx, req in enumerate(requests):
            # Track global request number
            global_req_idx = FixedPyTorchQwen25LM._global_request_count
            FixedPyTorchQwen25LM._global_request_count += 1
            
            print(f"---->[DEBUG] Processing request {req_idx} (global/2 : {global_req_idx/2})")
            
            # Debug first two requests
            if req_idx <= 1:
                print(f"\\n  Request {req_idx} (global: {global_req_idx}):")
            
            context, continuation = req.args
            
            if req_idx <= 1:
                print(f"    Context ends with: {repr(context[-50:])}")
                print(f"    Continuation: {repr(continuation)}")
            
            print(f"[DEBUG] Tokenizing request {req_idx}")
            # Tokenize context and continuation separately 
            context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
            continuation_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)
            
            if req_idx <= 1:
                print(f"    Context tokens: {len(context_tokens)}")
                print(f"    Continuation tokens: {continuation_tokens}")
            
            if len(continuation_tokens) == 0:
                print(f"[DEBUG] Empty continuation tokens for request {req_idx}")
                results.append((0.0, True))
                continue
            
            print(f"[DEBUG] Checking context overflow for request {req_idx}")
            # Handle context overflow
            if len(context_tokens) + len(continuation_tokens) > self.context_length:
                available_context = self.context_length - len(continuation_tokens)
                if available_context > 0:
                    context_tokens = context_tokens[-available_context:]
                else:
                    if len(requests) <= 10:  # Only show warning for small batches
                        print(f"Warning: Skipping sample - continuation too long ({len(continuation_tokens)} tokens)")
                    results.append((-float("inf"), False))
                    continue
            
            print(f"[DEBUG] About to score request {req_idx}")
            # FIXED: Use proper ANEMLL inference pattern for scoring
            try:
                if len(continuation_tokens) == 1:
                    print(f"[DEBUG] Using single token scoring for request {req_idx}")
                    # Single token scoring using proper ANEMLL pattern
                    score = self._score_single_token_proper(context_tokens, continuation_tokens[0])
                    print(f"[DEBUG] Single token scoring completed for request {req_idx}")
                else:
                    print(f"[DEBUG] Using multi-token scoring for request {req_idx}")
                    # Multi-token scoring using proper ANEMLL pattern
                    score = self._score_sequence_proper(context_tokens, continuation_tokens)
                    print(f"[DEBUG] Multi-token scoring completed for request {req_idx}")
                
                if req_idx <= 1:
                    print(f"    Score: {score:.6f}")
                
                results.append((score, True))  # Assume greedy for now
                print(f"[DEBUG] Request {req_idx} completed successfully")
                
            except Exception as e:
                print(f"[DEBUG] ERROR scoring request {req_idx}: {e}")
                import traceback
                traceback.print_exc()
                results.append((-float("inf"), False))
        
        print(f"[DEBUG] All requests completed, preparing final results")
        # Debug final results for first sample
        if len(results) == 2:  # Both choices completed
            print(f"\\n  Final results for first sample:")
            print(f"    Choice 0 (no) score: {results[0][0]:.6f}")
            print(f"    Choice 1 (yes) score: {results[1][0]:.6f}")
            predicted = 0 if results[0][0] > results[1][0] else 1
            print(f"    Predicted choice: {predicted} ({'no' if predicted == 0 else 'yes'})")
        
        print(f"[DEBUG] EXITING loglikelihood")
        return results
    
    def run_prefill(self, input_ids, batch_size=64):
        """Run prefill on the input sequence."""
        # Use provided causal mask or create one if not provided
        context_pos = input_ids.shape[1]
        causal_mask = self.make_causal_mask(self.context_length, 0)
        causal_mask = torch.tensor(causal_mask, dtype=torch.float16)
        
        # Process in batches
        batch_pos = 0
        while batch_pos < context_pos:
            batch_end = min(batch_pos + batch_size, context_pos)
            current_batch_size = batch_end - batch_pos
            
            # Get current batch
            batch_input = input_ids[:, batch_pos:batch_end]
            
            # Always pad to full batch size for prefill
            batch_input = F.pad(
                batch_input,
                (0, batch_size - current_batch_size),
                value=0
            )
            
            # Generate position IDs for full batch size
            position_ids = torch.arange(batch_pos, batch_pos+batch_size, dtype=torch.int32)  # Changed: Always use full batch size
            batch_causal_mask = causal_mask[:, :, batch_pos:batch_pos+batch_size, :]  # Changed: Use full batch size
            
            # Get device
            device = next(self.model.parameters()).device
            
            #call prefill
            self.model(
                batch_input.to(device),  # input_ids
                torch.zeros(1, batch_size, device=device),  # update_mask
                position_ids.to(device),  # position_ids
                batch_causal_mask.to(device),   # causal_mask
                torch.tensor(batch_pos, dtype=torch.long, device=device),  # current_pos
                IN_PREFILL=True
            )

            batch_pos = batch_end
        
        return torch.tensor([context_pos], dtype=torch.int32)

    def _score_single_token_proper(self, context_tokens, target_token):
        """Score single token using proper ANEMLL inference pattern from test_final_inference.py"""
        print(f"[DEBUG] ENTERING _score_single_token_proper")
        
        # CRITICAL: Clear KV cache before each new sequence to avoid device/state issues
        print(f"[DEBUG] Clearing KV cache")
        if hasattr(self.model.model, 'clear_kv_cache'):
            self.model.model.clear_kv_cache()
        
        # Create full sequence for prefill
        full_sequence = context_tokens
        prompt_length = len(full_sequence)
        print(f"[DEBUG] Prompt length: {prompt_length}")
        
        # Step 1: Prefill KV cache using proper pattern (test_final_inference.py lines 94-133)
        with torch.no_grad():
            device = next(self.model.parameters()).device
            
            if self.use_batch_prefill and prompt_length > 1:
                print(f"[DEBUG] Starting batch prefill")
                
                # Batch prefill: process all prompt tokens at once
                input_ids = torch.tensor([full_sequence], dtype=torch.long, device=device)
                prefill_position_ids = torch.arange(prompt_length, dtype=torch.long, device=device)
                
                # Create causal mask for prefill: only within prompt length
                prefill_causal_mask = torch.zeros((1, 1, prompt_length, self.context_length), dtype=torch.float16, device=device)
                
                # Apply causal mask: token i can attend to tokens 0 through i, -inf for future positions
                for i in range(prompt_length):
                    prefill_causal_mask[:, :, i, i+1:self.context_length] = float('-inf')
                
                # Run batch prefill (similar to test_final_inference.py lines 126-133)
                # Need to use batch / run prefill

                # call batched prefill
                self.run_prefill(
                    input_ids,  # input_ids
                )
                print(f"[DEBUG] Batch prefill completed")
                
            else:
                print(f"[DEBUG] Starting single-token prefill")
                # Single token prefill for consistency with test_final_inference.py
                for i, token_id in enumerate(full_sequence):
                    if i % 50 == 0:  # Print progress every 50 tokens
                        print(f"[DEBUG] Prefilling token {i}/{prompt_length}")
                    
                    single_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                    
                    # Single token prefill (same pattern as test_final_inference.py lines 102-109)
                    self.model(
                        single_token,  # input_ids
                        torch.zeros(1, 1, device=device), # update_mask (not used in prefill)
                        torch.tensor([i], dtype=torch.long, device=device),  # position_ids
                        self.causal_mask[:, :, i:i+1, :],  # causal_mask - single row
                        torch.tensor(i, dtype=torch.long, device=device),  # current_pos
                        IN_PREFILL=True
                    )
                print(f"[DEBUG] Single-token prefill completed")
            
            # Step 2: Generate next token to get logits (test_final_inference.py lines 139-160)
            current_pos = prompt_length
            print(f"[DEBUG] Current position: {current_pos}")
            
            if current_pos >= self.context_length:
                print(f"[DEBUG] Position out of bounds: {current_pos} >= {self.context_length}")
                return float('-inf')  # Position out of bounds
            
            # Get the last token for generation
            last_token = torch.tensor([[full_sequence[-1]]], dtype=torch.long, device=device)
            print(f"[DEBUG] Last token: {last_token}")
            
            # Create update mask for single token at current position
            update_mask = torch.zeros((1, 1, self.context_length, 1), dtype=torch.float16, device=device)
            update_mask[0, 0, current_pos, 0] = 1.0
            print(f"[DEBUG] Update mask created")
            
            # Generate with proper pattern (test_final_inference.py lines 153-160)
            print(f"[DEBUG] About to call model for generation")
            outputs = self.model(
                last_token.to(device)   ,  # input_ids
                update_mask.to(device),  # update_mask
                torch.tensor([current_pos], dtype=torch.long, device=device),  # position_ids
                self.causal_mask[:, :, current_pos:current_pos+1, :].to(device),  # causal_mask - single row
                torch.tensor(current_pos, dtype=torch.long, device=device),  # current_pos
                IN_PREFILL=False
            )
            print(f"[DEBUG] Model generation completed")
            
            # Extract logits (test_final_inference.py line 163)
            next_token_logits = outputs[0, -1, :]
            print(f"[DEBUG] Logits extracted, shape: {next_token_logits.shape}")
            
            # Calculate log probabilities (same as ANE: torch.log_softmax)
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            print(f"[DEBUG] Log probs calculated")
            
            # Return score for target token
            score = log_probs[target_token].item()
            print(f"[DEBUG] Target token {target_token} score: {score}")
            print(f"[DEBUG] EXITING _score_single_token_proper")
            return score
    
    def _score_sequence_proper(self, context_tokens, continuation_tokens):
        """Score sequence using proper ANEMLL inference pattern"""
        # For multi-token sequences, score each token sequentially using teacher forcing
        
        total_score = 0.0
        current_sequence = context_tokens.copy()
        
        for token in continuation_tokens:
            # Score this token given the current sequence
            token_score = self._score_single_token_proper(current_sequence, token)
            total_score += token_score
            
            # Teacher forcing: add the true token to sequence for next prediction
            current_sequence.append(token)
        
        return total_score
    
    def loglikelihood_rolling(self, requests):
        """Compute rolling log-likelihood"""
        print(f"[FixedPyTorchQwen25LM] loglikelihood_rolling called with {len(requests)} requests")
        return []
    
    def generate_until(self, requests):
        """Generate text until stopping condition"""
        print(f"[FixedPyTorchQwen25LM] generate_until called with {len(requests)} requests")
        return [""] * len(requests)

def test_fixed_pytorch_qwen25_evaluation(model_path, segment_size, limit=None):
    """Test FIXED PyTorch Qwen2.5 evaluation with proper ANEMLL pattern"""
    print("=" * 80)
    print(f"Testing FIXED PyTorch Qwen2.5 Evaluation (segment size: {segment_size})")
    print("Using proper ANEMLL inference pattern from test_final_inference.py")
    if limit:
        print(f"Limited to {limit} samples")
    print("=" * 80)
    
    # Load dataset
    dataset = load_dataset('boolq', split='validation')
    total_size = len(dataset)
    
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
    
    # Load model once
    print(f"\\nLoading FIXED PyTorch Qwen2.5 model...")
    lm = FixedPyTorchQwen25LM(model_path=model_path, max_tokens=2048)
    print(f"Model loaded successfully!")
    
    for segment_idx in range(num_segments):
        start_idx = segment_idx * segment_size
        end_idx = min((segment_idx + 1) * segment_size, total_size)
        current_segment_size = end_idx - start_idx
        
        print(f"\\nProcessing FIXED segment [{start_idx}..{end_idx-1}] ({current_segment_size} samples)")
        
        # Run evaluation
        results = simple_evaluate_segmented(
            model=lm,
            tasks=["boolq"],
            segment_start=start_idx,
            segment_size=current_segment_size,
            total_dataset_size=len(dataset)
        )
        
        # Extract results
        if "results" in results and "boolq" in results["results"]:
            boolq_results = results["results"]["boolq"]
            
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
            print(f"Warning: No results found for segment {segment_idx}")
    
    # Calculate overall accuracy
    if all_results:
        total_correct = sum(r["accuracy"] * r["num_examples"] for r in all_results)
        total_examples = sum(r["num_examples"] for r in all_results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        print(f"\\nOverall FIXED PyTorch Qwen2.5 Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"Total examples evaluated: {total_examples}")
        
        # Save results
        output_file = f"tests/dev/segmented_results/fixed_pytorch_qwen25_results_size_{segment_size}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        results_data = {
            "model": model_path,
            "segment_size": segment_size,
            "total_size": total_size,
            "overall_accuracy": overall_accuracy,
            "segments": all_results,
            "note": "Fixed implementation using proper ANEMLL inference pattern from test_final_inference.py"
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\\nResults saved to: {output_file}")
    else:
        print("\\nNo results collected!")

def main():
    parser = argparse.ArgumentParser(description='Test FIXED PyTorch Qwen2.5 with proper ANEMLL pattern')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B',
                       help='Path to model (HuggingFace ID)')
    parser.add_argument('--segment-size', type=int, default=10,
                       help='Size of each segment')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit total number of samples to evaluate')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with small test')
    
    args = parser.parse_args()
    
    if args.debug:
        print("Debug mode: testing with 10 samples")
        args.segment_size = 100
        args.limit = 100
    
    test_fixed_pytorch_qwen25_evaluation(args.model, args.segment_size, args.limit)

if __name__ == "__main__":
    main()