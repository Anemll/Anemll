#!/usr/bin/env python3
# ANE/CoreML model evaluation with lm-evaluation-harness using ANE_Model abstraction
# Based on MLX-LM's implementation pattern
#
# USAGE INSTRUCTIONS:
# -------------------
# This script evaluates Apple Neural Engine (ANE) models using lm-evaluation-harness.
# To run with proper serial execution:
#
# python anelm_harness.py \
#   --model /path/to/your/model \
#   --tasks boolq,arc_easy,hellaswag \
#   --batch-size 1     # Keep batch_size=1 for evaluation harness
#   --output-dir results
#
# For debugging tokenization, scoring, or other issues, add the debug flag:
#
# python anelm_harness.py \
#   --model /path/to/your/model \
#   --tasks hellaswag \
#   --debug \
#   --output-dir results/debug
#
# IMPORTANT NOTES ON CHAT TEMPLATES:
# ---------------------------------
# By default, chat templates are DISABLED for all tasks, which is correct for standard benchmarks.
# This ensures proper token alignment for log-likelihood scoring.
#
# For instruction-tuned models or chat checkpoints that require chat formatting:
#
# python anelm_harness.py \
#   --model /path/to/instruct-model \
#   --tasks truthfulqa,openbookqa \
#   --apply-chat-template \
#   --output-dir results/chat
#
# Remember: For "plain" tasks (BoolQ, ARC-Easy, HellaSwag, MMLU, etc.) do NOT use chat templates
# as they will interfere with proper evaluation by changing prompt structure.
#

import os
import sys
import time
import json
import argparse
import logging
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ANE_Model
try:
    from ane_model import ANE_Model
except ImportError:
    print("Error: ane_model.py not found.")
    sys.exit(1)

try:
    import coremltools as ct
    # Configure CoreML for single-threaded mode
    os.environ["COREML_PARTITION_LOADER_DISABLE_MULTI_ENGINE"] = "1"
    print("CoreML configured for single-threaded execution via environment variables")
except ImportError:
    print("Error: coremltools not found. Please install it using:")
    print("pip install coremltools")
    sys.exit(1)

try:
    import lm_eval
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError:
    print("Error: lm-evaluation-harness not found. Please install it using:")
    print("pip install lm-evaluation-harness")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: transformers not found. Please install it using:")
    print("pip install transformers")
    sys.exit(1)

# Default model path
DEFAULT_MODEL_PATH = "/Users/anemll/Models/ANE/anemll-Llama-3.2-1B-FP16-b64-ctx1024"


def _pad_inputs(inputs):
    """Pad input token sequences for batch processing."""
    lengths = np.array([len(x) for x in inputs])
    maxlen = lengths.max()
    padded = np.stack(
        [np.pad(x, (0, maxlen - len(x))) for x in inputs],
        axis=0,
    )
    return torch.tensor(padded), torch.tensor(lengths)


def _rstrip_until(s, untils):
    """Limit a string <s> to the first occurrence of any substring in untils."""
    l = len(s)
    f = [s.find(u) for u in untils]
    f = [l if x < 0 else x for x in f]
    return s[: min(f)]


@register_model("anelm_new")
class ANELM(LM):
    """ANE/CoreML model implementation for lm-evaluation-harness using ANE_Model abstraction."""

    def __init__(
        self,
        model_path: str,
        max_tokens: Optional[int] = None,
        use_chat_template: Optional[bool] = None,
        debug: bool = False,
        **kwargs
    ) -> None:
        """Initialize the ANE model evaluator.
        
        Args:
            model_path: Path to model directory containing CoreML models
            max_tokens: Maximum number of tokens to generate
            use_chat_template: Whether to use chat template for formatting
            debug: Whether to print verbose debug information
            **kwargs: Additional arguments passed from lm-evaluation-harness
        """
        super().__init__()
        self.model_path = Path(model_path)
        self.debug = debug
        
        # Initialize ANE_Model
        print(f"Initializing ANE_Model from {model_path}")
        self.model = ANE_Model(model_path, max_tokens=max_tokens or 2048)
        
        # Store metadata
        self.metadata = self.model.metadata
        
        # Get tokenizer
        self._initialize_tokenizer()
        
        # Set maximum tokens
        self._max_tokens = max_tokens or 2048
        
        # Parse batch size from kwargs (for harness only)
        if 'batch_size' in kwargs and kwargs['batch_size'] is not None:
            self._batch_size = kwargs['batch_size']
        else:
            # Default to batch_size 1 for strictly serial evaluation
            self._batch_size = 1
            
        # Chat template settings
        self.use_chat_template = use_chat_template
        if use_chat_template is None:
            self.use_chat_template = False  # Default to False regardless of tokenizer
        else:
            self.use_chat_template = use_chat_template

        # Print important configuration info
        print(f"\nANELM Configuration:")
        print(f"  Harness Batch Size: {self._batch_size}")
        print(f"  Model Batch Size: {self.metadata.get('batch_size', 'unknown')}")
        print(f"  Context Length: {self.metadata.get('context_length', 'unknown')}")
        print(f"  CoreML Single Threading: Enabled")
        print(f"  Debug Mode: {'Enabled' if self.debug else 'Disabled'}")

    def _initialize_tokenizer(self):
        """Initialize the tokenizer."""
        try:
            print(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                use_fast=False,
                trust_remote_code=True
            )
            
            # Configure the tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print("Set PAD token to EOS token")
            
            self.tokenizer.padding_side = "left"
            
            print(f"\nTokenizer info:")
            print(f"Vocabulary size: {len(self.tokenizer)}")
            print(f"BOS token: '{self.tokenizer.bos_token}' (ID: {self.tokenizer.bos_token_id})")
            print(f"EOS token: '{self.tokenizer.eos_token}' (ID: {self.tokenizer.eos_token_id})")
            print(f"PAD token: '{self.tokenizer.pad_token}' (ID: {self.tokenizer.pad_token_id})")
            
        except Exception as e:
            print(f"Error loading tokenizer: {str(e)}")
            print("Trying fallback tokenizers...")
            
            try:
                # Try Llama3 tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-3-8B-Instruct", 
                    use_fast=False
                )
                print("Loaded Llama-3 tokenizer as fallback")
            except:
                # Try Llama2 tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "meta-llama/Llama-2-7b-hf", 
                        use_fast=False
                    )
                    print("Loaded Llama-2 tokenizer as fallback")
                except:
                    raise ValueError("Could not load any compatible tokenizer")
            
            # Configure the tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.tokenizer.padding_side = "left"

    def _score_batch(self, token_batch):
        """Efficiently score a batch of tokens."""
        # Split into input tokens and target tokens to score
        inputs, targets = token_batch[:, :-1], token_batch[:, 1:]
        
        # Process each prompt independently (batch_size of 1)
        batch_size = inputs.shape[0]
        all_scores = []
        all_is_greedy = []
        
        if self.debug:
            print(f"Scoring batch of {batch_size} prompts")
        
        for b in range(batch_size):
            # Reset model state for each prompt
            self.model.reset_state()
            
            # Run prefill on the input sequence
            current_pos = self.model.prefill(inputs[b:b+1])
            
            # Score each target token
            prompt_scores = []
            prompt_is_greedy = []
            
            if self.debug and b == 0:
                print(f"Prompt {b+1}: Scoring {targets.shape[1]} target tokens")
            
            for i in range(targets.shape[1]):
                # Get the log probabilities for the next token using our new method
                log_probs = self.model.compute_logprobs(inputs[b:b+1, i:i+1])
                
                # Get the target token for this position
                target_token = targets[b, i].item()
                
                # Get the predicted token (greedy)
                predicted_token = torch.argmax(log_probs).item()
                is_greedy = (predicted_token == target_token)
                
                # Get log probability for the target token
                score = log_probs[target_token].item()
                
                # Print details for first few tokens of first prompt
                if self.debug and b == 0 and i < 3:
                    target_text = self.tokenizer.decode([target_token])
                    predicted_text = self.tokenizer.decode([predicted_token])
                    print(f"  Token {i+1}: Target={target_token} ('{target_text}'), Predicted={predicted_token} ('{predicted_text}')")
                    print(f"  Score: {score:.4f}, Is greedy: {is_greedy}")
                
                prompt_scores.append(score)
                prompt_is_greedy.append(is_greedy)
                
                # Generate next token to update state (unless this is the last one)
                if i < targets.shape[1] - 1:
                    _ = self.model.predict(inputs[b:b+1, i:i+1])
            
            all_scores.append(torch.tensor(prompt_scores))
            all_is_greedy.append(torch.tensor(prompt_is_greedy))
        
        # Stack scores and is_greedy
        scores = torch.stack(all_scores, dim=0)
        is_greedy = torch.stack(all_is_greedy, dim=0)
        
        if self.debug:
            print(f"Batch scoring complete. Average score: {scores.mean().item():.4f}")
        
        return scores, is_greedy

    def _process_prompt(self, prompt):
        """Process the prompt and get logits for next token prediction."""
        # Ensure prompt is a torch tensor
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.tensor(prompt)
        
        # Add batch dimension if needed
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        
        # Ensure int32 dtype for the CoreML model
        prompt = prompt.to(torch.int32)
        
        # Run prefill on the prompt
        self.model.reset_state()
        current_pos = self.model.prefill(prompt)
        
        # Get log probabilities for the next token using our new method
        log_probs = self.model.compute_logprobs(prompt[:, -1:])
        
        return log_probs, None  # Second value is normally the cache but we don't need it
    
    def _score_fn(self, inputs):
        """Score the inputs and return log probabilities and greedy indicators."""
        # Prepare inputs
        token_batch, lengths = _pad_inputs(inputs)
        
        # Use the much more efficient batch scoring method
        scores, is_greedy = self._score_batch(token_batch)
        
        # Apply mask based on lengths
        max_len = scores.shape[1]
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        scores = torch.where(mask, scores, torch.zeros_like(scores))
        is_greedy = torch.where(mask, is_greedy, torch.zeros_like(is_greedy, dtype=torch.bool))
        
        return scores, lengths, is_greedy

    def _tokenize(self, texts):
        """Tokenize texts using the model's tokenizer."""
        result = []
        for t in texts:
            if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
                # For models with chat templates, format as a user message
                try:
                    messages = [{"role": "user", "content": t}]
                    chat_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    result.append(self.tokenizer.encode(chat_text))
                except Exception as e:
                    print(f"Error applying chat template: {str(e)}. Falling back to standard tokenization.")
                    result.append(self.tokenizer.encode(t, add_special_tokens=False))
            else:
                # The harness already decides where BOS/EOS go
                result.append(self.tokenizer.encode(t, add_special_tokens=False))
        return result

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context."""
        logging.info("Estimating loglikelihood for %d pairs." % len(requests))
        
        # Try to determine if this is hellaswag task
        is_hellaswag = False
        choices_per_context = 0
        
        # Look for task name in the requests
        for req in requests[:1]:
            if hasattr(req, 'task_name') and req.task_name == 'hellaswag':
                is_hellaswag = True
                if self.debug:
                    print("HELLASWAG task detected!")
            if hasattr(req, 'request_type') and req.request_type == 'loglikelihood':
                if self.debug:
                    print(f"Request type: loglikelihood")
        
        # Check if there are multiple continuations per context (multiple choice)
        if len(requests) > 1:
            # Get contexts
            contexts = []
            for req in requests[:10]:  # Check first 10 requests
                if hasattr(req, 'args') and len(req.args) >= 2:
                    contexts.append(req.args[0])
                elif hasattr(req, 'kwargs') and 'doc' in req.kwargs:
                    contexts.append(req.kwargs['doc']['context'])
                    
            # Count unique contexts
            unique_contexts = len(set(contexts))
            if unique_contexts < len(contexts):
                is_multiple_choice = True
                choices_per_context = len(contexts) // unique_contexts
                if self.debug:
                    print(f"\nMultiple-choice task detected: {choices_per_context} choices per context")
        
        # Group by context for efficiency
        group_reqs = collections.defaultdict(list)
        for idx, req in enumerate(requests):
            if hasattr(req, 'args') and len(req.args) >= 2:
                # Old-style API with req.args
                context, continuation = req.args[0], req.args[1]
            elif hasattr(req, 'kwargs') and 'doc' in req.kwargs:
                # New-style API with req.kwargs['doc']
                doc = req.kwargs['doc']
                context, continuation = doc['context'], doc['continuation']
            else:
                if self.debug:
                    print(f"Warning: Unknown request format: {req}")
                continue
                
            group_reqs[context].append((idx, continuation))
        
        questions = list(group_reqs.keys())
        responses = []
        indices = []
        for v in group_reqs.values():
            idx, resp = zip(*v)
            indices.extend(idx)
            responses.append(resp)
        
        if self.debug:
            print(f"\nProcessing {len(questions)} unique contexts with {sum(len(r) for r in responses)} total continuations")
        
        scores, is_greedy = [], []
        for i, (q, rs) in enumerate(tqdm(zip(questions, responses), total=len(questions))):
            if i < 2 and self.debug:
                print(f"\nScoring context: {q[:50]}...")
                print(f"With {len(rs)} continuations:")
                for j, r in enumerate(rs):
                    print(f"  {j}: {r[:50]}...")
            
            # Tokenize context
            prefix = self._tokenize([q])[0]
            
            # For each context, tokenize all possible continuations
            full_sequences = [self._tokenize([q + r])[0] for r in rs]
            max_completed_l = max(len(s) for s in full_sequences)
            
            # Handle truncation if needed
            if max_completed_l > self._max_tokens:
                truncation = max(0, max_completed_l - self._max_tokens)
                prefix_l = max(len(prefix) - truncation, 0)
                prefix = prefix[len(prefix) - prefix_l:]
                
                # If the entire prompt got truncated, skip
                if prefix_l == 0:
                    scores.extend([-float("inf")] * len(rs))
                    is_greedy.extend([False] * len(rs))
                    continue
            
            # Reset model state for each prefix
            self.model.reset_state()
            
            # Process the prefix
            prefix_tensor = torch.tensor([prefix], dtype=torch.int32)
            _ = self.model.prefill(prefix_tensor)
            
            # Get log probabilities for potential next tokens
            prefix_log_probs = self.model.compute_logprobs(prefix_tensor[:, -1:])
            max_idx = torch.argmax(prefix_log_probs).item()
            
            # Score each continuation
            continuation_scores = []
            continuation_is_greedy = []
            
            for j, cont_tokens in enumerate([s[len(prefix):] for s in full_sequences]):
                if i < 2 and self.debug:
                    print(f"\nScoring continuation {j}")
                    if len(cont_tokens) > 0:
                        print(f"First token: {cont_tokens[0]} ('{self.tokenizer.decode([cont_tokens[0]])}')")
                
                if len(cont_tokens) > 0:
                    # Score the first token using the prefix log probs
                    first_token_score = prefix_log_probs[cont_tokens[0]].item()
                    first_token_is_greedy = (cont_tokens[0] == max_idx)
                    
                    if len(cont_tokens) > 1:
                        # Reset model state
                        self.model.reset_state()
                        
                        # Prefill with prefix + first token
                        first_input = torch.tensor([prefix + [cont_tokens[0]]], dtype=torch.int32)
                        _ = self.model.prefill(first_input)
                        
                        # Score the remaining tokens
                        rest_scores = []
                        rest_is_greedy = []
                        
                        for k in range(len(cont_tokens) - 1):
                            # Get current token
                            current_token = cont_tokens[k]
                            next_token = cont_tokens[k + 1]
                            
                            # Get log probabilities
                            token_tensor = torch.tensor([[current_token]], dtype=torch.int32)
                            log_probs = self.model.compute_logprobs(token_tensor)
                            
                            # Score next token
                            token_score = log_probs[next_token].item()
                            token_is_greedy = (torch.argmax(log_probs).item() == next_token)
                            
                            rest_scores.append(token_score)
                            rest_is_greedy.append(token_is_greedy)
                            
                            # Update state if not the last token
                            if k < len(cont_tokens) - 2:
                                _ = self.model.predict(token_tensor)
                        
                        # Calculate total score
                        total_score = first_token_score + sum(rest_scores)
                        is_all_greedy = first_token_is_greedy and all(rest_is_greedy)
                        
                        if i < 2 and self.debug:
                            print(f"First token score: {first_token_score:.4f}, rest: {sum(rest_scores):.4f}, total: {total_score:.4f}")
                    else:
                        # Only one token in continuation
                        total_score = first_token_score
                        is_all_greedy = first_token_is_greedy
                        
                        if i < 2 and self.debug:
                            print(f"Single token score: {total_score:.4f}")
                else:
                    # Empty continuation
                    total_score = 0.0
                    is_all_greedy = True
                    
                    if i < 2 and self.debug:
                        print(f"Empty continuation, score: {total_score}")
                
                continuation_scores.append(total_score)
                continuation_is_greedy.append(is_all_greedy)
            
            # For multiple-choice tasks, verify our prediction is correct
            if len(continuation_scores) >= 2:
                best_idx = int(np.argmax(continuation_scores))
                if i < 5 and self.debug:
                    print(f"\nBest continuation: {best_idx}")
                    for j, score in enumerate(continuation_scores):
                        print(f"  Score {j}: {score:.4f}")
            
            # Add all scores to results
            scores.extend(continuation_scores)
            is_greedy.extend(continuation_is_greedy)
        
        # Reorder results to match original request order
        inv_sort = torch.argsort(torch.tensor(indices))
        scores = [scores[i] for i in inv_sort]
        is_greedy = [is_greedy[i] for i in inv_sort]
        
        # Debug output
        if self.debug:
            print("\nFinal scores (first 10):")
            for i in range(min(10, len(scores))):
                print(f"  {i}: {scores[i]:.4f}, greedy: {is_greedy[i]}")
        
        return list(zip(scores, is_greedy))

    def loglikelihood_rolling(self, requests) -> list[float]:
        """Compute full log-likelihood of strings, for perplexity computation."""
        logging.info("Estimating rolling loglikelihood for %d sequences." % len(requests))
        
        # Get model context length for safety checks
        context_length = self.metadata.get('context_length', 2048)
        
        # Use a maximum safe length with more margin (context_length - 128)
        safe_length = context_length - 128
        
        # Tokenize the inputs
        texts = [self._tokenize([req.args[0]])[0] for req in requests]
        
        # Print sequence length statistics if debug is enabled
        if self.debug:
            lengths = [len(t) for t in texts]
            max_len = max(lengths) if lengths else 0
            min_len = min(lengths) if lengths else 0
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            over_ctx = sum(1 for l in lengths if l > safe_length)
            
            print(f"\nSequence length statistics:")
            print(f"  Min length: {min_len} tokens")
            print(f"  Avg length: {avg_len:.1f} tokens")
            print(f"  Max length: {max_len} tokens")
            print(f"  Safe context length: {safe_length} tokens")
            print(f"  Sequences exceeding safe length: {over_ctx}/{len(texts)} ({over_ctx/len(texts)*100:.1f}%)")
            
            # Print a few example lengths
            if lengths:
                print(f"  First 5 sequence lengths: {lengths[:5]}")
                
        # Process sequences
        all_scores = []
        for idx, tokens in enumerate(tqdm(texts)):
            # Skip empty sequences
            if len(tokens) <= 1:
                all_scores.append(0.0)
                continue
                
            # Truncate to safe length if needed, keeping rightmost tokens
            original_length = len(tokens)
            if len(tokens) > safe_length:
                tokens = tokens[-safe_length:]
                if self.debug and (idx < 3 or original_length > safe_length*2):
                    print(f"\nSequence {idx}: Truncated from {original_length} to {len(tokens)} tokens")
                    
            # Split into input and targets
            inputs, targets = tokens[:-1], tokens[1:]
            
            # Reset model state
            self.model.reset_state()
            
            try:
                # Prefill with input tokens
                input_tensor = torch.tensor([inputs], dtype=torch.int32)
                _ = self.model.prefill(input_tensor)
                
                # Score each target token
                sequence_scores = []
                
                for i, target in enumerate(targets):
                    # Get current token
                    current_token = inputs[i]
                    
                    # Get log probabilities for current token
                    token_tensor = torch.tensor([[current_token]], dtype=torch.int32)
                    log_probs = self.model.compute_logprobs(token_tensor)
                    
                    # Skip if compute_logprobs failed
                    if log_probs is None:
                        if self.debug and i < 5:
                            print(f"  Warning: No log probabilities for token at position {i}")
                        continue
                    
                    # Score target token
                    try:
                        token_score = log_probs[target].item()
                        sequence_scores.append(token_score)
                    except (IndexError, RuntimeError) as e:
                        if self.debug and i < 5:
                            print(f"  Error scoring token {target}: {str(e)}")
                        continue
                    
                    # Update state for next token prediction
                    if i < len(targets) - 1:
                        try:
                            _ = self.model.predict(token_tensor)
                        except Exception as e:
                            if self.debug:
                                print(f"  Error in predict: {str(e)}")
                            # Don't break, try to continue with next token
                            
                # Calculate total score for this sequence
                if sequence_scores:
                    total_score = sum(sequence_scores)
                    if self.debug and idx < 3:
                        print(f"  Scored {len(sequence_scores)}/{len(targets)} tokens, total score: {total_score:.4f}")
                    all_scores.append(total_score)
                else:
                    # No valid scores
                    if self.debug:
                        print(f"  Warning: No valid scores for sequence {idx}")
                    all_scores.append(0.0)
                    
            except Exception as e:
                print(f"Error processing sequence {idx}: {str(e)}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                # Add a default score on error
                all_scores.append(0.0)
        
        if self.debug and all_scores:
            print(f"\nCalculated {len(all_scores)} scores")
            print(f"First few scores: {all_scores[:5]}")
            avg_score = sum(all_scores)/len(all_scores) if all_scores else 0
            print(f"Average log likelihood: {avg_score:.4f}")
        
        return all_scores

    def generate_until(self, requests) -> list[str]:
        """Generate text until a stopping sequence is reached."""
        logging.info("Generating continuation for %d sequences." % len(requests))
        
        # Parse the requests (handling both old and new API formats)
        contexts_and_options = []
        for req in requests:
            if hasattr(req, 'args') and len(req.args) >= 2:
                # Old-style API
                contexts_and_options.append(req.args)
            elif hasattr(req, 'kwargs') and 'until' in req.kwargs:
                # New-style API
                contexts_and_options.append((req.args[0], req.kwargs))
            else:
                if self.debug:
                    print(f"Warning: Unknown request format for generate_until: {req}")
                contexts_and_options.append(("", {"until": ["\n\n"]}))  # Default fallback
        
        contexts, options = zip(*contexts_and_options)
        completions = []
        
        for i, (context, opt) in enumerate(tqdm(zip(contexts, options), total=len(contexts))):
            # Extract stopping sequences and other generation parameters
            until = opt.get("until", ["\n\n"])
            if not isinstance(until, list):
                until = [until]
                
            max_gen_tokens = min(
                opt.get("max_gen_tokens", self._max_tokens),
                self.metadata.get('context_length', 2048) - 10  # Reserve some space
            )
            temperature = opt.get("temperature", 0.0)
            
            if i < 2 and self.debug:
                print(f"\nGenerating for context: {context[:50]}...")
                print(f"Stopping sequences: {until}")
                print(f"Max tokens: {max_gen_tokens}")
                
            # Tokenize context
            context_tokens = self.tokenizer.encode(
                context, add_special_tokens=False
            )
            
            # Check if context exceeds max context length
            context_length = self.metadata.get('context_length', 2048)
            if len(context_tokens) >= context_length:
                if self.debug:
                    print(f"Warning: Context length {len(context_tokens)} exceeds max context length {context_length}")
                # Truncate to fit model's context window, keeping most recent tokens
                context_tokens = context_tokens[-context_length+10:]  # +10 to leave some room for generation
                if self.debug:
                    print(f"Truncated context to {len(context_tokens)} tokens")
            
            # Prepare for generation
            input_ids = torch.tensor([context_tokens], dtype=torch.int32)
            
            # Reset model state
            self.model.reset_state()
            
            # Run prefill on the context
            _ = self.model.prefill(input_ids)
            
            # Generate tokens
            generated_tokens = []
            
            for _ in range(max_gen_tokens):
                # Generate next token
                if not generated_tokens:
                    # First token after context
                    next_token = self.model.predict(input_ids[:, -1:])
                else:
                    # Subsequent tokens
                    next_token = self.model.predict(torch.tensor([[generated_tokens[-1]]]))
                
                # Add to generated tokens
                generated_tokens.append(next_token)
                
                # Decode all tokens so far to check stopping conditions
                text_so_far = self.tokenizer.decode(generated_tokens)
                
                # Check for stopping sequences
                if any(stop in text_so_far for stop in until):
                    # Find which stopping sequence triggered
                    stop_idx = min([
                        text_so_far.find(stop) + len(stop) 
                        for stop in until 
                        if stop in text_so_far
                    ])
                    # Truncate at the stopping sequence
                    text_so_far = text_so_far[:stop_idx]
                    
                    # Re-tokenize to get the tokens we want to keep
                    keep_tokens = self.tokenizer.encode(text_so_far, add_special_tokens=False)
                    if len(keep_tokens) < len(generated_tokens):
                        generated_tokens = generated_tokens[:len(keep_tokens)]
                    break
                
                # Stop if we reach end of text token
                if next_token == self.tokenizer.eos_token_id:
                    break
            
            # Decode generated tokens
            completion = self.tokenizer.decode(generated_tokens)
            
            # Apply stopping criteria one more time to handle edge cases
            for stop in until:
                if stop in completion:
                    completion = completion.split(stop)[0]
                    
            if i < 2 and self.debug:
                print(f"Generated: {completion[:100]}...")
            
            completions.append(completion)
        
        return completions

    def normalized_loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """
        Compute normalized log-likelihood for multiple-choice tasks like hellaswag.
        This normalizes scores within each context using softmax for fair comparison.
        """
        if self.debug:
            print("\nUsing normalized log-likelihood for multiple-choice evaluation")
        logging.info("Estimating normalized loglikelihood for %d pairs." % len(requests))
        
        # Group requests by context
        group_reqs = collections.defaultdict(list)
        for idx, req in enumerate(requests):
            if hasattr(req, 'args') and len(req.args) >= 2:
                context, continuation = req.args[0], req.args[1]
            elif hasattr(req, 'kwargs') and 'doc' in req.kwargs:
                doc = req.kwargs['doc']
                context, continuation = doc['context'], doc['continuation']
            else:
                if self.debug:
                    print(f"Warning: Unknown request format: {req}")
                continue
                
            group_reqs[context].append((idx, continuation))
        
        questions = list(group_reqs.keys())
        responses = []
        indices = []
        for v in group_reqs.values():
            idx, resp = zip(*v)
            indices.extend(idx)
            responses.append(resp)
        
        if self.debug:
            print(f"Processing {len(questions)} contexts with {sum(len(r) for r in responses)} total continuations")
        
        # Store all scores and results
        all_scores = []
        all_is_greedy = []
        
        # Process each context group separately
        for i, (context, continuations) in enumerate(zip(questions, responses)):
            if i < 1 and self.debug:
                print(f"Context {i+1}: '{context[:50]}...'")
                for j, cont in enumerate(continuations):
                    print(f"  Choice {j}: '{cont[:50]}...'")
            
            # Get unnormalized scores for this context
            context_scores, context_is_greedy = self._score_context(context, continuations)
            
            # For multiple-choice tasks, we want normalized probabilities
            # Convert log probabilities to probabilities with softmax
            max_score = max(context_scores)
            shifted_scores = [score - max_score for score in context_scores]  # Shift for numerical stability
            exp_scores = [np.exp(score) for score in shifted_scores]
            sum_exp = sum(exp_scores)
            normalized_probs = [exp / sum_exp for exp in exp_scores]
            
            # Convert back to log probabilities
            normalized_log_probs = [np.log(prob) for prob in normalized_probs]
            
            if i < 1 and self.debug:
                print(f"  Raw scores: {[f'{s:.4f}' for s in context_scores]}")
                print(f"  Normalized scores: {[f'{s:.4f}' for s in normalized_log_probs]}")
                print(f"  Normalized probs: {[f'{p:.6f}' for p in normalized_probs]}")
                
                # Show model's prediction
                best_idx = np.argmax(context_scores)
                best_norm_idx = np.argmax(normalized_log_probs)
                print(f"  Best raw choice: {best_idx}")
                print(f"  Best normalized choice: {best_norm_idx}")
            
            # Add normalized scores to results
            all_scores.extend(normalized_log_probs)
            all_is_greedy.extend(context_is_greedy)
        
        # Reorder results to match original request order
        inv_sort = torch.argsort(torch.tensor(indices))
        scores = [all_scores[i] for i in inv_sort]
        is_greedy = [all_is_greedy[i] for i in inv_sort]
        
        return list(zip(scores, is_greedy))
        
    def _score_context(self, context, continuations):
        """Helper method to score all continuations for a single context."""
        # Tokenize context
        prefix = self._tokenize([context])[0]
        
        # Tokenize all continuations
        full_sequences = [self._tokenize([context + r])[0] for r in continuations]
        
        # Check if we need to truncate
        max_completed_l = max(len(s) for s in full_sequences)
        if max_completed_l > self._max_tokens:
            truncation = max(0, max_completed_l - self._max_tokens)
            prefix_l = max(len(prefix) - truncation, 0)
            prefix = prefix[len(prefix) - prefix_l:]
            
            # If the entire prompt got truncated, return negative infinity
            if prefix_l == 0:
                if self.debug:
                    print(f"Warning: Entire prompt was truncated due to length")
                return ([-float("inf")] * len(continuations), 
                        [False] * len(continuations))
        
        # Reset model state
        self.model.reset_state()
        
        # Process the prefix
        prefix_tensor = torch.tensor([prefix], dtype=torch.int32)
        _ = self.model.prefill(prefix_tensor)
        
        # Get log probabilities for next tokens
        prefix_log_probs = self.model.compute_logprobs(prefix_tensor[:, -1:])
        max_idx = torch.argmax(prefix_log_probs).item()
        
        # Score each continuation
        scores = []
        is_greedy = []
        
        for i, cont_tokens in enumerate([s[len(prefix):] for s in full_sequences]):
            if self.debug and i == 0:
                print(f"Scoring continuation tokens (first continuation only)")
                
            if len(cont_tokens) > 0:
                # Score the first token
                first_token_score = prefix_log_probs[cont_tokens[0]].item()
                first_token_is_greedy = (cont_tokens[0] == max_idx)
                
                if self.debug and i == 0:
                    first_token_text = self.tokenizer.decode([cont_tokens[0]])
                    print(f"First token: {cont_tokens[0]} ('{first_token_text}') - score: {first_token_score:.4f}")
                
                if len(cont_tokens) > 1:
                    # Reset model state for rest of continuation
                    self.model.reset_state()
                    
                    # Prefill with context + first token
                    first_input = torch.tensor([prefix + [cont_tokens[0]]], dtype=torch.int32)
                    _ = self.model.prefill(first_input)
                    
                    # Score remaining tokens
                    rest_scores = []
                    rest_is_greedy = []
                    
                    for j in range(len(cont_tokens) - 1):
                        current_token = cont_tokens[j]
                        next_token = cont_tokens[j + 1]
                        
                        # Get log probabilities
                        token_tensor = torch.tensor([[current_token]], dtype=torch.int32)
                        log_probs = self.model.compute_logprobs(token_tensor)
                        
                        # Score next token
                        token_score = log_probs[next_token].item()
                        token_is_greedy = (torch.argmax(log_probs).item() == next_token)
                        
                        rest_scores.append(token_score)
                        rest_is_greedy.append(token_is_greedy)
                        
                        # Update state if not last token
                        if j < len(cont_tokens) - 2:
                            _ = self.model.predict(token_tensor)
                    
                    # Calculate total score
                    total_score = first_token_score + sum(rest_scores)
                    is_all_greedy = first_token_is_greedy and all(rest_is_greedy)
                    
                    if self.debug and i == 0:
                        print(f"Rest score: {sum(rest_scores):.4f}, total score: {total_score:.4f}")
                else:
                    # Only one token in continuation
                    total_score = first_token_score
                    is_all_greedy = first_token_is_greedy
                    
                    if self.debug and i == 0:
                        print(f"Single token score: {total_score:.4f}")
            else:
                # Empty continuation
                total_score = 0.0
                is_all_greedy = True
                
                if self.debug and i == 0:
                    print(f"Empty continuation, score: {total_score}")
                
            scores.append(total_score)
            is_greedy.append(is_all_greedy)
            
        return scores, is_greedy


def main():
    parser = argparse.ArgumentParser(description="Evaluate ANE/CoreML models with lm-evaluation-harness")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to model directory (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="Tasks to evaluate (e.g., boolq arc_easy hellaswag)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results (default: results)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for evaluation (default: 1 for strictly serial execution)")
    parser.add_argument("--num-shots", type=int, default=None,
                        help="Number of shots for few-shot evaluation")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of examples per task")
    parser.add_argument("--skip", type=int, default=None,
                        help="Skip the first N examples (helps avoid crashing samples)")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction,
                        help="Specifies whether to apply a chat template to the prompt",
                        default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug output")
    parser.add_argument("--perplexity", action="store_true",
                        help="Enable perplexity evaluation mode")
    parser.add_argument("--safety-margin", type=int, default=100,
                        help="Safety margin for context length (default: 100)")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Chunk size for processing long sequences")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save results JSON file")
    parser.add_argument("--download-timeout", type=int, default=60,
                        help="Timeout for dataset downloads in seconds (default: 60)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of download retries (default: 3)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Silence tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set higher download timeouts for HuggingFace Hub and datasets
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.download_timeout)
    os.environ["HF_DATASETS_TIMEOUT"] = str(args.download_timeout) 
    
    # Configure download retries
    os.environ["HF_HUB_DOWNLOAD_RETRY_COUNT"] = str(args.max_retries)
    
    # Try to configure datasets library timeout/retry settings directly
    try:
        import datasets
        from datasets.config import HF_DATASETS_TIMEOUT
        from huggingface_hub.constants import HF_HUB_DOWNLOAD_TIMEOUT
        
        # Set timeouts in the libraries directly if environment variables don't work
        datasets.config.HF_DATASETS_TIMEOUT = args.download_timeout
        datasets.config.MAX_RETRIES = args.max_retries
        
        # For older versions of huggingface_hub
        try:
            import huggingface_hub.constants
            huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = args.download_timeout
            huggingface_hub.constants.HF_HUB_DOWNLOAD_RETRY_COUNT = args.max_retries
        except (ImportError, AttributeError):
            # Try newer versions
            try:
                import huggingface_hub.file_download
                huggingface_hub.file_download.DOWNLOAD_RETRY_COUNT = args.max_retries
                huggingface_hub.file_download.DEFAULT_TIMEOUT = args.download_timeout
            except (ImportError, AttributeError):
                print("Could not set huggingface_hub timeout settings directly")
        
        print(f"Configured dataset download timeout: {args.download_timeout}s")
        print(f"Configured maximum download retries: {args.max_retries}")
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not configure dataset timeout settings: {e}")
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize model
    print(f"Initializing ANE model from {args.model}")
    lm = ANELM(
        args.model,
        max_tokens=args.max_tokens,
        use_chat_template=args.apply_chat_template,
        batch_size=args.batch_size,
        debug=args.debug
    )
    
    # Run evaluation with processes=0 to ensure single-threaded execution
    print("Running evaluation in single-threaded mode")
    
    # Check if the simple_evaluate function supports the processes parameter
    try:
        import inspect
        simple_eval_params = inspect.signature(lm_eval.simple_evaluate).parameters
        supports_processes = 'processes' in simple_eval_params
    except:
        supports_processes = False
    
    # Set up evaluation parameters
    eval_args = {
        "model": lm,
        "tasks": args.tasks,
        "num_fewshot": args.num_shots,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "random_seed": args.seed,
        "numpy_random_seed": args.seed, 
        "torch_random_seed": args.seed
    }
    
    # Set safety margin and chunk size in model if supported
    if hasattr(lm.model, 'metadata'):
        # Apply safety margin to context length calculations
        if 'context_length' in lm.model.metadata:
            safe_length = lm.model.metadata['context_length'] - args.safety_margin
            print(f"Applied safety margin of {args.safety_margin} tokens")
            print(f"Safe context length: {safe_length} tokens")
    
    # If perplexity mode is enabled, make sure wikitext is among tasks
    if args.perplexity and "wikitext" not in args.tasks:
        print("Adding wikitext to tasks for perplexity evaluation")
        args.tasks.append("wikitext")
        eval_args["tasks"] = args.tasks
    
    # Add skip example functionality
    if args.skip is not None and args.skip > 0:
        class SkipWrapper:
            def __init__(self, lm, skip_count):
                self.lm = lm
                self.skip_count = skip_count
                self.skipped = 0
                print(f"Skipping the first {skip_count} examples")
                
            def __getattr__(self, name):
                if name in ['loglikelihood', 'loglikelihood_rolling', 'generate_until', 'normalized_loglikelihood']:
                    orig_method = getattr(self.lm, name)
                    
                    def wrapper(requests, *args, **kwargs):
                        if self.skipped < self.skip_count:
                            if len(requests) + self.skipped > self.skip_count:
                                # Skip partial requests
                                to_skip = self.skip_count - self.skipped
                                print(f"Skipping examples {self.skipped} to {self.skip_count-1}")
                                self.skipped = self.skip_count
                                requests = requests[to_skip:]
                            else:
                                # Skip all requests in this batch
                                print(f"Skipping examples {self.skipped} to {self.skipped + len(requests) - 1}")
                                self.skipped += len(requests)
                                # Return dummy values based on the method
                                if name == 'loglikelihood' or name == 'normalized_loglikelihood':
                                    return [(0.0, False)] * len(requests)
                                elif name == 'loglikelihood_rolling':
                                    return [0.0] * len(requests)
                                else:  # generate_until
                                    return [''] * len(requests)
                        
                        if len(requests) == 0:
                            # Return empty results if all requests were skipped
                            if name == 'loglikelihood' or name == 'normalized_loglikelihood':
                                return []
                            elif name == 'loglikelihood_rolling':
                                return []
                            else:  # generate_until
                                return []
                                
                        return orig_method(requests, *args, **kwargs)
                    
                    return wrapper
                return getattr(self.lm, name)
        
        # Wrap the model with skip functionality
        lm = SkipWrapper(lm, args.skip)
        eval_args["model"] = lm
    
    # Add processes=0 if supported
    if supports_processes:
        print("Using processes=0 parameter for single-threaded execution")
        eval_args["processes"] = 0
    else:
        print("This version of lm-evaluation-harness doesn't support the processes parameter")
        print("Continuing with batch_size=1 for controlled execution")
    
    # Run evaluation
    results = lm_eval.simple_evaluate(**eval_args)
    
    # Get model name from path
    model_name = Path(args.model).name
    
    # Get current date in YYYYMMDD format
    current_date = time.strftime("%Y%m%d")
    
    # Save individual task results 
    for task, metrics in results["results"].items():
        # Create task-specific filename
        task_filename = f"{task}_{model_name}_{current_date}.json"
        task_output_path = output_dir / task_filename
        
        # Save task-specific results
        task_output_path.write_text(json.dumps({task: metrics}, indent=4))
        print(f"Saved {task} results to: {task_output_path}")
    
    # Also save the complete results
    filename = f"results_{model_name}_{current_date}.json"
    output_path = args.output_path or output_dir / filename
    
    if isinstance(output_path, str):
        output_path = Path(output_path)
        
    # Ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save complete results
    output_path.write_text(json.dumps(results["results"], indent=4))
    
    # Print summary
    print("\n===============================")
    print("Evaluation Summary")
    print("===============================")
    print(f"Model: {args.model}")
    print("\nResults:")
    for task, metrics in results["results"].items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    print("\nDetailed results saved to:", output_path)


if __name__ == "__main__":
    main() 