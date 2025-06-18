#!/usr/bin/env python3
# ANE/CoreML model evaluation with lm-evaluation-harness using ANE_Model abstraction
# Based on MLX-LM's implementation pattern
#
# USAGE INSTRUCTIONS:
# -------------------
# This script evaluates Apple Neural Engine (ANE) models using lm-evaluation-harness.
# To run with proper serial execution:
#
# python anelm_harness.py \\
#   --model /path/to/your/model \\
#   --tasks boolq,arc_easy,hellaswag \\
#   --batch-size 1     # Keep batch_size=1 for evaluation harness
#   --output-dir results
#
# For debugging tokenization, scoring, or other issues, add the debug flag:
#
# python anelm_harness.py \\
#   --model /path/to/your/model \\
#   --tasks hellaswag \\
#   --debug \\
#   --output-dir results/debug
#
# IMPORTANT NOTES ON CHAT TEMPLATES:
# ---------------------------------
# By default, chat templates are DISABLED for all tasks, which is correct for standard benchmarks.
# This ensures proper token alignment for log-likelihood scoring.
#
# For instruction-tuned models or chat checkpoints that require chat formatting:
#
# python anelm_harness.py \\
#   --model /path/to/instruct-model \\
#   --tasks truthfulqa,openbookqa \\
#   --apply-chat-template \\
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
    # print("CoreML configured for single-threaded execution via environment variables") # Less verbose
except ImportError:
    print("Error: coremltools not found. Please install it using:")
    print("pip install coremltools")
    sys.exit(1)

try:
    import lm_eval
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval.tasks import get_task_dict # For getting task objects
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
DEFAULT_MODEL_PATH = os.path.expandvars("$HOME/Models/ANE/anemll-Llama-3.2-1B-FP16-b64-ctx1024")


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
        
        if self.debug:
            print(f"Initializing ANE_Model from {model_path}")
        self.model = ANE_Model(model_path, max_tokens=max_tokens or 2048)
        
        self.metadata = self.model.metadata
        self._initialize_tokenizer()
        self._max_tokens = max_tokens or 2048
        
        if 'batch_size' in kwargs and kwargs['batch_size'] is not None:
            self._batch_size = kwargs['batch_size']
        else:
            self._batch_size = 1
            
        self.use_chat_template = use_chat_template
        if use_chat_template is None:
            self.use_chat_template = False
        else:
            self.use_chat_template = use_chat_template

        if self.debug:
            print(f"\\\\nANELM Configuration:")
            print(f"  Harness Batch Size: {self._batch_size}")
            print(f"  Model Batch Size: {self.metadata.get('batch_size', 'unknown')}")
            print(f"  Context Length: {self.metadata.get('context_length', 'unknown')}")
            print(f"  CoreML Single Threading: Enabled") # Assumed by env var
            print(f"  Debug Mode: {'Enabled' if self.debug else 'Disabled'}")

    def _initialize_tokenizer(self):
        """Initialize the tokenizer."""
        try:
            if self.debug:
                print(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                use_fast=False,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                if self.debug:
                    print("Set PAD token to EOS token")
            
            self.tokenizer.padding_side = "left"
            
            if self.debug:
                print(f"\\\\nTokenizer info:")
                print(f"Vocabulary size: {len(self.tokenizer)}")
                print(f"BOS token: '{self.tokenizer.bos_token}' (ID: {self.tokenizer.bos_token_id})")
                print(f"EOS token: '{self.tokenizer.eos_token}' (ID: {self.tokenizer.eos_token_id})")
                print(f"PAD token: '{self.tokenizer.pad_token}' (ID: {self.tokenizer.pad_token_id})")
            
        except Exception as e:
            print(f"Error loading tokenizer: {str(e)}")
            print("Trying fallback tokenizers...")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-3-8B-Instruct", 
                    use_fast=False
                )
                if self.debug: print("Loaded Llama-3 tokenizer as fallback")
            except:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "meta-llama/Llama-2-7b-hf", 
                        use_fast=False
                    )
                    if self.debug: print("Loaded Llama-2 tokenizer as fallback")
                except:
                    raise ValueError("Could not load any compatible tokenizer")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.tokenizer.padding_side = "left"

    def _score_batch(self, token_batch):
        """Efficiently score a batch of tokens."""
        inputs, targets = token_batch[:, :-1], token_batch[:, 1:]
        batch_size = inputs.shape[0]
        all_scores = []
        all_is_greedy = []
        
        if self.debug:
            print(f"Scoring batch of {batch_size} prompts")
        
        for b in range(batch_size):
            self.model.reset_state()
            _ = self.model.prefill(inputs[b:b+1])
            prompt_scores = []
            prompt_is_greedy = []
            
            if self.debug and b == 0:
                print(f"Prompt {b+1}: Scoring {targets.shape[1]} target tokens")
            
            for i in range(targets.shape[1]):
                log_probs = self.model.compute_logprobs(inputs[b:b+1, i:i+1])
                target_token = targets[b, i].item()
                predicted_token = torch.argmax(log_probs).item()
                is_greedy = (predicted_token == target_token)
                score = log_probs[target_token].item()
                
                if self.debug and b == 0 and i < 3:
                    target_text = self.tokenizer.decode([target_token])
                    predicted_text = self.tokenizer.decode([predicted_token])
                    print(f"  Token {i+1}: Target={target_token} ('{target_text}'), Predicted={predicted_token} ('{predicted_text}')")
                    print(f"  Score: {score:.4f}, Is greedy: {is_greedy}")
                
                prompt_scores.append(score)
                prompt_is_greedy.append(is_greedy)
                
                if i < targets.shape[1] - 1:
                    _ = self.model.predict(inputs[b:b+1, i:i+1])
            
            all_scores.append(torch.tensor(prompt_scores))
            all_is_greedy.append(torch.tensor(prompt_is_greedy))
        
        scores = torch.stack(all_scores, dim=0)
        is_greedy = torch.stack(all_is_greedy, dim=0)
        
        if self.debug:
            print(f"Batch scoring complete. Average score: {scores.mean().item():.4f}")
        
        return scores, is_greedy

    def _process_prompt(self, prompt):
        """Process the prompt and get logits for next token prediction."""
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.tensor(prompt)
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        prompt = prompt.to(torch.int32)
        self.model.reset_state()
        _ = self.model.prefill(prompt)
        log_probs = self.model.compute_logprobs(prompt[:, -1:])
        return log_probs, None
    
    def _score_fn(self, inputs):
        """Score the inputs and return log probabilities and greedy indicators."""
        token_batch, lengths = _pad_inputs(inputs)
        scores, is_greedy = self._score_batch(token_batch)
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
                try:
                    messages = [{"role": "user", "content": t}]
                    chat_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    result.append(self.tokenizer.encode(chat_text))
                except Exception as e:
                    if self.debug:
                        print(f"Error applying chat template: {str(e)}. Falling back to standard tokenization.")
                    result.append(self.tokenizer.encode(t, add_special_tokens=False))
            else:
                result.append(self.tokenizer.encode(t, add_special_tokens=False))
        return result

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context."""
        if self.debug:
            logging.info("Estimating loglikelihood for %d pairs." % len(requests))
        
        choices_per_context = 0
        if len(requests) > 1:
            contexts = []
            for req in requests[:10]:
                if hasattr(req, 'args') and len(req.args) >= 2:
                    contexts.append(req.args[0])
                elif hasattr(req, 'kwargs') and 'doc' in req.kwargs:
                    contexts.append(req.kwargs['doc']['context'])
            unique_contexts = len(set(contexts))
            if unique_contexts > 0 and unique_contexts < len(contexts): # check unique_contexts > 0
                choices_per_context = len(contexts) // unique_contexts
                if self.debug:
                    print(f"\\\\nMultiple-choice task detected: {choices_per_context} choices per context")
        
        group_reqs = collections.defaultdict(list)
        for idx, req in enumerate(requests):
            if hasattr(req, 'args') and len(req.args) >= 2:
                context, continuation = req.args[0], req.args[1]
            elif hasattr(req, 'kwargs') and 'doc' in req.kwargs:
                doc = req.kwargs['doc']
                context, continuation = doc['context'], doc['continuation']
            else:
                if self.debug: print(f"Warning: Unknown request format: {req}")
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
            print(f"\\\\nProcessing {len(questions)} unique contexts with {sum(len(r) for r in responses)} total continuations")
        
        scores, is_greedy = [], []
        for i, (q, rs) in enumerate(tqdm(zip(questions, responses), total=len(questions), disable=self.debug)): # Disable tqdm in debug
            if i < 2 and self.debug:
                print(f"\\\\nScoring context: {q[:50]}...")
                print(f"With {len(rs)} continuations:")
                for j, r_text in enumerate(rs):
                    print(f"  {j}: {r_text[:50]}...")
            
            prefix = self._tokenize([q])[0]
            full_sequences = [self._tokenize([q + r_text])[0] for r_text in rs]
            max_completed_l = max(len(s) for s in full_sequences)
            
            if max_completed_l > self._max_tokens:
                truncation = max(0, max_completed_l - self._max_tokens)
                prefix_l = max(len(prefix) - truncation, 0)
                prefix = prefix[len(prefix) - prefix_l:]
                if prefix_l == 0:
                    scores.extend([-float("inf")] * len(rs))
                    is_greedy.extend([False] * len(rs))
                    continue
            
            self.model.reset_state()
            prefix_tensor = torch.tensor([prefix], dtype=torch.int32)
            _ = self.model.prefill(prefix_tensor)
            prefix_log_probs = self.model.compute_logprobs(prefix_tensor[:, -1:])
            max_idx = torch.argmax(prefix_log_probs).item()
            
            continuation_scores = []
            continuation_is_greedy = []
            
            for j, cont_tokens in enumerate([s[len(prefix):] for s in full_sequences]):
                if i < 2 and self.debug:
                    print(f"\\\\nScoring continuation {j}")
                    if len(cont_tokens) > 0:
                        print(f"First token: {cont_tokens[0]} ('{self.tokenizer.decode([cont_tokens[0]])}')")
                
                if len(cont_tokens) > 0:
                    first_token_score = prefix_log_probs[cont_tokens[0]].item()
                    first_token_is_greedy = (cont_tokens[0] == max_idx)
                    
                    if len(cont_tokens) > 1:
                        self.model.reset_state()
                        first_input = torch.tensor([prefix + [cont_tokens[0]]], dtype=torch.int32)
                        _ = self.model.prefill(first_input)
                        rest_scores_list = []
                        rest_is_greedy_list = []
                        
                        for k in range(len(cont_tokens) - 1):
                            current_token = cont_tokens[k]
                            next_token = cont_tokens[k + 1]
                            token_tensor = torch.tensor([[current_token]], dtype=torch.int32)
                            log_probs_cont = self.model.compute_logprobs(token_tensor)
                            token_score = log_probs_cont[next_token].item()
                            token_is_greedy_val = (torch.argmax(log_probs_cont).item() == next_token)
                            rest_scores_list.append(token_score)
                            rest_is_greedy_list.append(token_is_greedy_val)
                            if k < len(cont_tokens) - 2: _ = self.model.predict(token_tensor)
                        total_score = first_token_score + sum(rest_scores_list)
                        is_all_greedy = first_token_is_greedy and all(rest_is_greedy_list)
                        if i < 2 and self.debug:
                            print(f"First token score: {first_token_score:.4f}, rest: {sum(rest_scores_list):.4f}, total: {total_score:.4f}")
                    else:
                        total_score = first_token_score
                        is_all_greedy = first_token_is_greedy
                        if i < 2 and self.debug: print(f"Single token score: {total_score:.4f}")
                else:
                    total_score = 0.0
                    is_all_greedy = True
                    if i < 2 and self.debug: print(f"Empty continuation, score: {total_score}")
                
                continuation_scores.append(total_score)
                continuation_is_greedy.append(is_all_greedy)
            
            if len(continuation_scores) >= 2 and i < 5 and self.debug:
                best_idx = int(np.argmax(continuation_scores))
                print(f"\\\\nBest continuation: {best_idx}")
                for j_score_idx, score_val in enumerate(continuation_scores):
                    print(f"  Score {j_score_idx}: {score_val:.4f}")
            
            scores.extend(continuation_scores)
            is_greedy.extend(continuation_is_greedy)
        
        inv_sort = torch.argsort(torch.tensor(indices))
        scores = [scores[i_val] for i_val in inv_sort]
        is_greedy = [is_greedy[i_val] for i_val in inv_sort]
        
        if self.debug:
            print("\\\\nFinal scores (first 10):")
            for i_final in range(min(10, len(scores))):
                print(f"  {i_final}: {scores[i_final]:.4f}, greedy: {is_greedy[i_final]}")
        
        return list(zip(scores, is_greedy))

    def loglikelihood_rolling(self, requests) -> list[float]:
        """Compute full log-likelihood of strings, for perplexity computation."""
        if self.debug:
            logging.info("Estimating rolling loglikelihood for %d sequences." % len(requests))
        
        context_length = self.metadata.get('context_length', 2048)
        safe_length = context_length - 128
        texts = [self._tokenize([req.args[0]])[0] for req in requests]
        
        if self.debug:
            lengths = [len(t) for t in texts]
            max_len = max(lengths) if lengths else 0
            min_len = min(lengths) if lengths else 0
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            over_ctx = sum(1 for l in lengths if l > safe_length)
            print(f"\\\\nSequence length statistics:")
            print(f"  Min length: {min_len} tokens")
            print(f"  Avg length: {avg_len:.1f} tokens")
            print(f"  Max length: {max_len} tokens")
            print(f"  Safe context length: {safe_length} tokens")
            print(f"  Sequences exceeding safe length: {over_ctx}/{len(texts)} ({over_ctx/len(texts)*100:.1f}%)")
            if lengths: print(f"  First 5 sequence lengths: {lengths[:5]}")
                
        all_scores = []
        for idx, tokens in enumerate(tqdm(texts, disable=self.debug)):
            if len(tokens) <= 1:
                all_scores.append(0.0)
                continue
            original_length = len(tokens)
            if len(tokens) > safe_length:
                tokens = tokens[-safe_length:]
                if self.debug and (idx < 3 or original_length > safe_length*2):
                    print(f"\\\\nSequence {idx}: Truncated from {original_length} to {len(tokens)} tokens")
                    
            inputs, targets = tokens[:-1], tokens[1:]
            self.model.reset_state()
            try:
                input_tensor = torch.tensor([inputs], dtype=torch.int32)
                _ = self.model.prefill(input_tensor)
                sequence_scores = []
                for i, target in enumerate(targets):
                    current_token = inputs[i]
                    token_tensor = torch.tensor([[current_token]], dtype=torch.int32)
                    log_probs = self.model.compute_logprobs(token_tensor)
                    if log_probs is None:
                        if self.debug and i < 5: print(f"  Warning: No log probabilities for token at position {i}")
                        continue
                    try:
                        token_score = log_probs[target].item()
                        sequence_scores.append(token_score)
                    except (IndexError, RuntimeError) as e:
                        if self.debug and i < 5: print(f"  Error scoring token {target}: {str(e)}")
                        continue
                    if i < len(targets) - 1:
                        try:
                            _ = self.model.predict(token_tensor)
                        except Exception as e:
                            if self.debug: print(f"  Error in predict: {str(e)}")
                if sequence_scores:
                    total_score = sum(sequence_scores)
                    if self.debug and idx < 3:
                        print(f"  Scored {len(sequence_scores)}/{len(targets)} tokens, total score: {total_score:.4f}")
                    all_scores.append(total_score)
                else:
                    if self.debug: print(f"  Warning: No valid scores for sequence {idx}")
                    all_scores.append(0.0)
            except Exception as e:
                if self.debug:
                    print(f"Error processing sequence {idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                all_scores.append(0.0)
        
        if self.debug and all_scores:
            print(f"\\\\nCalculated {len(all_scores)} scores")
            print(f"First few scores: {all_scores[:5]}")
            avg_score = sum(all_scores)/len(all_scores) if all_scores else 0
            print(f"Average log likelihood: {avg_score:.4f}")
        
        return all_scores

    def generate_until(self, requests) -> list[str]:
        """Generate text until a stopping sequence is reached."""
        if self.debug:
            logging.info("Generating continuation for %d sequences." % len(requests))
        
        contexts_and_options = []
        for req in requests:
            if hasattr(req, 'args') and len(req.args) >= 2:
                contexts_and_options.append(req.args)
            elif hasattr(req, 'kwargs') and 'until' in req.kwargs:
                contexts_and_options.append((req.args[0], req.kwargs))
            else:
                if self.debug: print(f"Warning: Unknown request format for generate_until: {req}")
                contexts_and_options.append(("", {"until": ["\\\\n\\\\n"]}))
        
        contexts, options = zip(*contexts_and_options)
        completions = []
        
        for i, (context, opt) in enumerate(tqdm(zip(contexts, options), total=len(contexts), disable=self.debug)):
            until = opt.get("until", ["\\\\n\\\\n"])
            if not isinstance(until, list): until = [until]
            max_gen_tokens = min(opt.get("max_gen_tokens", self._max_tokens), self.metadata.get('context_length', 2048) - 10)
            
            if i < 2 and self.debug:
                print(f"\\\\nGenerating for context: {context[:50]}...")
                print(f"Stopping sequences: {until}")
                print(f"Max tokens: {max_gen_tokens}")
                
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
            context_length = self.metadata.get('context_length', 2048)
            if len(context_tokens) >= context_length:
                if self.debug:
                    print(f"Warning: Context length {len(context_tokens)} exceeds max context length {context_length}")
                    context_tokens = context_tokens[-context_length+10:]
                    print(f"Truncated context to {len(context_tokens)} tokens")
            
            input_ids = torch.tensor([context_tokens], dtype=torch.int32)
            self.model.reset_state()
            _ = self.model.prefill(input_ids)
            generated_tokens = []
            
            for _ in range(max_gen_tokens):
                next_token_val = self.model.predict(input_ids[:, -1:] if not generated_tokens else torch.tensor([[generated_tokens[-1]]]))
                generated_tokens.append(next_token_val)
                text_so_far = self.tokenizer.decode(generated_tokens)
                if any(stop in text_so_far for stop in until):
                    stop_idx = min([text_so_far.find(stop) + len(stop) for stop in until if stop in text_so_far])
                    text_so_far = text_so_far[:stop_idx]
                    keep_tokens = self.tokenizer.encode(text_so_far, add_special_tokens=False)
                    if len(keep_tokens) < len(generated_tokens): generated_tokens = generated_tokens[:len(keep_tokens)]
                    break
                if next_token_val == self.tokenizer.eos_token_id: break
            
            completion = self.tokenizer.decode(generated_tokens)
            for stop_seq in until:
                if stop_seq in completion: completion = completion.split(stop_seq)[0]
            if i < 2 and self.debug: print(f"Generated: {completion[:100]}...")
            completions.append(completion)
        
        return completions

    def normalized_loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """
        Compute normalized log-likelihood for multiple-choice tasks like hellaswag.
        This normalizes scores within each context using softmax for fair comparison.
        """
        if self.debug:
            print("\\\\nUsing normalized log-likelihood for multiple-choice evaluation")
            logging.info("Estimating normalized loglikelihood for %d pairs." % len(requests))
        
        group_reqs = collections.defaultdict(list)
        for idx, req in enumerate(requests):
            if hasattr(req, 'args') and len(req.args) >= 2:
                context, continuation = req.args[0], req.args[1]
            elif hasattr(req, 'kwargs') and 'doc' in req.kwargs:
                doc = req.kwargs['doc']
                context, continuation = doc['context'], doc['continuation']
            else:
                if self.debug: print(f"Warning: Unknown request format: {req}")
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
        
        all_scores = []
        all_is_greedy = []
        
        for i, (context, continuations) in enumerate(tqdm(zip(questions, responses), total=len(questions), disable=self.debug)):
            if i < 1 and self.debug:
                print(f"Context {i+1}: '{context[:50]}...'")
                for j, cont_text in enumerate(continuations):
                    print(f"  Choice {j}: '{cont_text[:50]}...'")
            
            context_scores, context_is_greedy = self._score_context(context, continuations)
            max_score_val = max(context_scores) if context_scores else 0 # Handle empty context_scores
            shifted_scores = [score - max_score_val for score in context_scores]
            exp_scores = [np.exp(score) for score in shifted_scores]
            sum_exp = sum(exp_scores)
            normalized_probs = [exp / sum_exp if sum_exp > 0 else 0 for exp in exp_scores] # Avoid division by zero
            normalized_log_probs = [np.log(prob) if prob > 0 else -float('inf') for prob in normalized_probs] # Avoid log(0)
            
            if i < 1 and self.debug:
                print(f"  Raw scores: {[f'{s:.4f}' for s in context_scores]}")
                print(f"  Normalized scores: {[f'{s:.4f}' for s in normalized_log_probs]}")
                print(f"  Normalized probs: {[f'{p:.6f}' for p in normalized_probs]}")
                best_idx = np.argmax(context_scores) if context_scores else -1
                best_norm_idx = np.argmax(normalized_log_probs) if normalized_log_probs else -1
                print(f"  Best raw choice: {best_idx}")
                print(f"  Best normalized choice: {best_norm_idx}")
            
            all_scores.extend(normalized_log_probs)
            all_is_greedy.extend(context_is_greedy)
        
        inv_sort = torch.argsort(torch.tensor(indices))
        scores = [all_scores[i_val] for i_val in inv_sort]
        is_greedy = [all_is_greedy[i_val] for i_val in inv_sort]
        
        return list(zip(scores, is_greedy))
        
    def _score_context(self, context, continuations):
        """Helper method to score all continuations for a single context."""
        prefix = self._tokenize([context])[0]
        full_sequences = [self._tokenize([context + r])[0] for r in continuations]
        max_completed_l = max(len(s) for s in full_sequences) if full_sequences else 0
        
        if max_completed_l > self._max_tokens:
            truncation = max(0, max_completed_l - self._max_tokens)
            prefix_l = max(len(prefix) - truncation, 0)
            prefix = prefix[len(prefix) - prefix_l:]
            if prefix_l == 0:
                if self.debug: print(f"Warning: Entire prompt was truncated due to length")
                return ([-float("inf")] * len(continuations), [False] * len(continuations))
        
        self.model.reset_state()
        prefix_tensor = torch.tensor([prefix], dtype=torch.int32)
        _ = self.model.prefill(prefix_tensor)
        prefix_log_probs = self.model.compute_logprobs(prefix_tensor[:, -1:])
        max_idx = torch.argmax(prefix_log_probs).item()
        
        scores = []
        is_greedy = []
        
        for i_cont, cont_tokens in enumerate([s[len(prefix):] for s in full_sequences]):
            if self.debug and i_cont == 0: print(f"Scoring continuation tokens (first continuation only)")
                
            if len(cont_tokens) > 0:
                first_token_score = prefix_log_probs[cont_tokens[0]].item()
                first_token_is_greedy = (cont_tokens[0] == max_idx)
                if self.debug and i_cont == 0:
                    first_token_text = self.tokenizer.decode([cont_tokens[0]])
                    print(f"First token: {cont_tokens[0]} ('{first_token_text}') - score: {first_token_score:.4f}")
                
                if len(cont_tokens) > 1:
                    self.model.reset_state()
                    first_input = torch.tensor([prefix + [cont_tokens[0]]], dtype=torch.int32)
                    _ = self.model.prefill(first_input)
                    rest_scores_list = []
                    rest_is_greedy_list = []
                    for j_token_idx in range(len(cont_tokens) - 1):
                        current_token = cont_tokens[j_token_idx]
                        next_token = cont_tokens[j_token_idx + 1]
                        token_tensor = torch.tensor([[current_token]], dtype=torch.int32)
                        log_probs_cont = self.model.compute_logprobs(token_tensor)
                        token_score = log_probs_cont[next_token].item()
                        token_is_greedy_val = (torch.argmax(log_probs_cont).item() == next_token)
                        rest_scores_list.append(token_score)
                        rest_is_greedy_list.append(token_is_greedy_val)
                        if j_token_idx < len(cont_tokens) - 2: _ = self.model.predict(token_tensor)
                    total_score = first_token_score + sum(rest_scores_list)
                    is_all_greedy = first_token_is_greedy and all(rest_is_greedy_list)
                    if self.debug and i_cont == 0: print(f"Rest score: {sum(rest_scores_list):.4f}, total score: {total_score:.4f}")
                else:
                    total_score = first_token_score
                    is_all_greedy = first_token_is_greedy
                    if self.debug and i_cont == 0: print(f"Single token score: {total_score:.4f}")
            else:
                total_score = 0.0
                is_all_greedy = True
                if self.debug and i_cont == 0: print(f"Empty continuation, score: {total_score}")
                
            scores.append(total_score)
            is_greedy.append(is_all_greedy)
            
        return scores, is_greedy


def format_duration_from_seconds(seconds: float) -> str:
    """Formats a duration in seconds into a HHh:MMm:SSs string."""
    s = int(round(seconds))
    hours = s // 3600
    minutes = (s % 3600) // 60
    secs = s % 60
    return f"{hours:02d}h:{minutes:02d}m:{secs:02d}s"

def main():
    script_start_time = time.monotonic()
    current_date_str = time.strftime("%Y%m%d")
    current_time_str = time.strftime("%I:%M:%p") # AM/PM format

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
    parser.add_argument("--cmd-line-str", type=str, default="",
                        help="Command line string from the calling script (e.g., run_eval.sh)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.download_timeout)
    os.environ["HF_DATASETS_TIMEOUT"] = str(args.download_timeout) 
    os.environ["HF_HUB_DOWNLOAD_RETRY_COUNT"] = str(args.max_retries)
    
    try:
        import datasets
        # datasets.config.HF_DATASETS_TIMEOUT = args.download_timeout # May cause issues if set directly
        # datasets.config.MAX_RETRIES = args.max_retries
        if args.debug:
            print(f"Configured dataset download timeout (attempted via env): {args.download_timeout}s")
            print(f"Configured maximum download retries (attempted via env): {args.max_retries}")
    except (ImportError, AttributeError) as e:
        if args.debug:
            print(f"Warning: Could not directly configure datasets library settings: {e}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.debug:
        print(f"Initializing ANE model from {args.model} for harness")
    lm = ANELM(
        args.model,
        max_tokens=args.max_tokens,
        use_chat_template=args.apply_chat_template,
        batch_size=args.batch_size,
        debug=args.debug
    )
    
    if args.debug:
        print("Running evaluation in single-threaded mode")
    
    try:
        import inspect
        simple_eval_params = inspect.signature(lm_eval.simple_evaluate).parameters
        supports_processes = 'processes' in simple_eval_params
    except:
        supports_processes = False
    
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
    
    if hasattr(lm.model, 'metadata') and 'context_length' in lm.model.metadata:
        safe_length = lm.model.metadata['context_length'] - args.safety_margin
        if args.debug:
            print(f"Applied safety margin of {args.safety_margin} tokens")
            print(f"Safe context length: {safe_length} tokens")
    
    if args.perplexity and "wikitext" not in args.tasks:
        if args.debug: print("Adding wikitext to tasks for perplexity evaluation")
        args.tasks.append("wikitext")
        eval_args["tasks"] = args.tasks
    
    if args.skip is not None and args.skip > 0:
        class SkipWrapper:
            def __init__(self, lm, skip_count):
                self.lm = lm
                self.skip_count = skip_count
                self.skipped = 0
                if args.debug: print(f"Skipping the first {skip_count} examples")
                
            def __getattr__(self, name):
                if name in ['loglikelihood', 'loglikelihood_rolling', 'generate_until', 'normalized_loglikelihood']:
                    orig_method = getattr(self.lm, name)
                    
                    def wrapper(requests, *w_args, **w_kwargs): # Renamed to avoid conflict
                        if self.skipped < self.skip_count:
                            if len(requests) + self.skipped > self.skip_count:
                                to_skip = self.skip_count - self.skipped
                                if args.debug: print(f"Skipping examples {self.skipped} to {self.skip_count-1}")
                                self.skipped = self.skip_count
                                requests = requests[to_skip:]
                            else:
                                if args.debug: print(f"Skipping examples {self.skipped} to {self.skipped + len(requests) - 1}")
                                self.skipped += len(requests)
                                if name == 'loglikelihood' or name == 'normalized_loglikelihood': return [(0.0, False)] * len(requests)
                                elif name == 'loglikelihood_rolling': return [0.0] * len(requests)
                                else: return [''] * len(requests)
                        
                        if len(requests) == 0:
                            if name == 'loglikelihood' or name == 'normalized_loglikelihood': return []
                            elif name == 'loglikelihood_rolling': return []
                            else: return []
                                
                        return orig_method(requests, *w_args, **w_kwargs)
                    return wrapper
                return getattr(self.lm, name)
        
        lm = SkipWrapper(lm, args.skip)
        eval_args["model"] = lm
    
    if supports_processes:
        if args.debug: print("Using processes=0 parameter for single-threaded execution")
        eval_args["processes"] = 0
    else:
        if args.debug:
            print("This version of lm-evaluation-harness doesn't support the processes parameter")
            print("Continuing with batch_size=1 for controlled execution")
    
    eval_results_obj = lm_eval.simple_evaluate(**eval_args)
    
    script_end_time = time.monotonic()
    script_duration_seconds = script_end_time - script_start_time
    script_duration_formatted = format_duration_from_seconds(script_duration_seconds)

    if not args.tasks or len(args.tasks) == 0:
        print("Error: No tasks specified for metadata injection.")
        task_name_for_metadata = "unknown_task" 
    else:
        task_name_for_metadata = args.tasks[0]
    
    samples_processed = 0
    num_samples_in_dataset = 0
    try:
        task_dict_for_metadata = get_task_dict([task_name_for_metadata], model_name=args.model) # Pass model_name for specific task versions
        if task_name_for_metadata in task_dict_for_metadata:
            task_obj_for_metadata = task_dict_for_metadata[task_name_for_metadata]
            docs_iterable = []
            if task_obj_for_metadata.has_test_docs():
                docs_iterable = task_obj_for_metadata.test_docs()
            elif task_obj_for_metadata.has_validation_docs():
                docs_iterable = task_obj_for_metadata.validation_docs()
            all_task_docs = list(docs_iterable) if docs_iterable else []
            num_samples_in_dataset = len(all_task_docs)
            if args.limit is not None:
                samples_processed = min(args.limit, num_samples_in_dataset)
            else:
                samples_processed = num_samples_in_dataset
        else:
            if args.debug: print(f"Warning: Task '{task_name_for_metadata}' not found in task_dict for dataset size determination.")

    except Exception as e:
        if args.debug:
            print(f"Warning: Could not accurately determine dataset size for task '{task_name_for_metadata}': {e}")
        num_samples_in_dataset = 0 
        samples_processed = args.limit if args.limit is not None else 0

    iter_per_sec = (samples_processed / script_duration_seconds) if script_duration_seconds > 0 and samples_processed > 0 else 0.0

    metadata_to_add = {
        "samples_processed": samples_processed,
        "samples_indataset": num_samples_in_dataset,
        "date": current_date_str,
        "time": current_time_str,
        "duration": script_duration_formatted,
        "IterPerSec": round(iter_per_sec, 2),
        "cmdline": args.cmd_line_str
    }

    processed_results_for_json = {}
    if "results" in eval_results_obj:
        for task_name, task_metrics in eval_results_obj["results"].items():
            updated_task_metrics = task_metrics.copy()
            if task_name == task_name_for_metadata:
                updated_task_metrics.update(metadata_to_add)
            processed_results_for_json[task_name] = updated_task_metrics
    else:
        if args.debug: print("Warning: 'results' key not found in evaluation output object.")
        processed_results_for_json = {task_name_for_metadata: metadata_to_add} # Create a basic structure

    model_name_for_file = Path(args.model).name
    if args.output_path:
        output_file_path = Path(args.output_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_path.write_text(json.dumps(processed_results_for_json, indent=4))
        print(f"\\\\nDetailed results including metadata saved to: {output_file_path}")
    else:
        if args.debug: print("\\\\nWarning: --output-path not specified by calling script. Saving results to output_dir.")
        for task, metrics_dict_with_metadata in processed_results_for_json.items():
            task_filename = f"{task}_{model_name_for_file}_{current_date_str}.json" 
            task_specific_output_path = output_dir / task_filename
            task_specific_output_path.write_text(json.dumps({task: metrics_dict_with_metadata}, indent=4))
            if args.debug: print(f"Saved {task} results to: {task_specific_output_path}")
        combined_filename = f"results_{model_name_for_file}_{current_date_str}.json"
        combined_output_path = output_dir / combined_filename
        combined_output_path.write_text(json.dumps(processed_results_for_json, indent=4))
        if args.debug: print(f"Saved combined results (with metadata for primary task) to: {combined_output_path}")

    print("\\\\n===============================")
    print("Evaluation Summary (Original Metrics)")
    print("===============================")
    print(f"Model: {args.model}")
    if "results" in eval_results_obj:
        for task_name, original_metrics in eval_results_obj["results"].items():
            print(f"\\\\n{task_name}:")
            for metric, value in original_metrics.items():
                print(f"  {metric}: {value}")
    
    print(f"\\\\nScript duration: {script_duration_formatted}")

if __name__ == "__main__":
    main() 