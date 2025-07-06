#!/usr/bin/env python3
"""
PyTorch Qwen2.5 evaluation using full lm-evaluation-harness pipeline like MLX
Keeps optimized single prefill scoring logic for BoolQ pairs
"""

import argparse
import os
import sys
import json
from pathlib import Path
from importlib.metadata import version

# Set offline mode to prevent network calls
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# Performance optimizations
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Add paths for imports
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import collections
import time

# PyTorch performance optimizations
torch.set_num_threads(4)
if torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True

# Import your custom Qwen2.5 model
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def make_causal_mask(length, start): # start = tokens -1 (last token)
    """Create causal attention mask."""
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return mask

@register_model("pytorch_qwen25")
class PyTorchQwen25LM(LM):
    """PyTorch Qwen2.5 LM wrapper using full lm-evaluation-harness pipeline
    Keeps optimized single prefill scoring logic for BoolQ pairs"""
    
    def __init__(self, model_path: str, max_tokens: int = 2048, debug: bool = False, log_incorrect_answers: bool = False, skip: int = 0, **kwargs):
        super().__init__()
        self.model_path = model_path
        self._max_tokens = max_tokens
        self._rank = 0
        self._world_size = 1
        self.debug = debug
        self.log_incorrect_answers = log_incorrect_answers
        self.skip = skip
        
        print(f"Loading PyTorch Qwen2.5 model from: {model_path}")
        
        # Check if it's a local path or HuggingFace ID
        if not os.path.exists(model_path):
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
        
        # Create config and set context length before model creation
        config = Qwen25Config.from_json(f'{model_path}/config.json')
        
        # Set a larger context length to handle long BoolQ passages
        config.context_length = 1024  # Increase to handle longer contexts (up to 728+ tokens seen)
        config.state_length = 1024    # Also set state_length for KV cache size
        
        # Enable KV cache
        self.model = Qwen25ForCausalLM(config, disable_kv_cache=False)
        
        # Load pretrained weights
        print(f"Loading pretrained weights...")
        success = self.model.load_pretrained_weights(model_path)
        if not success:
            raise RuntimeError(f"Failed to load pretrained weights from {model_path}")
        
        self.model.eval()
        
        print("Using CPU (ANEMLL model has MPS compatibility issues)")
        self.device_name = 'cpu'
        
        print(f"PyTorch Qwen2.5 model loaded with KV cache enabled")
        
        # Store context length for proper mask creation
        self.context_length = config.context_length
        

        
        causal_mask_data = make_causal_mask(self.context_length, 0)
        self.causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
        
        print(f"Context length: {self.context_length}")
        
        # Pre-compute token IDs for " no" and " yes"
        self.no_token_id = self.tokenizer.encode(" no", add_special_tokens=False)[0]
        self.yes_token_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
        print(f"Token IDs: ' no'={self.no_token_id}, ' yes'={self.yes_token_id}")
    
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
        return self.device_name
    
    def loglikelihood(self, requests):
        """OPTIMIZED: Process BoolQ pairs together to avoid duplicate prefills"""
        print(f"[PyTorchQwen25LM] Processing {len(requests)} requests")
        
        results = []
        
        # Group by common prefix for ground truth tracking
        group_reqs = collections.defaultdict(list)
        request_metadata = {}
        
        for idx, req in enumerate(requests):
            # Extract request data and try to find ground truth
            context = None
            continuation = None
            doc = None
            
            if hasattr(req, 'arguments') and len(req.arguments) >= 2:
                # New API with req.arguments and req.doc
                context, continuation = req.arguments[0], req.arguments[1]
                doc = req.doc if hasattr(req, 'doc') else None
            elif hasattr(req, 'args') and len(req.args) >= 2:
                # Fallback: Old-style API with req.args
                context, continuation = req.args[0], req.args[1]
                doc = req.doc if hasattr(req, 'doc') else None
            elif hasattr(req, 'kwargs') and 'doc' in req.kwargs:
                # Fallback: kwargs-style API
                doc = req.kwargs['doc']
                context, continuation = doc['context'], doc['continuation']
            else:
                print(f"Warning: Unknown request format: {req}")
                continue
            
            # Store metadata about this request for ground truth detection
            request_metadata[idx] = {
                'doc': doc,
                'req_obj': req,
                'context': context,
                'continuation': continuation
            }
            
            # Store doc alongside for ground truth access
            group_reqs[context].append((idx, continuation, doc))
        
        # Track results for incorrect answer logging
        if self.log_incorrect_answers:
            self._question_results = []
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            progress_bar = tqdm(range(0, len(requests), 2), desc="Scoring pairs", unit="pair")
        except ImportError:
            progress_bar = range(0, len(requests), 2)
        
        question_index = 0
        
        # Process requests in pairs for BoolQ (context + " no", context + " yes")
        for i in progress_bar:
            if i + 1 >= len(requests):
                # Handle odd number of requests
                print(f"Warning: Odd number of requests, processing last one separately")
                req = requests[i]
                context, continuation = req.args
                score = self._score_single_continuation(context, continuation)
                results.append((score, True))
                break
            
            # Get the pair of requests
            req1, req2 = requests[i], requests[i+1]
            context1, cont1 = req1.args
            context2, cont2 = req2.args
            
            # Verify they have the same context (BoolQ should)
            if context1 != context2:
                print(f"Warning: Contexts don't match for pair {i//2}, processing separately")
                score1 = self._score_single_continuation(context1, cont1)
                score2 = self._score_single_continuation(context2, cont2)
                results.extend([(score1, True), (score2, True)])
                continue
            
            # OPTIMIZED: Score both continuations with one prefill
            score1, score2 = self._score_both_continuations(i, context1, cont1, cont2)
            results.extend([(score1, True), (score2, True)])
            
            # Store information for incorrect answer logging
            if self.log_incorrect_answers:
                # Get current scores for this question
                current_question_scores = [score1, score2]
                current_options = [cont1, cont2]
                
                # Find the answer with the highest score (least negative log probability)
                selected_idx = max(range(len(current_question_scores)), key=lambda idx: current_question_scores[idx])
                selected_answer = current_options[selected_idx]
                selected_score = current_question_scores[selected_idx]
                
                # Extract ground truth using the first doc
                first_doc = None
                if hasattr(req1, 'doc'):
                    first_doc = req1.doc
                elif hasattr(req2, 'doc'):
                    first_doc = req2.doc
                
                correct_idx = self._gold_idx(first_doc, current_options)
                
                self._question_results.append({
                    'question_idx': question_index,
                    'context': context1,
                    'options': list(current_options),
                    'selected_idx': selected_idx,
                    'selected_answer': selected_answer,
                    'selected_score': selected_score,
                    'all_scores': current_question_scores.copy(),
                    'correct_idx': correct_idx,
                    'ground_truth_doc': first_doc
                })
                
            question_index += 1
            
            if i == 0:  # Debug first pair
                print(f"  First pair scores: {cont1.strip()}={score1:.4f}, {cont2.strip()}={score2:.4f}")
        
        # Log incorrect answers if requested
        if self.log_incorrect_answers and hasattr(self, '_question_results'):
            self._log_incorrect_answers(results)
        
        print(f"[PyTorchQwen25LM] Completed {len(results)} results")
        return results
    
    def _score_both_continuations(self, i, context, cont1, cont2):
        """OPTIMIZED: Score both continuations with a single prefill"""
        print(f"[PyTorchQwen25LM] _score_both_continuations called with {context}, {cont1}, {cont2}")

        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        no_token_id = self.tokenizer.encode(cont1, add_special_tokens=False)[0]
        yes_token_id = self.tokenizer.encode(cont2, add_special_tokens=False)[0]
        cont1_tokens = self.tokenizer.encode(cont1, add_special_tokens=False)
        cont2_tokens = self.tokenizer.encode(cont2, add_special_tokens=False)

        if len(cont1_tokens) != 1 or len(cont2_tokens) != 1:
            print(f"Warning: Multi-token continuation, simplified scoring")
            return -float('inf')
        
        cont1_token = cont1_tokens[0]
        cont2_token = cont2_tokens[0]
        #print(f"Q{i}")
        #print(f"\nContext: {repr(context)}")
        #print(f"Context tokens length: {len(context_tokens)}")
        #print(f"First 5 context tokens: {context_tokens[:5]}")
        #print(f"Last 5 context tokens: {context_tokens[-5:]}")
        #print(f"Token IDs: ' no'={no_token_id}, ' yes'={yes_token_id}")
   

        context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
        no_token_id = self.tokenizer.encode(" no", add_special_tokens=False)[0]
        yes_token_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]

        
        # Create simple causal mask
        seq_len = len(context_tokens)
        causal_mask = torch.full((1, 1, seq_len, seq_len), float('-inf'))
        for i in range(seq_len):
            causal_mask[0, 0, i, :i+1] = 0

        causal_mask= make_causal_mask(self.context_length, 0)
        causal_mask[:, :, seq_len:seq_len+1, :],
        causal_mask = torch.tensor(causal_mask, dtype=torch.float16)

        
        # Run inference
        input_ids = torch.tensor(context_tokens[-1:], dtype=torch.long).unsqueeze(0)  # Shape: [1, last token is the continuation token
        position_ids = torch.arange(1,seq_len, dtype=torch.long)
        current_pos = torch.tensor(seq_len, dtype=torch.long)
        update_mask = torch.zeros((1, seq_len), dtype=torch.float16)  # Dummy update mask

        #print(f"input_ids: {input_ids}")
        #print(f"current_pos: {current_pos}")
        #print(f"current_pos: {position_ids}")


        with torch.no_grad():
            print(f"[PyTorchQwen25LM] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            outputs = self.model(input_ids, update_mask, position_ids, causal_mask, current_pos, IN_PREFILL=False)
            logits = outputs[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Get scores for both tokens
            score1 = log_probs[cont1_token].item()
            score2 = log_probs[cont2_token].item()

            return score1, score2



        # Clear KV cache
        if hasattr(self.model.model, 'clear_kv_cache'):
            self.model.model.clear_kv_cache()
        
        # Tokenize
        context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
        cont1_tokens = self.tokenizer.encode(cont1, add_special_tokens=False)
        cont2_tokens = self.tokenizer.encode(cont2, add_special_tokens=False)
        
        # For BoolQ, continuations should be single tokens
        if len(cont1_tokens) != 1 or len(cont2_tokens) != 1:
            print(f"Warning: Multi-token continuations, falling back to separate scoring")
            return self._score_single_continuation(context, cont1), self._score_single_continuation(context, cont2)
        
        cont1_token = cont1_tokens[0]
        cont2_token = cont2_tokens[0]
        
        # Debug output
        if self.debug:
            print(f"\n[DEBUG] Context: {repr(context)}")
            print(f"[DEBUG] Continuations: {repr(cont1)} | {repr(cont2)}")
            print(f"[DEBUG] Context tokens length: {len(context_tokens)}")
            print(f"[DEBUG] First 5 context tokens: {context_tokens[:5]}")
            print(f"[DEBUG] Last 5 context tokens: {context_tokens[-5:]}")
            print(f"[DEBUG] Continuation token IDs: {cont1_token} | {cont2_token}")
        
        # Handle context overflow
        if len(context_tokens) > self.context_length - 1:
            context_tokens = context_tokens[-(self.context_length - 1):]
        
        prompt_length = len(context_tokens)
        
        with torch.no_grad():
            device = next(self.model.parameters()).device
            
            # Batch prefill
            input_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)
            
            # Run batched prefill
            batch_pos = 0
            batch_size = 64
            while batch_pos < prompt_length:
                batch_end = min(batch_pos + batch_size, prompt_length)
                current_batch_size = batch_end - batch_pos
                
                # Get current batch
                batch_input = input_ids[:, batch_pos:batch_end]
                
                # Pad to batch size
                import torch.nn.functional as F
                batch_input = F.pad(batch_input, (0, batch_size - current_batch_size), value=0)
                
                # Position IDs and masks
                position_ids = torch.arange(batch_pos, batch_pos + batch_size, dtype=torch.long, device=device)
                batch_causal_mask = self.causal_mask[:, :, batch_pos:batch_pos + batch_size, :]
                update_mask = torch.zeros(1, batch_size, device=device)
                
                # Prefill
                self.model(
                    batch_input,
                    update_mask,
                    position_ids,
                    batch_causal_mask,
                    torch.tensor(batch_pos, dtype=torch.long, device=device),
                    IN_PREFILL=True
                )
                
                batch_pos = batch_end
            
            # Single generation step to get logits
            current_pos = prompt_length
            last_token = torch.tensor([[context_tokens[-1]]], dtype=torch.long, device=device)
            
            update_mask = torch.zeros((1, 1, self.context_length, 1), dtype=torch.float16, device=device)
            update_mask[0, 0, current_pos, 0] = 1.0
            
            outputs = self.model(
                last_token,
                update_mask,
                torch.tensor([current_pos], dtype=torch.long, device=device),
                self.causal_mask[:, :, current_pos:current_pos+1, :],
                torch.tensor(current_pos, dtype=torch.long, device=device),
                IN_PREFILL=False
            )
            
            # Extract logits and compute log probabilities
            logits = outputs[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Get scores for both tokens
            score1 = log_probs[cont1_token].item()
            score2 = log_probs[cont2_token].item()
            
            return score1, score2
    
    def _score_single_continuation(self, context, continuation):
        """Fallback for scoring a single continuation"""
        print(f"[PyTorchQwen25LM] _score_single_continuation called with {context}, {continuation}")    

        # Clear KV cache
        if hasattr(self.model.model, 'clear_kv_cache'):
            self.model.model.clear_kv_cache()
        
        # Tokenize
        context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
        cont_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)
        
        if len(cont_tokens) != 1:
            print(f"Warning: Multi-token continuation, simplified scoring")
            return -float('inf')
        
        cont_token = cont_tokens[0]
        
        # Handle context overflow
        if len(context_tokens) > self.context_length - 1:
            context_tokens = context_tokens[-(self.context_length - 1):]
        
        prompt_length = len(context_tokens)


        with torch.no_grad():
            device = next(self.model.parameters()).device
            input_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)
            
            # Batch prefill
            batch_pos = 0
            batch_size = 64
            while batch_pos < prompt_length:
                batch_end = min(batch_pos + batch_size, prompt_length)
                current_batch_size = batch_end - batch_pos
                
                batch_input = input_ids[:, batch_pos:batch_end]
                
                import torch.nn.functional as F
                batch_input = F.pad(batch_input, (0, batch_size - current_batch_size), value=0)
                
                position_ids = torch.arange(batch_pos, batch_pos + batch_size, dtype=torch.long, device=device)
                batch_causal_mask = self.causal_mask[:, :, batch_pos:batch_pos + batch_size, :]
                update_mask = torch.zeros(1, batch_size, device=device)
                
                self.model(
                    batch_input,
                    update_mask,
                    position_ids,
                    batch_causal_mask,
                    torch.tensor(batch_pos, dtype=torch.long, device=device),
                    IN_PREFILL=True
                )
                
                batch_pos = batch_end
            
            # Single generation step
            current_pos = prompt_length
            last_token = torch.tensor([[context_tokens[-1]]], dtype=torch.long, device=device)
            
            update_mask = torch.zeros((1, 1, self.context_length, 1), dtype=torch.float16, device=device)
            update_mask[0, 0, current_pos, 0] = 1.0
            
            outputs = self.model(
                last_token,
                update_mask,
                torch.tensor([current_pos], dtype=torch.long, device=device),
                self.causal_mask[:, :, current_pos:current_pos+1, :],
                torch.tensor(current_pos, dtype=torch.long, device=device),
                IN_PREFILL=False
            )
            
            logits = outputs[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            score = log_probs[cont_token].item()
            
            return score
    
    def loglikelihood_rolling(self, requests):
        """Compute rolling log-likelihood"""
        print(f"[PyTorchQwen25LM] loglikelihood_rolling called with {len(requests)} requests")
        return []
    
    def generate_until(self, requests):
        """Generate text until stopping condition"""
        print(f"[PyTorchQwen25LM] generate_until called with {len(requests)} requests")
        return [""] * len(requests)
    
    def _gold_idx(self, doc, opts):
        """Extract the correct answer index from the document."""
        if doc is None:
            return None
        # Unwrap nested 'doc' field if present (newer harness may nest the original doc)
        if isinstance(doc.get('doc', None), dict):
            return self._gold_idx(doc['doc'], opts)
        
        # BoolQ: answer field contains True/False -> 0/1
        if "answer" in doc:
            return 0 if doc["answer"] else 1
        
        # ARC easy/challenge: answerKey contains "A"-"D" -> 0-3
        if "answerKey" in doc:
            answer_key = doc["answerKey"]
            if isinstance(answer_key, str) and len(answer_key) == 1:
                return "ABCD".index(answer_key.upper())
        
        # HellaSwag and others: label contains integer index
        if "label" in doc:
            return int(doc["label"])
        
        # Other common patterns
        if "gold" in doc:
            return int(doc["gold"])

        # New-style API uses 'target' for ground truth
        if "target" in doc:
            val = doc["target"]
            try:
                return int(val)
            except (TypeError, ValueError):
                if isinstance(val, str):
                    low = val.lower()
                    if low in ("true", "yes"):
                        return 1
                    if low in ("false", "no"):
                        return 0
            return None

        return None
    
    def _log_incorrect_answers(self, loglikelihood_results):
        """Log detailed information about incorrect answers using proper ground truth."""
        print(f"\n[INCORRECT ANSWER ANALYSIS] Analyzing {len(self._question_results)} questions...")
        
        incorrect_count = 0
        total_questions = len(self._question_results)
        
        # Open log file for writing
        import os
        log_file = os.path.join(os.getcwd(), "incorrect_answers_pytorch.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== INCORRECT ANSWER LOG (PyTorch) ===\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for q_info in self._question_results:
                question_idx = q_info['question_idx']
                context = q_info['context']
                options = q_info['options']
                selected_idx = q_info['selected_idx']
                selected_answer = q_info['selected_answer']
                selected_score = q_info['selected_score']
                all_scores = q_info['all_scores']
                correct_idx = q_info.get('correct_idx')
                ground_truth_doc = q_info.get('ground_truth_doc')
                
                # Determine if the answer was incorrect
                # For PyTorch, we need to map from question index to result index
                # Each question generates 2 results (for the 2 options), so result index = question_idx * 2
                result_idx = question_idx * 2
                if result_idx < len(loglikelihood_results):
                    is_greedy_match = loglikelihood_results[result_idx][1]  # Use first option's greedy flag as reference
                else:
                    is_greedy_match = False
                
                if correct_idx is None:
                    is_incorrect = not is_greedy_match
                else:
                    is_incorrect = (selected_idx != correct_idx)
                
                if is_incorrect:
                    incorrect_count += 1
                    # Prepare correct-answer info if available
                    if correct_idx is not None and 0 <= correct_idx < len(options):
                        correct_answer = options[correct_idx]
                        correct_score = all_scores[correct_idx]
                        score_diff = selected_score - correct_score
                    else:
                        # Fallback: display raw ground-truth fields if available
                        if ground_truth_doc is not None:
                            if 'answer' in ground_truth_doc:
                                correct_answer = ground_truth_doc['answer']
                            elif 'answerKey' in ground_truth_doc:
                                correct_answer = ground_truth_doc['answerKey']
                            elif 'label' in ground_truth_doc:
                                correct_answer = ground_truth_doc['label']
                            elif 'gold' in ground_truth_doc:
                                correct_answer = ground_truth_doc['gold']
                            elif 'target' in ground_truth_doc:
                                correct_answer = ground_truth_doc['target']
                            else:
                                correct_answer = '<unknown>'
                        else:
                            correct_answer = '<unknown>'
                        correct_score = None
                        score_diff = None

                    # Calculate absolute question number
                    absolute_question_num = self.skip + question_idx + 1
                    
                    print(f"\n[INCORRECT] Question {absolute_question_num}:")
                    print(f"  Context: {repr(context[:200])}{'...' if len(context) > 200 else ''}")
                    print(f"  Options: {options}")
                    print(f"  Selected: '{selected_answer}' (index {selected_idx}, score: {selected_score:.4f})")
                    if correct_score is not None:
                        print(f"  Correct: '{correct_answer}' (index {correct_idx}, score: {correct_score:.4f})")
                        print(f"  Score difference: {score_diff:.4f}")
                    else:
                        print(f"  Correct: {correct_answer}")

                    # Write to log file
                    f.write(f"QUESTION {absolute_question_num} (INCORRECT):\n")
                    f.write(f"Context: {context}\n")
                    f.write(f"Options: {options}\n")
                    f.write(f"Selected Answer: '{selected_answer}' (index {selected_idx})\n")
                    if correct_score is not None:
                        f.write(f"Correct Answer: '{correct_answer}' (index {correct_idx})\n")
                        f.write(f"Selected Score: {selected_score:.4f}\n")
                        f.write(f"Correct Score: {correct_score:.4f}\n")
                        f.write(f"Score Difference: {score_diff:.4f}\n")
                    else:
                        f.write(f"Correct Answer: {correct_answer}\n")
                    f.write(f"All Scores: {[f'{s:.4f}' for s in all_scores]}\n")

                    # Log ground truth source information if available
                    if ground_truth_doc:
                        gt_fields = []
                        for field in ['answer', 'answerKey', 'label', 'gold', 'target']:
                            if field in ground_truth_doc:
                                gt_fields.append(f"{field}={ground_truth_doc[field]}")
                        f.write(f"Ground Truth: {', '.join(gt_fields) if gt_fields else 'Unknown'}\n")
                    f.write("=" * 50 + "\n\n")
        
        print(f"\n[SUMMARY] Found {incorrect_count} incorrect answers out of {total_questions} questions")
        if total_questions > 0:
            print(f"Accuracy: {((total_questions - incorrect_count) / total_questions * 100):.1f}%")
        print(f"Detailed log saved to: {log_file}")

def main():
    parser = argparse.ArgumentParser(
        "Evaluate PyTorch Qwen2.5 model using full lm-evaluation-harness pipeline."
    )
    parser.add_argument("--model", help="Model to evaluate", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", default=["boolq"])
    parser.add_argument(
        "--output-dir", default=".", help="Output directory for result files."
    )
    parser.add_argument(
        "--output-path", default=None, help="Specific output file path (overrides auto-generated path)."
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-shots", type=int, default=None, help="Number of shots")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens to generate. Defaults to the model's max context length.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="Limit the number of examples per task.",
        type=int,
    )
    parser.add_argument(
        "--skip",
        default=0,
        help="Skip the first N examples per task.",
        type=int,
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--fewshot-as-multiturn",
        action="store_true",
        help="Whether to provide the fewshot examples as a multiturn "
        "conversation or a single user turn.",
        default=False,
    )
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        help="Specifies whether to apply a chat template to the prompt. "
        "For base models, this defaults to False.",
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output with detailed prompt information",
        default=False,
    )
    parser.add_argument(
        "--log-incorrect-answers",
        action="store_true",
        help="Log detailed information about incorrect answers to incorrect_answers_pytorch.log",
        default=False,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    #args.debug = True
    #args.skip = 30
    #args.limit = 1
    #args.skip = 55 # DEBUG!


    # Silence tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # Load PyTorch model
    lm = PyTorchQwen25LM(
        model_path=args.model,
        max_tokens=args.max_tokens or 2048,
        debug=args.debug,
        log_incorrect_answers=args.log_incorrect_answers,
        skip=args.skip
    )

    # Handle skip parameter by creating custom samples dict
    samples = None
    if args.skip > 0:
        # Create samples dict that skips the first N examples
        samples = {}
        for task in args.tasks:
            start_idx = args.skip
            end_idx = start_idx + args.limit if args.limit else start_idx + 100
            samples[task] = list(range(start_idx, end_idx))
        # When using samples, don't use limit
        eval_limit = None
    else:
        eval_limit = args.limit

    # For base models, default to no chat template
    use_chat_template = args.apply_chat_template
    if use_chat_template is None:
        use_chat_template = False

    print(f"Using chat template: {use_chat_template}")
    print(f"Skip: {args.skip}, Limit: {args.limit}")
    if samples:
        print(f"Samples dict: {samples}")


    # Run full lm-evaluation-harness pipeline
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        apply_chat_template=use_chat_template,
        num_fewshot=args.num_shots,
        limit=eval_limit,
        samples=samples,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    # Save results
    file_keys = ["eval", args.model.replace("/", "_"), version("lm_eval")]
    if args.num_shots is not None:
        file_keys += [f"{args.num_shots:02d}"]
    file_keys += args.tasks
    filename = "_".join(file_keys)
    
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = output_dir / filename
    
    output_path.write_text(json.dumps(results["results"], indent=4))
    print("Results:")
    for result in results["results"].values():
        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()