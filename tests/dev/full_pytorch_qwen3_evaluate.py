#!/usr/bin/env python3
"""
PyTorch Qwen3 evaluation using full lm-evaluation-harness pipeline
Supports both single-token answers (BoolQ) and multi-token answers (arc_challenge)
Based on full_pytorch_qwen25_evaluate.py but adapted for Qwen3 model
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

# Import your custom Qwen3 model
from anemll.models import qwen_model
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig

# Override the global device setting
qwen_model.TEST_DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

def make_causal_mask(length, start): # start = tokens -1 (last token)
    """Create causal attention mask."""
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return mask

@register_model("pytorch_qwen3")
class PyTorchQwen3LM(LM):
    """PyTorch Qwen3 LM wrapper using full lm-evaluation-harness pipeline
    Supports both BoolQ (single-token) and arc_challenge (multi-token) patterns"""
    
    def __init__(self, model_path: str, max_tokens: int = 2048, debug: bool = False, log_incorrect_answers: bool = False, skip: int = 0, **kwargs):
        super().__init__()
        self.model_path = model_path
        self._max_tokens = max_tokens
        self._rank = 0
        self._world_size = 1
        self.debug = debug
        self.log_incorrect_answers = log_incorrect_answers
        self.skip = skip
        
        print(f"Loading PyTorch Qwen3 model from: {model_path}")
        
        # Check if it's a local path or HuggingFace ID
        if os.path.exists(model_path):
            print(f"Using local model path: {model_path}")
        else:
            # It's a HuggingFace model ID
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
        config = QwenConfig.from_json(f'{model_path}/config.json')
        
        # Set a larger context length to handle long sequences like arc_challenge
        config.context_length = 2048  # Increase to handle longer contexts (arc_challenge can be ~1500+ tokens)
        config.state_length = 2048    # Also set state_length for KV cache size
        
        # Enable KV cache
        self.model = QwenForCausalLM(config, disable_kv_cache=False)
        
        # Load pretrained weights
        print(f"Loading pretrained weights...")
        success = self.model.load_pretrained_weights(model_path)
        if not success:
            raise RuntimeError(f"Failed to load pretrained weights from {model_path}")
        
        self.model.eval()
        
        # Try to use MPS if available
        if torch.backends.mps.is_available():
            print("Using MPS (Metal Performance Shaders)")
            self.device_name = 'mps'
            self.model = self.model.to('mps')
        else:
            print("Using CPU (MPS not available)")
            self.device_name = 'cpu'
        
        print(f"PyTorch Qwen3 model loaded with KV cache enabled")
        
        # Store context length for proper mask creation
        self.context_length = config.context_length
        
        causal_mask_data = make_causal_mask(self.context_length, 0)
        self.causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
        if self.device_name == 'mps':
            self.causal_mask = self.causal_mask.to('mps')
        
        print(f"Context length: {self.context_length}")
        
        # Pre-compute token IDs for " no" and " yes" (useful for BoolQ)
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
        """Process both single-token (BoolQ) and multi-token (arc_challenge) requests"""
        print(f"[PyTorchQwen3LM] Processing {len(requests)} requests")
        
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
            progress_bar = tqdm(group_reqs.items(), desc="Scoring questions", unit="question")
        except ImportError:
            progress_bar = group_reqs.items()
        
        question_index = 0
        
        # Process each question (context) and its associated answer choices
        for context, choices_list in progress_bar:
            # Sort choices by request index to maintain consistent order
            choices_list = sorted(choices_list, key=lambda x: x[0])
            choices = [choice[1] for choice in choices_list]  # Extract continuation texts
            choice_docs = [choice[2] for choice in choices_list]  # Extract docs
            
            if self.debug:
                print(f"\n[DEBUG] Question {question_index}: {len(choices)} choices")
                print(f"[DEBUG] Context: {repr(context[:100])}...")
                print(f"[DEBUG] Choices: {choices}")
            
            # Score all choices for this question
            question_scores = []
            for choice_idx, choice in enumerate(choices):
                score = self._score_one_choice(context, choice)
                question_scores.append(score)
                
                if self.debug:
                    print(f"[DEBUG] Choice {choice_idx+1}: '{choice}' -> {score:.4f}")
            
            # Add scores to results in the original request order
            for choice_info, score in zip(choices_list, question_scores):
                results.append((score, True))  # True for is_greedy (simplified)
            
            # Store information for incorrect answer logging
            if self.log_incorrect_answers:
                # Find the answer with the highest score (least negative log probability)
                selected_idx = max(range(len(question_scores)), key=lambda idx: question_scores[idx])
                selected_answer = choices[selected_idx]
                selected_score = question_scores[selected_idx]
                
                # Extract ground truth using the first doc
                first_doc = choice_docs[0] if choice_docs else None
                correct_idx = self._gold_idx(first_doc, choices)
                
                self._question_results.append({
                    'question_idx': question_index,
                    'context': context,
                    'options': list(choices),
                    'selected_idx': selected_idx,
                    'selected_answer': selected_answer,
                    'selected_score': selected_score,
                    'all_scores': question_scores.copy(),
                    'correct_idx': correct_idx,
                    'ground_truth_doc': first_doc
                })
                
            question_index += 1
            
            if question_index == 1:  # Debug first question
                print(f"  First question scores: {[f'{s:.4f}' for s in question_scores]}")
        
        # Log incorrect answers if requested
        if self.log_incorrect_answers and hasattr(self, '_question_results'):
            self._log_incorrect_answers(results)
        
        print(f"[PyTorchQwen3LM] Completed {len(results)} results")
        return results
    
    def _score_one_choice(self, context, continuation):
        """Score a single answer choice using teacher forcing.
        
        Supports both single-token (BoolQ) and multi-token (arc_challenge) answers.
        
        Args:
            context: String context/prompt
            continuation: String answer to score
            
        Returns:
            total_logprob: Sum of log probabilities for all answer tokens
        """
        if self.debug:
            print(f"\n[DEBUG] _score_one_choice: '{continuation}' ")
            
        # Clear KV cache
        if self.debug:
            print(f"  Clearing KV cache...")
        # Reset KV cache by zeroing it out
        if hasattr(self.model.model, 'kv_cache_0'):
            self.model.model.kv_cache_0.zero_()
            if self.debug:
                print(f"  KV cache cleared")
        
        # Tokenize
        context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
        cont_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)
        
        if len(cont_tokens) == 0:
            print(f"Warning: Empty continuation tokens")
            return -float('inf')
        
        # Check if total sequence would exceed context length
        total_length = len(context_tokens) + len(cont_tokens)
        if total_length > self.context_length:
            if self.debug:
                print(f"  Skipping: total length {total_length} exceeds context length {self.context_length}")
            return -float("inf")
        
        # Truncate context if needed (shouldn't happen due to check above, but for safety)
        if len(context_tokens) > self.context_length - len(cont_tokens):
            context_tokens = context_tokens[-(self.context_length - len(cont_tokens)):]
        
        if self.debug:
            print(f"  Context tokens: {len(context_tokens)}, Answer tokens: {len(cont_tokens)}")
            print(f"  Context: {self.tokenizer.decode(context_tokens[:10])}...")
            print(f"  Answer: '{self.tokenizer.decode(cont_tokens)}'")
        
        # Score using teacher forcing approach with prefill
        with torch.no_grad():
            device = 'mps' if self.device_name == 'mps' else 'cpu'
            if self.debug:
                print(f"  Device: {device}")
                
            # Step 1: Prefill with context tokens
            if self.debug:
                print(f"  Starting prefill with {len(context_tokens)} context tokens")
                import time
                start_time = time.time()
                
            # Prepare context for prefill
            context_input = torch.tensor([context_tokens], dtype=torch.long, device=device)
            context_len = len(context_tokens)
            
            # Create position IDs for prefill
            position_ids = torch.arange(context_len, dtype=torch.long, device=device)
            
            # Create causal mask for prefill - use full context length mask
            prefill_mask = self.causal_mask.to(device)
            
            # Create update mask for prefill
            update_mask = torch.zeros((1, context_len), dtype=torch.float16, device=device)
            
            # Call model in prefill mode
            if self.debug:
                print(f"  Calling model.prefill_kv_cache...")
            self.model.prefill_kv_cache(
                input_ids=context_input,
                position_ids=position_ids,
                start_pos=0,
                causal_mask=prefill_mask
            )
            
            if self.debug:
                elapsed = time.time() - start_time
                print(f"  Prefill completed in {elapsed:.2f}s")
            
            # Step 2: Teacher forcing through answer tokens
            total_logprob = 0.0
            prev_token = context_tokens[-1]  # Start with last context token
            
            for i, target_token in enumerate(cont_tokens):
                if self.debug:
                    print(f"  Processing answer token {i+1}/{len(cont_tokens)}")
                    step_start = time.time()
                    
                # Current position in the sequence
                current_pos = context_len + i
                
                # Input is the previous token
                input_ids = torch.tensor([[prev_token]], dtype=torch.long, device=device)
                
                # Position ID for this single token
                position_ids = torch.tensor([current_pos], dtype=torch.long, device=device)
                
                # Update mask for single token generation
                update_mask = torch.zeros((1, 1, self.context_length, 1), dtype=torch.float16, device=device)
                update_mask[0, 0, current_pos, 0] = 1.0
                
                # Causal mask for this position
                causal_mask = self.causal_mask[:, :, current_pos:current_pos+1, :].to(device)
                
                # Generate next token prediction
                if self.debug:
                    print(f"    Calling model forward (single token)...")
                outputs = self.model(
                    input_ids,
                    update_mask,
                    position_ids,
                    causal_mask,
                    torch.tensor(current_pos, dtype=torch.long, device=device),
                    IN_PREFILL=False
                )
                
                if self.debug:
                    step_elapsed = time.time() - step_start
                    print(f"    Model forward completed in {step_elapsed:.2f}s")
                
                # Extract logits
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # First element is usually logits
                else:
                    logits = outputs
                
                # Get logits for the generated token
                token_logits = logits[0, -1, :]  # [vocab_size]
                log_probs = torch.log_softmax(token_logits, dim=-1)
                
                # Score for the target token
                token_logprob = log_probs[target_token].item()
                total_logprob += token_logprob
                
                if self.debug:
                    predicted_token = torch.argmax(token_logits).item()
                    predicted_text = self.tokenizer.decode([predicted_token])
                    target_text = self.tokenizer.decode([target_token])
                    prev_text = self.tokenizer.decode([prev_token])
                    print(f"    Token {i+1}: prev='{prev_text}' -> target='{target_text}' (logprob={token_logprob:.4f})")
                    if predicted_token != target_token:
                        print(f"      (Model predicted: '{predicted_text}')")
                
                # Update prev_token for next iteration (teacher forcing)
                prev_token = target_token
        
        if self.debug:
            print(f"  Total log probability: {total_logprob:.4f}")
        
        return total_logprob
    
    def loglikelihood_rolling(self, requests):
        """Compute rolling log-likelihood"""
        print(f"[PyTorchQwen3LM] loglikelihood_rolling called with {len(requests)} requests")
        return []
    
    def generate_until(self, requests):
        """Generate text until stopping condition"""
        print(f"[PyTorchQwen3LM] generate_until called with {len(requests)} requests")
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
        log_file = os.path.join(os.getcwd(), "incorrect_answers_pytorch_qwen3.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== INCORRECT ANSWER LOG (PyTorch Qwen3) ===\n")
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
                if correct_idx is None:
                    # If no ground truth available, can't determine correctness
                    is_incorrect = False
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
        "Evaluate PyTorch Qwen3 model using full lm-evaluation-harness pipeline."
    )
    parser.add_argument("--model", help="Model to evaluate", default="/Volumes/Models/Huggingface/qwen3-0.6b")
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
        help="Log detailed information about incorrect answers to incorrect_answers_pytorch_qwen3.log",
        default=False,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Silence tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load PyTorch model
    lm = PyTorchQwen3LM(
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