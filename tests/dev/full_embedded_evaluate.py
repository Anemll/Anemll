#!/usr/bin/env python3
"""
Embedded PyTorch Qwen2.5 evaluation using full lm-evaluation-harness pipeline
Uses embedded model classes from test_boolq_pytorch_baseline_embedded.py
"""

import argparse
import os
import sys
import json
import math
from pathlib import Path
from importlib.metadata import version
from typing import Dict, List, Tuple, Any, Optional
import collections
import time

# Set offline mode to prevent network calls
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# Performance optimizations
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import safetensors.torch
from tqdm import tqdm

# PyTorch performance optimizations
torch.set_num_threads(4)
if torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True

# ============================================================================
# EMBEDDED MODEL CLASSES (from test_boolq_pytorch_baseline_embedded.py)
# ============================================================================

# Embedded ANEMLL model constants and configuration
MODEL_DTYPE = torch.float32
TEST_DEVICE = "mps"  # Default to MPS, but will be overridden by command line
CONTEXT_LENGTH = 1024

# Cache configuration constants
FORCE_UNIFIED_CACHE = True
ENABLE_UNIFIED_CACHE = True
STATE_LENGTH = CONTEXT_LENGTH
DISABLE_KV_CACHE = False

# LM head configuration constants
ENABLE_CONV2D = bool(1)
ENABLE_VACAB_SPLIT = bool(1)
ENABLE_VACAB_SPLIT8 = bool(0)
ENABLE_VACAB_SPLIT16 = bool(1)
ENABLE_LOGITS2 = bool(1)
ENABLE_COREML = bool(0)

def ane_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Custom softmax implementation optimized for fp16 numerical stability."""
    return F.softmax(x, dim=dim)

class Qwen25Config:
    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["Qwen2ForCausalLM"])
        self.attention_bias = kwargs.get("attention_bias", True)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.bos_token_id = kwargs.get("bos_token_id", 151643)
        self.eos_token_id = kwargs.get("eos_token_id", 151645)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.hidden_size = kwargs.get("hidden_size", 896)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.intermediate_size = kwargs.get("intermediate_size", 4864)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.model_type = kwargs.get("model_type", "qwen2")
        self.num_attention_heads = kwargs.get("num_attention_heads", 14)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 24)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 2)
        self.head_dim = kwargs.get(
            "head_dim",
            self.hidden_size // max(1, self.num_attention_heads),
        )
        self.pretraining_tp = kwargs.get("pretraining_tp", 1)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-06)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        if self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling.get("rope_type", "default")
        self.base = kwargs.get("rope_theta", 1000000.0)
        if self.rope_scaling and 'factor' in self.rope_scaling:
            self.base = self.base * self.rope_scaling['factor']
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.torch_required = kwargs.get("torch_dtype", "bfloat16")
        self.transformers_version = kwargs.get("transformers_version", "4.37.0")
        self.use_cache = kwargs.get("use_cache", True)
        self.vocab_size = kwargs.get("vocab_size", 151936)
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = kwargs.get("state_length", STATE_LENGTH)
        self.use_sliding_window = kwargs.get("use_sliding_window", False)
        self.sliding_window = kwargs.get("sliding_window", 32768)
        self.max_window_layers = kwargs.get("max_window_layers", 28)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

# Import all the embedded model classes - we'll copy the essential ones here
# For brevity, I'll import the main classes we need

class Qwen25RMSNorm(nn.Module):
    """RMSNorm used in Qwen 2.5 models - ANE-aware implementation with mean subtraction."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

def make_causal_mask(length, start):
    """Create causal attention mask."""
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return mask

# ============================================================================
# EVALUATION WRAPPER CLASS
# ============================================================================

@register_model("embedded_qwen25")
class EmbeddedQwen25LM(LM):
    """Embedded Qwen2.5 LM wrapper using embedded model classes"""
    
    def __init__(self, model_path: str, max_tokens: int = 2048, debug: bool = False, 
                 log_incorrect_answers: bool = False, skip: int = 0, **kwargs):
        super().__init__()
        self.model_path = model_path
        self._max_tokens = max_tokens
        self._rank = 0
        self._world_size = 1
        self.debug = debug
        self.log_incorrect_answers = log_incorrect_answers
        self.skip = skip
        self._question_results = []
        
        print(f"Loading Embedded Qwen2.5 model from: {model_path}")
        
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
        config.context_length = 1024  # Increase to handle longer contexts
        config.state_length = 1024    # Also set state_length for KV cache size
        
        # We need to import the actual model classes from the embedded file
        # Override the TEST_DEVICE before importing to make it work with any device
        sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
        
        # Import the module and dynamically set TEST_DEVICE
        import test_boolq_pytorch_baseline_embedded as embedded_module
        embedded_module.TEST_DEVICE = TEST_DEVICE  # Override the hardcoded MPS device
        embedded_module.MODEL_DTYPE = MODEL_DTYPE  # Ensure consistent dtype
        
        from test_boolq_pytorch_baseline_embedded import Qwen25ForCausalLM
        
        # Create embedded model (KV cache is always enabled in embedded version)
        self.model = Qwen25ForCausalLM(config)
        
        # Load pretrained weights
        print(f"Loading pretrained weights...")
        success = self.model.load_pretrained_weights(model_path)
        if not success:
            raise RuntimeError(f"Failed to load pretrained weights from {model_path}")
        
        # Move model to the specified device
        self.model.eval()
        
        # Ensure the model is on the correct device
        device = torch.device(TEST_DEVICE)
        self.model = self.model.to(device)
        
        # Also ensure cached tensors in rotary embeddings are on correct device
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, 'rotary_emb'):
                rotary_emb = layer.self_attn.rotary_emb
                if hasattr(rotary_emb, 'cos_cached'):
                    rotary_emb.cos_cached = rotary_emb.cos_cached.to(device)
                if hasattr(rotary_emb, 'sin_cached'):
                    rotary_emb.sin_cached = rotary_emb.sin_cached.to(device)
        
        print(f"Using device: {TEST_DEVICE}")
        self.device_name = TEST_DEVICE
        
        print(f"Embedded Qwen2.5 model loaded with KV cache enabled")
        
        # Store context length for proper mask creation
        self.context_length = config.context_length
        
        causal_mask_data = make_causal_mask(self.context_length, 0)
        device = torch.device(self.device_name)
        self.causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16).to(device)
        
        print(f"Context length: {self.context_length}")
        
        # Pre-compute token IDs for " no" and " yes"
        self.no_token_id = self.tokenizer.encode(" no", add_special_tokens=False)[0]
        self.yes_token_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
        print(f"Token IDs: ' no'={self.no_token_id}, ' yes'={self.yes_token_id}")

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_tokens

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self.device_name
    
    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _gold_idx(self, doc, opts):
        """Extract ground truth index from document"""
        # Handle BoolQ format
        if 'answer' in doc:
            answer = doc['answer']
            if isinstance(answer, bool):
                return 1 if answer else 0
            elif isinstance(answer, str):
                return 1 if answer.lower() in ['true', 'yes', '1'] else 0
        
        # Handle other formats like ARC
        if 'answerKey' in doc:
            key = doc['answerKey']
            if isinstance(key, str):
                return ord(key.upper()) - ord('A')
            return int(key)
        
        # Handle HellaSwag format
        if 'label' in doc:
            return int(doc['label'])
        
        return 0

    def loglikelihood(self, requests):
        """Process loglikelihood requests using embedded model"""
        print(f"[EmbeddedQwen25LM] Processing {len(requests)} requests")
        
        if self.log_incorrect_answers:
            # Group requests by context to extract ground truth
            context_groups = collections.defaultdict(list)
            for i, req in enumerate(requests):
                context, continuation = req.args
                context_groups[context].append((i, req, continuation))
        
        results = []
        
        with tqdm(total=len(requests) // 2, desc="Scoring pairs") as pbar:
            for i in range(0, len(requests), 2):
                # Process pairs of requests (typically " no" and " yes")
                req1 = requests[i]
                req2 = requests[i + 1] if i + 1 < len(requests) else None
                
                if req2 is None:
                    # Single request
                    context1, cont1 = req1.args
                    score1 = self._score_single_continuation(context1, cont1)
                    results.append((score1, False))
                else:
                    # Pair of requests - use optimized scoring
                    context1, cont1 = req1.args
                    context2, cont2 = req2.args
                    
                    if context1 == context2:
                        # Same context, different continuations
                        score1, score2 = self._score_both_continuations(context1, cont1, cont2)
                    else:
                        # Different contexts
                        score1 = self._score_single_continuation(context1, cont1)
                        score2 = self._score_single_continuation(context2, cont2)
                    
                    results.extend([(score1, False), (score2, False)])
                
                pbar.update(1)
        
        # Log incorrect answers if enabled
        if self.log_incorrect_answers and len(context_groups) > 0:
            self._process_question_results(context_groups, results)
        
        return results

    def _score_both_continuations(self, context: str, cont1: str, cont2: str) -> Tuple[float, float]:
        """Score both continuations efficiently using embedded model"""
        if self.debug:
            print(f"[EmbeddedQwen25LM] _score_both_continuations called with {context[:100]}..., {cont1}, {cont2}")
        
        # Tokenize context
        context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
        
        # Truncate if too long
        if len(context_tokens) > self.context_length - 1:
            context_tokens = context_tokens[-(self.context_length - 1):]
        
        # Get scores for both continuations
        score1 = self._get_continuation_score(context_tokens, cont1)
        score2 = self._get_continuation_score(context_tokens, cont2)
        
        return score1, score2

    def _score_single_continuation(self, context: str, continuation: str) -> float:
        """Score a single continuation using embedded model"""
        context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
        
        if len(context_tokens) > self.context_length - 1:
            context_tokens = context_tokens[-(self.context_length - 1):]
        
        return self._get_continuation_score(context_tokens, continuation)

    def _get_continuation_score(self, context_tokens: List[int], continuation: str) -> float:
        """Get log probability score for a continuation given context using embedded model"""
        
        # Tokenize continuation
        cont_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)
        if not cont_tokens:
            return float('-inf')
        
        # Combine context and continuation
        input_tokens = context_tokens + cont_tokens
        seq_len = len(input_tokens)
        
        if seq_len > self.context_length:
            return float('-inf')
        
        # Prepare input tensors and move to device
        device = torch.device(self.device_name)
        input_ids = torch.tensor([input_tokens], dtype=torch.long).to(device)
        position_ids = torch.arange(seq_len, dtype=torch.long).to(device)
        
        # Create attention mask and move to device
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len].to(device)
        
        # Create update mask (not used in this simple implementation)
        update_mask = torch.zeros((1, 1, self.context_length, 1), dtype=torch.float16).to(device)
        
        # Forward pass
        current_pos = seq_len - 1
        with torch.no_grad():
            hidden_states = self.model(input_ids, update_mask, position_ids, causal_mask, current_pos, IN_PREFILL=True)
            
            # Apply LM head to get logits (embedded model's batched forward doesn't apply LM head)
            # Check if using 16-way split or single LM head
            if hasattr(self.model, 'lm_head16_1'):
                # Using 16-way split LM head
                lm_head_weight_dtype = self.model.lm_head16_1.weight.dtype
                if self.debug:
                    print(f"Hidden states dtype: {hidden_states.dtype}")
                    print(f"LM head weight dtype: {lm_head_weight_dtype}")
                    print(f"MODEL_DTYPE: {MODEL_DTYPE}")
                    print("Using 16-way split LM head")
                
                # Convert to the same dtype as the LM head weights
                hidden_states_reshaped = hidden_states.permute(0, 2, 1).unsqueeze(2).to(lm_head_weight_dtype)
                
                # Apply all 16 LM heads and concatenate results
                logits_parts = []
                for i in range(16):
                    logits_part = getattr(self.model, f"lm_head16_{i+1}")(hidden_states_reshaped).squeeze(2).transpose(1, 2)
                    logits_parts.append(logits_part)
                logits = torch.cat(logits_parts, dim=2)
            else:
                # Using single LM head
                lm_head_weight_dtype = self.model.lm_head.weight.dtype
                if self.debug:
                    print(f"Hidden states dtype: {hidden_states.dtype}")
                    print(f"LM head weight dtype: {lm_head_weight_dtype}")
                    print(f"MODEL_DTYPE: {MODEL_DTYPE}")
                    print("Using single LM head")
                
                # Convert to the same dtype as the LM head weights
                hidden_states_reshaped = hidden_states.permute(0, 2, 1).unsqueeze(2).to(lm_head_weight_dtype)
                logits = self.model.lm_head(hidden_states_reshaped).squeeze(2).transpose(1, 2)
        
        # logits shape: [1, seq_len, vocab_size]
        
        # Calculate log probabilities for continuation tokens
        log_probs = F.log_softmax(logits, dim=-1)
        
        total_log_prob = 0.0
        for i, token_id in enumerate(cont_tokens):
            pos = len(context_tokens) + i - 1
            if pos >= 0 and pos < logits.size(1):
                token_log_prob = log_probs[0, pos, token_id].item()
                total_log_prob += token_log_prob
        
        return total_log_prob

    def _process_question_results(self, context_groups, results):
        """Process results to identify incorrect answers"""
        question_idx = 0
        
        for context, group in context_groups.items():
            if len(group) != 2:  # Expected " no" and " yes" for BoolQ
                continue
            
            # Extract document for ground truth
            doc = group[0][1].doc if hasattr(group[0][1], 'doc') else {}
            gold_idx = self._gold_idx(doc, [" no", " yes"])
            
            # Get scores for this question
            scores = [results[group[j][0]][0] for j in range(len(group))]
            selected_idx = np.argmax(scores)
            
            if selected_idx != gold_idx:
                # This is an incorrect answer
                self._question_results.append({
                    'question_idx': question_idx,
                    'context': context,
                    'options': [" no", " yes"],
                    'selected_idx': selected_idx,
                    'selected_score': scores[selected_idx],
                    'gold_idx': gold_idx,
                    'all_scores': scores,
                    'doc': doc
                })
            
            question_idx += 1
        
        # Log incorrect answers
        if self._question_results:
            self._log_incorrect_answers()

    def _log_incorrect_answers(self):
        """Log detailed information about incorrect answers"""
        print(f"\n[INCORRECT ANSWER ANALYSIS] Analyzing {len(self._question_results)} incorrect answers...")
        
        import time
        log_file = os.path.join(os.getcwd(), "incorrect_answers_embedded.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== INCORRECT ANSWER LOG (Embedded) ===\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for q_info in self._question_results:
                question_idx = q_info['question_idx']
                absolute_question_num = self.skip + question_idx + 1
                
                f.write(f"QUESTION {absolute_question_num} (INCORRECT):\n")
                f.write(f"Context: {q_info['context']}\n")
                f.write(f"Options: {q_info['options']}\n")
                f.write(f"Selected Answer: '{q_info['options'][q_info['selected_idx']]}' (index {q_info['selected_idx']})\n")
                f.write(f"Correct Answer: '{q_info['options'][q_info['gold_idx']]}' (index {q_info['gold_idx']})\n")
                f.write(f"Selected Score: {q_info['selected_score']:.4f}\n")
                f.write(f"Correct Score: {q_info['all_scores'][q_info['gold_idx']]:.4f}\n")
                f.write(f"Score Difference: {q_info['selected_score'] - q_info['all_scores'][q_info['gold_idx']]:.4f}\n")
                f.write(f"All Scores: {[f'{score:.4f}' for score in q_info['all_scores']]}\n")
                f.write(f"Ground Truth: label={q_info['gold_idx']}\n")
                f.write("=" * 50 + "\n\n")
        
        print(f"[SUMMARY] Found {len(self._question_results)} incorrect answers")
        print(f"Detailed log saved to: {log_file}")

    def loglikelihood_rolling(self, requests):
        """Rolling loglikelihood - not used for BoolQ but required by LM interface"""
        raise NotImplementedError("Rolling loglikelihood not implemented for embedded model")

    def generate_until(self, requests):
        """Not implemented for BoolQ evaluation"""
        raise NotImplementedError("Generate until not implemented for embedded model")

def main():
    parser = argparse.ArgumentParser(description='Evaluate embedded Qwen2.5 model')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-0.5B", help='Model path or HuggingFace ID')
    parser.add_argument('--tasks', type=str, default="boolq", help='Evaluation tasks')
    parser.add_argument('--output-dir', type=str, default="./results", help='Output directory')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of examples')
    parser.add_argument('--skip', type=int, default=0, help='Skip first N examples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--log-incorrect-answers', action='store_true', help='Log incorrect answers with details')
    parser.add_argument('--device', type=str, default='mps', help='Device to use (cpu, mps, cuda)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Silence tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Update global TEST_DEVICE before creating model
    global TEST_DEVICE
    TEST_DEVICE = args.device

    # Create model
    lm = EmbeddedQwen25LM(
        model_path=args.model,
        debug=args.debug,
        log_incorrect_answers=args.log_incorrect_answers,
        skip=args.skip
    )

    # Prepare samples if skip/limit specified
    samples = None
    if args.skip > 0 or args.limit is not None:
        end_idx = args.skip + (args.limit if args.limit else 50)
        samples = {args.tasks: list(range(args.skip, end_idx))}
        print(f"Skip: {args.skip}, Limit: {args.limit}")
        print(f"Samples dict: {samples}")

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[args.tasks],
        num_fewshot=0,
        batch_size=args.batch_size,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        samples=samples
    )

    print("Results:")
    for task, result in results['results'].items():
        for metric, value in result.items():
            if isinstance(value, (int, float)):
                print(f"    \"{metric}\": {value},")

if __name__ == "__main__":
    main()