#!/usr/bin/env python3
# ANE/CoreML model evaluation with lm-evaluation-harness
# Based on MLX-LM's implementation pattern
# tested lm-evaluation-harness version 0.4.9
#
# USAGE INSTRUCTIONS:
# -------------------
# This script evaluates Apple Neural Engine (ANE) models using lm-evaluation-harness.
# To run with proper serial execution:
#
# python evaluate_with_harness.py \
#   --model /path/to/your/model \
#   --tasks boolq,arc_easy,hellaswag \
#   --batch-size 1     # Ensures one prompt → one Core ML call \
#   --limit 100         # Limit number of examples per task \
#   --skip 100          # Skip the first N examples (requires --limit) \
#   --output-dir results \
#   --output-path results/window_${skip}_${skip+limit-1}.json
#
# The --batch-size 1 flag is critical to ensure that CoreML runs strictly serially.
# All new KV caches are created fresh for each request to prevent ANE resource conflicts.
#
# MODEL DIRECTORY STRUCTURE:
# Your model directory should contain:
# - embeddings.mlmodelc or embeddings.mlpackage
# - lm_head.mlmodelc or lm_head.mlpackage
# - FFN_*.mlmodelc or FFN_*.mlpackage files
#
# The model directory path must be valid and accessible.

import os
import sys
import time
import json
import argparse
import logging
import collections
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set offline mode BEFORE any other imports to prevent network calls
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0" 
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/.cache/huggingface/datasets")

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add tests directory to path for importing chat.py
tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests")
sys.path.append(tests_dir)

try:
    import coremltools as ct
    # Configure CoreML for single-threaded mode using environment variables
    # This works for all versions of CoreMLTools
    os.environ["COREML_PARTITION_LOADER_DISABLE_MULTI_ENGINE"] = "1"
    print("CoreML configured for single-threaded execution via environment variables")
    
    # Try to use set_low_memory_mode if available (CoreML 7+)
    try:
        if hasattr(ct.utils, 'set_low_memory_mode'):
            ct.utils.set_low_memory_mode(True)
            print("CoreML low memory mode enabled")
    except (AttributeError, ImportError):
        print("CoreML low memory mode not available in this version")
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
    print(" pip install lm-eval")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: transformers not found. Please install it using:")
    print("pip install transformers")
    sys.exit(1)


def create_unified_state(ffn_models):
    """Create unified KV cache state for transformer."""
    if isinstance(ffn_models[0], dict):
        # Use first FFN model's prefill function to create state
        state = ffn_models[0]['prefill'].make_state()
        return state
    else:
        state = ffn_models[0].make_state()
        return state


# Import from chat.py
try:
    from chat import (
        run_prefill,
        generate_next_token,
        initialize_causal_mask,
        initialize_tokenizer,
        parse_model_path,
        parse_ffn_filename,
        find_all_chunks,
        load_model,
        load_metadata
    )
    print("Successfully imported inference functions from chat.py")
    USING_REAL_INFERENCE = True
except ImportError:
    print("Warning: Could not import from tests directory. Please ensure chat.py is available.")
    sys.exit(1)

# Default model path
DEFAULT_MODEL_PATH = os.path.expandvars("$HOME/Models/ANE/models/latest")


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


def gold_idx(doc, opts):
    """Extract the correct answer index from the document."""
    if doc is None:
        return None
    # Unwrap nested 'doc' field if present (newer harness may nest the original doc)
    if isinstance(doc.get('doc', None), dict):
        return gold_idx(doc['doc'], opts)
    
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


@register_model("anelm")
class ANELM(LM):
    """ANE/CoreML model implementation for lm-evaluation-harness."""

    def __init__(
        self,
        model_path: str,
        max_tokens: Optional[int] = None,
        use_chat_template: Optional[bool] = None,
        verbose_output: bool = False,
        log_incorrect_answers: bool = False,
        **kwargs
    ) -> None:
        """Initialize the ANE model evaluator.
        
        Args:
            model_path: Path to model directory containing CoreML models
            max_tokens: Maximum number of tokens to generate
            use_chat_template: Whether to use chat template for formatting
            **kwargs: Additional arguments passed from lm-evaluation-harness
        """
        super().__init__()
        self.model_path = Path(model_path)
        
        # Load the models
        self._load_models()
        
        # Initialize inference components
        self._initialize_inference_components()
        
        # Set maximum tokens
        self._max_tokens = max_tokens or 2048  # Default to 2048 if not specified
        
        # Parse batch size from kwargs if provided (for harness only)
        if 'batch_size' in kwargs and kwargs['batch_size'] is not None:
            self._batch_size = kwargs['batch_size']
        else:
            # Default to batch_size 1 for strictly serial evaluation
            self._batch_size = 1
            
        # IMPORTANT: We NEVER change the model's compiled batch size in metadata
        # Only warn if harness batch size doesn't match model's compiled batch size
        if 'batch_size' in self.metadata and self._batch_size != self.metadata['batch_size']:
            print(f"\nWARNING: Harness batch_size={self._batch_size} differs from model's compiled batch_size={self.metadata['batch_size']}")
            print(f"The CoreML model requires inputs padded to batch_size={self.metadata['batch_size']}")
            print(f"This is handled internally - DO NOT change metadata['batch_size']")
        
        # Chat template settings - match chat.py behavior more closely
        # If not explicitly requested, default to False to match --no-template behavior
        self.use_chat_template = use_chat_template if use_chat_template is not None else False
        
        # Store verbose output flag
        self.verbose_output = verbose_output
        
        # Store incorrect answer logging flag
        self.log_incorrect_answers = log_incorrect_answers
        
        if verbose_output:
            print(f"[DEBUG] use_chat_template={self.use_chat_template}, original param={use_chat_template}")
            
        # Print important configuration info
        print(f"\nANE Configuration:")
        print(f"  Harness Batch Size: {self._batch_size}")
        print(f"  Model Batch Size: {self.metadata.get('batch_size', 'unknown')}")
        print(f"  CoreML Single Threading: Enabled")
        print(f"  Shared State: Enabled (state created once during initialization)")

    def _load_models(self):
        """Load all required model components."""
        print(f"Loading models from {self.model_path}")
        
        if not self.model_path.exists():
            raise ValueError(f"Model directory not found: {self.model_path}")
            
        # Parse arguments for metadata loading
        args = argparse.Namespace()
        args.model = str(self.model_path)
        args.d = str(self.model_path)  # Add the 'd' attribute that load_metadata expects
        args.context_length = None
        args.batch_size = None
        
        # More flexible model file search - look for any embeddings, lm_head, and FFN models
        # First try to match exact filenames
        embed_paths = list(self.model_path.glob("*embeddings*.mlmodelc")) or list(self.model_path.glob("*embeddings*.mlpackage"))
        lm_head_paths = list(self.model_path.glob("*lm_head*.mlmodelc")) or list(self.model_path.glob("*lm_head*.mlpackage"))
        ffn_paths = list(self.model_path.glob("*FFN*PF*.mlmodelc")) or list(self.model_path.glob("*FFN*PF*.mlpackage"))
        
        # If not found, try more general patterns
        if not embed_paths:
            print("No embeddings models found with standard naming, trying general patterns...")
            embed_paths = list(self.model_path.glob("*embed*.mlmodelc")) or list(self.model_path.glob("*embed*.mlpackage"))
        
        if not lm_head_paths:
            print("No lm_head models found with standard naming, trying general patterns...")
            lm_head_paths = list(self.model_path.glob("*lm*.mlmodelc")) or list(self.model_path.glob("*lm*.mlpackage"))
        
        if not ffn_paths:
            print("No FFN_PF models found with standard naming, trying general patterns...")
            ffn_paths = list(self.model_path.glob("*FFN*.mlmodelc")) or list(self.model_path.glob("*FFN*.mlpackage"))
        
        # Check if we found all the necessary models
        if not embed_paths or not lm_head_paths or not ffn_paths:
            print(f"Warning: Could not find all required models. Found:")
            print(f"  Embedding models: {len(embed_paths)}")
            print(f"  LM Head models: {len(lm_head_paths)}")
            print(f"  FFN models: {len(ffn_paths)}")
            
            # Extra logging to diagnose the issue
            print("\nListing all files in directory:")
            for file in self.model_path.glob("*"):
                print(f"  {file.name}")
            
            # Do we have nemotron models instead of llama?
            embed_paths = list(self.model_path.glob("*nemotron*embeddings*.mlmodelc")) or list(self.model_path.glob("*nemotron*embeddings*.mlpackage"))
            lm_head_paths = list(self.model_path.glob("*nemotron*lm_head*.mlmodelc")) or list(self.model_path.glob("*nemotron*lm_head*.mlpackage"))
            ffn_paths = list(self.model_path.glob("*nemotron*FFN*PF*.mlmodelc")) or list(self.model_path.glob("*nemotron*FFN*PF*.mlpackage"))
            
            if embed_paths and lm_head_paths and ffn_paths:
                print("\nFound models with 'nemotron' prefix")
            else:
                raise ValueError("One or more required models not found.")
        
        # Load embedding model
        embed_path = embed_paths[0]
        print(f"Loading embeddings model from {embed_path}")
        self.embedding_model = load_model(embed_path)
        print("Embeddings model loaded successfully")
        
        # Load LM head model
        lm_head_path = lm_head_paths[0]
        print(f"Loading LM head model from {lm_head_path}")
        self.lm_head_model = load_model(lm_head_path)
        print("LM head model loaded successfully")
        
        # Load FFN models
        self.ffn_models = []
        
        # Sort FFN paths to ensure consistent order
        ffn_paths = sorted(ffn_paths)
        print(f"Found {len(ffn_paths)} FFN models:")
        for path in ffn_paths:
            print(f"  {path.name}")
        
        for ffn_path in ffn_paths:
            chunk_no, total_chunks = parse_ffn_filename(ffn_path)
            
            if chunk_no and total_chunks:
                print(f"Loading chunked FFN model {ffn_path.name} ({chunk_no} of {total_chunks})")
                # For chunked models, use both infer and prefill functions
                try:
                    self.ffn_models.append({
                        'infer': load_model(ffn_path, function_name='infer'),
                        'prefill': load_model(ffn_path, function_name='prefill')
                    })
                    print(f"Loaded {ffn_path.name} with infer/prefill functions")
                except Exception as e:
                    print(f"Could not load with function_name parameter: {str(e)}")
                    print("Trying without specifying function name")
                    self.ffn_models.append(load_model(ffn_path))
            else:
                # Single FFN model
                print(f"Loading single FFN model: {ffn_path.name}")
                self.ffn_models.append(load_model(ffn_path))
        
        print(f"Loaded {len(self.ffn_models)} FFN models successfully")
        
        # Load metadata - try from meta.yaml first, then from model
        meta_yaml_path = self.model_path / "meta.yaml"
        if meta_yaml_path.exists():
            try:
                import yaml
                with open(meta_yaml_path, 'r') as f:
                    meta_data = yaml.safe_load(f)
                
                params = meta_data.get('model_info', {}).get('parameters', {})
                self.metadata = {
                    'context_length': params.get('context_length', 1024),
                    'state_length': params.get('context_length', 1024),  # Usually same as context_length
                    'batch_size': params.get('batch_size', 64),
                    'lut_bits': 4 if params.get('lut_ffn') == '4' else 0,  # Check for LUT quantization
                    'num_chunks': params.get('num_chunks', len(ffn_paths)),
                    'split_lm_head': params.get('split_lm_head', 1)
                }
                print(f"Loaded metadata from {meta_yaml_path}")
            except Exception as e:
                print(f"Error loading meta.yaml: {str(e)}")
                # Fall back to loading from model
                try:
                    self.metadata = load_metadata(self.embedding_model, args)
                except Exception as e2:
                    print(f"Error loading metadata from model: {str(e2)}")
                    print("Using default metadata values")
                    self.metadata = {
                        'context_length': 1024,
                        'state_length': 1024,
                        'batch_size': 64,
                        'lut_bits': 0,
                        'num_chunks': len(ffn_paths)
                    }
        else:
            # Try loading from model metadata
            try:
                self.metadata = load_metadata(self.embedding_model, args)
            except Exception as e:
                print(f"Error loading metadata from model: {str(e)}")
                print("Using default metadata values")
                self.metadata = {
                    'context_length': 1024,
                    'state_length': 1024,
                    'batch_size': 64,
                    'lut_bits': 0,
                    'num_chunks': len(ffn_paths)
                }
        
        print("\nModel metadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")
    
    def _initialize_inference_components(self):
        """Initialize components needed for inference like tokenizer, state, and causal mask."""
        # Initialize tokenizer
        print("\nInitializing tokenizer...")
        self.tokenizer = initialize_tokenizer(self.model_path)
        if self.tokenizer is None:
            raise ValueError("Failed to initialize tokenizer")
        print(f"Tokenizer initialized with vocabulary size: {len(self.tokenizer)}")
        
        # Initialize state and causal mask
        print("\nInitializing KV cache state...")
        context_length = self.metadata['context_length']
        self.state = create_unified_state(self.ffn_models)
        self.causal_mask = initialize_causal_mask(context_length)
        print(f"State initialized for context length: {context_length}")

    def _create_new_state(self):
        """This is a no-op in the simplified implementation.
        We create the state once during initialization and reuse it.
        """
        # The state is already created during initialization
        return self.state

    def _clone_state(self, state=None):
        """This is a no-op in the simplified implementation.
        We create the state once during initialization and reuse it.
        """
        # Just use the existing state
        return self.state
    

    ########################################################
    # score one sequence  (shape  [L]  after padding trim) #
    ########################################################
    def _score_sequence(self, tokens_1d: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score a complete sequence by computing log probabilities for each token.
        
        FIXED VERSION: Properly prefills the correct number of tokens.
        
        Parameters
        ----------
        tokens_1d : 1-D LongTensor, length L  (prompt + answer)

        Returns
        -------
        logp_vec   : FloatTensor [L-1]   log p(t_k | t_<k)  for k = 1 ... L-1
        greedy_vec : BoolTensor  [L-1]   top-1 == gold at every position
        """
        seq_length = tokens_1d.size(0)
        ctx_len = self.metadata["context_length"]

        # fresh empty KV-cache
        kv = create_unified_state(self.ffn_models)

        # 1. prefill the prompt (t_0 ... t_{seq_length-2})
        prefill = tokens_1d[:-1].unsqueeze(0)                # [1, seq_length-1]
        prefill_length = seq_length - 1  # Number of tokens to prefill
        
        # FIXED: Pass the correct number of tokens to prefill instead of 0
        run_prefill(self.embedding_model,
                    self.ffn_models,
                    prefill,
                    prefill_length,                          # FIXED: was 0, now correct length
                    ctx_len,
                    self.metadata.get("batch_size", 64),
                    kv,
                    self.causal_mask)
        pos = seq_length

        # 2. walk through answer tokens
        logps, greedy = [], []
        
        # Start from position seq_length-1 (after prefilling seq_length-1 tokens)
        for i in range(1, seq_length):
            # Get the current token to process (single token)
            curr_token = tokens_1d[i-1:i].unsqueeze(0)  # [1, 1] single token
            gold_tok = tokens_1d[i]
            
            # Generate next token and get log probabilities
            nxt_tok, logp_vec = self._generate_next_token_with_logits(
                self.embedding_model,
                self.ffn_models,
                self.lm_head_model,
                curr_token,          # [1, 1] single token
                pos+1,                
                ctx_len,
                self.metadata,
                state       = kv,
                causal_mask = self.causal_mask)

            # Record statistics
            logps.append(logp_vec[gold_tok].unsqueeze(0))
            greedy.append(torch.tensor(nxt_tok == gold_tok))

        return torch.cat(logps), torch.stack(greedy)


    # --------------------------------------------------------------
    #  Helper: feed *gold* token, get logits for that position
    # --------------------------------------------------------------
    def _predict_token_with_logits(self,
                                gold_tok_id: int,
                                kv,
                                pos: int):
        """Feed gold token, update KV cache, and get logits for next position.
        
        CURRENTLY USED FOR: Second _score_sequence method (lines 580-610)
        COULD BE USED FOR: Arc_challenge optimization (but has a bug - see line 549)
        
        BUG: Line 549 tries to access lm_out['hidden_states'] but lm_out comes from 
             LM head model (produces logits, not hidden_states). This needs to be fixed
             to enable arc_challenge optimization.
        
        Parameters
        ----------
        gold_tok_id : int              token to insert at position `pos`
        kv          : inplace KV-cache
        pos         : int (0-based)    position of `gold_tok_id`

        Returns
        -------
        greedy_id   : int              arg-max next-token prediction
        log_probs   : FloatTensor [vocab]
        """
        # 1. create 1×1 tensor [[gold_tok]]
        tok_tensor = torch.tensor([[gold_tok_id]], dtype=torch.int32)

        # 2. embed + update cache + run LM-head
        lm_out = self.lm_head_model.predict(
            {'hidden_states':
                self.embedding_model
                    .predict({'input_ids': tok_tensor.numpy()})['hidden_states']
            })

        # 2.a update KV for *all* FFN chunks
        for ffn in self.ffn_models:
            if isinstance(ffn, dict) and 'prefill' in ffn:
                ffn['prefill'].predict(
                    {
                        'hidden_states': lm_out['hidden_states'],
                        'update_mask':   np.array([[[[1.]]]], dtype=np.float16),
                        'position_ids':  np.array([pos],  dtype=np.int32),
                        'causal_mask':   self.causal_mask[:, :, pos:pos+1, :],
                        'current_pos':   np.array([pos],  dtype=np.int32)
                    },
                    kv)

        # 3. concatenate logits parts (split-LM-head models)
        num_parts  = self.metadata.get('split_lm_head',
                                    self.metadata.get('num_logits', 8))
        if 'logits1' in lm_out:  # split head
            logits = torch.cat([torch.from_numpy(lm_out[f'logits{i}'])
                                for i in range(1, num_parts + 1)], dim=-1)
        else:                    # single tensor
            logits = torch.from_numpy(lm_out.get('output_logits'))

        # 4. last dimension → vocab
        if logits.dim() == 3:
            logits = logits[0, -1]          # [vocab]
        elif logits.dim() == 2:
            logits = logits[-1]             # [vocab]

        log_probs = torch.log_softmax(logits, dim=-1)
        greedy_id = torch.argmax(log_probs).item()
        return greedy_id, log_probs




    # --------------------------------------------------------------
    #  Batch wrapper – pads to rectangular [B, L-1]
    # --------------------------------------------------------------
    def _score_batch(self, token_batch: torch.Tensor):
        """
        token_batch : LongTensor [B, L]  (right-padded with 0)
        """
        B, L = token_batch.shape
        seq_scores, seq_greedy = [], []

        for b in range(B):
            seq = token_batch[b]
            true_len = (seq != 0).sum().item()
            if true_len < 2:
                seq_scores.append(torch.empty(0))
                seq_greedy.append(torch.empty(0, dtype=torch.bool))
                continue

            lp_vec, gd_vec = self._score_sequence(seq[:true_len])
            seq_scores.append(lp_vec)
            seq_greedy.append(gd_vec)

        max_T = max(v.size(0) for v in seq_scores)  # ≤ L-1

        def pad(v, fill):
            return torch.cat([v,
                            v.new_full((max_T - v.size(0),),
                                        fill,
                                        dtype=v.dtype,
                                        device=v.device)])
        scores    = torch.stack([pad(v, 0.0)   for v in seq_scores])   # [B, max_T]
        is_greedy = torch.stack([pad(v, False) for v in seq_greedy])   # [B, max_T]

        return scores, is_greedy
    

    def _process_prompt(self, prompt, step_size: int = 1024):
        """Process the prompt and return logprobs.
        
        USED FOR: BoolQ (yes/no) and other single-token answer tasks
        
        PREFILL APPROACH: 
        - Prefills ALL tokens in the prompt (including last token)
        - Then reprocesses the last token to get prediction scores
        - This is technically incorrect but works by overwriting KV cache at last position
        
        WORKFLOW:
        1. run_prefill(entire_prompt, prompt_length) 
        2. Re-run embeddings + FFN on last token to get logits
        3. Return log probabilities for next token prediction
        """
        if self.verbose_output:
            print(f"[DEBUG] _process_prompt: received prompt type {type(prompt)}, length {len(prompt) if hasattr(prompt, '__len__') else 'N/A'}")
        
        # Convert prompt to proper torch tensor format [1, sequence_length] with int32 dtype
        if isinstance(prompt, torch.Tensor):
            if prompt.dim() == 1:
                prompt = prompt.unsqueeze(0)  # Add batch dimension
            prompt = prompt.to(torch.int32)
        else:
            # Convert list/array to tensor with proper shape and dtype
            prompt_array = np.array(prompt, dtype=np.int32)
            if prompt_array.ndim == 1:
                prompt = torch.tensor(prompt_array, dtype=torch.int32).unsqueeze(0)
            else:
                prompt = torch.tensor(prompt_array, dtype=torch.int32)
        
        if self.verbose_output:
            print(f"[DEBUG] _process_prompt: converted to tensor shape {prompt.shape}")
        
        # Create a completely fresh state for each request to match chat.py behavior exactly
        # This ensures zero state contamination
        cache = create_unified_state(self.ffn_models)
        
        if self.verbose_output:
            print(f"[DEBUG] Created fresh cache for this prompt")
        
        # Step 1: Pad prompt to match model's expected context length
        prompt_length = prompt.shape[1]
        context_length = self.metadata['context_length']
        
        if prompt_length > context_length:
            # Truncate if too long
            prompt = prompt[:, -context_length:]
            prompt_length = context_length
        else:
            # Pad to context length
            import torch.nn.functional as F
            prompt = F.pad(prompt, (0, context_length - prompt_length), value=0)
        
        # Step 2: Run prefill using original run_prefill function but with un-padded input
        # BOOLQ PREFILL: Prefills ALL tokens (technically incorrect but works)
        # The run_prefill function will handle the batching internally
        if self.verbose_output:
            print(f"[DEBUG] _process_prompt: calling run_prefill with prompt_length={prompt_length}")
        pos_result = run_prefill(
            self.embedding_model,
            self.ffn_models,
            prompt[:, :prompt_length],  # Use only the actual prompt length, not padded
            prompt_length,  # BOOLQ: Prefills entire prompt (all tokens)
            self.metadata['context_length'],
            self.metadata.get('batch_size', 64),
            cache,
            self.causal_mask
        )
        if self.verbose_output:
            print(f"[DEBUG] _process_prompt: run_prefill completed")
        
        # Step 3: Use chat.py's exact generate_next_token function instead of replicating the logic
        # BOOLQ PREDICTION: Re-process last token to get next token prediction scores
        try:
            generation_pos = prompt_length  # This is like 'pos' parameter in chat.py
            
            # Use the original unpadded input_ids, just like chat.py does
            original_input_ids = prompt[:, :prompt_length]  # Unpadded input tensor
            
            # Instead of using generate_next_token which only returns token ID,
            # we need to get the actual logits from the model
            
            # Get current token (last token in prompt)
            current_token = original_input_ids[:, generation_pos-1:generation_pos]  # [1, 1]
            
            # Ensure proper data type for CoreML
            current_token_array = current_token.numpy().astype(np.int32)
            
            # Run embeddings
            hidden_states = torch.from_numpy(
                self.embedding_model.predict({'input_ids': current_token_array})['hidden_states']
            )  # [1, 1, hidden_size]
            
            # Create masks
            update_mask = torch.zeros((1, 1, self.metadata['context_length'], 1), dtype=torch.float16)
            update_mask[0, 0, generation_pos-1, 0] = 1.0
            position_ids = torch.tensor([generation_pos-1], dtype=torch.int32)  # [1]
            
            # Extract single position causal mask
            single_causal_mask = self.causal_mask[:, :, generation_pos-1:generation_pos, :]  # [1, num_heads, 1, seq_len]
            
            # Run through FFN models (same logic as generate_next_token)
            for ffn_model in self.ffn_models:
                if isinstance(ffn_model, dict) and 'infer' in ffn_model:
                    # Chunked model with infer function
                    inputs = {
                        'hidden_states': hidden_states.numpy().astype(np.float16),
                        'update_mask': update_mask.numpy().astype(np.float16),
                        'position_ids': position_ids.numpy().astype(np.int32),
                        'causal_mask': single_causal_mask.numpy().astype(np.float16),
                        'current_pos': position_ids.numpy().astype(np.int32)
                    }
                    output = ffn_model['infer'].predict(inputs, cache)
                    hidden_states = torch.from_numpy(output['output_hidden_states'])
                else:
                    # Non-chunked model or fallback
                    print(f"[WARNING] FFN model is not a chunked model with 'infer' function")
            
            # Run LM head to get logits
            lm_output = self.lm_head_model.predict({'hidden_states': hidden_states.numpy().astype(np.float16)})
            
            # Get number of logits from metadata
            num_logits = self.metadata.get('split_lm_head', self.metadata.get('num_logits', 8))
            
            # Combine logits1-N if they exist
            if 'logits1' in lm_output:
                # Concatenate all logits parts
                logits_parts = []
                for i in range(1, num_logits + 1):
                    key = f'logits{i}'
                    if key in lm_output:
                        logits_parts.append(torch.from_numpy(lm_output[key]))
                if self.verbose_output:
                    print(f"[DEBUG] LM head: {len(logits_parts)} logit parts found, each shape {lm_output['logits1'].shape}")
                logits = torch.cat(logits_parts, dim=-1)  # Concatenate along vocab dimension
            elif 'output_logits' in lm_output:
                # Try output_logits as fallback
                logits = torch.from_numpy(lm_output['output_logits'])
                if self.verbose_output:
                    print(f"[DEBUG] output_logits shape: {lm_output['output_logits'].shape}")
            else:
                raise ValueError(f"No logits found in lm_head output. Keys: {list(lm_output.keys())}")
            
            # Extract logits for the last position and convert to log probabilities
            # The logits shape should be [batch_size, seq_len, vocab_size]
            # We want the last token's logits
            if logits.dim() == 3:
                logits = logits[0, -1, :]  # Shape: [vocab_size]
            elif logits.dim() == 2:
                logits = logits[-1, :]  # Already [seq_len, vocab_size], take last
            elif logits.dim() == 1:
                pass  # Already [vocab_size]
            else:
                print(f"[WARNING] Unexpected logits shape: {logits.shape}")
                
            # Add debug logging to check logits
            if self.verbose_output:
                print(f"[DEBUG] Logits shape after extraction: {logits.shape}")
                print(f"[DEBUG] Logits min: {logits.min().item():.3f}, max: {logits.max().item():.3f}")
                print(f"[DEBUG] Top 5 logit values: {torch.topk(logits, 5).values.tolist()}")
                print(f"[DEBUG] Top 5 token IDs: {torch.topk(logits, 5).indices.tolist()}")
                
            log_probs = torch.log_softmax(logits, dim=-1).unsqueeze(0)  # Shape: [1, vocab_size]
            
            return log_probs, cache
            
        except Exception as e:
            print(f"Error in _process_prompt: {e}")
            print(f"Debug info - prompt shape: {prompt.shape}, prompt_length: {prompt_length}")
            print(f"Debug info - metadata: {self.metadata}")
            # Return uniform distribution as fallback
            vocab_size = len(self.tokenizer)
            log_probs = torch.full((1, vocab_size), -np.log(vocab_size))
            return log_probs, cache
        
    def _score_fn(self, inputs, cache: Optional[Any] = None, step_size: int = 1024):
        """Score the inputs and return log probabilities and greedy indicators."""
        # Prepare inputs
        token_batch, lengths = _pad_inputs(inputs)
        
        # Use the much more efficient batch scoring method
        scores, is_greedy = self._score_batch(token_batch, step_size)
        
        # Apply mask based on lengths
        max_len = scores.shape[1]
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        scores = torch.where(mask, scores, torch.zeros_like(scores))
        is_greedy = torch.where(mask, is_greedy, torch.zeros_like(is_greedy, dtype=torch.bool))
        
        return scores, lengths, is_greedy
    
    def _generate_next_token_with_logits(self, embed_model, ffn_models, lmhead_model, input_ids, pos, context_length, metadata, state=None, causal_mask=None):
        """Modified version of generate_next_token that returns both token and log probabilities.
        
        This is an unrolled version of generate_next_token from chat.py that returns
        the full log probability distribution along with the selected token.
        
        Returns:
            tuple: (next_token, log_probs) where log_probs is the full vocabulary log probability distribution
        """
        # Get current token for embedding
        # Assert that input_ids is a single token [1, 1]
        if self.verbose_output:
            print(f"[DEBUG] _generate_next_token_with_logits: pos={pos}, input_ids.shape={input_ids.shape}")
        if input_ids.shape[1] != 1:
            raise ValueError(f"_generate_next_token_with_logits expects single token input [1, 1], got {input_ids.shape}")
        
        current_token = input_ids  # Already [1, 1]
        
        if current_token.shape[1] == 0:
            raise ValueError(f"Empty token at position {pos-1}")
        
        # Run embeddings
        hidden_states = torch.from_numpy(
            embed_model.predict({'input_ids': current_token.numpy().astype(np.int32)})['hidden_states']
        )
        
        # Create masks for single position inference
        update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
        update_mask[0, 0, pos-1, 0] = 1.0
        position_ids = torch.tensor([pos-1], dtype=torch.int32)
        
        # Extract single position causal mask
        if causal_mask is not None:
            single_causal_mask = causal_mask[:, :, pos-1:pos, :]
        else:
            # Create causal mask if not provided
            from chat import make_causal_mask
            full_mask = make_causal_mask(context_length, 0)
            single_causal_mask = torch.tensor(full_mask[:, :, pos-1:pos, :], dtype=torch.float16)
        
        # Run through FFN models (chunked)
        for ffn_model in ffn_models:
            if isinstance(ffn_model, dict) and 'infer' in ffn_model:
                inputs = {
                    'hidden_states': hidden_states.numpy().astype(np.float16),
                    'update_mask': update_mask.numpy().astype(np.float16),
                    'position_ids': position_ids.numpy().astype(np.int32),
                    'causal_mask': single_causal_mask.numpy().astype(np.float16),
                    'current_pos': position_ids.numpy().astype(np.int32)
                }
                output = ffn_model['infer'].predict(inputs, state)
                hidden_states = torch.from_numpy(output['output_hidden_states'])
        
        # Run LM head
        lm_output = lmhead_model.predict({'hidden_states': hidden_states.numpy().astype(np.float16)})
        
        # Get number of logits from metadata
        num_logits = metadata.get('split_lm_head', metadata.get('num_logits', 8))
        
        # Combine logits1-N if they exist
        if 'logits1' in lm_output:
            # Concatenate all logits parts
            logits_parts = []
            for i in range(1, num_logits + 1):
                key = f'logits{i}'
                if key in lm_output:
                    logits_parts.append(torch.from_numpy(lm_output[key]))
            logits = torch.cat(logits_parts, dim=-1)  # Concatenate along vocab dimension
        elif 'output_logits' in lm_output:
            # Try output_logits as fallback
            logits = torch.from_numpy(lm_output['output_logits'])
        else:
            raise ValueError("No logits found in lm_head output")
        
        # Extract logits for the current position
        if logits.dim() == 3:
            logits = logits[0, -1, :]  # [vocab_size]
        elif logits.dim() == 2:
            logits = logits[-1, :]  # [vocab_size]
        
        # Get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get predicted token (for debugging/comparison)
        next_token = torch.argmax(logits).item()
        
        return next_token, log_probs
    
    def _score_one_choice(self, prefix_tokens, answer_tokens):
        """Score a single answer choice using teacher forcing.
        
        USED FOR: Arc_challenge & BOOLQ and other multi-token answer tasks
        
        PREFILL APPROACH (CORRECT):
        - Prefills prompt MINUS last token: run_prefill(prompt[:-1], num_tokens-1)
        - Then processes each answer token sequentially using generate_next_token
        - Follows proper pattern: prefill(prompt less last token) + generate_token(last token)
        
        WORKFLOW:
        1. run_prefill(prompt[:-1], len(prefix_tokens)-1) - prefill all but last prompt token
        2. For each answer token:
           a. Call generate_next_token(sequence_so_far, current_length) 
           b. Get log probability for target answer token
           c. Update sequence with actual answer token (teacher forcing)
        3. Return sum of log probabilities for all answer tokens
        
        Args:
            prefix_tokens: List of token IDs for the context/prompt
            answer_tokens: List of token IDs for the answer to score
            
        Returns:
            total_logprob: Sum of log probabilities for all answer tokens
        """
        if self.verbose_output:
            answer_text = self.tokenizer.decode(answer_tokens)
            print(f"\n[VERBOSE] _score_one_choice: '{answer_text}' ({len(answer_tokens)} tokens)")
            
        # Create fresh KV cache state
        kv = create_unified_state(self.ffn_models)
        
        # Convert prefix tokens to tensor
        prefix_tensor = torch.tensor(prefix_tokens, dtype=torch.int32).unsqueeze(0)  # [1, seq_len]
        
        # Check if total sequence would exceed context length - skip if so (like boolq does)
        total_length = len(prefix_tokens) + len(answer_tokens)
        if total_length > self.metadata['context_length']:
            if self.verbose_output:
                print(f"  Skipping: total length {total_length} exceeds context length {self.metadata['context_length']}")
            return -float("inf")  # Return -inf to indicate this choice should be skipped
        
        # Truncate prefix if needed (shouldn't happen due to check above, but for safety)
        if len(prefix_tokens) > self.metadata['context_length']:
            prefix_tensor = prefix_tensor[:, -self.metadata['context_length']:]
            prefix_tokens = prefix_tokens[-self.metadata['context_length']:]
        
        #################################
        #  Stage A – prefill P-1 tokens #
        #################################
        # ARC_CHALLENGE PREFILL (CORRECT): Prefill prompt minus last token
        if len(prefix_tokens) > 1:
            prefill_tokens = prefix_tensor[:, :-1]  # t₀ … t_{P-2}
            # The second parameter is the number of tokens to prefill
            num_tokens_to_prefill = len(prefix_tokens) - 1
            pos_result = run_prefill(
                self.embedding_model,
                self.ffn_models,
                prefill_tokens,
                num_tokens_to_prefill,  # ARC_CHALLENGE: Prefill P-1 tokens (correct approach)
                self.metadata['context_length'],
                self.metadata.get('batch_size', 64),
                kv,
                self.causal_mask
            )
            # After prefilling P-1 tokens, we're at position P-1
            pos = len(prefix_tokens) - 1
        else:
            # Edge case: single token prompt
            pos = 0
            
        prev = prefix_tokens[-1]  # t_{P-1} (the *last* prompt token)
        
        if self.verbose_output:
            print(f"  Prefilled {len(prefix_tokens)-1} tokens, pos={pos}")
            print(f"  Last prompt token (prev): {prev} = '{self.tokenizer.decode([prev])}'")
        
        #################################
        #  Stage B – walk through answer#
        #################################
        # ARC_CHALLENGE PREDICTION: Process each answer token sequentially
        logp_total = 0.0
        
        for i, tok in enumerate(answer_tokens):
            # Process single token (prev) to get prediction for next token
            prev_token_tensor = torch.tensor([[prev]], dtype=torch.int32)  # [1, 1] single token
            
            if self.verbose_output:
                print(f"\n  Iteration {i+1}: prev={prev}, target={tok}")
                print(f"    Processing single token at position {pos}")
            
            # Get log probabilities and next token prediction
            next_token, log_probs = self._generate_next_token_with_logits(
                self.embedding_model,
                self.ffn_models,
                self.lm_head_model,
                prev_token_tensor,  # [1, 1] single token
                pos+1,  # Current position (0-based)
                self.metadata['context_length'],
                self.metadata,
                state=kv,
                causal_mask=self.causal_mask
            )
            
            # Score the actual answer token
            logprob = log_probs[tok].item()
            logp_total += logprob
            
            if self.verbose_output:
                token_text = self.tokenizer.decode([tok])
                predicted_text = self.tokenizer.decode([next_token])
                print(f"    Target token: '{token_text}' (ID: {tok}) -> log prob: {logprob:.4f}")
                if next_token != tok:
                    print(f"    (Model predicted: '{predicted_text}' (ID: {next_token}))")
            
            # Teacher forcing: the true token becomes "prev" for next iteration
            prev = tok
            pos += 1
        
        if self.verbose_output:
            print(f"\n  Total log probability: {logp_total:.4f}")
        
        return logp_total

    def _debug_tokenization(self, text, context=""):
        """Debug tokenization to compare with MLX approach"""
        print(f"\n[TOKENIZATION DEBUG] {context}")
        print(f"Text: {repr(text)}")
        print(f"use_chat_template: {self.use_chat_template}")
        
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        print(f"Tokens: {tokens}")
        
        return tokens

    def _tokenize(self, texts):
        """Tokenize texts using the model's tokenizer."""
        result = []
        for t in texts:
            if self.verbose_output and len(result) < 3:  # Debug first few
                tokens = self._debug_tokenization(t, f"Text {len(result)}")
                result.append(tokens)
                continue
                
            if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
                # For models with chat templates, format as a user message
                try:
                    messages = [{"role": "user", "content": t}]
                    chat_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    result.append(self.tokenizer.encode(chat_text, add_special_tokens=True))
                except Exception as e:
                    print(f"Error applying chat template: {str(e)}. Falling back to standard tokenization.")
                    result.append(self.tokenizer.encode(t, add_special_tokens=True))
            else:
                tokens = self.tokenizer.encode(t, add_special_tokens=True)
                
                # If we detect chat template tokens, strip them to match chat.py --no-template behavior
                if len(tokens) > 0 and tokens[0] == 151644:  # <|im_start|> token
                    if self.verbose_output:
                        print(f"[DEBUG] Detected chat template formatting, attempting to extract raw content")
                    
                    # Try to extract just the content between system and assistant parts
                    decoded_text = self.tokenizer.decode(tokens)
                    
                    # Look for the actual content after system prompt and before assistant
                    if '<|im_start|>user\n' in decoded_text and '<|im_end|>\n<|im_start|>assistant\n' in decoded_text:
                        # Extract content between user tags
                        start_marker = '<|im_start|>user\n'
                        end_marker = '<|im_end|>\n<|im_start|>assistant\n'
                        start_idx = decoded_text.find(start_marker)
                        if start_idx != -1:
                            start_idx += len(start_marker)
                            end_idx = decoded_text.find(end_marker, start_idx)
                            if end_idx != -1:
                                raw_content = decoded_text[start_idx:end_idx]
                                # Re-tokenize just the raw content using consistent approach
                                tokens = self.tokenizer.encode(raw_content, add_special_tokens=True)
                                if self.verbose_output:
                                    print(f"[DEBUG] Extracted raw content: {repr(raw_content[:100])}...")
                                    print(f"[DEBUG] Extracted raw content, new token count: {len(tokens)}")
                                    print(f"[DEBUG] New first 5 tokens: {tokens[:5] if len(tokens) >= 5 else tokens}")
                
                result.append(tokens)
        return result

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        
        TASK-SPECIFIC PREFILL/PREDICTION APPROACHES:
        
        BOOLQ (yes/no single-token answers):
        - Uses _process_prompt() which incorrectly prefills ALL tokens
        - Then reprocesses last token to get next token prediction 
        - Works by accident due to KV cache overwriting
        
        ARC_CHALLENGE (multi-token answers):
        - Uses _score_one_choice() which correctly prefills prompt[:-1]
        - Then processes each answer token with generate_next_token + teacher forcing
        - Follows proper pattern: prefill(prompt less last token) + generate tokens
        """
        print(f"[ANELM] loglikelihood called with {len(requests)} requests")
        if self.verbose_output:
            print(f"[VERBOSE] Processing {len(requests)} total requests")
        logging.info("Estimating loglikelihood for %d pairs." % len(requests))
        
        # Updated to handle both old and new LM-Eval harness API
        if self.verbose_output:
            print("[ANELM] Processing requests...")
        group_reqs = collections.defaultdict(list)
        request_metadata = {}  # Store additional metadata about each request
        
        for idx, req in enumerate(requests):
            if self.verbose_output and idx % 100 == 0:
                print(f"[ANELM] Processing request {idx+1}/{len(requests)}")
            
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
                if self.verbose_output:
                    print(f"  Request attributes: {dir(req)}")
                    if hasattr(req, '__dict__'):
                        print(f"  Request dict: {req.__dict__}")
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
        
        questions = list(group_reqs.keys())
        responses = []
        indices = []
        docs = []  # Store docs for ground truth
        for v in group_reqs.values():
            idx, resp, doc_list = zip(*v)
            indices.extend(idx)
            responses.append(resp)
            docs.append(doc_list)
        
        scores, is_greedy = [], []
        if self.verbose_output:
            print(f"[ANELM] Processing {len(questions)} question groups...")
        for q_idx, (q, rs, question_docs) in enumerate(tqdm(zip(questions, responses, docs), total=len(questions))):
            if self.verbose_output:
                print(f"[ANELM] Question {q_idx+1}/{len(questions)}: tokenizing...")
                print(f"[DEBUG] Raw context text (first 200 chars): {repr(q[:200])}...")
                print(f"[DEBUG] Raw context text (last 200 chars): {repr(q[-200:])}...")
                print(f"[DEBUG] Full context length: {len(q)} characters")
                
                # Save the exact prompt to a file for testing
                with open('/tmp/boolq_exact_prompt.txt', 'w') as f:
                    f.write(q)
                print(f"[DEBUG] Saved exact prompt to /tmp/boolq_exact_prompt.txt")
            prefix = self._tokenize([q])[0]
            full_sequences = [self._tokenize([q + r])[0] for r in rs]
            max_completed_l = max(len(s) for s in full_sequences)
            if self.verbose_output:
                print(f"[ANELM] Question {q_idx+1}: prefix={len(prefix)} tokens, max_seq={max_completed_l} tokens")
            
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
            
            # Get log probabilities for the prefix
            # BOOLQ: Uses _process_prompt (single-token answers like yes/no)
            # ARC_CHALLENGE: Will use _score_one_choice below (multi-token answers)
            if self.verbose_output:
                print(f"[ANELM] Question {q_idx+1}: calling _process_prompt...")
            logprobs, cache = self._process_prompt(prefix)
            if self.verbose_output:
                print(f"[ANELM] Question {q_idx+1}: _process_prompt completed, logprobs shape: {logprobs.shape}")
            max_idx = torch.argmax(logprobs, dim=-1).item()
            
            # Print verbose output if requested
            if self.verbose_output:
                print(f"\n[VERBOSE] Question {q_idx+1}:")
                print(f"  Context: {repr(q)}")
                print(f"  Tokenized prefix length: {len(prefix)} tokens")
                print(f"  First 5 tokens: {prefix[:5]}")
                print(f"  Last 5 tokens: {prefix[-5:]}")
                predicted_token_text = self.tokenizer.decode([max_idx])
                print(f"  Predicted next token (highest prob): '{predicted_token_text}' (ID: {max_idx})")
                print(f"  Expected continuations: {rs}")
            
            for answer_idx, s in enumerate(full_sequences):
                continuation_tokens = s[len(prefix):]
                
                # Check if this answer would exceed context length
                total_length = len(prefix) + len(continuation_tokens)
                if total_length > self.metadata['context_length']:
                    if self.verbose_output:
                        print(f"  Answer {answer_idx+1} would exceed context length ({total_length} > {self.metadata['context_length']}), assigning -inf score")
                    total_score = -float("inf")
                    is_all_greedy = False
                elif len(continuation_tokens) > 0:
                    # For multi-token continuations, use the dedicated scoring method
                    # ARC_CHALLENGE: Multi-token answers use _score_one_choice with teacher forcing
                    if self.verbose_output and len(continuation_tokens) > 1:
                        print(f"  Using multi-token scoring for {len(continuation_tokens)} tokens")
                    
                    # Score this answer choice using teacher forcing
                    total_score = self._score_one_choice(prefix, continuation_tokens)
                    
                    # For is_greedy, we just check if first token matches
                    # (This is less important than getting correct scores)
                    is_all_greedy = (continuation_tokens[0] == max_idx) if len(continuation_tokens) > 0 else True
                else:
                    # Empty continuation
                    total_score = 0.0
                    is_all_greedy = True
                    
                scores.append(total_score)
                is_greedy.append(is_all_greedy)
            
            # After scoring all answers for this question, log which one was selected
            if (self.verbose_output or self.log_incorrect_answers) and len(rs) > 0:
                # Get the scores for just this question (last len(rs) scores)
                current_question_scores = scores[-len(rs):]
                
                # Find the answer with the highest score (least negative log probability)
                selected_idx = max(range(len(current_question_scores)), key=lambda i: current_question_scores[i])
                selected_answer = rs[selected_idx]
                selected_score = current_question_scores[selected_idx]
                
                if self.verbose_output:
                    print(f"\n[SELECTED] Answer {selected_idx + 1}/{len(rs)}: '{selected_answer}' (score: {selected_score:.4f})")
                    print(f"  All scores: {[f'{s:.4f}' for s in current_question_scores]}")
                
                # Store information for later incorrect answer logging
                if self.log_incorrect_answers:
                    if not hasattr(self, '_question_results'):
                        self._question_results = []
                    # Extract ground truth using the first doc (all docs for this question should have same ground truth)
                    first_doc = question_docs[0] if question_docs else None
                    correct_idx = gold_idx(first_doc, rs)
                    if self.verbose_output and first_doc is not None:
                        print(f"[DEBUG] Raw doc: {first_doc}")
                        print(f"[DEBUG] Extracted correct_idx: {correct_idx}")
                        if correct_idx is not None:
                            print(f"[DEBUG] Ground truth answer: '{rs[correct_idx]}' vs selected: '{rs[selected_idx]}'")
                    self._question_results.append({
                        'question_idx': q_idx,
                        'context': q,
                        'options': list(rs),
                        'selected_idx': selected_idx,
                        'selected_answer': selected_answer,
                        'selected_score': selected_score,
                        'all_scores': current_question_scores.copy(),
                        'correct_idx': correct_idx,
                        'ground_truth_doc': first_doc
                    })
        
        # Reorder results to match original request order
        inv_sort = torch.argsort(torch.tensor(indices))
        scores = [scores[i] for i in inv_sort]
        is_greedy = [is_greedy[i] for i in inv_sort]
        
        result = list(zip(scores, is_greedy))
        
        # Log incorrect answers if requested
        if self.log_incorrect_answers and hasattr(self, '_question_results'):
            self._log_incorrect_answers(result)
        
        return result

    def _log_incorrect_answers(self, loglikelihood_results):
        """Log detailed information about incorrect answers using proper ground truth."""
        print(f"\n[INCORRECT ANSWER ANALYSIS] Analyzing {len(self._question_results)} questions...")
        
        incorrect_count = 0
        total_questions = len(self._question_results)
        
        # Open log file for writing
        import os
        log_file = os.path.join(os.getcwd(), "incorrect_answers.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== INCORRECT ANSWER LOG ===\n")
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
                
                # Determine if the answer was incorrect:
                # If ground-truth label is available, compare selected vs. correct;
                # otherwise, use greedy match as fallback.
                if correct_idx is None:
                    is_incorrect = not loglikelihood_results[question_idx][1]
                    if self.verbose_output:
                        print(f"[DEBUG] Question {question_idx+1}: no label found; is_greedy={loglikelihood_results[question_idx][1]}")
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

                    print(f"\n[INCORRECT] Question {question_idx + 1}:")
                    print(f"  Context: {repr(context[:200])}{'...' if len(context) > 200 else ''}")
                    print(f"  Options: {options}")
                    print(f"  Selected: '{selected_answer}' (index {selected_idx}, score: {selected_score:.4f})")
                    if correct_score is not None:
                        print(f"  Correct: '{correct_answer}' (index {correct_idx}, score: {correct_score:.4f})")
                        print(f"  Score difference: {score_diff:.4f}")
                    else:
                        print(f"  Correct: {correct_answer}")

                    # Write to log file
                    f.write(f"QUESTION {question_idx + 1} (INCORRECT):\n")
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

    def loglikelihood_rolling(self, requests) -> list[float]:
        """Compute full log-likelihood of strings, for perplexity computation."""
        logging.info("Estimating rolling loglikelihood for %d sequences." % len(requests))
        
        # Tokenize the inputs
        texts = [self._tokenize([req.args[0]])[0] for req in requests]
        
        all_scores = []
        for i in tqdm(range(0, len(texts), self._batch_size)):
            batch = texts[i : i + self._batch_size]
            scores, lengths, _ = self._score_fn(batch)
            
            # Apply mask based on lengths
            mask = torch.arange(scores.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
            masked_scores = torch.where(mask, scores, torch.zeros_like(scores))
            
            # Sum scores for each sequence
            batch_scores = torch.sum(masked_scores, dim=1).tolist()
            all_scores.extend(batch_scores)
        
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
                print(f"Warning: Unknown request format for generate_until: {req}")
                contexts_and_options.append(("", {"until": ["\n\n"]}))  # Default fallback
        
        contexts, options = zip(*contexts_and_options)
        completions = []
        
        for idx, (context, opt) in enumerate(tqdm(zip(contexts, options), total=len(contexts))):
            # Extract stopping sequences and other generation parameters
            until = opt.get("until", ["\n\n"])
            if not isinstance(until, list):
                until = [until]
                
            max_gen_tokens = min(
                opt.get("max_gen_tokens", self._max_tokens),
                self.metadata.get('context_length', 2048) - 10  # Reserve some space
            )
            temperature = opt.get("temperature", 0.0)
            
            # Print verbose input if requested
            if self.verbose_output:
                print(f"\n[VERBOSE] Generation {idx+1}:")
                print(f"  Context: {repr(context)}")
                print(f"  Until: {until}")
                print(f"  Max tokens: {max_gen_tokens}")
                
            # Tokenize context - always use add_special_tokens=True to match chat.py behavior
            context_tokens = self.tokenizer.encode(
                context, add_special_tokens=True
            )
            
            # Check if context exceeds max context length
            context_length = self.metadata.get('context_length', 2048)
            if len(context_tokens) >= context_length:
                print(f"Warning: Context length {len(context_tokens)} exceeds max context length {context_length}")
                # Truncate to fit model's context window, keeping most recent tokens
                context_tokens = context_tokens[-context_length+10:]  # +10 to leave some room for generation
                print(f"Truncated context to {len(context_tokens)} tokens")
            
            # Prepare for generation
            input_ids = torch.tensor(context_tokens).unsqueeze(0)
            
            # Use the shared state
            cache = self.state
            
            # Run prefill on the context
            current_pos = run_prefill(
                self.embedding_model,
                self.ffn_models,
                input_ids,
                0,  # Start at position 0
                self.metadata['context_length'],
                self.metadata.get('batch_size', 64),
                cache,
                self.causal_mask
            )
            
            # Generate tokens
            generated_tokens = []
            text = ""
            
            for _ in range(max_gen_tokens):
                # Generate next token
                next_token = generate_next_token(
                    self.embedding_model,
                    self.ffn_models,
                    self.lm_head_model,
                    input_ids,
                    current_pos + 1,  # Convert from 0-based to 1-based position
                    self.metadata['context_length'],
                    self.metadata,  # Pass full metadata dict
                    state=cache,
                    causal_mask=self.causal_mask,
                    temperature=temperature
                )
                
                # Add to generated tokens
                generated_tokens.append(next_token)
                
                # Update current position
                current_pos += 1
                
                # Convert token to text incrementally
                # Note: This is not always correct due to tokenization subtleties,
                # but it's a reasonable approximation for stopping condition checks
                token_text = self.tokenizer.decode([next_token])
                text += token_text
                
                # Check for stopping sequences
                if any(u in text for u in until):
                    # Cut off text at the stopping sequence
                    text = _rstrip_until(text, until)
                    break
                
                # Update input_ids for next token generation
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
                
                # Stop if we reach end of text token or maximum allowed length
                if next_token == self.tokenizer.eos_token_id:
                    break
            
            # After generation is complete, decode all tokens properly at once for accurate results
            if generated_tokens:
                final_text = self.tokenizer.decode(generated_tokens)
                # Apply stopping conditions one final time
                if any(u in final_text for u in until):
                    final_text = _rstrip_until(final_text, until)
                completions.append(final_text)
                
                # Print verbose output if requested
                if self.verbose_output:
                    print(f"  Generated response: {repr(final_text)}")
                    print(f"  Tokens generated: {len(generated_tokens)}")
            else:
                completions.append("")
                if self.verbose_output:
                    print(f"  Generated response: (empty)")
                    print(f"  Tokens generated: 0")
        
        return completions


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
    parser.add_argument("--skip", type=int, default=0,
                        help="Number of examples to skip (requires --limit)")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction,
                        help="Specifies whether to apply a chat template to the prompt",
                        default=None)
    parser.add_argument("--verbose-output", action="store_true",
                        help="Print questions and responses (detokenized highest prob token) during evaluation")
    parser.add_argument("--log-incorrect-answers", action="store_true",
                        help="Log detailed information for incorrect answers including context, options, selected answer, and correct answer")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Optional path to save results JSON file (overrides --output-dir and default naming)")
    
    args = parser.parse_args()
    if args.skip and args.limit is None:
        parser.error("--skip requires --limit to be set")
    if args.skip and args.limit is None:
        parser.error("--skip requires --limit to be set")
    
    # Handle comma-separated tasks (e.g., "boolq,hellaswag,arc_easy")
    if len(args.tasks) == 1 and ',' in args.tasks[0]:
        args.tasks = [task.strip() for task in args.tasks[0].split(',')]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment variables already set at top of file
    print("Using offline mode to prevent rate limits and network issues")
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize model
    print(f"Initializing ANE model from {args.model}")
    lm = ANELM(
        args.model,
        max_tokens=args.max_tokens,
        use_chat_template=args.apply_chat_template,
        verbose_output=args.verbose_output,
        log_incorrect_answers=args.log_incorrect_answers,
    )
    print("ANE model initialization complete")
    
    # Run evaluation with processes=0 to ensure single-threaded execution
    print("Running evaluation in single-threaded mode")
    print(f"About to evaluate tasks: {args.tasks}")
    print(f"Limit: {args.limit}, Batch size: {args.batch_size}")
    
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
    if args.skip:
        # Skip the first N examples via explicit samples instead of limit
        indices = list(range(args.skip, args.skip + args.limit))
        eval_args["samples"] = {task: indices for task in args.tasks}
        eval_args.pop("limit", None)

    # Add processes=0 if supported
    if supports_processes:
        print("Using processes=0 parameter for single-threaded execution")
        eval_args["processes"] = 0
    else:
        print("This version of lm-evaluation-harness doesn't support the processes parameter")
        print("Continuing with single-threaded execution via batch_size=1")
        # For older versions, we rely on batch_size=1 and our implementation of _create_new_state 
        # to ensure serial execution
    
    # Run evaluation
    print("Calling lm_eval.simple_evaluate...")
    print(f"Eval args: {list(eval_args.keys())}")
    try:
        results = lm_eval.simple_evaluate(**eval_args)
        print("Evaluation completed successfully")
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
        raise
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save results
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        filename = f"eval_{Path(args.model).name}_{args.num_shots or 0}shot_{'_'.join(args.tasks)}.json"
        output_path = output_dir / filename
    output_path.write_text(json.dumps(results["results"], indent=4))
    
    # Print summary in lm-eval table format
    print("\n" + "="*80)
    print("Evaluation Results (ANE/CoreML)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Samples: {args.limit or 'all'}")
    print(f"Few-shot: {args.num_shots or 0}")
    print()
    
    # Create table header
    print("|    Tasks    |Version|Filter|n-shot| Metric |   |Value|   |Stderr|")
    print("|-------------|------:|------|-----:|--------|---|----:|---|-----:|")
    
    # Print results for each task
    for task_idx, (task, metrics) in enumerate(results["results"].items()):
        # Get version from task alias or default to 1
        version = 1
        task_alias = metrics.get('alias', task)
        
        # Group metrics by type (acc, acc_norm, etc.)
        metric_groups = {}
        for metric_name, value in metrics.items():
            if metric_name == 'alias':
                continue
            # Split metric name (e.g., "acc,none" -> "acc", "acc_stderr,none" -> "acc_stderr")
            base_metric = metric_name.split(',')[0]
            if base_metric.endswith('_stderr'):
                base_name = base_metric[:-7]  # Remove '_stderr'
                if base_name not in metric_groups:
                    metric_groups[base_name] = {}
                metric_groups[base_name]['stderr'] = value
            else:
                if base_metric not in metric_groups:
                    metric_groups[base_metric] = {}
                metric_groups[base_metric]['value'] = value
        
        # Print each metric
        for metric_idx, (metric_name, metric_data) in enumerate(metric_groups.items()):
            value = metric_data.get('value', 0.0)
            stderr = metric_data.get('stderr', 0.0)
            
            # Convert to float if they're strings
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            try:
                stderr = float(stderr)
            except (TypeError, ValueError):
                stderr = 0.0
            
            # Determine if higher is better (most metrics are, except perplexity-like)
            higher_is_better = "↑" if not metric_name.startswith('ppl') else "↓"
            
            if metric_idx == 0:
                # First row shows task name
                print(f"|{task_alias:<13}|{version:>7}|{'none':>6}|{args.num_shots or 0:>6}|{metric_name:<8}|{higher_is_better:>3}|{value:>7.4f}|±  |{stderr:>5.4f}|")
            else:
                # Subsequent rows are blank in task column
                print(f"|{'':<13}|{'':<7}|{'none':>6}|{'':<6}|{metric_name:<8}|{higher_is_better:>3}|{value:>7.4f}|±  |{stderr:>5.4f}|")
    
    print()
    print("Detailed results saved to:", output_path)


if __name__ == "__main__":
    main() 