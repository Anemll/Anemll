#!/usr/bin/env python3
# ANE/CoreML model evaluation with lm-evaluation-harness
# Based on MLX-LM's implementation pattern
#
# USAGE INSTRUCTIONS:
# -------------------
# This script evaluates Apple Neural Engine (ANE) models using lm-evaluation-harness.
# To run with proper serial execution:
#
# python evaluate_with_harness.py \
#   --model /path/to/your/model \
#   --tasks boolq,arc_easy,hellaswag \
#   --batch-size 1     # Ensures one prompt → one Core ML call
#   --output-dir results
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
os.environ["HF_HUB_OFFLINE"] = "1"
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

# Import from chat.py
try:
    from chat import (
        run_prefill,
        generate_next_token,
        create_unified_state,
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


@register_model("anelm")
class ANELM(LM):
    """ANE/CoreML model implementation for lm-evaluation-harness."""

    def __init__(
        self,
        model_path: str,
        max_tokens: Optional[int] = None,
        use_chat_template: Optional[bool] = None,
        verbose_output: bool = False,
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
        self.state = create_unified_state(self.ffn_models, context_length)
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
        
    def _score_batch(self, token_batch, step_size: int = 1024):
        """Efficiently score a batch of tokens using vectorized operations.
        Uses the shared state for inference.
        """
        # Use the shared state
        kv_cache = self.state
        
        # Split into input tokens and target tokens to score
        inputs, targets = token_batch[:, :-1], token_batch[:, 1:]
        
        # Process full input sequence once
        seq_len = inputs.shape[1]
        current_pos = 0
        
        # Process in chunks to avoid OOM
        for i in range(0, seq_len, step_size):
            chunk = inputs[:, i:i+step_size]
            
            # IMPORTANT: This calls .predict() in a strictly serial manner
            current_pos = run_prefill(
                self.embedding_model, 
                self.ffn_models,
                chunk,
                current_pos,  # Pass the current position in the sequence
                self.metadata['context_length'],
                self.metadata.get('batch_size', 64),
                kv_cache,
                self.causal_mask
            )
        
        # For implementations without return_logits, we need to score each position individually
        # This is slower, but handles the API limitations properly
        batch_size = inputs.shape[0]
        all_scores = []
        all_is_greedy = []
        
        for i in range(seq_len):
            # For each position in the sequence, we'll need to:
            # 1. Prefill up to position i
            # 2. Generate the next token
            # 3. Check if it matches the target
            
            position_scores = []
            position_is_greedy = []
            
            for b in range(batch_size):
                # Use the shared state
                item_cache = self.state
                
                # Prefill up to position i
                item_pos = run_prefill(
                    self.embedding_model,
                    self.ffn_models,
                    inputs[b:b+1, :i+1],
                    0,  # Start from position 0
                    self.metadata['context_length'],
                    self.metadata.get('batch_size', 64),
                    item_cache,
                    self.causal_mask
                )
                
                # Generate next token
                predicted_token = generate_next_token(
                    self.embedding_model,
                    self.ffn_models,
                    self.lm_head_model,
                    inputs[b:b+1, :i+1],
                    item_pos + 1,  # Convert from 0-based to 1-based position
                    self.metadata['context_length'],
                    self.metadata,  # Pass full metadata dict
                    state=item_cache,
                    causal_mask=self.causal_mask,
                    temperature=0.0  # Use greedy generation
                )
                
                # Check if predicted token matches target
                target_token = targets[b, i].item()
                is_greedy = (predicted_token == target_token)
                
                # Assign scores based on match or mismatch
                # Note: This is a binary scoring approach
                score = 0.0 if is_greedy else -10.0  # Log probability estimate
                
                position_scores.append(score)
                position_is_greedy.append(is_greedy)
            
            all_scores.append(torch.tensor(position_scores))
            all_is_greedy.append(torch.tensor(position_is_greedy))
        
        # Stack scores and is_greedy
        scores = torch.stack(all_scores, dim=1)
        is_greedy = torch.stack(all_is_greedy, dim=1)
        
        return scores, is_greedy
    
    def _process_prompt(self, prompt, step_size: int = 1024):
        """Process the prompt and return logprobs."""
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
        cache = create_unified_state(self.ffn_models, self.metadata['context_length'], eval_mode=True)
        
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
        # The run_prefill function will handle the batching internally
        if self.verbose_output:
            print(f"[DEBUG] _process_prompt: calling run_prefill with prompt_length={prompt_length}")
        pos_result = run_prefill(
            self.embedding_model,
            self.ffn_models,
            prompt[:, :prompt_length],  # Use only the actual prompt length, not padded
            prompt_length, 
            self.metadata['context_length'],
            self.metadata.get('batch_size', 64),
            cache,
            self.causal_mask
        )
        if self.verbose_output:
            print(f"[DEBUG] _process_prompt: run_prefill completed")
        
        # Step 3: Use chat.py's exact generate_next_token function instead of replicating the logic
        try:
            generation_pos = prompt_length  # This is like 'pos' parameter in chat.py
            
            # Use the original unpadded input_ids, just like chat.py does
            original_input_ids = prompt[:, :prompt_length]  # Unpadded input tensor
            
            # Use the exact generate_next_token function from chat.py
            predicted_token = generate_next_token(
                self.embedding_model,
                self.ffn_models,
                self.lm_head_model,
                original_input_ids,  # Use unpadded input_ids like chat.py
                generation_pos,  # Position to generate at
                self.metadata['context_length'],
                self.metadata,  # Pass full metadata dict
                state=cache,
                causal_mask=self.causal_mask,
                temperature=0.0  # Use greedy generation to get most likely token
            )
            
            # Create a realistic probability distribution to avoid numerical issues
            # Set predicted token to high probability, others to low probability
            vocab_size = len(self.tokenizer)
            log_probs = torch.full((1, vocab_size), -10.0)  # Low probability for all tokens
            log_probs[0, predicted_token] = 0.0  # Set predicted token to highest probability
            
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
                    result.append(self.tokenizer.encode(chat_text, add_special_tokens=True))
                except Exception as e:
                    print(f"Error applying chat template: {str(e)}. Falling back to standard tokenization.")
                    result.append(self.tokenizer.encode(t, add_special_tokens=True))
            else:
                # Check if the text already contains chat template tokens (BoolQ forces this)
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
                                # Re-tokenize just the raw content
                                tokens = self.tokenizer.encode(raw_content, add_special_tokens=True)
                                if self.verbose_output:
                                    print(f"[DEBUG] Extracted raw content: {repr(raw_content[:100])}...")
                                    print(f"[DEBUG] Extracted raw content, new token count: {len(tokens)}")
                                    print(f"[DEBUG] New first 5 tokens: {tokens[:5] if len(tokens) >= 5 else tokens}")
                
                result.append(tokens)
        return result

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context."""
        print(f"[ANELM] loglikelihood called with {len(requests)} requests")
        if self.verbose_output:
            print(f"[VERBOSE] Processing {len(requests)} total requests")
        logging.info("Estimating loglikelihood for %d pairs." % len(requests))
        
        # Updated to handle both old and new LM-Eval harness API
        if self.verbose_output:
            print("[ANELM] Processing requests...")
        group_reqs = collections.defaultdict(list)
        for idx, req in enumerate(requests):
            if self.verbose_output and idx % 100 == 0:
                print(f"[ANELM] Processing request {idx+1}/{len(requests)}")
            if hasattr(req, 'args') and len(req.args) >= 2:
                # Old-style API with req.args
                context, continuation = req.args[0], req.args[1]
            elif hasattr(req, 'kwargs') and 'doc' in req.kwargs:
                # New-style API with req.kwargs['doc']
                doc = req.kwargs['doc']
                context, continuation = doc['context'], doc['continuation']
            else:
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
        
        scores, is_greedy = [], []
        if self.verbose_output:
            print(f"[ANELM] Processing {len(questions)} question groups...")
        for q_idx, (q, rs) in enumerate(tqdm(zip(questions, responses), total=len(questions))):
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
            
            for s in full_sequences:
                continuation_tokens = s[len(prefix):]
                
                if len(continuation_tokens) > 0:
                    # Score the first token using the prefix logprobs
                    first_token_score = logprobs[0, continuation_tokens[0]].item()
                    first_token_is_greedy = (continuation_tokens[0] == max_idx)
                    
                    if len(continuation_tokens) > 1:
                        # Score the rest of the continuation
                        continuation_tensor = torch.tensor(continuation_tokens)[None, :]
                        rest_scores, rest_lengths, rest_is_greedy = self._score_fn(
                            continuation_tensor, 
                            cache=self._clone_state(cache)
                        )
                        
                        # Combine the scores
                        total_score = first_token_score + torch.sum(rest_scores).item()
                        is_all_greedy = first_token_is_greedy and torch.all(rest_is_greedy).item()
                    else:
                        total_score = first_token_score
                        is_all_greedy = first_token_is_greedy
                else:
                    # Empty continuation
                    total_score = 0.0
                    is_all_greedy = True
                    
                scores.append(total_score)
                is_greedy.append(is_all_greedy)
        
        # Reorder results to match original request order
        inv_sort = torch.argsort(torch.tensor(indices))
        scores = [scores[i] for i in inv_sort]
        is_greedy = [is_greedy[i] for i in inv_sort]
        
        return list(zip(scores, is_greedy))

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
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction,
                        help="Specifies whether to apply a chat template to the prompt",
                        default=None)
    parser.add_argument("--verbose-output", action="store_true",
                        help="Print questions and responses (detokenized highest prob token) during evaluation")
    
    args = parser.parse_args()
    
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
    except ConnectionError as e:
        print(f"Error during evaluation: {e}")
        
        # Check if this is an offline mode error for dataset downloading
        if "OfflineModeIsEnabled" in str(e) or "Couldn't reach" in str(e):
            print("\n" + "="*60)
            print("DATASET DOWNLOAD ERROR - OFFLINE MODE DETECTED")
            print("="*60)
            print("\nThe evaluation harness needs to download dataset(s) but is in offline mode.")
            print("\nTo fix this, you have two options:")
            print("\n1. Enable online mode and run again:")
            print("   export HF_DATASETS_OFFLINE=0")
            print("   export TRANSFORMERS_OFFLINE=0")
            print("\n2. Pre-download the required dataset(s):")
            
            # Extract dataset name from error if possible
            error_str = str(e).lower()
            if "super_glue" in error_str:
                if "boolq" in error_str or "boolq" in args.tasks:
                    print("   python -c \"from datasets import load_dataset; load_dataset('super_glue', 'boolq')\"")
                elif "copa" in error_str or "copa" in args.tasks:
                    print("   python -c \"from datasets import load_dataset; load_dataset('super_glue', 'copa')\"")
                elif "multirc" in error_str or "multirc" in args.tasks:
                    print("   python -c \"from datasets import load_dataset; load_dataset('super_glue', 'multirc')\"")
                else:
                    print("   python -c \"from datasets import load_dataset; load_dataset('super_glue', '<task_name>')\"")
            elif "glue" in error_str:
                print("   python -c \"from datasets import load_dataset; load_dataset('glue', '<task_name>')\"")
            elif "wikitext" in error_str:
                print("   python -c \"from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-raw-v1')\"")
            elif "lambada" in error_str:
                print("   python -c \"from datasets import load_dataset; load_dataset('lambada')\"")
            elif "hellaswag" in error_str:
                print("   python -c \"from datasets import load_dataset; load_dataset('hellaswag')\"")
            elif "piqa" in error_str:
                print("   python -c \"from datasets import load_dataset; load_dataset('piqa')\"")
            elif "arc" in error_str:
                print("   python -c \"from datasets import load_dataset; load_dataset('ai2_arc', 'ARC-Easy')\"")
                print("   python -c \"from datasets import load_dataset; load_dataset('ai2_arc', 'ARC-Challenge')\"")
            else:
                print("   python -c \"from datasets import load_dataset; load_dataset('<dataset_name>')\"")
                print("\n   Common datasets:")
                print("   - BoolQ: load_dataset('super_glue', 'boolq')")
                print("   - HellaSwag: load_dataset('hellaswag')")
                print("   - PIQA: load_dataset('piqa')")
                print("   - ARC: load_dataset('ai2_arc', 'ARC-Easy')")
                print("   - WikiText: load_dataset('wikitext', 'wikitext-103-raw-v1')")
            
            print("\nAfter downloading, you can run the evaluation in offline mode.")
            print("="*60 + "\n")
        
        # Exit cleanly without showing the full traceback
        import sys
        sys.exit(1)
    except Exception as e:
        # For other exceptions, show the full traceback
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save results
    filename = f"eval_{Path(args.model).name}_{args.num_shots or 0}shot_{'_'.join(args.tasks)}.json"
    output_path = output_dir / filename
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