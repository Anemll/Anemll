# chat.py
#!/usr/bin/env python3
# chat.py
# Copyright (c) 2025 Anemll
# Licensed under the MIT License

import argparse
import os
import re
import glob
from pathlib import Path
import json
import sys
import coremltools as ct
from transformers import LlamaTokenizer, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import queue
import threading
import time
import yaml
import sys
import resource


def _get_rss_mb() -> float:
    """Best-effort RSS in MB (macOS/Linux)."""
    try:
        # On macOS ru_maxrss is bytes; on Linux it is kilobytes.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > 10_000_000:  # heuristic: likely bytes
            return rss / (1024 * 1024)
        return rss / 1024  # kB -> MB
    except Exception:
        return -1.0


def _maybe_report_mem(label: str, enabled: bool):
    if not enabled:
        return
    rss_mb = _get_rss_mb()
    if rss_mb >= 0:
        print(f"[mem] {label}: rss≈{rss_mb:.1f} MB")

# ANSI color codes
LIGHT_BLUE = "\033[94m"
DARK_BLUE = "\033[34m"
LIGHT_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# Add at top with other constants
WARMUP_TOKEN_LIMIT = 10  # Maximum tokens to generate during warmup

class TokenPrinter:
    """Handles background printing of generated tokens."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = None
        self.buffer = ""
        self.lock = threading.Lock()
        self.thinking = True  # Track if we're still in thinking mode
        self.decoding_buffer = []  # Buffer for token IDs
        self.all_token_ids = []   # Full history for correct SentencePiece decode
        self.prev_text = ""       # Previously decoded text (for diffing)
        # Add token counting and timing
        self.start_time = time.time()
        self.token_count = 0
        self.start()

    def start(self):
        """Start the printer thread."""
        if self.thread is None:
            self.thread = threading.Thread(target=self._print_worker)
            self.thread.daemon = True
            self.thread.start()

    def add_token(self, token_id):
        """Add a token to the print queue."""
        if not self.stop_event.is_set():
            self.token_queue.put(token_id)
            self.token_count += 1

    def drain_buffer(self, eval_mode=False):
        """Decode token IDs from decoding_buffer in the main thread.

        Uses full-sequence decode + diff to preserve SentencePiece spaces.
        Decoding tokens one-at-a-time strips the leading ▁ (space) that
        SentencePiece encodes into most word-initial tokens.
        """
        if not self.decoding_buffer:
            return

        # Move new tokens into the full history
        with self.lock:
            self.all_token_ids.extend(self.decoding_buffer)
            self.decoding_buffer.clear()

        # Decode the FULL token sequence and diff against previous decode
        try:
            full_text = self.tokenizer.decode(self.all_token_ids)
        except Exception:
            # Fallback: decode only the new tokens (may lose spaces, but won't crash)
            try:
                full_text = self.prev_text + self.tokenizer.decode(
                    self.all_token_ids[len(self.all_token_ids) - 1:]
                )
            except Exception:
                return

        token_str = full_text[len(self.prev_text):]
        self.prev_text = full_text

        if not token_str:
            return

        # Store the text in buffer for later saving to file
        with self.lock:
            self.buffer += token_str

        # Skip printing in eval mode
        if eval_mode:
            return

        # Color-handling logic
        if self.thinking and "</think>" in token_str:
            self.thinking = False
            parts = token_str.split("</think>")
            if len(parts) > 0:
                print(parts[0] + "</think>", end='', flush=True)
                if len(parts) > 1:
                    print(LIGHT_BLUE + parts[1], end='', flush=True)
        else:
            if not self.thinking:
                print(LIGHT_BLUE + token_str, end='', flush=True)
            else:
                print(token_str, end='', flush=True)

    def _print_worker(self):
        """Worker thread that takes token_ids from the queue."""
        while not self.stop_event.is_set():
            try:
                token_id = self.token_queue.get(timeout=0.01)
                with self.lock:
                    self.decoding_buffer.append(token_id)
                self.token_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nError: Token printer error: {str(e)}")
                break

    def stop(self, eval_mode=False):
        """Stop the printer thread."""
        if self.thread and self.thread.is_alive():
            # Ensure any remaining tokens are processed
            self.drain_buffer()
            self.stop_event.set()
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass
            # Calculate and print tokens/s with shorter format in blue (unless in eval mode)
            if not eval_mode:
                elapsed = time.time() - self.start_time
                if elapsed > 0 and self.token_count > 0:
                    tokens_per_sec = self.token_count / elapsed
                    print(f"\n{DARK_BLUE}{tokens_per_sec:.1f} t/s{RESET_COLOR}")
                else:
                    print(RESET_COLOR)  # Reset color at the end
        return self.buffer

def parse_model_path(path):
    """Parse model path and return full path with .mlmodelc or .mlpackage extension."""
    path = Path(path)
    
    # If path exists exactly as specified, return it
    if path.exists():
        return str(path)
        
    # Try with both extensions
    candidates = [
        path,  # Original path
        path.with_suffix('.mlmodelc'),  # With .mlmodelc
        path.with_suffix('.mlpackage'),  # With .mlpackage
        Path(str(path) + '.mlmodelc'),  # Handle case where extension is included
        Path(str(path) + '.mlpackage')
    ]
    
    # Try all possible paths
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    
    # If embeddings with LUT suffix not found, try without LUT suffix
    if "_lut" in str(path) and "embeddings" in str(path):
        print(f"Failed to find {path}, trying without LUT suffix...")
        # Remove LUT suffix
        path_no_lut = str(path).split("_lut")[0]
        path_no_lut = Path(path_no_lut)
        
        # Try candidates without LUT suffix
        candidates_no_lut = [
            path_no_lut,
            path_no_lut.with_suffix('.mlmodelc'),
            path_no_lut.with_suffix('.mlpackage'),
            Path(str(path_no_lut) + '.mlmodelc'),
            Path(str(path_no_lut) + '.mlpackage')
        ]
        
        for candidate in candidates_no_lut:
            if candidate.exists():
                return str(candidate)
        
        # Add no-LUT candidates to the list for error reporting
        candidates.extend(candidates_no_lut)

    # If FFN path isn't chunked, try to find chunked variants.
    path_str = str(path)
    base_str = str(path.with_suffix('')) if path.suffix in ('.mlmodelc', '.mlpackage') else path_str
    if "_chunk_" not in base_str:
        chunk_pattern = f"{base_str}_chunk_*of*"
        chunk_candidates = sorted(glob.glob(chunk_pattern + ".mlmodelc"))
        if not chunk_candidates:
            chunk_candidates = sorted(glob.glob(chunk_pattern + ".mlpackage"))
        if chunk_candidates:
            return str(Path(chunk_candidates[0]))
        candidates.extend([Path(p) for p in sorted(glob.glob(chunk_pattern + ".mlmodelc"))])
        candidates.extend([Path(p) for p in sorted(glob.glob(chunk_pattern + ".mlpackage"))])
            
    # If we get here, no valid path was found
    print("\nError: Model not found. Tried following paths:")
    for candidate in candidates:
        print(f"  {candidate}")
    raise FileNotFoundError(f"Model not found: {path}")

def build_stop_token_ids(tokenizer):
    """Collect token IDs that should stop generation."""
    def _get_token_id_if_present(token_str):
        if not token_str:
            return None
        # Avoid calling get_vocab() as it can segfault on some tokenizers (e.g., Gemma)
        # Use convert_tokens_to_ids() directly instead
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(token_id, list):
                if len(token_id) == 1:
                    token_id = token_id[0]
                else:
                    return None
            if token_id is None:
                return None
            if tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id:
                return None
            return token_id
        except Exception:
            return None

    stop_ids = set()
    eos_token_ids = tokenizer.eos_token_id
    if isinstance(eos_token_ids, list):
        stop_ids.update(eos_token_ids)
    elif eos_token_ids is not None:
        stop_ids.add(eos_token_ids)

    for token_str in ("<|endoftext|>", "<end_of_turn>", "<|eot_id|>"):
        token_id = _get_token_id_if_present(token_str)
        if token_id is not None:
            stop_ids.add(token_id)

    return stop_ids


def _parse_lm_head_chunk_sizes(raw_sizes):
    """Parse lm_head_chunk_sizes from metadata (list or comma-separated string)."""
    if raw_sizes is None:
        return None
    if isinstance(raw_sizes, (list, tuple)):
        sizes = []
        for x in raw_sizes:
            try:
                v = int(x)
                if v > 0:
                    sizes.append(v)
            except Exception:
                return None
        return sizes if sizes else None
    if isinstance(raw_sizes, str):
        raw_sizes = raw_sizes.strip()
        if not raw_sizes:
            return None
        # Accept either "1,2,3" or "[1, 2, 3]"
        raw_sizes = raw_sizes.strip("[]")
        parts = [p.strip() for p in raw_sizes.split(",") if p.strip()]
        if not parts:
            return None
        try:
            sizes = [int(p) for p in parts]
        except Exception:
            return None
        return sizes if all(v > 0 for v in sizes) else None
    return None


def _resolve_argmax_chunk_layout(metadata, num_chunks, argmax_idx_flat=None):
    """Resolve per-chunk sizes and offsets for argmax local->global index mapping."""
    # Preferred: exact chunk sizes from metadata.
    chunk_sizes = _parse_lm_head_chunk_sizes(metadata.get("lm_head_chunk_sizes"))
    if chunk_sizes and len(chunk_sizes) == num_chunks and all(v > 0 for v in chunk_sizes):
        offsets = [0]
        for size in chunk_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        return chunk_sizes, offsets

    # Next-best: derive from vocab_size + split distribution used by converters.
    vocab_size = metadata.get("vocab_size")
    try:
        vocab_size = int(vocab_size) if vocab_size is not None else None
    except Exception:
        vocab_size = None
    if vocab_size and vocab_size > 0 and num_chunks > 0:
        base, rem = divmod(vocab_size, num_chunks)
        chunk_sizes = [base + (1 if i < rem else 0) for i in range(num_chunks)]
        offsets = [0]
        for size in chunk_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        return chunk_sizes, offsets

    # Fallback defaults by architecture.
    arch = str(metadata.get("architecture", "")).lower()
    if arch.startswith("gemma"):
        fallback_vocab = 262144
    elif arch.startswith("qwen"):
        fallback_vocab = 151936
    elif arch.startswith("llama"):
        fallback_vocab = 128257
    else:
        fallback_vocab = None
    if fallback_vocab and num_chunks > 0:
        base, rem = divmod(fallback_vocab, num_chunks)
        chunk_sizes = [base + (1 if i < rem else 0) for i in range(num_chunks)]
        offsets = [0]
        for size in chunk_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        return chunk_sizes, offsets

    # Last resort: infer a minimum width from observed local indices.
    inferred = 1
    if argmax_idx_flat is not None and len(argmax_idx_flat) > 0:
        try:
            inferred = max(int(np.max(argmax_idx_flat)) + 1, 1)
        except Exception:
            inferred = 1
    chunk_sizes = [inferred] * max(num_chunks, 1)
    offsets = [i * inferred for i in range(max(num_chunks, 1))]
    return chunk_sizes, offsets

def parse_ffn_filename(path):
    """Parse FFN model filename to extract chunk information."""
    path = Path(path)
    # Support multiple naming conventions:
    # - FFN_PF_lut6_chunk_01of04 (legacy/prefill style)
    # - gemma3_1b_FFN_lut6_chunk_01of04 (new Gemma3 style)
    # - any_prefix_FFN_*_chunk_NNofNN
    pattern = r'FFN[^/]*_chunk_(\d+)of(\d+)'
    match = re.search(pattern, path.name)

    if match:
        current_chunk = int(match.group(1))
        total_chunks = int(match.group(2))
        return current_chunk, total_chunks
    return None, None

def find_all_chunks(base_path):
    """Find all chunk files matching the base FFN path pattern."""
    path = Path(base_path)
    pattern = re.sub(r'_chunk_\d+of\d+', '_chunk_*', str(path))
    return sorted(glob.glob(pattern))

def load_model(path, function_name=None, compute_unit=None):
    """Load a CoreML model, handling both .mlmodelc and .mlpackage formats.

    For single-function compiled models, function_name is ignored.
    For multi-function ML Program models, function_name selects the function.
    """
    def _available_functions(model):
        if not hasattr(model, "get_spec"):
            return []
        spec = model.get_spec()
        functions = []
        mlprog = getattr(spec, "mlProgram", None)
        if mlprog is not None and hasattr(mlprog, "functions"):
            try:
                functions.extend(list(mlprog.functions.keys()))
            except Exception:
                pass
        desc = getattr(spec, "description", None)
        if desc is not None and hasattr(desc, "functions"):
            try:
                functions.extend([f.name for f in desc.functions])
            except Exception:
                pass
        seen = set()
        ordered = []
        for fn in functions:
            if fn not in seen:
                seen.add(fn)
                ordered.append(fn)
        return ordered

    def _check_function_exists(path_obj, fn_name, cu):
        try:
            if path_obj.suffix == '.mlmodelc':
                probe = ct.models.CompiledMLModel(str(path_obj), cu)
            else:
                probe = ct.models.MLModel(str(path_obj), compute_units=cu)
        except Exception:
            return  # If we can't probe, let the real load error surface.
        available = _available_functions(probe)
        if available and fn_name not in available:
            raise RuntimeError(
                f"Model is missing '{fn_name}' function. "
                f"Available: {', '.join(available)}\n"
                f"Model path: {path_obj}"
            )

    path = Path(path)
    if compute_unit is None:
        compute_unit = ct.ComputeUnit.CPU_AND_NE

    try:
        if path.suffix == '.mlmodelc':
            # For compiled models (.mlmodelc), use CompiledMLModel
            if function_name:
                _check_function_exists(path, function_name, compute_unit)
                try:
                    return ct.models.CompiledMLModel(str(path), compute_unit, function_name=function_name)
                except RuntimeError as e:
                    # Multi-function compiled models don't support function_name parameter
                    # Fall back to .mlpackage if available
                    if "functionName" in str(e) or "function_name" in str(e):
                        mlpackage_path = path.with_suffix('.mlpackage')
                        if mlpackage_path.exists():
                            print(f"  Note: Using .mlpackage for multi-function model: {mlpackage_path.name}")
                            _check_function_exists(mlpackage_path, function_name, compute_unit)
                            # Don't specify compute_units to allow make_state() to work
                            return ct.models.MLModel(str(mlpackage_path), function_name=function_name)
                    raise
            else:
                return ct.models.CompiledMLModel(str(path), compute_unit)
        else:
            # For packages (.mlpackage)
            # Don't specify compute_units to allow make_state() to work
            if function_name:
                _check_function_exists(path, function_name, compute_unit)
                return ct.models.MLModel(str(path), function_name=function_name)
            else:
                return ct.models.MLModel(str(path))

    except RuntimeError as e:
        if "valid manifest does not exist" in str(e):
            print(f"\nError: Could not load compiled model at {path}")
            print("This might be because:")
            print("1. The model is not properly compiled")
            print("2. The model was compiled for a different OS version")
            print("3. The model needs to be recompiled")
            print("\nTry using the .mlpackage version instead, or recompile the model.")
        raise

def load_metadata(model,args):
    # Extract metadata and config parameters
    metadata = {}
    if hasattr(model, 'user_defined_metadata'):
        meta = model.user_defined_metadata
        
        # Extract key parameters with defaults
        metadata['context_length'] = int(meta.get('com.anemll.context_length', 512))
        metadata['state_length'] = int(meta.get('com.anemll.state_length', metadata['context_length']))  # Added state_length
        metadata['batch_size'] = int(meta.get('com.anemll.batch_size', 64))
        metadata['lut_bits'] = int(meta.get('com.anemll.lut_bits', 0))
        metadata['num_chunks'] = int(meta.get('com.anemll.num_chunks', 1))
        if 'com.anemll.vocab_size' in meta:
            try:
                metadata['vocab_size'] = int(meta.get('com.anemll.vocab_size', 0))
            except Exception:
                pass
        if 'com.anemll.lm_head_chunk_sizes' in meta:
            metadata['lm_head_chunk_sizes'] = _parse_lm_head_chunk_sizes(
                meta.get('com.anemll.lm_head_chunk_sizes')
            )
        if getattr(args, 'update_mask_prefill', None) is not None:
            metadata['update_mask_prefill'] = bool(args.update_mask_prefill)
        if getattr(args, 'prefill_dynamic_slice', None) is not None:
            metadata['prefill_dynamic_slice'] = bool(args.prefill_dynamic_slice)
        
        if not args.eval:
            print("\nExtracted Parameters:")
            print(f"  Context Length: {metadata['context_length']}")
            print(f"  State Length: {metadata['state_length']}")
            print(f"  Prefill Batch Size: {metadata['batch_size']}")
            print(f"  LUT Bits: {metadata['lut_bits']}")
            print(f"  Number of Chunks: {metadata['num_chunks']}")
            
            # Print model info
            print("\nModel Info:")
            if 'com.anemll.info' in meta:
                print(f"  {meta['com.anemll.info']}")
            if 'com.github.apple.coremltools.version' in meta:
                print(f"  CoreML Tools: {meta['com.github.apple.coremltools.version']}")
            
            # Print model input/output shapes
            print("\nModel Shapes:")
            if hasattr(model, 'input_description'):
                print("  Inputs:")
                try:
                    if hasattr(model.input_description, 'items'):
                        for name, desc in model.input_description.items():
                            print(f"    {name}: {desc}")
                    else:
                        print(f"    {model.input_description}")
                except:
                    print(f"    Input description: {type(model.input_description)}")
            if hasattr(model, 'output_description'):
                print("  Outputs:")
                try:
                    if hasattr(model.output_description, 'items'):
                        for name, desc in model.output_description.items():
                            print(f"    {name}: {desc}")
                    else:
                        print(f"    {model.output_description}")
                except:
                    print(f"    Output description: {type(model.output_description)}")
    else:
        if not args.eval:
            print("\nWarning: No metadata found in model")

        # Check if model directory name contains context length pattern (ctxXXX)
        ctx_len = 512
        if args.context_length is  None:
            import re
            ctx_match = re.search(r'ctx(\d+)', str(args.d))
            if ctx_match:
                ctx_len0 = int(ctx_match.group(1))
                if 512 <= ctx_len0 <= 8096:
                    ctx_len = ctx_len0
                    print(f"\nDetected context length {ctx_len} from directory name")
            else:
                print(f"\nWarning: No context length found in directory  {ctx_len} from directory name {args.d}")
        else:
            ctx_len = args.context_length

        # Use defaults or values from args
        metadata['context_length'] = ctx_len
        metadata['state_length'] = ctx_len
        # Get batch size from args or use default
        metadata['batch_size'] = getattr(args, 'batch_size', 64)
        metadata['lut_bits'] = 4
        metadata['num_chunks'] = getattr(args, 'num_chunks', 4)
        metadata['update_mask_prefill'] = bool(getattr(args, 'update_mask_prefill', False))
        metadata['prefill_dynamic_slice'] = bool(getattr(args, 'prefill_dynamic_slice', False))
        if not args.eval:
            print("\nUsing parameters:")
            print(f"  Context Length: {metadata['context_length']}")
            print(f"  State Length: {metadata['state_length']}")
            print(f"  Prefill Batch Size: {metadata['batch_size']}")
            print(f"  LUT Bits: {metadata['lut_bits']}")
            print(f"  Number of Chunks: {metadata['num_chunks']}")

    # Override with values from args if they exist
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        metadata['batch_size'] = args.batch_size
        if not args.eval:
            print(f"\nOverriding batch size from args: {args.batch_size}")
    if hasattr(args, 'num_chunks') and args.num_chunks is not None:
        metadata['num_chunks'] = args.num_chunks
        if not args.eval:
            print(f"\nOverriding num chunks from args: {args.num_chunks}")
    if getattr(args, 'update_mask_prefill', None) is not None:
        metadata['update_mask_prefill'] = bool(args.update_mask_prefill)
    if getattr(args, 'prefill_dynamic_slice', None) is not None:
        metadata['prefill_dynamic_slice'] = bool(args.prefill_dynamic_slice)
    
    return metadata

def detect_cache_type(ffn_model, eval_mode=False):
    """Detect KV cache type from model's state specification.

    Works with both .mlpackage (MLModel) and .mlmodelc (CompiledMLModel).
    For compiled models, tries make_state() and inspects the state object.

    Returns:
        dict with:
            - 'has_global_cache': True if model has kv_cache_global state
            - 'has_local_cache': True if model has kv_cache_local state
            - 'has_unified_cache': True if model has kv_cache_0 state (legacy)
            - 'cache_type': 'split' for local+global, 'unified' for kv_cache_0
            - 'global_cache_chunk_idx': index of chunk with global cache (for multi-chunk)
    """
    result = {
        'has_global_cache': False,
        'has_local_cache': False,
        'has_unified_cache': False,
        'cache_type': 'unknown',
        'global_cache_chunk_idx': None
    }

    def check_model_state(model):
        """Check a single model for state types."""
        state_names = []

        # Method 1: Try get_spec() for .mlpackage
        try:
            spec = model.get_spec()
            # Check top-level stateInput (older format)
            if hasattr(spec, 'description') and hasattr(spec.description, 'stateInput'):
                state_names = [s.name for s in spec.description.stateInput]
            # Check top-level state (newer format)
            if not state_names and hasattr(spec, 'description') and hasattr(spec.description, 'state'):
                state_names = [s.name for s in spec.description.state if s.name]
            # Method 1b: For multi-function ML Programs, check functions[].state
            if not state_names and hasattr(spec, 'description') and hasattr(spec.description, 'functions'):
                for func in spec.description.functions:
                    if hasattr(func, 'state'):
                        for s in func.state:
                            if s.name and s.name not in state_names:
                                state_names.append(s.name)
        except (AttributeError, Exception):
            pass

        # Method 2: For compiled models, try make_state() and inspect
        if not state_names:
            try:
                state = model.make_state()
                # Try different ways to get state names from MLState
                if hasattr(state, 'keys'):
                    state_names = list(state.keys())
                elif hasattr(state, '_state') and hasattr(state._state, 'keys'):
                    state_names = list(state._state.keys())
                elif hasattr(state, '__dict__'):
                    # Check internal dict for state names
                    for key, val in state.__dict__.items():
                        if 'kv_cache' in str(key) or 'kv_cache' in str(val):
                            state_names.append(str(key))
                # Try repr/str for state names
                if not state_names:
                    state_repr = repr(state)
                    if 'kv_cache_global' in state_repr:
                        state_names.append('kv_cache_global')
                    if 'kv_cache_local' in state_repr:
                        state_names.append('kv_cache_local')
                    if 'kv_cache_0' in state_repr:
                        state_names.append('kv_cache_0')
            except Exception:
                pass

        return state_names

    try:
        # Handle list of models (multi-chunk)
        if isinstance(ffn_model, list):
            models_to_check = ffn_model
        else:
            models_to_check = [ffn_model]

        all_state_names = set()
        for idx, fm in enumerate(models_to_check):
            if isinstance(fm, dict):
                model = fm.get('prefill') or fm.get('infer')
            else:
                model = fm

            if model is None:
                continue

            state_names = check_model_state(model)
            all_state_names.update(state_names)

            # Track which chunk has global cache
            has_global = any('kv_cache_global' in name for name in state_names)
            if has_global and result['global_cache_chunk_idx'] is None:
                result['global_cache_chunk_idx'] = idx

        result['has_global_cache'] = any('kv_cache_global' in name for name in all_state_names)
        result['has_local_cache'] = any('kv_cache_local' in name for name in all_state_names)
        result['has_unified_cache'] = any('kv_cache_0' in name for name in all_state_names)

        if result['has_local_cache'] and result['has_global_cache']:
            result['cache_type'] = 'split'
        elif result['has_unified_cache']:
            result['cache_type'] = 'unified'
        elif result['has_local_cache']:
            result['cache_type'] = 'local_only'

        # Fallback: if we couldn't detect via spec, try make_state() on each chunk
        # and look for global cache in the state repr
        if result['global_cache_chunk_idx'] is None and result['cache_type'] == 'unknown':
            if not eval_mode:
                print("  Trying fallback detection via make_state()...")
            for idx, fm in enumerate(models_to_check):
                try:
                    if isinstance(fm, dict):
                        model = fm.get('prefill') or fm.get('infer')
                    else:
                        model = fm
                    if model is None:
                        continue
                    state = model.make_state()
                    state_repr = repr(state)
                    has_global = 'kv_cache_global' in state_repr
                    has_local = 'kv_cache_local' in state_repr
                    has_unified = 'kv_cache_0' in state_repr or 'kv_cache' in state_repr
                    if not eval_mode:
                        print(f"    Chunk {idx}: global={has_global}, local={has_local}, unified={has_unified}")
                    if has_global:
                        result['global_cache_chunk_idx'] = idx
                        result['has_global_cache'] = True
                        if has_local:
                            result['has_local_cache'] = True
                            result['cache_type'] = 'split'
                        break
                    elif has_unified and not result['has_unified_cache']:
                        result['has_unified_cache'] = True
                        result['cache_type'] = 'unified'
                    elif has_local and not result['has_local_cache']:
                        result['has_local_cache'] = True
                        result['cache_type'] = 'local_only'
                except Exception as e:
                    if not eval_mode:
                        print(f"    Chunk {idx}: error - {e}")

        if not eval_mode:
            print(f"\nCache type detection: {result['cache_type']}")
            if all_state_names:
                print(f"  State inputs: {sorted(all_state_names)}")
            if result['global_cache_chunk_idx'] is not None:
                print(f"  Global cache found in chunk: {result['global_cache_chunk_idx']}")

    except Exception as e:
        if not eval_mode:
            print(f"Warning: Could not detect cache type: {e}")

    return result

def load_models(args,metadata):
    """Load all required models and extract metadata."""
    if not args.eval:
        print("\nLoading models...")
    _maybe_report_mem("start load_models", getattr(args, "mem_report", False))

    # Determine compute unit
    compute_unit = ct.ComputeUnit.CPU_ONLY if getattr(args, 'cpu', False) else ct.ComputeUnit.CPU_AND_NE
    if not args.eval and getattr(args, 'cpu', False):
        print("Running in CPU-only mode")

    try:
        # Load embeddings model
        if not args.eval:
            print("\nLoading embeddings model...")
        embed_path = parse_model_path(args.embed)
        if not args.eval:
            print(f"Loading from: {embed_path}")
        embed_model = load_model(embed_path, compute_unit=compute_unit)
        if not args.eval:
            print("Embeddings model loaded successfully")
        _maybe_report_mem("after embeddings load", getattr(args, "mem_report", False))
        metadata = load_metadata(embed_model,args)
        

        
        # Load LM head model
        if not args.eval:
            print("\nLoading LM head model...")
        lmhead_path = parse_model_path(args.lmhead)
        if not args.eval:
            print(f"Loading from: {lmhead_path}")
        lmhead_model = load_model(lmhead_path, compute_unit=compute_unit)
        if not args.eval:
            print("LM head model loaded successfully")
        _maybe_report_mem("after lmhead load", getattr(args, "mem_report", False))
        
        # Parse FFN path and find chunks if needed
        if not args.eval:
            print("\nLoading FFN+PREFILL model(s)...")
        ffn_path = parse_model_path(args.ffn)
        chunk_no, total_chunks = parse_ffn_filename(ffn_path)

        ffn_models = []
        if chunk_no and total_chunks:
            if not args.eval:
                if getattr(args, "split_rotate", False):
                    print(f"\nDetected split-rotate chunked FFN+PF models ({total_chunks} chunks)")
                else:
                    print(f"\nDetected chunked FFN+PREFILL model ({total_chunks} chunks)")
            # Find and load all chunks
            ffn_chunk_paths = find_all_chunks(ffn_path)
            if len(ffn_chunk_paths) != total_chunks:
                raise ValueError(f"Found {len(ffn_chunk_paths)} chunks but filename indicates {total_chunks} chunks")

            if getattr(args, "split_rotate", False):
                if not args.pf:
                    raise ValueError("split-rotate requires --pf (prefill model path)")
                pf_path = parse_model_path(args.pf)
                pf_chunk_paths = find_all_chunks(pf_path)
                if len(pf_chunk_paths) != total_chunks:
                    raise ValueError(f"Found {len(pf_chunk_paths)} PF chunks but filename indicates {total_chunks} chunks")
                for ffn_chunk_path, pf_chunk_path in zip(ffn_chunk_paths, pf_chunk_paths):
                    if not args.eval:
                        print(f"\nLoading FFN chunk: {Path(ffn_chunk_path).name}")
                        print(f"Loading PF chunk: {Path(pf_chunk_path).name}")
                    try:
                        chunk_dict = {
                            'infer': load_model(ffn_chunk_path, function_name='infer', compute_unit=compute_unit),
                            'prefill': load_model(pf_chunk_path, function_name='prefill', compute_unit=compute_unit)
                        }
                        try:
                            chunk_dict['infer_rotate'] = load_model(ffn_chunk_path, function_name='infer_rotate', compute_unit=compute_unit)
                        except Exception:
                            pass
                        try:
                            chunk_dict['prefill_rotate'] = load_model(pf_chunk_path, function_name='prefill_rotate', compute_unit=compute_unit)
                        except Exception:
                            pass
                        ffn_models.append(chunk_dict)
                        if not args.eval:
                            print("Chunk loaded successfully")
                        _maybe_report_mem(f"after split-rotate chunk load {Path(ffn_chunk_path).name}", getattr(args, "mem_report", False))
                    except Exception as e:
                        if not args.eval:
                            print(f"Error loading split-rotate chunk {ffn_chunk_path}: {str(e)}")
                        raise
            else:
                for chunk_path in ffn_chunk_paths:
                    if not args.eval:
                        print(f"\nLoading FFN+PREFILL chunk: {Path(chunk_path).name}")
                    try:
                        # For chunked models, we need both infer and prefill functions
                        chunk_dict = {
                            'infer': load_model(chunk_path, function_name='infer', compute_unit=compute_unit),
                            'prefill': load_model(chunk_path, function_name='prefill', compute_unit=compute_unit)
                        }
                        # Try to load rotation functions only if context > sliding_window
                        # If context_length <= sliding_window, rotation is never needed
                        sliding_window = getattr(args, 'sliding_window', None)
                        context_length = getattr(args, 'context_length', None)
                        needs_rotation = (sliding_window is not None and
                                         context_length is not None and
                                         context_length > sliding_window)

                        if needs_rotation:
                            try:
                                chunk_dict['infer_rotate'] = load_model(chunk_path, function_name='infer_rotate', compute_unit=compute_unit)
                                chunk_dict['prefill_rotate'] = load_model(chunk_path, function_name='prefill_rotate', compute_unit=compute_unit)
                                if not args.eval:
                                    print("  Rotation functions loaded (4-function model)")
                            except Exception:
                                # Rotation functions not available - standard 2-function model
                                pass
                        elif not args.eval and sliding_window is not None:
                            print(f"  Skipping rotation functions (context {context_length} <= sliding_window {sliding_window})")
                        ffn_models.append(chunk_dict)
                        if not args.eval:
                            print("Chunk loaded successfully")
                        _maybe_report_mem(f"after FFN chunk load {Path(chunk_path).name}", getattr(args, "mem_report", False))
                    except Exception as e:
                        if not args.eval:
                            print(f"Error loading chunk {chunk_path}: {str(e)}")
                        raise
            metadata = load_metadata(ffn_models[0],args)

        else:
            if not args.eval:
                print("\nLoading single FFN model...")
            ffn_models.append(load_model(ffn_path, compute_unit=compute_unit))
            if not args.eval:
                print("FFN model loaded successfully")
            _maybe_report_mem("after FFN load", getattr(args, "mem_report", False))

        # Detect cache type from first FFN model
        # Detect cache type from ALL FFN models (some chunks may not have global cache)
        cache_info = detect_cache_type(ffn_models, eval_mode=args.eval)
        metadata['has_global_cache'] = cache_info['has_global_cache']
        metadata['has_local_cache'] = cache_info['has_local_cache']
        metadata['cache_type'] = cache_info['cache_type']
        metadata['global_cache_chunk_idx'] = cache_info['global_cache_chunk_idx']

        return embed_model, ffn_models, lmhead_model, metadata
        
    except Exception as e:
        print(f"\nError loading models: {str(e)}")
        print("\nPlease ensure all model files exist and are accessible.")
        print("Expected files:")
        print(f"  Embeddings: {args.embed}")
        print(f"  LM Head: {args.lmhead}")
        print(f"  FFN: {args.ffn}")
        if getattr(args, "split_rotate", False):
            print(f"  PF: {args.pf}")
        raise

# At the top of the file, make this a default path

def initialize_tokenizer(model_path=None, eval_mode=False):
    """Initialize and configure the tokenizer."""
    try:

        
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), 
            use_fast=False,
            trust_remote_code=True
        )

        # Try to load a chat template if the tokenizer doesn't have one.
        if getattr(tokenizer, "chat_template", None) in (None, "") and model_path:
            template = None
            config_path = Path(model_path) / "tokenizer_config.json"
            if config_path.exists():
                try:
                    config_data = json.loads(config_path.read_text())
                    template = config_data.get("chat_template")
                except Exception as e:
                    if not eval_mode:
                        print(f"Warning: Failed to read tokenizer_config.json chat_template: {e}")
            if template is None:
                jinja_path = Path(model_path) / "chat_template.jinja"
                if jinja_path.exists():
                    template = jinja_path.read_text()
            if template:
                tokenizer.chat_template = template
                if not eval_mode:
                    print("Loaded chat_template from model files")
        
        if not eval_mode:
            print("\nTokenizer Configuration:")
            print(f"Tokenizer type: {type(tokenizer)}")
            print(f"Tokenizer name: {tokenizer.__class__.__name__}")
            print(f"Vocabulary size: {len(tokenizer)}")
            print(f"Model max length: {tokenizer.model_max_length}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if not eval_mode:
                print("Set PAD token to EOS token")
        
        tokenizer.padding_side = "left"
        
        if not eval_mode:
            print(f"\nSpecial Tokens:")
            print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
            print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
            print(f"BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
            print(f"UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")

        return tokenizer
        
    except Exception as e:
        print(f"\nError: Failed to load tokenizer from {model_path}")
        print(f"Error details: {str(e)}")
        print(f"Error type: {type(e)}")
        print("\nThis appears to be a tokenizer loading issue.")
        
        # Check if it's the specific Qwen tokenizer file issue
        if "expected str, bytes or os.PathLike object, not NoneType" in str(e):
            print("\nThis error suggests the tokenizer files are missing or incomplete.")
            print("For Qwen models, you need the original model directory with tokenizer files.")
            print("Try using: --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/YOUR_SNAPSHOT_ID")
        else:
            print("Please provide the path to a compatible model directory with tokenizer files.")
        import traceback
        traceback.print_exc()
        raise



def make_causal_mask(length, start):
    """Create causal attention mask."""
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return mask

def make_update_mask(mask_len, batch_pos, batch_size):
    """Create update mask for batched KV writes."""
    update_mask = np.zeros((1, 1, mask_len, batch_size), dtype=np.float16)
    for i in range(batch_size):
        pos = batch_pos + i
        if pos < mask_len:
            update_mask[0, 0, pos, i] = 1.0
    return update_mask

def _predict_with_optional_update_mask(model, inputs, state, update_mask):
    if update_mask is None:
        return model.predict(inputs, state)

    supports = getattr(model, "_supports_update_mask", None)
    if supports is False:
        return model.predict(inputs, state)

    inputs_with_mask = dict(inputs)
    inputs_with_mask["update_mask"] = update_mask

    if supports is True:
        return model.predict(inputs_with_mask, state)

    try:
        output = model.predict(inputs_with_mask, state)
        model._supports_update_mask = True
        return output
    except RuntimeError as e:
        if "update_mask" in str(e):
            model._supports_update_mask = False
            return model.predict(inputs, state)
        raise

def initialize_causal_mask(context_length, eval_mode=False):
    """Initialize causal mask for transformer attention."""
    causal_mask = make_causal_mask(context_length, 0)
    causal_mask = torch.tensor(causal_mask, dtype=torch.float16)
    if not eval_mode:
        print(f"\nInitialized causal mask for context length {context_length}")
    return causal_mask

def run_prefill(embed_model, ffn_models, input_ids, context_pos, context_length, batch_size=64, state=None,
                causal_mask=None, sliding_window=None, single_token_mode=False, use_update_mask=True):
    """Run prefill on the input sequence.

    For Gemma3 with 4-function models:
    - Uses 'prefill' for positions < sliding_window
    - Uses 'prefill_rotate' for positions >= sliding_window (if available)
    """
    # Use provided causal mask or create one if not provided
    if causal_mask is None:
        causal_mask = make_causal_mask(context_length, 0)
        causal_mask = torch.tensor(causal_mask, dtype=torch.float16)

    # Check if rotation functions are available
    has_rotation = isinstance(ffn_models[0], dict) and 'prefill_rotate' in ffn_models[0]

    # If no rotation or no sliding_window, use standard prefill
    if not has_rotation or sliding_window is None:
        sliding_window = context_length  # Effectively disables rotation mode

    # Process in full batches only with prefill
    batch_pos = 0
    mask_len = max(context_length, sliding_window or 0)
    if not single_token_mode:
        while batch_pos + batch_size <= context_pos:
            batch_end = batch_pos + batch_size

            # Get current batch (exactly batch_size tokens)
            batch_input = input_ids[:, batch_pos:batch_end]

            # Generate position IDs for full batch size
            position_ids = torch.arange(batch_pos, batch_pos + batch_size, dtype=torch.int32)
            batch_causal_mask = causal_mask[:, :, batch_pos:batch_pos + batch_size, :]

            # Run embeddings
            hidden_states = torch.from_numpy(
                embed_model.predict({
                    'input_ids': batch_input.numpy().astype(np.int32)
                })['hidden_states']
            )

            # Determine which prefill function to use based on position
            prefill_func_name = 'prefill_rotate' if batch_pos >= sliding_window and has_rotation else 'prefill'

            # Run through FFN chunks with state
            # Handle per-chunk states (list) or unified state (single object)
            for chunk_idx, ffn_model in enumerate(ffn_models):
                if isinstance(ffn_model, dict):
                    inputs = {
                        'hidden_states': hidden_states.numpy().astype(np.float16),  # [1, 64, hidden_size]
                        'position_ids': position_ids.numpy().astype(np.int32),    # [64]
                        'causal_mask': batch_causal_mask.numpy().astype(np.float16), # [1, 1, 64, context_length]
                        'current_pos': np.array([batch_pos], dtype=np.int32)  # [1]
                    }
                    # Use per-chunk state if state is a list, otherwise use unified state
                    chunk_state = state[chunk_idx] if isinstance(state, list) else state
                    update_mask = make_update_mask(mask_len, batch_pos, batch_size) if use_update_mask else None
                    output = _predict_with_optional_update_mask(ffn_model[prefill_func_name], inputs, chunk_state, update_mask)
                    hidden_states = torch.from_numpy(output['output_hidden_states'])

            batch_pos = batch_end

    # Process remaining tokens one-at-a-time using infer (avoids padding issues)
    while batch_pos < context_pos:
        token = input_ids[:, batch_pos:batch_pos + 1]
        position_ids = torch.tensor([batch_pos], dtype=torch.int32)
        single_causal_mask = causal_mask[:, :, batch_pos:batch_pos + 1, :]

        # Determine which infer function to use
        infer_func_name = 'infer_rotate' if (batch_pos >= sliding_window and has_rotation) else 'infer'

        # Run embeddings
        hidden_states = torch.from_numpy(
            embed_model.predict({
                'input_ids': token.numpy().astype(np.int32)
            })['hidden_states']
        )

        for chunk_idx, ffn_model in enumerate(ffn_models):
            if isinstance(ffn_model, dict):
                inputs = {
                    'hidden_states': hidden_states.numpy().astype(np.float16),
                    'position_ids': position_ids.numpy().astype(np.int32),
                    'causal_mask': single_causal_mask.numpy().astype(np.float16),
                    'current_pos': np.array([batch_pos], dtype=np.int32)
                }
                chunk_state = state[chunk_idx] if isinstance(state, list) else state
                output = ffn_model[infer_func_name].predict(inputs, chunk_state)
                hidden_states = torch.from_numpy(output['output_hidden_states'])

        batch_pos += 1

    return torch.tensor([context_pos], dtype=torch.int32)

def generate_next_token(embed_model, ffn_models, lmhead_model, input_ids, pos, context_length, metadata, state=None, causal_mask=None, temperature=0.0):
    """Generate the next token.

    For Gemma3 with 4-function models:
    - Uses 'infer' for positions < sliding_window
    - Uses 'infer_rotate' for positions >= sliding_window (if available)
    """
    debug = metadata.get('debug', False)
    attention_size = metadata.get('attention_size', context_length)
    sliding_window = metadata.get('sliding_window', None)

    # Check if rotation functions are available
    has_rotation = isinstance(ffn_models[0], dict) and 'infer_rotate' in ffn_models[0]

    # Determine which infer function to use
    # Use infer_rotate for positions >= sliding_window (0-indexed, so pos-1 is the actual position)
    use_rotation = has_rotation and sliding_window is not None and (pos - 1) >= sliding_window
    infer_func_name = 'infer_rotate' if use_rotation else 'infer'

    # Get current token
    current_token = input_ids[:, pos-1:pos]  # [1, 1]

    # Ensure proper data type for CoreML
    current_token_array = current_token.numpy().astype(np.int32)

    # Run embeddings
    hidden_states = torch.from_numpy(
        embed_model.predict({'input_ids': current_token_array})['hidden_states']
    )  # [1, 1, hidden_size]

    # Create masks
    update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
    update_mask[0, 0, pos-1, 0] = 1.0
    position_ids = torch.tensor([pos-1], dtype=torch.int32)  # [1]

    # Use provided causal mask or create one if not provided
    if causal_mask is None:
        causal_mask_data = make_causal_mask(context_length, 0)
        single_causal_mask = torch.tensor(causal_mask_data[:, :, pos-1:pos, :], dtype=torch.float16)  # [1, 1, 1, context_length]
    else:
        single_causal_mask = causal_mask[:, :, pos-1:pos, :]

    if debug:
        print(f"\n[DEBUG] generate_next_token: pos={pos}, context_length={context_length}, attention_size={attention_size}")
        print(f"[DEBUG]   position_ids={position_ids.item()}, current_token={current_token.item()}")
        print(f"[DEBUG]   causal_mask shape={single_causal_mask.shape}")
        print(f"[DEBUG]   hidden_states shape={hidden_states.shape}")

    # Run through FFN chunks with state
    # Handle per-chunk states (list) or unified state (single object)
    for chunk_idx, ffn_model in enumerate(ffn_models):
        if isinstance(ffn_model, dict):
            # Build inputs dict - only include inputs that the model expects
            inputs = {
                'hidden_states': hidden_states.numpy().astype(np.float16),
                'position_ids': position_ids.numpy().astype(np.int32),
                'causal_mask': single_causal_mask.numpy().astype(np.float16),
                'current_pos': position_ids.numpy().astype(np.int32)
            }
            # Add update_mask only if model expects it (older models)
            # Get model input names from the spec
            try:
                model_inputs = {inp.name for inp in ffn_model[infer_func_name].get_spec().description.input}
            except:
                model_inputs = set()
            if 'update_mask' in model_inputs:
                inputs['update_mask'] = update_mask.numpy().astype(np.float16)
            if debug:
                print(f"[DEBUG]   FFN {infer_func_name} inputs: position_ids={inputs['position_ids']}, current_pos={inputs['current_pos']}")
                print(f"[DEBUG]   FFN {infer_func_name} causal_mask shape={inputs['causal_mask'].shape}")
            try:
                # Use per-chunk state if state is a list, otherwise use unified state
                chunk_state = state[chunk_idx] if isinstance(state, list) else state
                output = ffn_model[infer_func_name].predict(inputs, chunk_state)
            except Exception as e:
                print(f"\n[ERROR] FFN {infer_func_name} failed at pos={pos}, position_ids={position_ids.item()}")
                print(f"[ERROR]   context_length={context_length}, attention_size={attention_size}")
                print(f"[ERROR]   causal_mask shape={single_causal_mask.shape}")
                print(f"[ERROR]   Exception: {e}")
                raise
            hidden_states = torch.from_numpy(output['output_hidden_states'])
    
    # Run LM head
    lm_output = lmhead_model.predict({'hidden_states': hidden_states.numpy().astype(np.float16)})

    # Check if model uses argmax_in_model mode (outputs argmax_idx/argmax_val instead of logits)
    argmax_in_model = metadata.get('argmax_in_model', False)

    # Debug: show LM head output keys if argmax mode expected but not found
    if argmax_in_model and 'argmax_idx' not in lm_output:
        print(f"\n[WARNING] argmax_in_model=True but model outputs: {list(lm_output.keys())}")
        print("[WARNING] Model may need to be reconverted with --argmax flag")
        # Fall through to logits processing

    if argmax_in_model and 'argmax_idx' in lm_output:
        # Argmax-in-model mode: find the chunk with highest value and compute global index
        # Model outputs LOCAL indices (0 to chunk_size-1) for each chunk
        # We compute: global_idx = local_idx + (best_chunk * chunk_size)
        argmax_idx = lm_output['argmax_idx']  # shape: [num_chunks], LOCAL indices
        argmax_val = lm_output['argmax_val']  # shape: [num_chunks]

        # Flatten arrays
        argmax_idx_flat = argmax_idx.flatten()
        argmax_val_flat = argmax_val.flatten()

        # Find best chunk (highest value)
        best_chunk = int(np.argmax(argmax_val_flat))
        local_idx = int(argmax_idx_flat[best_chunk])

        num_chunks = len(argmax_idx_flat)
        chunk_sizes, chunk_offsets = _resolve_argmax_chunk_layout(
            metadata,
            num_chunks,
            argmax_idx_flat=argmax_idx_flat,
        )
        global_idx = local_idx + chunk_offsets[best_chunk]

        if metadata.get('debug_argmax', False):
            print(f"\nLM head argmax mode (chunked):")
            print(f"  argmax_idx shape: {argmax_idx.shape}, dtype: {argmax_idx.dtype}")
            print(f"  argmax_val shape: {argmax_val.shape}, dtype: {argmax_val.dtype}")
            print(f"  best_chunk={best_chunk}, local_idx={local_idx}, global_idx={global_idx}")
            print(f"  chunk_size={chunk_sizes[best_chunk]}, chunk_offset={chunk_offsets[best_chunk]}")
            print(f"  best_val={argmax_val_flat[best_chunk]:.4f}")

        return global_idx

    # Get number of logits from metadata, using split_lm_head if available
    # First check for split_lm_head (new), then num_logits (legacy), default to 8
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
    else:
        # Try output_logits as fallback
        logits = torch.from_numpy(lm_output['output_logits'])

    # Apply temperature and sample
    if temperature > 0:
        logits = logits / temperature
        probs = F.softmax(logits[0, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
    else:
        next_token = torch.argmax(logits[0, -1, :]).item()

    return next_token

def create_unified_state(ffn_models, context_length, eval_mode=False, metadata=None):
    """Create unified KV cache state for transformer.

    For split cache models (Gemma3), uses the chunk with global cache to ensure
    both kv_cache_local and kv_cache_global are properly initialized.

    For models with inconsistent state declarations (old converter), creates
    per-chunk states to avoid passing unsupported state types.

    Args:
        ffn_models: List of FFN model chunks
        context_length: Context length (unused but kept for API compatibility)
        eval_mode: Suppress output if True
        metadata: Optional metadata dict with 'global_cache_chunk_idx'

    Returns:
        Either a single state (if all chunks have same state interface) or
        a list of states (one per chunk, for inconsistent models)
    """
    # First, check if all chunks have consistent state interfaces
    # by checking if each chunk's state has global cache
    chunk_has_global = []
    for idx, fm in enumerate(ffn_models):
        if isinstance(fm, dict):
            model = fm.get('prefill') or fm.get('infer')
        else:
            model = fm

        has_global = False
        try:
            state = model.make_state()
            state_repr = repr(state)
            has_global = 'kv_cache_global' in state_repr
        except Exception:
            pass
        chunk_has_global.append(has_global)

    # Check if all chunks have same interface
    all_have_global = all(chunk_has_global)
    none_have_global = not any(chunk_has_global)
    consistent = all_have_global or none_have_global

    if not eval_mode:
        print(f"\nChunk global cache support: {chunk_has_global}")
        print(f"  Consistent interface: {consistent}")

    if consistent:
        # All chunks have same state interface - use unified state
        state_chunk_idx = 0
        if metadata and metadata.get('global_cache_chunk_idx') is not None:
            state_chunk_idx = metadata['global_cache_chunk_idx']

        if isinstance(ffn_models[0], dict):
            state = ffn_models[state_chunk_idx]['prefill'].make_state()
        else:
            state = ffn_models[state_chunk_idx].make_state()

        if not eval_mode:
            print(f"Created unified transformer state from chunk {state_chunk_idx}")
        return state
    else:
        # Inconsistent state interfaces - create per-chunk states
        # This is a workaround for models converted before the fix
        if not eval_mode:
            print("WARNING: Chunks have inconsistent state interfaces!")
            print("  Creating per-chunk states (KV cache won't be fully shared)")
            print("  Please reconvert model with latest converter for proper support")

        states = []
        for idx, fm in enumerate(ffn_models):
            if isinstance(fm, dict):
                model = fm.get('prefill') or fm.get('infer')
            else:
                model = fm
            states.append(model.make_state())

        return states

def chat_loop(embed_model, ffn_models, lmhead_model, tokenizer, metadata, state, causal_mask=None, auto_prompt=None, warmup=False, save_file=None, max_tokens=None, no_template=False, eval_mode=False, no_think=False):
    """Interactive chat loop."""
    context_length = metadata.get('context_length')
    batch_size = metadata.get('batch_size', 64)
    update_mask_prefill = metadata.get('update_mask_prefill', False) if metadata else False
    prefill_dynamic_slice = metadata.get('prefill_dynamic_slice', False) if metadata else False
    single_token_mode = False
    if not (update_mask_prefill or prefill_dynamic_slice):
        single_token_mode = True
        if not warmup and not eval_mode:
            print("No update_mask_prefill or prefill_dynamic_slice; forcing single-token prefill for compatibility.")
    
    if not warmup and not eval_mode:
        print(f"\nUsing context length: {context_length}")
        print("\nStarting chat session. Press Ctrl+D to exit.")
        print("Type your message and press Enter to chat.")
    
    # Check if tokenizer has chat template and if it works
    template_kwargs = {"enable_thinking": False} if no_think else {}
    has_chat_template = False
    try:
        # Test if chat template works
        test_messages = [{"role": "user", "content": "test"}]
        tokenizer.apply_chat_template(test_messages, return_tensors="pt", **template_kwargs)
        has_chat_template = True
        if not warmup and not eval_mode:
            print("\nUsing chat template for prompts")
    except:
        if not warmup and not eval_mode:
            print("\nUsing manual formatting for prompts")

    stop_token_ids = build_stop_token_ids(tokenizer)
    
    conversation = []
    
    try:
        while True:
            try:
                if not warmup and not eval_mode:
                    print(f"\n{LIGHT_GREEN}You:{RESET_COLOR}", end=' ', flush=True)
                if auto_prompt is not None:
                    user_input = auto_prompt
                    if not warmup and not eval_mode:
                        print(user_input)
                else:
                    user_input = input().strip()
            except EOFError:
                if not warmup and not eval_mode:
                    print("\nExiting chat...")
                break
                
            if not user_input:
                continue
            
            # Format prompt based on no_template flag and tokenizer capabilities
            if no_template:
                # Use raw input without any chat template formatting
                input_ids = tokenizer(
                    user_input,
                    return_tensors="pt",
                    add_special_tokens=True
                ).input_ids.to(torch.int32)
                if not warmup and not eval_mode:
                    print("Using raw input without chat template")
            elif has_chat_template:
                messages = [{"role": "user", "content": user_input}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    **template_kwargs
                ).to(torch.int32)
            else:
                # Manual formatting for Llama models without chat template
                formatted_prompt = f"[INST] {user_input} [/INST]"
                input_ids = tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    add_special_tokens=True
                ).input_ids.to(torch.int32)
            
            context_pos = input_ids.size(1)
            
            if not warmup and not eval_mode:
                print(f"\n{LIGHT_BLUE}Assistant:{RESET_COLOR}", end=' ', flush=True)
            
            # Initialize token printer
            token_printer = TokenPrinter(tokenizer)
            tokens_generated = 0  # Track number of tokens
            
            try:
                # Start prefill timing
                prefill_start = time.time()
                
                # Run prefill with state and causal mask
                # Ensure batch_size is not None
                if batch_size is None:
                    batch_size = 64
                    if not eval_mode:
                        print(f"Warning: batch_size was None, using default: {batch_size}")
                
                # Get sliding_window for rotation support (Gemma3)
                sliding_window = metadata.get('sliding_window', None)

                _ = run_prefill(
                    embed_model,
                    ffn_models,
                    input_ids,
                    context_pos,
                    context_length,
                    batch_size,
                    state,
                    causal_mask,
                    sliding_window,
                    single_token_mode=single_token_mode,
                    use_update_mask=update_mask_prefill,
                )
                
                # Calculate prefill timing
                prefill_time = time.time() - prefill_start
                prefill_tokens = context_pos  # Number of tokens in input
                prefill_tokens_per_sec = prefill_tokens / prefill_time if prefill_time > 0 else 0
                
                # Generation loop with state
                input_ids = input_ids
                pos = context_pos
                inference_start = time.time()
                inference_tokens = 0
                
                while pos < context_length - 1:
                    # Generate next token with causal mask
                    next_token = generate_next_token(
                        embed_model,
                        ffn_models,
                        lmhead_model,
                        input_ids,
                        pos,
                        context_length,
                        metadata,
                        state,
                        causal_mask
                    )
                    
                    # Add token to sequence
                    if pos < input_ids.size(1):
                        input_ids[0, pos] = next_token
                    else:
                        input_ids = torch.cat([
                            input_ids,
                            torch.tensor([[next_token]], dtype=torch.int32)
                        ], dim=1)
                    
                    # Add to printer only if not in warmup
                    if not warmup:
                        token_printer.add_token(next_token)
                        token_printer.drain_buffer(eval_mode)
                    
                    pos += 1
                    tokens_generated += 1
                    inference_tokens += 1
                    
                    # Check limits
                    if warmup and tokens_generated >= WARMUP_TOKEN_LIMIT:
                        break
                    
                    # Check max_tokens limit
                    if max_tokens is not None and tokens_generated >= max_tokens:
                        break
                        
                    if next_token in stop_token_ids:
                        break
                
                # Calculate inference timing
                inference_time = time.time() - inference_start
                inference_tokens_per_sec = inference_tokens / inference_time if inference_time > 0 else 0
                
                # Get final response and add to conversation
                if not warmup:
                    response = token_printer.stop(eval_mode)
                    if eval_mode:
                        # In eval mode, only print the model response
                        print(response, end='')
                    else:
                        # Print timing stats
                        prefill_ms = prefill_time * 1000  # Convert to milliseconds
                        print(f"\nPrefill: {prefill_ms:.1f}ms ({prefill_tokens_per_sec:.1f} t/s)")
                        print(f"Inference: {inference_tokens_per_sec:.1f} t/s")
                        print(f"Total: Generated {tokens_generated} tokens in {prefill_time + inference_time:.2f}s")
                    conversation.append({"role": "assistant", "content": response})
                    
                    # Save response to file if requested
                    if save_file and not eval_mode:
                        try:
                            # Add small delay to ensure all tokens are processed
                            time.sleep(0.5)
                            
                            # Make sure response ends with EOS token if it's supposed to
                            if response and not response.endswith("<|eot_id|>") and not response.endswith("</s>") and not response.endswith("<end_of_turn>"):
                                if tokenizer.eos_token:
                                    eos_text = tokenizer.decode([tokenizer.eos_token_id])
                                    if not response.endswith(eos_text):
                                        print(f"\n{DARK_BLUE}Adding missing EOS token for consistency{RESET_COLOR}")
                                        response += eos_text
                            
                            with open(save_file, 'w') as f:
                                f.write(response)
                            print(f"\n{DARK_BLUE}Response saved to file: {save_file}{RESET_COLOR}")
                        except Exception as e:
                            print(f"\n{DARK_BLUE}Error saving to file: {str(e)}{RESET_COLOR}")
                else:
                    token_printer.stop(eval_mode)  # Clean up without printing stats
                
                # Exit after one response in auto_prompt mode
                if auto_prompt is not None:
                    break
                
            except KeyboardInterrupt:
                if not eval_mode:
                    print("\nGeneration interrupted")
                token_printer.stop(eval_mode)
                continue
                
    except Exception as e:
        print(f"\nError in chat loop: {str(e)}")
        import traceback
        traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with CoreML LLaMA, gil resolved  (c) 2025 Anemll')
    
    # Add meta.yaml option
    parser.add_argument('--meta', type=str, help='Path to meta.yaml to load all parameters')
    
    # Model paths
    parser.add_argument('--d', '--dir', type=str, default='.',
                       help='Directory containing model files (default: current directory)')
    parser.add_argument('--embed', type=str, required=False,
                       help='Path to embeddings model (relative to --dir)')
    parser.add_argument('--ffn', type=str, required=False,
                       help='Path to FFN model (can be chunked, relative to --dir)')
    parser.add_argument('--pf', type=str, required=False,
                       help='Path to prefill model (split-rotate, relative to --dir)')
    parser.add_argument('--lmhead', type=str, required=False,
                       help='Path to LM head model (relative to --dir)')
    parser.add_argument('--tokenizer', type=str, required=False,
                       help='Path to tokenizer')
    
    # Add new argument for auto-generation
    parser.add_argument('--prompt', type=str,
                       help='If specified, run once with this prompt and exit')
    
    # Add save option
    parser.add_argument('--save', type=str,
                       help='Save assistant\'s response to specified file')
    
    # Add max-tokens option
    parser.add_argument('--max-tokens', type=int,
                       help='Maximum number of tokens to generate')
    
    # Add no-warmup flag
    parser.add_argument('--nw', action='store_true',
                       help='Skip warmup phase')
    
    # Add no-template flag  
    parser.add_argument('--no-template', action='store_true',
                       help='Prefill the question itself and start inference directly without chat template')
    parser.add_argument('--no-think', action='store_true',
                       help='Disable thinking mode in chat templates (e.g., Qwen enable_thinking)')
    
    # Add eval mode flag
    parser.add_argument('--eval', action='store_true',
                       help='Evaluation mode: suppress all output except model response')

    # Add CPU-only mode
    parser.add_argument('--cpu', action='store_true',
                       help='Run on CPU only (no ANE/GPU)')
    parser.add_argument('--mem-report', action='store_true',
                       help='Print approximate RSS after large steps (debugging)')

    # Model configuration
    parser.add_argument('--context-length', type=int,
                       help='Context length for the model (default: 512), if not provided, it will be detected from the model directory name ctxNUMBER')
    parser.add_argument('--sliding-window', type=int,
                       help='Sliding-window size for models with cache rotation support (e.g., Gemma3)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for prefill (default: 64)')
    parser.add_argument('--num-logits', type=int, default=8,
                       help='Number of logits outputs from LM head (default: 8, legacy)')
    parser.add_argument('--split-lm-head', type=int,
                       help='Number of logits splits from LM head (default: 8 for llama, 16 for qwen)')
    parser.add_argument('--debug-argmax', action='store_true',
                       help='Enable debug output for argmax mode (print indices and values)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output (position, state, shapes)')
    parser.add_argument('--split-rotate', action='store_true',
                       help='Split rotate mode: FFN and PF are separate model files')

    args = parser.parse_args()

    def _strip_model_ext(value):
        if value is None:
            return None
        return value.replace('.mlmodelc', '').replace('.mlpackage', '')
    
    # If meta.yaml is provided, load parameters from it
    if args.meta:
        try:
            with open(args.meta, 'r') as f:
                meta = yaml.safe_load(f)
            params = meta['model_info']['parameters']

            # Set model directory to meta.yaml directory if not specified
            if not args.d or args.d == '.':
                args.d = str(Path(args.meta).parent)

            # Check if this is a monolithic model
            model_type = meta['model_info'].get('model_type', 'chunked')
            args.is_monolithic = (model_type == 'monolithic')

            if args.is_monolithic:
                # Monolithic model configuration
                prefix = params.get('model_prefix', 'qwen')
                lut_bits = params.get('lut_bits', 'none')
                lut_suffix = f"_lut{lut_bits}" if lut_bits != 'none' else ''

                # Set monolithic model path
                args.monolithic_model = params.get('monolithic_model', f'{prefix}_monolithic_full{lut_suffix}.mlmodelc')

                # Set other parameters
                if args.context_length is None:
                    args.context_length = int(params['context_length'])
                # state_length for split cache models (defaults to context_length if not specified)
                args.state_length = int(params.get('state_length', args.context_length))
                if args.batch_size is None:
                    args.batch_size = int(params['batch_size'])
                args.num_chunks = 1  # Monolithic has no chunks

                # Set split_lm_head, but allow CLI override
                if args.split_lm_head is None:
                    if 'split_lm_head' in params:
                        args.split_lm_head = int(params['split_lm_head'])
                    else:
                        args.split_lm_head = 16 if 'qwen' in prefix.lower() else 8

                # Check for argmax_in_model flag
                args.argmax_in_model = params.get('argmax_in_model', False)
                args.vocab_size = params.get('vocab_size', None)
                args.lm_head_chunk_sizes = params.get('lm_head_chunk_sizes', None)
                # Prefill behavior flags
                args.update_mask_prefill = params.get('update_mask_prefill', False)
                args.prefill_dynamic_slice = params.get('prefill_dynamic_slice', False)
                # Sliding window (cache rotation) is optional; keep None when model does not use SWA.
                if getattr(args, 'sliding_window', None) is None:
                    if 'sliding_window' in params:
                        args.sliding_window = int(params['sliding_window'])
                    elif prefix.lower().startswith('gemma3'):
                        args.sliding_window = 512
                    else:
                        args.sliding_window = None

                # Set tokenizer path
                if not args.tokenizer:
                    if 'tokenizer_path' in params:
                        args.tokenizer = params['tokenizer_path']
                    else:
                        args.tokenizer = args.d

                if not args.eval:
                    print(f"\nLoaded MONOLITHIC model from {args.meta}:")
                    print(f"  Model: {args.monolithic_model}")
                    print(f"  Context Length: {args.context_length}")
                    print(f"  State Length: {args.state_length}")
                    print(f"  Batch Size: {args.batch_size}")
                    print(f"  Split LM Head: {args.split_lm_head}")
                    print(f"  Argmax in Model: {args.argmax_in_model}")
                    print(f"  Models Directory: {args.d}")
            else:
                # Standard chunked model configuration
                args.is_monolithic = False
                prefix = params.get('model_prefix', 'llama')
                lut_ffn = f"_lut{params['lut_ffn']}" if params['lut_ffn'] != 'none' else ''
                lut_lmhead = f"_lut{params['lut_lmhead']}" if params['lut_lmhead'] != 'none' else ''
                lut_embeddings = f"_lut{params['lut_embeddings']}" if params['lut_embeddings'] != 'none' else ''
                num_chunks = int(params['num_chunks'])
                args.split_rotate = bool(params.get('split_rotate', False)) or getattr(args, 'split_rotate', False)

                # Set model paths if not specified
                if not args.lmhead:
                    if 'lm_head' in params:
                        args.lmhead = _strip_model_ext(params['lm_head'])
                    else:
                        args.lmhead = f'{prefix}_lm_head{lut_lmhead}'
                if not args.embed:
                    if 'embeddings' in params:
                        args.embed = _strip_model_ext(params['embeddings'])
                    else:
                        args.embed = f'{prefix}_embeddings{lut_embeddings}'
                if args.split_rotate:
                    if not args.ffn:
                        if 'ffn' in params:
                            args.ffn = _strip_model_ext(params['ffn'])
                        else:
                            args.ffn = f'{prefix}_FFN{lut_ffn}_chunk_01of{num_chunks:02d}_combined'
                    if not args.pf:
                        if 'pf' in params:
                            args.pf = _strip_model_ext(params['pf'])
                        else:
                            args.pf = f'{prefix}_PF{lut_ffn}_chunk_01of{num_chunks:02d}'
                else:
                    if not args.ffn:
                        if 'ffn' in params:
                            ffn_candidate = _strip_model_ext(params['ffn'])
                            ffn_path = Path(ffn_candidate)
                            if "_chunk_" not in ffn_candidate:
                                default_ffn = f'{prefix}_FFN_PF{lut_ffn}_chunk_01of{num_chunks:02d}'
                                base_dir = ffn_path.parent if ffn_path.is_absolute() else Path(args.d)
                                if (base_dir / f"{default_ffn}.mlmodelc").exists() or (base_dir / f"{default_ffn}.mlpackage").exists():
                                    args.ffn = str(base_dir / default_ffn) if ffn_path.is_absolute() else default_ffn
                                else:
                                    args.ffn = ffn_candidate
                            else:
                                args.ffn = ffn_candidate
                        else:
                            args.ffn = f'{prefix}_FFN_PF{lut_ffn}_chunk_01of{num_chunks:02d}'
                if not args.tokenizer:
                    if 'tokenizer_path' in params:
                        args.tokenizer = params['tokenizer_path']
                    else:
                        args.tokenizer = args.d

                # Set other parameters if not overridden by command line
                if args.context_length is None:
                    args.context_length = int(params['context_length'])
                if args.batch_size is None:
                    args.batch_size = int(params['batch_size'])
                args.num_chunks = num_chunks
                if 'num_logits' in params:
                    args.num_logits = int(params['num_logits'])
                # attention_size is used for causal mask (sliding window for Gemma3)
                args.attention_size = int(params.get('attention_size', args.context_length))

                # sliding_window for Gemma3 rotation support (default 512 for Gemma3)
                # Only set if not overridden by command line
                if getattr(args, 'sliding_window', None) is None:
                    if 'sliding_window' in params:
                        args.sliding_window = int(params['sliding_window'])
                    elif prefix.lower().startswith('gemma3'):
                        args.sliding_window = 512  # Default Gemma3 sliding window
                    else:
                        args.sliding_window = None  # No rotation for other models

                # Set split_lm_head, but allow CLI override
                if args.split_lm_head is None:
                    if 'split_lm_head' in params:
                        args.split_lm_head = int(params['split_lm_head'])
                    else:
                        args.split_lm_head = 8

                # Check for argmax_in_model flag (for chunked models)
                args.argmax_in_model = params.get('argmax_in_model', False)
                args.vocab_size = params.get('vocab_size', None)
                args.lm_head_chunk_sizes = params.get('lm_head_chunk_sizes', None)
                # Prefill behavior flags
                args.update_mask_prefill = params.get('update_mask_prefill', False)
                args.prefill_dynamic_slice = params.get('prefill_dynamic_slice', False)

                if not args.eval:
                    print(f"\nLoaded parameters from {args.meta}:")
                    print(f"  Context Length: {args.context_length}")
                    print(f"  Batch Size: {args.batch_size}")
                    print(f"  Num Chunks: {args.num_chunks}")
                    print(f"  Num Logits: {args.num_logits}")
                    print(f"  Split LM Head: {args.split_lm_head}")
                    print(f"  Argmax in Model: {args.argmax_in_model}")
                    if args.split_rotate:
                        print("  Split Rotate: True")
                        print(f"  PF: {args.pf}")
                    print(f"  Models Directory: {args.d}")
                    print(f"  Embeddings: {args.embed}")
                    print(f"  LM Head: {args.lmhead}")
                    print(f"  FFN: {args.ffn}")
            
        except Exception as e:
            print(f"\nError loading meta.yaml: {str(e)}")
            sys.exit(1)
    else:
        # If no meta.yaml, set defaults
        args.is_monolithic = False
        if not hasattr(args, 'split_lm_head') or args.split_lm_head is None:
            args.split_lm_head = args.num_logits  # Use num_logits as fallback

    return args


def load_monolithic_model(args, metadata):
    """Load monolithic model with infer, infer_rotate, and prefill functions."""
    if not args.eval:
        print("\nLoading monolithic model...")
    _maybe_report_mem("start load_monolithic_model", getattr(args, "mem_report", False))

    # Determine compute unit
    compute_unit = ct.ComputeUnit.CPU_ONLY if getattr(args, 'cpu', False) else ct.ComputeUnit.CPU_AND_NE
    if not args.eval and getattr(args, 'cpu', False):
        print("Running in CPU-only mode")

    model_path = str(Path(args.d) / args.monolithic_model)
    model_path = parse_model_path(model_path)

    if not args.eval:
        print(f"Loading from: {model_path}")

    def _progress_bar(done, total, label, width=18):
        if total <= 0:
            total = 1
        filled = int(width * done / total)
        bar = "[" + ("#" * filled) + ("." * (width - filled)) + "]"
        sys.stdout.write(f"\r{bar} {done}/{total} {label}")
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\n")

    # Decide whether to attempt rotate functions
    attempt_rotate = False
    context_length = getattr(args, "context_length", None)
    sliding_window = getattr(args, "sliding_window", None)
    if context_length is not None and sliding_window is not None:
        attempt_rotate = context_length > sliding_window

    functions_to_load = [("infer", True), ("prefill", True)]
    if attempt_rotate:
        functions_to_load += [("infer_rotate", False), ("prefill_rotate", False)]

    infer_model = None
    prefill_model = None
    infer_rotate_model = None
    prefill_rotate_model = None
    loaded = []
    missing = []

    total = len(functions_to_load)
    for idx, (name, required) in enumerate(functions_to_load, start=1):
        if not args.eval:
            _progress_bar(idx, total, name)
        try:
            model = load_model(model_path, function_name=name, compute_unit=compute_unit)
            loaded.append(name)
            if name == "infer":
                infer_model = model
            elif name == "prefill":
                prefill_model = model
            elif name == "infer_rotate":
                infer_rotate_model = model
            elif name == "prefill_rotate":
                prefill_rotate_model = model
        except Exception:
            if required:
                raise
            missing.append(name)

    _maybe_report_mem("after load monolithic functions", getattr(args, "mem_report", False))

    if not args.eval:
        summary = "Monolithic model loaded (" + ", ".join(loaded) + ")"
        if missing:
            summary += f" [missing: {', '.join(missing)}]"
        print(summary)

    # Extract metadata from model
    metadata = load_metadata(infer_model, args)

    return infer_model, infer_rotate_model, prefill_model, prefill_rotate_model, metadata


def run_monolithic_prefill(model, input_ids, context_pos, context_length, batch_size, state, causal_mask,
                           infer_model=None, single_token_mode=False, mask_len=None, use_update_mask=True):
    """Run prefill on monolithic model."""
    batch_pos = 0
    if mask_len is None:
        mask_len = context_length

    if not single_token_mode:
        while batch_pos + batch_size <= context_pos:
            batch_end = batch_pos + batch_size

            # Get current batch
            batch_input = input_ids[:, batch_pos:batch_end]

            # Generate position IDs for full batch size
            position_ids = torch.arange(batch_pos, batch_pos + batch_size, dtype=torch.int32)
            batch_causal_mask = causal_mask[:, :, batch_pos:batch_pos + batch_size, :]

            # Run monolithic prefill (input_ids -> logits directly)
            inputs = {
                'input_ids': batch_input.numpy().astype(np.int32),
                'position_ids': position_ids.numpy().astype(np.int32),
                'causal_mask': batch_causal_mask.numpy().astype(np.float16),
                'current_pos': np.array([batch_pos], dtype=np.int32)
            }
            update_mask = make_update_mask(mask_len, batch_pos, batch_size) if use_update_mask else None
            _predict_with_optional_update_mask(model, inputs, state, update_mask)

            batch_pos = batch_end

    # Process remaining tokens one-at-a-time using infer
    single_model = infer_model if infer_model is not None else model
    while batch_pos < context_pos:
        token = input_ids[:, batch_pos:batch_pos + 1]
        position_ids = torch.tensor([batch_pos], dtype=torch.int32)
        single_causal_mask = causal_mask[:, :, batch_pos:batch_pos + 1, :]

        inputs = {
            'input_ids': token.numpy().astype(np.int32),
            'position_ids': position_ids.numpy().astype(np.int32),
            'causal_mask': single_causal_mask.numpy().astype(np.float16),
            'current_pos': np.array([batch_pos], dtype=np.int32)
        }
        single_model.predict(inputs, state)
        batch_pos += 1

    return torch.tensor([context_pos], dtype=torch.int32)


def run_monolithic_prefill_with_rotation(prefill_model, prefill_rotate_model, input_ids, context_pos,
                                         context_length, batch_size, state, causal_mask, sliding_window,
                                         infer_rotate_model=None, infer_model=None,
                                         single_token_mode=False, mask_len=None, use_update_mask=True):
    """Run prefill with rotation support for long contexts.

    When context_pos > sliding_window, this splits the prefill into two phases:
    - Phase 1: Fill mode (prefill_model) for positions 0 to sliding_window-1
    - Phase 2: Rotation mode (prefill_rotate_model) for positions sliding_window to context_pos-1

    If prefill_rotate_model is None or context_pos <= sliding_window, falls back to standard prefill.
    """
    # If no rotation model or short context, use standard prefill
    if prefill_rotate_model is None or sliding_window is None or context_pos <= sliding_window:
        return run_monolithic_prefill(prefill_model, input_ids, context_pos, context_length,
                                      batch_size, state, causal_mask,
                                      infer_model=infer_model,
                                      single_token_mode=single_token_mode,
                                      mask_len=mask_len,
                                      use_update_mask=use_update_mask)

    if mask_len is None:
        mask_len = max(context_length, sliding_window)

    # In single-token mode, route every token through infer / infer_rotate to avoid
    # batch prefill paths entirely.
    if single_token_mode:
        batch_pos = 0
        while batch_pos < context_pos:
            token = input_ids[:, batch_pos:batch_pos + 1]
            position_ids = torch.tensor([batch_pos], dtype=torch.int32)
            single_causal_mask = causal_mask[:, :, batch_pos:batch_pos + 1, :]

            use_rotation = infer_rotate_model is not None and batch_pos >= sliding_window
            single_model = infer_rotate_model if use_rotation else (
                infer_model if infer_model is not None else prefill_model
            )

            inputs = {
                'input_ids': token.numpy().astype(np.int32),
                'position_ids': position_ids.numpy().astype(np.int32),
                'causal_mask': single_causal_mask.numpy().astype(np.float16),
                'current_pos': position_ids.numpy().astype(np.int32)
            }
            single_model.predict(inputs, state)
            batch_pos += 1

        return torch.tensor([context_pos], dtype=torch.int32)

    # Phase 1: Fill mode for positions 0 to sliding_window-1
    batch_pos = 0
    while batch_pos + batch_size <= sliding_window:
        batch_end = batch_pos + batch_size

        batch_input = input_ids[:, batch_pos:batch_end]

        position_ids = torch.arange(batch_pos, batch_pos + batch_size, dtype=torch.int32)
        batch_causal_mask = causal_mask[:, :, batch_pos:batch_pos + batch_size, :]

        inputs = {
            'input_ids': batch_input.numpy().astype(np.int32),
            'position_ids': position_ids.numpy().astype(np.int32),
            'causal_mask': batch_causal_mask.numpy().astype(np.float16),
            'current_pos': np.array([batch_pos], dtype=np.int32)
        }
        update_mask = make_update_mask(mask_len, batch_pos, batch_size) if use_update_mask else None
        _predict_with_optional_update_mask(prefill_model, inputs, state, update_mask)
        batch_pos = batch_end

    # Handle any non-divisible fill tail before switching to rotate mode.
    while batch_pos < sliding_window:
        token = input_ids[:, batch_pos:batch_pos + 1]
        position_ids = torch.tensor([batch_pos], dtype=torch.int32)
        single_causal_mask = causal_mask[:, :, batch_pos:batch_pos + 1, :]

        inputs = {
            'input_ids': token.numpy().astype(np.int32),
            'position_ids': position_ids.numpy().astype(np.int32),
            'causal_mask': single_causal_mask.numpy().astype(np.float16),
            'current_pos': position_ids.numpy().astype(np.int32)
        }
        fill_single_model = infer_model if infer_model is not None else prefill_model
        fill_single_model.predict(inputs, state)
        batch_pos += 1

    # Phase 2: Rotation mode for positions sliding_window to context_pos-1
    batch_pos = max(batch_pos, sliding_window)
    # Process full batches with prefill_rotate
    while batch_pos + batch_size <= context_pos:
        batch_end = batch_pos + batch_size

        batch_input = input_ids[:, batch_pos:batch_end]
        position_ids = torch.arange(batch_pos, batch_end, dtype=torch.int32)
        batch_causal_mask = causal_mask[:, :, batch_pos:batch_end, :]

        inputs = {
            'input_ids': batch_input.numpy().astype(np.int32),
            'position_ids': position_ids.numpy().astype(np.int32),
            'causal_mask': batch_causal_mask.numpy().astype(np.float16),
            'current_pos': np.array([batch_pos], dtype=np.int32)
        }
        update_mask = make_update_mask(mask_len, batch_pos, batch_size) if use_update_mask else None
        _predict_with_optional_update_mask(prefill_rotate_model, inputs, state, update_mask)
        batch_pos = batch_end

    # Handle remainder tokens without padding (token-by-token rotation)
    if batch_pos < context_pos:
        if infer_rotate_model is not None:
            while batch_pos < context_pos:
                token = input_ids[:, batch_pos:batch_pos + 1]
                position_ids = torch.tensor([batch_pos], dtype=torch.int32)
                single_causal_mask = causal_mask[:, :, batch_pos:batch_pos + 1, :]

                inputs = {
                    'input_ids': token.numpy().astype(np.int32),
                    'position_ids': position_ids.numpy().astype(np.int32),
                    'causal_mask': single_causal_mask.numpy().astype(np.float16),
                    'current_pos': position_ids.numpy().astype(np.int32)
                }
                infer_rotate_model.predict(inputs, state)
                batch_pos += 1
        elif infer_model is not None:
            while batch_pos < context_pos:
                token = input_ids[:, batch_pos:batch_pos + 1]
                position_ids = torch.tensor([batch_pos], dtype=torch.int32)
                single_causal_mask = causal_mask[:, :, batch_pos:batch_pos + 1, :]

                inputs = {
                    'input_ids': token.numpy().astype(np.int32),
                    'position_ids': position_ids.numpy().astype(np.int32),
                    'causal_mask': single_causal_mask.numpy().astype(np.float16),
                    'current_pos': position_ids.numpy().astype(np.int32)
                }
                infer_model.predict(inputs, state)
                batch_pos += 1
        else:
            # Fallback: do nothing special if no infer model is available
            pass

    return torch.tensor([context_pos], dtype=torch.int32)


def generate_next_token_monolithic(model, input_ids, pos, context_length, metadata, state, causal_mask, temperature=0.0):
    """Generate next token using monolithic model."""
    # Get current token
    current_token = input_ids[:, pos-1:pos]  # [1, 1]

    # Create inputs
    position_ids = torch.tensor([pos-1], dtype=torch.int32)
    single_causal_mask = causal_mask[:, :, pos-1:pos, :]

    # Run monolithic infer
    inputs = {
        'input_ids': current_token.numpy().astype(np.int32),
        'position_ids': position_ids.numpy().astype(np.int32),
        'causal_mask': single_causal_mask.numpy().astype(np.float16),
        'current_pos': position_ids.numpy().astype(np.int32)
    }
    output = model.predict(inputs, state)

    # Check if model uses argmax_in_model mode (outputs 2 tensors instead of logits)
    argmax_in_model = metadata.get('argmax_in_model', False)

    if argmax_in_model and 'argmax_idx' in output:
        # Argmax-in-model mode: find the chunk with highest value and compute global index
        # Model outputs LOCAL indices (0 to chunk_size-1) for each chunk
        # We compute: global_idx = local_idx + (best_chunk * chunk_size)
        argmax_idx = output['argmax_idx']  # shape: [num_chunks], LOCAL indices
        argmax_val = output['argmax_val']  # shape: [num_chunks]

        # Flatten in case of extra dimensions
        argmax_idx_flat = argmax_idx.flatten()
        argmax_val_flat = argmax_val.flatten()

        # Find chunk with highest value
        best_chunk = int(np.argmax(argmax_val_flat))
        local_idx = int(argmax_idx_flat[best_chunk])

        num_chunks = len(argmax_idx_flat)
        chunk_sizes, chunk_offsets = _resolve_argmax_chunk_layout(
            metadata,
            num_chunks,
            argmax_idx_flat=argmax_idx_flat,
        )
        global_idx = local_idx + chunk_offsets[best_chunk]

        # Debug: print shapes and values
        if metadata.get('debug_argmax', False):
            print(f"\n=== Argmax Debug (pos={pos}) ===")
            print(f"argmax_idx shape: {argmax_idx.shape}, dtype: {argmax_idx.dtype}")
            print(f"argmax_val shape: {argmax_val.shape}, dtype: {argmax_val.dtype}")
            print("Per-chunk results (LOCAL indices):")

            # Find top 3 chunks by value for comparison
            sorted_indices = np.argsort(argmax_val_flat)[::-1][:3]

            for i in range(len(argmax_idx_flat)):
                local = int(argmax_idx_flat[i])
                val = float(argmax_val_flat[i])
                chunk_size = chunk_sizes[i]
                chunk_offset = chunk_offsets[i]
                computed_global = local + chunk_offset
                in_range = 0 <= local < chunk_size
                marker = " <-- SELECTED" if i == best_chunk else ""
                if i in sorted_indices and i != best_chunk:
                    marker += f" (top-{list(sorted_indices).index(i)+1})"
                range_ok = "✓" if in_range else f"✗ (expected 0-{chunk_size-1})"
                print(
                    f"  Chunk {i:2d}: local={local:5d}, global={computed_global:6d}, "
                    f"offset={chunk_offset:6d}, size={chunk_size:5d}, val={val:8.4f}, range={range_ok}{marker}"
                )

            print(f"Result: best_chunk={best_chunk}, local_idx={local_idx}, global_idx={global_idx}, best_val={argmax_val_flat[best_chunk]:.4f}")

            # Value comparison: show if there are close competing values
            top_values = [float(argmax_val_flat[i]) for i in sorted_indices]
            if len(top_values) >= 2:
                val_diff = abs(top_values[0] - top_values[1])
                print(f"Value comparison: top-1={top_values[0]:.6f}, top-2={top_values[1]:.6f}, diff={val_diff:.6f}")
                if val_diff < 0.01:
                    print(f"  WARNING: Values are very close - possible precision issue!")

        return global_idx

    # Get number of logits from metadata
    num_logits = metadata.get('split_lm_head', metadata.get('num_logits', 8))

    # Combine logits1-N if they exist
    if 'logits1' in output:
        logits_parts = []
        for i in range(1, num_logits + 1):
            key = f'logits{i}'
            if key in output:
                logits_parts.append(torch.from_numpy(output[key]))
        logits = torch.cat(logits_parts, dim=-1)
    elif 'logits' in output:
        logits = torch.from_numpy(output['logits'])
    else:
        # Try other common output names
        for key in output.keys():
            if 'logit' in key.lower():
                logits = torch.from_numpy(output[key])
                break

    # Apply temperature and sample
    if temperature > 0:
        logits = logits / temperature
        probs = F.softmax(logits[0, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
    else:
        next_token = torch.argmax(logits[0, -1, :]).item()

    return next_token


def chat_loop_monolithic(infer_model, prefill_model, tokenizer, metadata, state, causal_mask=None,
                         auto_prompt=None, warmup=False, save_file=None, max_tokens=None,
                         no_template=False, eval_mode=False, infer_rotate_model=None, prefill_rotate_model=None,
                         no_think=False):
    """Chat loop for monolithic models.

    Args:
        infer_model: Model for single-token inference (fill mode, pos < sliding_window)
        prefill_model: Model for batch prefill (fill mode, for positions 0 to sliding_window-1)
        tokenizer: Tokenizer
        metadata: Model metadata dict
        state: CoreML state object
        causal_mask: Causal mask tensor
        auto_prompt: Optional auto-prompt string
        warmup: If True, skip output
        save_file: Optional file to save conversation
        max_tokens: Maximum tokens to generate
        no_template: If True, don't use chat template
        eval_mode: If True, minimal output for evaluation
        infer_rotate_model: Optional model for single-token inference with cache rotation
                           (rotation mode, pos >= sliding_window). If None, uses infer_model.
        prefill_rotate_model: Optional model for batch prefill with cache rotation
                              (rotation mode, for positions >= sliding_window). If None,
                              uses prefill_model for all positions (legacy behavior).
    """
    context_length = metadata.get('context_length')
    batch_size = metadata.get('batch_size', 64)
    sliding_window = metadata.get('sliding_window', None)  # For switching between infer modes
    mask_len = max(metadata.get('state_length', context_length), sliding_window or 0)
    update_mask_prefill = metadata.get('update_mask_prefill', False)
    single_token_mode = False
    if not update_mask_prefill:
        single_token_mode = True
        if not eval_mode and not warmup:
            print("update_mask_prefill not set; forcing single-token prefill for compatibility.")

    if not warmup and not eval_mode:
        print(f"\nUsing context length: {context_length}")
        if infer_rotate_model is not None:
            print(f"Cache rotation: ENABLED (infer_rotate function available)")
        else:
            print(f"Cache rotation: NOT AVAILABLE (using infer for all positions)")
        print("\nStarting chat session. Press Ctrl+D to exit.")

    # Check chat template
    template_kwargs = {"enable_thinking": False} if no_think else {}
    has_chat_template = False
    try:
        test_messages = [{"role": "user", "content": "test"}]
        tokenizer.apply_chat_template(test_messages, return_tensors="pt", **template_kwargs)
        has_chat_template = True
        if not warmup and not eval_mode:
            print("\nUsing chat template for prompts")
    except:
        if not warmup and not eval_mode:
            print("\nUsing manual formatting for prompts")

    stop_token_ids = build_stop_token_ids(tokenizer)

    try:
        while True:
            try:
                if not warmup and not eval_mode:
                    print(f"\n{LIGHT_GREEN}You:{RESET_COLOR}", end=' ', flush=True)
                if auto_prompt is not None:
                    user_input = auto_prompt
                    if not warmup and not eval_mode:
                        print(user_input)
                else:
                    user_input = input().strip()
            except EOFError:
                if not warmup and not eval_mode:
                    print("\nExiting chat...")
                break

            if not user_input:
                continue

            # Format prompt
            if no_template:
                input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=True).input_ids.to(torch.int32)
            elif has_chat_template:
                messages = [{"role": "user", "content": user_input}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    **template_kwargs
                ).to(torch.int32)
            else:
                formatted_prompt = f"[INST] {user_input} [/INST]"
                input_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(torch.int32)

            context_pos = input_ids.size(1)

            if not warmup and not eval_mode:
                print(f"\n{LIGHT_BLUE}Assistant:{RESET_COLOR}", end=' ', flush=True)

            token_printer = TokenPrinter(tokenizer)
            tokens_generated = 0

            try:
                prefill_start = time.time()

                # Run prefill with monolithic model (uses rotation for pos >= sliding_window if available)
                _ = run_monolithic_prefill_with_rotation(
                    prefill_model, prefill_rotate_model, input_ids, context_pos, context_length,
                    batch_size, state, causal_mask, sliding_window, infer_rotate_model,
                    infer_model=infer_model, single_token_mode=single_token_mode, mask_len=mask_len,
                    use_update_mask=update_mask_prefill
                )

                prefill_time = time.time() - prefill_start
                prefill_tokens_per_sec = context_pos / prefill_time if prefill_time > 0 else 0

                # Generation loop
                pos = context_pos
                inference_start = time.time()
                inference_tokens = 0

                while pos < context_length - 1:
                    # Switch on the actual token position passed to the model (pos - 1).
                    current_pos = pos - 1
                    if sliding_window is not None and current_pos >= sliding_window and infer_rotate_model is not None:
                        current_infer_model = infer_rotate_model
                    else:
                        current_infer_model = infer_model

                    next_token = generate_next_token_monolithic(
                        current_infer_model, input_ids, pos, context_length, metadata, state, causal_mask
                    )

                    if pos < input_ids.size(1):
                        input_ids[0, pos] = next_token
                    else:
                        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.int32)], dim=1)

                    if not warmup:
                        token_printer.add_token(next_token)
                        token_printer.drain_buffer(eval_mode)

                    pos += 1
                    tokens_generated += 1
                    inference_tokens += 1

                    if warmup and tokens_generated >= WARMUP_TOKEN_LIMIT:
                        break
                    if max_tokens is not None and tokens_generated >= max_tokens:
                        break

                    if next_token in stop_token_ids:
                        break

                inference_time = time.time() - inference_start
                inference_tokens_per_sec = inference_tokens / inference_time if inference_time > 0 else 0

                if not warmup:
                    response = token_printer.stop(eval_mode)
                    if eval_mode:
                        print(response, end='')
                    else:
                        prefill_ms = prefill_time * 1000
                        print(f"\nPrefill: {prefill_ms:.1f}ms ({prefill_tokens_per_sec:.1f} t/s)")
                        print(f"Inference: {inference_tokens_per_sec:.1f} t/s")
                        print(f"Total: Generated {tokens_generated} tokens in {prefill_time + inference_time:.2f}s")

                    if save_file and not eval_mode:
                        try:
                            time.sleep(0.5)
                            with open(save_file, 'w') as f:
                                f.write(response)
                            print(f"\n{DARK_BLUE}Response saved to file: {save_file}{RESET_COLOR}")
                        except Exception as e:
                            print(f"\n{DARK_BLUE}Error saving to file: {str(e)}{RESET_COLOR}")
                else:
                    token_printer.stop(eval_mode)

                if auto_prompt is not None:
                    break

            except KeyboardInterrupt:
                if not eval_mode:
                    print("\nGeneration interrupted")
                token_printer.stop(eval_mode)
                continue

    except Exception as e:
        print(f"\nError in chat loop: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()
    
    # Convert directory to absolute path
    model_dir = Path(args.d).resolve()
    if not model_dir.exists():
        if not args.eval:
            print(f"\nError: Model directory not found: {model_dir}")
        return 1
        
    if not args.eval:
        print(f"\nUsing model directory: {model_dir}")
        print(f"Context length: {args.context_length}")
    
    try:
        # Handle tokenizer path
        if args.tokenizer is None:
            args.tokenizer = str(model_dir)

        # Check if tokenizer directory exists and has required files
        tokenizer_path = Path(args.tokenizer)
        if not tokenizer_path.exists():
            if not args.eval:
                print(f"\nError: Tokenizer directory not found: {args.tokenizer}")
            return 1

        required_files = ['tokenizer.json', 'tokenizer_config.json']
        missing_files = [f for f in required_files if not (tokenizer_path / f).exists()]

        if missing_files and not args.eval:
            print(f"\nWarning: Tokenizer directory missing required files: {missing_files}")
            print(f"Current tokenizer path: {args.tokenizer}")
            print("\nFor Qwen models, you may need to specify the original model directory:")
            print("  python chat.py --meta /tmp/qwen/meta.yaml --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/YOUR_SNAPSHOT_ID")

        args.tokenizer = str(Path(args.tokenizer).resolve())
        if not args.eval:
            print(f"Using tokenizer path: {args.tokenizer}")

        # Load tokenizer
        tokenizer = initialize_tokenizer(args.tokenizer, args.eval)
        if tokenizer is None:
            raise RuntimeError("Failed to initialize tokenizer")

        metadata = {}

        # Branch based on model type
        if getattr(args, 'is_monolithic', False):
            # MONOLITHIC MODEL PATH
            infer_model, infer_rotate_model, prefill_model, prefill_rotate_model, metadata = load_monolithic_model(args, metadata)

            # Override context length from command line if provided
            if args.context_length is not None:
                metadata['context_length'] = args.context_length
            # Use state_length from args (parsed from YAML) or default to context_length
            metadata['state_length'] = getattr(args, 'state_length', metadata['context_length'])

            # Set metadata values
            metadata['batch_size'] = getattr(args, 'batch_size', 64)
            metadata['split_lm_head'] = getattr(args, 'split_lm_head', 16)
            metadata['argmax_in_model'] = getattr(args, 'argmax_in_model', False)
            metadata['debug_argmax'] = getattr(args, 'debug_argmax', False)
            metadata['debug'] = getattr(args, 'debug', False)
            metadata['sliding_window'] = getattr(args, 'sliding_window', None)
            metadata['vocab_size'] = getattr(args, 'vocab_size', None)
            metadata['lm_head_chunk_sizes'] = _parse_lm_head_chunk_sizes(
                getattr(args, 'lm_head_chunk_sizes', None)
            )

            if not args.eval:
                print(f"\nMonolithic metadata: {metadata}")

            # Create state from infer model
            state = infer_model.make_state()
            if not args.eval:
                print("\nCreated unified transformer state for monolithic model")
            _maybe_report_mem("after monolithic make_state", getattr(args, "mem_report", False))

            # Initialize causal mask - use state_length for split cache models
            causal_mask = initialize_causal_mask(metadata['state_length'], args.eval)

            # Warmup runs
            if not args.nw and not args.eval:
                for _ in range(2):
                    chat_loop_monolithic(
                        infer_model=infer_model,
                        infer_rotate_model=infer_rotate_model,
                        prefill_model=prefill_model,
                        prefill_rotate_model=prefill_rotate_model,
                        tokenizer=tokenizer,
                        metadata=metadata,
                        state=state,
                        causal_mask=causal_mask,
                        warmup=True,
                        auto_prompt="who are you?",
                        no_template=args.no_template,
                        eval_mode=args.eval,
                        no_think=args.no_think
                    )

            # Main run
            chat_loop_monolithic(
                infer_model=infer_model,
                infer_rotate_model=infer_rotate_model,
                prefill_model=prefill_model,
                prefill_rotate_model=prefill_rotate_model,
                tokenizer=tokenizer,
                metadata=metadata,
                state=state,
                causal_mask=causal_mask,
                warmup=False,
                auto_prompt=args.prompt,
                save_file=args.save,
                max_tokens=args.max_tokens,
                no_template=args.no_template,
                eval_mode=args.eval,
                no_think=args.no_think
            )

        else:
            # CHUNKED MODEL PATH (original code)
            # Update paths to be relative to model directory
            args.embed = str(model_dir / args.embed)
            args.ffn = str(model_dir / args.ffn)
            args.lmhead = str(model_dir / args.lmhead)

            # Load models and extract metadata
            embed_model, ffn_models, lmhead_model, metadata = load_models(args, metadata)

            if not args.eval:
                print(f"\nMetadata befor args.context_length: {metadata}")

            # Override context length from command line if provided
            if args.context_length is not None:
                metadata['context_length'] = args.context_length
                metadata['state_length'] = args.context_length
                if not args.eval:
                    print(f"\nOverriding context length from command line: {args.context_length}")

            # Add num_logits to metadata (legacy support)
            metadata['num_logits'] = getattr(args, 'num_logits', 8)

            # Add split_lm_head to metadata (preferred)
            metadata['split_lm_head'] = getattr(args, 'split_lm_head', getattr(args, 'num_logits', 8))

            # Add debug flag
            metadata['debug'] = getattr(args, 'debug', False)

            # Add argmax_in_model flag for chunked models
            metadata['argmax_in_model'] = getattr(args, 'argmax_in_model', False)
            metadata['debug_argmax'] = getattr(args, 'debug_argmax', False)
            metadata['vocab_size'] = getattr(args, 'vocab_size', None)
            metadata['lm_head_chunk_sizes'] = _parse_lm_head_chunk_sizes(
                getattr(args, 'lm_head_chunk_sizes', None)
            )

            if not args.eval:
                print(f"\nMetadata after load_models: {metadata}")
                print(f"Using split_lm_head value: {metadata.get('split_lm_head', 8)}")
                if metadata.get('argmax_in_model'):
                    print("Argmax mode enabled for LM head")

            # Create unified state once (use chunk with global cache if available)
            state = create_unified_state(ffn_models, metadata['context_length'], args.eval, metadata=metadata)
            _maybe_report_mem("after chunked make_state", getattr(args, "mem_report", False))

            # Initialize causal mask once
            # For Gemma3 with split cache, use attention_size (sliding window) for causal mask
            attention_size = getattr(args, 'attention_size', metadata['context_length'])
            metadata['attention_size'] = attention_size

            # Add sliding_window for Gemma3 rotation support
            sliding_window = getattr(args, 'sliding_window', None)
            metadata['sliding_window'] = sliding_window
            if sliding_window is not None and not args.eval:
                context_len = metadata['context_length']
                if context_len > sliding_window:
                    print(f"Sliding window: {sliding_window} (rotation enabled for pos >= {sliding_window})")
                else:
                    print(f"Sliding window: {sliding_window} (rotation disabled - context {context_len} <= sliding_window)")

            causal_mask = initialize_causal_mask(attention_size, args.eval)

            # Warmup runs to prevent Python GIL issues with CoreML
            if not args.nw and not args.eval:
                for _ in range(2):
                    chat_loop(
                        embed_model=embed_model,
                        ffn_models=ffn_models,
                        lmhead_model=lmhead_model,
                        tokenizer=tokenizer,
                        metadata=metadata,
                        state=state,
                        causal_mask=causal_mask,
                        warmup=True,
                        auto_prompt="who are you?",
                        no_template=args.no_template,
                        eval_mode=args.eval,
                        no_think=args.no_think
                    )

            # Main run
            chat_loop(
                embed_model=embed_model,
                ffn_models=ffn_models,
                lmhead_model=lmhead_model,
                tokenizer=tokenizer,
                metadata=metadata,
                state=state,
                causal_mask=causal_mask,
                warmup=False,
                auto_prompt=args.prompt,
                save_file=args.save,
                max_tokens=args.max_tokens,
                no_template=args.no_template,
                eval_mode=args.eval,
                no_think=args.no_think
            )
        
    except Exception as e:
        if not args.eval:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
