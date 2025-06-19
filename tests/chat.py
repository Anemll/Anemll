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

    def drain_buffer(self):
        """Decode token IDs from decoding_buffer in the main thread."""
        if not self.decoding_buffer:
            return

        # Decode all tokens at once in the main thread
        token_str = self.tokenizer.decode(self.decoding_buffer)
        self.decoding_buffer.clear()
        
        # Store the text in buffer for later saving to file
        with self.lock:
            self.buffer += token_str

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

    def stop(self):
        """Stop the printer thread."""
        if self.thread and self.thread.is_alive():
            # Ensure any remaining tokens are processed
            self.drain_buffer()
            self.stop_event.set()
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass
            # Calculate and print tokens/s with shorter format in blue
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
            print(f"Found model at: {candidate}")
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
                print(f"Found model at: {candidate}")
                return str(candidate)
        
        # Add no-LUT candidates to the list for error reporting
        candidates.extend(candidates_no_lut)
            
    # If we get here, no valid path was found
    print("\nError: Model not found. Tried following paths:")
    for candidate in candidates:
        print(f"  {candidate}")
    raise FileNotFoundError(f"Model not found: {path}")

def parse_ffn_filename(path):
    """Parse FFN model filename to extract chunk information."""
    path = Path(path)
    pattern = r'FFN_PF.*_chunk_(\d+)of(\d+)'
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

def load_model(path, function_name=None):
    """Load a CoreML model, handling both .mlmodelc and .mlpackage formats."""
    path = Path(path)
    compute_unit = ct.ComputeUnit.CPU_AND_NE
    
    try:
        if path.suffix == '.mlmodelc':
            # For compiled models (.mlmodelc), use CompiledMLModel
            if function_name:
                return ct.models.CompiledMLModel(str(path), compute_unit, function_name=function_name)
            else:
                return ct.models.CompiledMLModel(str(path), compute_unit)
        else:
            # For packages (.mlpackage)
            if function_name:
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
        print("\nUsing parameters:")
        print(f"  Context Length: {metadata['context_length']}")
        print(f"  State Length: {metadata['state_length']}")
        print(f"  Prefill Batch Size: {metadata['batch_size']}")
        print(f"  LUT Bits: {metadata['lut_bits']}")
        print(f"  Number of Chunks: {metadata['num_chunks']}")

    # Override with values from args if they exist
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        metadata['batch_size'] = args.batch_size
        print(f"\nOverriding batch size from args: {args.batch_size}")
    if hasattr(args, 'num_chunks') and args.num_chunks is not None:
        metadata['num_chunks'] = args.num_chunks
        print(f"\nOverriding num chunks from args: {args.num_chunks}")
    
    return metadata
    
def load_models(args,metadata):
    """Load all required models and extract metadata."""
    print("\nLoading models...")
    
    try:
        # Load embeddings model
        print("\nLoading embeddings model...")
        embed_path = parse_model_path(args.embed)
        print(f"Loading from: {embed_path}")
        embed_model = load_model(embed_path)
        print("Embeddings model loaded successfully")
        metadata = load_metadata(embed_model,args)
        

        
        # Load LM head model
        print("\nLoading LM head model...")
        lmhead_path = parse_model_path(args.lmhead)
        print(f"Loading from: {lmhead_path}")
        lmhead_model = load_model(lmhead_path)
        print("LM head model loaded successfully")
        
        # Parse FFN path and find chunks if needed
        print("\nLoading FFN+PREFILL model(s)...")
        ffn_path = parse_model_path(args.ffn)
        chunk_no, total_chunks = parse_ffn_filename(ffn_path)
        
        ffn_models = []
        if chunk_no and total_chunks:
            print(f"\nDetected chunked FFN+PREFILL model ({total_chunks} chunks)")
            # Find and load all chunks
            chunk_paths = find_all_chunks(ffn_path)
            if len(chunk_paths) != total_chunks:
                raise ValueError(f"Found {len(chunk_paths)} chunks but filename indicates {total_chunks} chunks")
                
            for chunk_path in chunk_paths:
                print(f"\nLoading FFN+PREFILL chunk: {Path(chunk_path).name}")
                try:
                    # For chunked models, we need both infer and prefill functions
                    ffn_models.append({
                        'infer': load_model(chunk_path, function_name='infer'),
                        'prefill': load_model(chunk_path, function_name='prefill')
                    })
                    print("Chunk loaded successfully")
                except Exception as e:
                    print(f"Error loading chunk {chunk_path}: {str(e)}")
                    raise
            metadata = load_metadata(ffn_models[0],args)

        else:
            print("\nLoading single FFN model...")
            ffn_models.append(load_model(ffn_path))
            print("FFN model loaded successfully")
        
        return embed_model, ffn_models, lmhead_model, metadata
        
    except Exception as e:
        print(f"\nError loading models: {str(e)}")
        print("\nPlease ensure all model files exist and are accessible.")
        print("Expected files:")
        print(f"  Embeddings: {args.embed}")
        print(f"  LM Head: {args.lmhead}")
        print(f"  FFN: {args.ffn}")
        raise

# At the top of the file, make this a default path

def initialize_tokenizer(model_path=None):
    """Initialize and configure the tokenizer."""
    try:

        
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), 
            use_fast=False,
            trust_remote_code=True
        )
        
        print("\nTokenizer Configuration:")
        print(f"Tokenizer type: {type(tokenizer)}")
        print(f"Tokenizer name: {tokenizer.__class__.__name__}")
        print(f"Vocabulary size: {len(tokenizer)}")
        print(f"Model max length: {tokenizer.model_max_length}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Set PAD token to EOS token")
        
        tokenizer.padding_side = "left"
        
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

def initialize_causal_mask(context_length):
    """Initialize causal mask for transformer attention."""
    causal_mask = make_causal_mask(context_length, 0)
    causal_mask = torch.tensor(causal_mask, dtype=torch.float16)
    print(f"\nInitialized causal mask for context length {context_length}")
    return causal_mask

def run_prefill(embed_model, ffn_models, input_ids, context_pos, context_length, batch_size=64, state=None, causal_mask=None):
    """Run prefill on the input sequence."""
    # Use provided causal mask or create one if not provided
    if causal_mask is None:
        causal_mask = make_causal_mask(context_length, 0)
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
        
        # Run embeddings
        hidden_states = torch.from_numpy(
            embed_model.predict({
                'input_ids': batch_input.numpy()
            })['hidden_states']
        )
        
        # Run through FFN chunks with state
        for ffn_model in ffn_models:
            if isinstance(ffn_model, dict):
                inputs = {
                    'hidden_states': hidden_states.numpy(),  # [1, 64, hidden_size]
                    'position_ids': position_ids.numpy(),    # [64]
                    'causal_mask': batch_causal_mask.numpy(), # [1, 1, 64, context_length]
                    'current_pos': np.array([batch_pos], dtype=np.int32)  # [1]
                }
                output = ffn_model['prefill'].predict(inputs, state)
                hidden_states = torch.from_numpy(output['output_hidden_states'])
        
        batch_pos = batch_end
    
    return torch.tensor([context_pos], dtype=torch.int32)

def generate_next_token(embed_model, ffn_models, lmhead_model, input_ids, pos, context_length, metadata, state=None, causal_mask=None, temperature=0.0):
    """Generate the next token."""
    # Get current token
    current_token = input_ids[:, pos-1:pos]  # [1, 1]
    
    # Run embeddings
    hidden_states = torch.from_numpy(
        embed_model.predict({'input_ids': current_token.numpy()})['hidden_states']
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
    
    # Run through FFN chunks with state
    for ffn_model in ffn_models:
        if isinstance(ffn_model, dict):
            inputs = {
                'hidden_states': hidden_states.numpy(),
                'update_mask': update_mask.numpy(),
                'position_ids': position_ids.numpy(),
                'causal_mask': single_causal_mask.numpy(),
                'current_pos': position_ids.numpy()
            }
            output = ffn_model['infer'].predict(inputs, state)
            hidden_states = torch.from_numpy(output['output_hidden_states'])
    
    # Run LM head
    lm_output = lmhead_model.predict({'hidden_states': hidden_states.numpy()})
    # Debug print
    #print("\nLM Head output keys:", list(lm_output.keys()))
    
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

def create_unified_state(ffn_models, context_length):
    """Create unified KV cache state for transformer."""
    if isinstance(ffn_models[0], dict):
        # Use first FFN model's prefill function to create state
        state = ffn_models[0]['prefill'].make_state()
        print(f"\nCreated unified transformer state for {len(ffn_models)} chunks")
        return state
    else:
        state = ffn_models[0].make_state()
        print("\nCreated unified transformer state")
        return state

def chat_loop(embed_model, ffn_models, lmhead_model, tokenizer, metadata, state, causal_mask=None, auto_prompt=None, warmup=False, save_file=None, max_tokens=None):
    """Interactive chat loop."""
    context_length = metadata.get('context_length')
    batch_size = metadata.get('batch_size', 64)
    
    if not warmup:
        print(f"\nUsing context length: {context_length}")
        print("\nStarting chat session. Press Ctrl+D to exit.")
        print("Type your message and press Enter to chat.")
    
    # Check if tokenizer has chat template and if it works
    has_chat_template = False
    try:
        # Test if chat template works
        test_messages = [{"role": "user", "content": "test"}]
        tokenizer.apply_chat_template(test_messages, return_tensors="pt")
        has_chat_template = True
        if not warmup:
            print("\nUsing chat template for prompts")
    except:
        if not warmup:
            print("\nUsing manual formatting for prompts")
    
    conversation = []
    
    try:
        while True:
            try:
                if not warmup:
                    print(f"\n{LIGHT_GREEN}You:{RESET_COLOR}", end=' ', flush=True)
                if auto_prompt is not None:
                    user_input = auto_prompt
                    if not warmup:
                        print(user_input)
                else:
                    user_input = input().strip()
            except EOFError:
                if not warmup:
                    print("\nExiting chat...")
                break
                
            if not user_input:
                continue
            
            # Format prompt based on tokenizer capabilities
            if has_chat_template:
                messages = [{"role": "user", "content": user_input}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True
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
            
            if not warmup:
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
                    print(f"Warning: batch_size was None, using default: {batch_size}")
                
                _ = run_prefill(
                    embed_model,
                    ffn_models,
                    input_ids,
                    context_pos,
                    context_length,
                    batch_size,
                    state,
                    causal_mask
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
                        token_printer.drain_buffer()
                    
                    pos += 1
                    tokens_generated += 1
                    inference_tokens += 1
                    
                    # Check limits
                    if warmup and tokens_generated >= WARMUP_TOKEN_LIMIT:
                        break
                    
                    # Check max_tokens limit
                    if max_tokens is not None and tokens_generated >= max_tokens:
                        break
                        
                    # Check for all possible EOS tokens
                    eos_token_ids = tokenizer.eos_token_id
                    if isinstance(eos_token_ids, list):
                        if next_token in eos_token_ids:
                            break
                    else:
                        if next_token == eos_token_ids:
                            break
                
                # Calculate inference timing
                inference_time = time.time() - inference_start
                inference_tokens_per_sec = inference_tokens / inference_time if inference_time > 0 else 0
                
                # Get final response and add to conversation
                if not warmup:
                    response = token_printer.stop()
                    # Print timing stats
                    prefill_ms = prefill_time * 1000  # Convert to milliseconds
                    print(f"\nPrefill: {prefill_ms:.1f}ms ({prefill_tokens_per_sec:.1f} t/s)")
                    print(f"Inference: {inference_tokens_per_sec:.1f} t/s")
                    print(f"Total: Generated {tokens_generated} tokens in {prefill_time + inference_time:.2f}s")
                    conversation.append({"role": "assistant", "content": response})
                    
                    # Save response to file if requested
                    if save_file:
                        try:
                            # Add small delay to ensure all tokens are processed
                            time.sleep(0.5)
                            
                            # Make sure response ends with EOS token if it's supposed to
                            if response and not response.endswith("<|eot_id|>") and not response.endswith("</s>"):
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
                    token_printer.stop()  # Clean up without printing stats
                
                # Exit after one response in auto_prompt mode
                if auto_prompt is not None:
                    break
                
            except KeyboardInterrupt:
                print("\nGeneration interrupted")
                token_printer.stop()
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
    
    # Model configuration
    parser.add_argument('--context-length', type=int,
                       help='Context length for the model (default: 512), if not provided, it will be detected from the model directory name ctxNUMBER')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for prefill (default: 64)')
    parser.add_argument('--num-logits', type=int, default=8,
                       help='Number of logits outputs from LM head (default: 8, legacy)')
    parser.add_argument('--split-lm-head', type=int, 
                       help='Number of logits splits from LM head (default: 8 for llama, 16 for qwen)')
    
    args = parser.parse_args()
    
    # If meta.yaml is provided, load parameters from it
    if args.meta:
        try:
            with open(args.meta, 'r') as f:
                meta = yaml.safe_load(f)
            params = meta['model_info']['parameters']
            
            # Set model directory to meta.yaml directory if not specified
            if not args.d or args.d == '.':
                args.d = str(Path(args.meta).parent)
            
            # Build model paths based on parameters
            prefix = params.get('model_prefix', 'llama')  # Default to 'llama' if not specified
            lut_ffn = f"_lut{params['lut_ffn']}" if params['lut_ffn'] != 'none' else ''
            lut_lmhead = f"_lut{params['lut_lmhead']}" if params['lut_lmhead'] != 'none' else ''
            lut_embeddings = f"_lut{params['lut_embeddings']}" if params['lut_embeddings'] != 'none' else ''
            num_chunks = int(params['num_chunks'])
            
            # Set model paths if not specified
            if not args.lmhead:
                args.lmhead = f'{prefix}_lm_head{lut_lmhead}'
            if not args.embed:
                args.embed = f'{prefix}_embeddings{lut_embeddings}'  # Changed from lm_head to embeddings
            if not args.ffn:
                args.ffn = f'{prefix}_FFN_PF{lut_ffn}_chunk_01of{num_chunks:02d}'
            if not args.tokenizer:
                # Check if there's a tokenizer_path parameter in meta.yaml
                if 'tokenizer_path' in params:
                    args.tokenizer = params['tokenizer_path']
                else:
                    # Default to the model directory, but this might need manual override
                    args.tokenizer = args.d
            
            # Set other parameters if not overridden by command line
            if args.context_length is None:
                args.context_length = int(params['context_length'])
            if args.batch_size is None:
                args.batch_size = int(params['batch_size'])
            args.num_chunks = num_chunks
            # Add num_logits parameter with default of 8, override command line if present in meta
            if 'num_logits' in params:
                args.num_logits = int(params['num_logits'])
            
            # Add split_lm_head parameter with default of 8
            if 'split_lm_head' in params:
                args.split_lm_head = int(params['split_lm_head'])
            else:
                args.split_lm_head = 8  # Default value for backward compatibility
            
            print(f"\nLoaded parameters from {args.meta}:")
            print(f"  Context Length: {args.context_length}")
            print(f"  Batch Size: {args.batch_size}")
            print(f"  Num Chunks: {args.num_chunks}")
            print(f"  Num Logits: {args.num_logits}")
            print(f"  Split LM Head: {args.split_lm_head}")
            print(f"  Models Directory: {args.d}")
            print(f"  Embeddings: {args.embed}")
            print(f"  LM Head: {args.lmhead}")
            print(f"  FFN: {args.ffn}")
            
        except Exception as e:
            print(f"\nError loading meta.yaml: {str(e)}")
            sys.exit(1)
    else:
        # If no meta.yaml, set default split_lm_head if not provided
        if not hasattr(args, 'split_lm_head') or args.split_lm_head is None:
            args.split_lm_head = args.num_logits  # Use num_logits as fallback
    
    return args

def main():
    args = parse_args()
    
    # Convert directory to absolute path
    model_dir = Path(args.d).resolve()
    if not model_dir.exists():
        print(f"\nError: Model directory not found: {model_dir}")
        return 1
        
    print(f"\nUsing model directory: {model_dir}")
    print(f"Context length: {args.context_length}")
    
    try:
        # Update paths to be relative to model directory
        args.embed = str(model_dir / args.embed)
        args.ffn = str(model_dir / args.ffn)
        args.lmhead = str(model_dir / args.lmhead)
        
        # Handle tokenizer path separately since it's not relative to model_dir
        if args.tokenizer is None:
            args.tokenizer = str(model_dir)
        
        # Check if tokenizer directory exists and has required files
        tokenizer_path = Path(args.tokenizer)
        if not tokenizer_path.exists():
            print(f"\nError: Tokenizer directory not found: {args.tokenizer}")
            return 1
        
        # Check if tokenizer has the required files
        required_files = ['tokenizer.json', 'tokenizer_config.json']
        missing_files = [f for f in required_files if not (tokenizer_path / f).exists()]
        
        if missing_files:
            print(f"\nWarning: Tokenizer directory missing required files: {missing_files}")
            print(f"Current tokenizer path: {args.tokenizer}")
            print("\nFor Qwen models, you may need to specify the original model directory:")
            print("  python chat.py --meta /tmp/qwen/meta.yaml --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/YOUR_SNAPSHOT_ID")
            print("\nOr add 'tokenizer_path' to your meta.yaml file.")
    
        args.tokenizer = str(Path(args.tokenizer).resolve())  # Convert to absolute path
        print(f"Using tokenizer path: {args.tokenizer}")
        
        metadata = {}
        # Load models and extract metadata
        embed_model, ffn_models, lmhead_model, metadata = load_models(args,metadata)
        
        print(f"\nMetadata befor args.context_length: {metadata}")

        # Override context length from command line if provided
        if args.context_length is not None:
            metadata['context_length'] = args.context_length
            metadata['state_length'] = args.context_length  # Also update state_length
            print(f"\nOverriding context length from command line: {args.context_length}")
        
        # Add num_logits to metadata (legacy support)
        metadata['num_logits'] = getattr(args, 'num_logits', 8)
        
        # Add split_lm_head to metadata (preferred)
        metadata['split_lm_head'] = getattr(args, 'split_lm_head', getattr(args, 'num_logits', 8))
        
        print(f"\nMetadata after load_models: {metadata}")
        print(f"Using split_lm_head value: {metadata.get('split_lm_head', 8)}")
        
        # Load tokenizer with resolved path
        tokenizer = initialize_tokenizer(args.tokenizer)
        if tokenizer is None:
            raise RuntimeError("Failed to initialize tokenizer")
        
        # Create unified state once
        state = create_unified_state(ffn_models, metadata['context_length'])
        
        # Initialize causal mask once
        causal_mask = initialize_causal_mask(metadata['context_length'])
        
        # Warmup runs to prevent Python GIL issues with CoreML !
        if not args.nw:
            for _ in range(2):
                chat_loop(
                    embed_model=embed_model,
                    ffn_models=ffn_models,
                    lmhead_model=lmhead_model,
                    tokenizer=tokenizer,
                    metadata=metadata,
                    state=state,
                    causal_mask=causal_mask,  # Pass the causal mask
                    warmup=True,
                    auto_prompt="who are you?"
                )
        
        # Main run
        chat_loop(
            embed_model=embed_model,
            ffn_models=ffn_models,
            lmhead_model=lmhead_model,
            tokenizer=tokenizer,
            metadata=metadata,
            state=state,
            causal_mask=causal_mask,  # Pass the causal mask
            warmup=False,
            auto_prompt=args.prompt,
            save_file=args.save
        )
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
