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

# Add at the top with other constants
WARMUP_TOKEN_LIMIT = 10  # Maximum tokens to generate during warmup
THINKING_MODE = False
THINKING_PROMPT = """You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."""
DEBUG_LEVEL = 0  # Default debug level

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
        # Timing and stats tracking
        self.start_time = time.time()
        self.token_count = 0
        self.prefill_time = 0
        self.inference_time = 0
        self.context_pos = 0
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
            self.stop_event.set()
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass
            print(RESET_COLOR)  # Reset color at the end
        return self.buffer

    def set_timing(self, prefill_time, inference_time, context_pos):
        """Set timing information."""
        self.prefill_time = prefill_time
        self.inference_time = inference_time
        self.context_pos = context_pos

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

def parse_args():
    parser = argparse.ArgumentParser(description='Full Chat with CoreML LLaMA with context window shifting, gil resolved (c) 2025 Anemll')
    
    # Add meta.yaml option
    parser.add_argument('--meta', type=str, help='Path to meta.yaml to load all parameters')
    
    # Add existing arguments
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
    
    # Add no-warmup flag
    parser.add_argument('--nw', action='store_true',
                       help='Skip warmup phase')
    
    # Add debug level
    parser.add_argument('--debug-level', type=int, default=0,
                       help='Debug level (0=none, 1=print prompts, 2=more verbose)')
    
    # Model configuration
    parser.add_argument('--context-length', type=int,
                       help='Context length for the model (default: 512), if not provided, it will be detected from the model directory name ctxNUMBER')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for prefill (default: 64)')
    
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
                args.tokenizer = args.d
            
            # Set other parameters if not overridden by command line
            if args.context_length is None:
                args.context_length = int(params['context_length'])
            if args.batch_size is None:
                args.batch_size = int(params['batch_size'])
            args.num_chunks = num_chunks
            
            # Parse split_lm_head parameter from meta.yaml
            if 'split_lm_head' in params:
                args.split_lm_head = int(params['split_lm_head'])
            else:
                args.split_lm_head = 8  # Default value
            
            print(f"\nLoaded parameters from {args.meta}:")
            print(f"  Context Length: {args.context_length}")
            print(f"  Batch Size: {args.batch_size}")
            print(f"  Num Chunks: {args.num_chunks}")
            print(f"  Split LM Head: {args.split_lm_head}")
            print(f"  Models Directory: {args.d}")
            print(f"  Embeddings: {args.embed}")
            print(f"  LM Head: {args.lmhead}")
            print(f"  FFN: {args.ffn}")
            
        except Exception as e:
            print(f"\nError loading meta.yaml: {str(e)}")
            sys.exit(1)
    
    return args

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
        print("\nThis code requires a Llama 3.2 model for chat template functionality.")
        print("Please provide the path to a Llama 3.2 model directory.")
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

def run_prefill(embed_model, ffn_models, input_ids, current_pos, context_length, batch_size, state, causal_mask):
    """Run prefill on the input sequence."""
    #print(f"[DEBUG] Running prefill from 0 to {current_pos}")
    
    # Process in batches
    batch_pos = 0
    while batch_pos < current_pos:
        batch_end = min(batch_pos + batch_size, current_pos)
        current_batch_size = batch_end - batch_pos
        
        #print(f"[DEBUG] Prefill batch {batch_pos}-{batch_end} (size={current_batch_size})")
        
        # Get current batch
        batch_input = input_ids[:, batch_pos:batch_end]
        
        # Pad to full batch size
        batch_input = F.pad(
            batch_input,
            (0, batch_size - current_batch_size),
            value=0
        )
        
        # Generate position IDs for this batch
        position_ids = torch.arange(batch_pos, batch_pos + batch_size, dtype=torch.int32)
        
        # Use the pre-initialized causal mask and extract the batch portion
        batch_causal_mask = causal_mask[:, :, batch_pos:batch_pos + batch_size, :]
        
        # Run embeddings
        hidden_states = torch.from_numpy(
            embed_model.predict({'input_ids': batch_input.numpy()})['hidden_states']
        )
        
        # Run through FFN chunks
        for ffn_model in ffn_models:
            if isinstance(ffn_model, dict):
                inputs = {
                    'hidden_states': hidden_states.numpy(),
                    'position_ids': position_ids.numpy(),
                    'causal_mask': batch_causal_mask.numpy(),
                    'current_pos': np.array([batch_pos], dtype=np.int32)
                }
                output = ffn_model['prefill'].predict(inputs, state)
                hidden_states = torch.from_numpy(output['output_hidden_states'])
        
        batch_pos = batch_end
    
    return torch.tensor([current_pos], dtype=torch.int32)

def generate_next_token(embed_model, ffn_models, lmhead_model, input_ids, pos, context_length, state, causal_mask, metadata=None, temperature=0.0):
    """Generate the next token."""
    # Get current token
    current_token = input_ids[:, pos-1:pos]
    
    # Run embeddings
    hidden_states = torch.from_numpy(
        embed_model.predict({'input_ids': current_token.numpy()})['hidden_states']
    )
    
    # Create masks
    update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
    update_mask[0, 0, pos-1, 0] = 1.0
    position_ids = torch.tensor([pos-1], dtype=torch.int32)
    
    # Use the pre-initialized causal mask and extract the single position portion
    single_causal_mask = causal_mask[:, :, pos-1:pos, :]
    
    # Run through FFN chunks
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
    
    # Run LM head and get next token
    lm_output = lmhead_model.predict({'hidden_states': hidden_states.numpy()})
    
    if 'logits1' in lm_output:
        logits_parts = []
        for i in range(1, metadata.get('split_lm_head', 8) + 1):
            key = f'logits{i}'
            if key in lm_output:
                logits_parts.append(torch.from_numpy(lm_output[key]))
        logits = torch.cat(logits_parts, dim=-1)
    else:
        logits = torch.from_numpy(lm_output['output_logits'])
    
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

def initialize_causal_mask(context_length):
    """Initialize causal mask for transformer attention."""
    causal_mask = make_causal_mask(context_length, 0)
    causal_mask = torch.tensor(causal_mask, dtype=torch.float16)
    print(f"\nInitialized causal mask for context length {context_length}")
    return causal_mask

def get_user_input():
    """Get input from user, handling special key combinations."""
    global THINKING_MODE
    try:
        import termios
        import tty
        import sys

        def _getch():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        buffer = []
        while True:
            char = _getch()
            
            # Debug: print the character code
            print(f"\nKey pressed: {repr(char)} (hex: {hex(ord(char))})")
            
            # Check for Enter key
            if char == '\r' or char == '\n':
                print()  # Move to next line
                input_text = ''.join(buffer)
                # Check if the command is /t
                if input_text == '/t':
                    THINKING_MODE = not THINKING_MODE
                    print(f"Thinking mode {'ON' if THINKING_MODE else 'OFF'}")
                    buffer = []  # Clear buffer
                    print(f"\n{LIGHT_GREEN}You{' (thinking)' if THINKING_MODE else ''}:{RESET_COLOR}", end=' ', flush=True)
                    continue
                return input_text
                
            # Handle backspace
            if char == '\x7f':  # backspace
                if buffer:
                    buffer.pop()
                    sys.stdout.write('\b \b')  # Erase character
                    sys.stdout.flush()
                continue
                
            # Handle Ctrl-C
            if char == '\x03':  # Ctrl-C
                print("^C")
                raise KeyboardInterrupt
                
            # Print character and add to buffer
            sys.stdout.write(char)
            sys.stdout.flush()
            buffer.append(char)
            
    except ImportError:
        # Fallback for systems without termios
        return input("> ")

def chat_loop(embed_model, ffn_models, lmhead_model, tokenizer, metadata, state, causal_mask, auto_prompt=None, warmup=False):
    """Interactive chat loop."""
    global THINKING_MODE
    global DEBUG_LEVEL
    context_length = metadata.get('context_length')
    batch_size = metadata.get('batch_size', 64)
    
    if not warmup:
        print(f"\nUsing context length: {context_length}")
        print("\nStarting chat session. Press Ctrl+D to exit.")
        print("Type your message and press Enter to chat. Use /t to toggle thinking mode.")
        print(f"Thinking mode is {'ON' if THINKING_MODE else 'OFF'}")
    
    # Keep track of conversation history
    conversation = []
    
    try:
        while True:
            try:
                if not warmup:
                    print(f"\n{LIGHT_GREEN}You{' (thinking)' if THINKING_MODE else ''}:{RESET_COLOR}", end=' ', flush=True)
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

            # Handle /t command
            if user_input == "/t":
                THINKING_MODE = not THINKING_MODE
                print(f"Thinking mode {'ON' if THINKING_MODE else 'OFF'}")
                continue
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            # Format using chat template with full history
            if THINKING_MODE:
                # Add thinking prompt to system message
                conversation_with_thinking = [{"role": "system", "content": THINKING_PROMPT}] + conversation
                base_input_ids = tokenizer.apply_chat_template(
                    conversation_with_thinking,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(torch.int32)
                
                # Print full prompt if debug level >= 1
                if DEBUG_LEVEL >= 1 and not warmup:
                    print(f"\n{DARK_BLUE}Debug: Full prompt with thinking:{RESET_COLOR}")
                    print(tokenizer.decode(base_input_ids[0]))
            else:
                base_input_ids = tokenizer.apply_chat_template(
                    conversation,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(torch.int32)
                
                # Print full prompt if debug level >= 1
                if DEBUG_LEVEL >= 1 and not warmup:
                    print(f"\n{DARK_BLUE}Debug: Full prompt:{RESET_COLOR}")
                    print(tokenizer.decode(base_input_ids[0]))
            
            # Check if we need to trim history
            while base_input_ids.size(1) > context_length - 100:  # Leave room for response
                # Remove oldest message pair (user + assistant)
                if len(conversation) > 2:
                    conversation = conversation[2:]  # Remove oldest pair
                    base_input_ids = tokenizer.apply_chat_template(
                        conversation,
                        return_tensors="pt",
                        add_generation_prompt=True
                    ).to(torch.int32)
                else:
                    # If only current message remains and still too long, truncate
                    base_input_ids = base_input_ids[:, -context_length//2:]
                    break
            
            context_pos = base_input_ids.size(1)
            
            # Pad sequence to context_size
            input_ids = F.pad(
                base_input_ids,
                (0, context_length - context_pos),
                value=0
            )
            
            if not warmup:
                print(f"\n{LIGHT_BLUE}Assistant:{RESET_COLOR}", end=' ', flush=True)
            
            # split_lm_head should already be in metadata from caller
            
            # Initialize token printer and collect response
            token_printer = TokenPrinter(tokenizer)
            response_tokens = []
            generation_start_time = time.time()
            
            try:
                # Run prefill on entire context
                current_pos = run_prefill(
                    embed_model,
                    ffn_models,
                    input_ids,
                    context_pos,
                    context_length,
                    batch_size,
                    state,
                    causal_mask
                )
                #print(f"\n[DEBUG] After initial prefill - current_pos: {current_pos}")
                
                # Generation loop
                pos = context_pos
                tokens_generated = 0
                inference_start = time.time()  # Start inference timing
                
                while True:
                    # Check if we need to shift window
                    if pos >= context_length - 2:
                        # Calculate shift to maintain full batches
                        batch_size = metadata.get('batch_size', 64)
                        # Calculate max batches that fit in context
                        max_batches = context_length // batch_size
                        desired_batches = max(1, max_batches - 2)  # Leave room for new tokens
                        new_size = min(desired_batches * batch_size, context_length - batch_size)
                        
                        # Create shifted input_ids
                        tmp = torch.zeros((1, context_length), dtype=torch.int32)
                        tmp[:,0:new_size] = input_ids[:,pos-new_size:pos]
                        input_ids = tmp
                        
                        # Reset state and run prefill
                        # keep the same state
                        #state = create_unified_state(ffn_models, context_length)
                        current_pos = run_prefill(
                            embed_model,
                            ffn_models,
                            input_ids,
                            new_size,  # Prefill the entire shifted content
                            context_length,
                            batch_size,
                            state,
                            causal_mask
                        )
                        
                        # Start generating from the next position
                        pos = new_size  # Don't back up, continue from where we left off
                        
                        #print(f"\n[DEBUG] After shift - next token will be at pos {pos}")
                        #print(f"[DEBUG] Context before next token: {tokenizer.decode(input_ids[0, pos-40:pos])}")
                        
                        window_shifted = True
                    
                    # Generate next token
                    next_token = generate_next_token(
                        embed_model,
                        ffn_models,
                        lmhead_model,
                        input_ids,
                        pos,
                        context_length,
                        state,
                        causal_mask,
                        metadata
                    )
                    
                    # Add token
                    input_ids[0, pos] = next_token
                    if not warmup:
                        token_printer.add_token(next_token)
                        token_printer.drain_buffer()
                    response_tokens.append(next_token)
                    
                    pos += 1
                    tokens_generated += 1
                    
                    # In warmup mode, limit tokens
                    if warmup and tokens_generated >= WARMUP_TOKEN_LIMIT:
                        break
                    
                    # Check for all possible EOS tokens
                    eos_token_ids = tokenizer.eos_token_id
                    if isinstance(eos_token_ids, list):
                        if next_token in eos_token_ids:
                            break
                    else:
                        if next_token == eos_token_ids:
                            break
                
                inference_time = time.time() - inference_start  # Calculate inference time
                
                # Add assistant response to conversation
                response_text = token_printer.stop()
                conversation.append({"role": "assistant", "content": response_text})
                
                # Print stats only if not in warmup
                if not warmup:
                    total_time = time.time() - generation_start_time
                    prefill_time = total_time - inference_time
                    inference_tokens_per_sec = len(response_tokens) / inference_time if inference_time > 0 else 0
                    prefill_ms = prefill_time * 1000
                    prefill_tokens_per_sec = context_pos / prefill_time if prefill_time > 0 else 0
                    print(f"{DARK_BLUE}{inference_tokens_per_sec:.1f} t/s, "
                          f"TTFT: {prefill_ms:.1f}ms ({prefill_tokens_per_sec:.1f} t/s), "
                          f"{len(response_tokens)} tokens{RESET_COLOR}")
                
                if auto_prompt is not None:
                    break
                
            except KeyboardInterrupt:
                if not warmup:
                    print("\nGeneration interrupted")
                token_printer.stop()
                continue
                
    except Exception as e:
        if not warmup:
            print(f"\nError in chat loop: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    args = parse_args()
    global DEBUG_LEVEL
    DEBUG_LEVEL = args.debug_level
    
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
        
        if not Path(args.tokenizer).exists():
            print(f"\nError: Tokenizer directory not found: {args.tokenizer}")
            return 1
    
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
        
        print(f"\nMetadata after load_models: {metadata}")
        
        # Load tokenizer with resolved path
        tokenizer = initialize_tokenizer(args.tokenizer)
        if tokenizer is None:
            raise RuntimeError("Failed to initialize tokenizer")
        
        # Create unified state once
        state = create_unified_state(ffn_models, metadata['context_length'])
        
        # Initialize causal mask once
        causal_mask = initialize_causal_mask(metadata['context_length'])
        
        # Add split_lm_head to metadata for generate_next_token
        metadata['split_lm_head'] = getattr(args, 'split_lm_head', 8)
        
        # Warmup runs to prevent Python GIL issues with CoreML !
        if not args.nw:
            for i in range(2):
                chat_loop(
                    embed_model=embed_model,
                    ffn_models=ffn_models,
                    lmhead_model=lmhead_model,
                    tokenizer=tokenizer,
                    metadata=metadata,
                    state=state,  # Pass the state
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
            state=state,  # Pass the state
            causal_mask=causal_mask,  # Pass the causal mask
            warmup=False,
            auto_prompt=args.prompt
        )
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 