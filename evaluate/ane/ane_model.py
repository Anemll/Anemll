#!/usr/bin/env python3
# ANE_Model class for handling ANE/CoreML model operations
# Designed to provide a clean abstraction for LM evaluation harness

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

try:
    import coremltools as ct
except ImportError:
    print("Error: coremltools not found. Please install it using:")
    print("pip install coremltools")
    sys.exit(1)

# Ensure single-threaded CoreML execution
os.environ["COREML_PARTITION_LOADER_DISABLE_MULTI_ENGINE"] = "1"

# Add tests directory to path for importing relevant functions
tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests")
sys.path.append(tests_dir)

# Import necessary functions from chat.py
try:
    from chat import (
        parse_model_path,
        parse_ffn_filename,
        find_all_chunks,
        make_causal_mask,
        initialize_causal_mask
    )
    print("Successfully imported helper functions from chat.py")
except ImportError:
    print("Warning: Could not import from chat.py, implementing local versions")
    
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


class ANE_Model:
    """
    ANE_Model provides a clean abstraction for CoreML model operations with proper state handling.
    
    This class manages:
    1. Loading and handling CoreML models
    2. Processing input tensors with correct shapes and dtypes
    3. Tracking position state between prefill and predict operations
    4. Providing a consistent interface for LM evaluation

    IMPORTANT STATE MANAGEMENT NOTES:
    - self.current_position always points to the NEXT EMPTY SLOT in the KV cache
    - Each token is written to the current position, then position is advanced
    - compute_logprobs() and predict() both write to position and advance it
    - Never write to position-1 as this causes state corruption
    - For scoring/perplexity calculations, each position should be written to exactly once
    """
    
    def __init__(self, model_path: Union[str, Path], max_tokens: int = 2048):
        """
        Initialize the ANE_Model with model path.
        
        Args:
            model_path: Path to model directory containing CoreML models
            max_tokens: Maximum number of tokens to generate
        """
        self.model_path = Path(model_path)
        self.max_tokens = max_tokens
        
        # Model components
        self.embedding_model = None
        self.lm_head_model = None
        self.ffn_models = []
        
        # Model metadata
        self.metadata = {}
        
        # State tracking
        self.kv_cache = None
        self.causal_mask = None
        self.current_position = 0
        self.debug = 0
        # Load models
        self._load_models()
        
        # Initialize state and causal mask
        self._initialize_state()
    
    def _load_models(self):
        """Load all required model components."""
        print(f"Loading models from {self.model_path}")
        
        if not self.model_path.exists():
            raise ValueError(f"Model directory not found: {self.model_path}")
            
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
        self.embedding_model = self._load_model(embed_path)
        print("Embeddings model loaded successfully")
        
        # Load LM head model
        lm_head_path = lm_head_paths[0]
        print(f"Loading LM head model from {lm_head_path}")
        self.lm_head_model = self._load_model(lm_head_path)
        print("LM head model loaded successfully")
        
        # Load FFN models
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
                        'infer': self._load_model(ffn_path, function_name='infer'),
                        'prefill': self._load_model(ffn_path, function_name='prefill')
                    })
                    print(f"Loaded {ffn_path.name} with infer/prefill functions")
                except Exception as e:
                    print(f"Could not load with function_name parameter: {str(e)}")
                    print("Trying without specifying function name")
                    self.ffn_models.append(self._load_model(ffn_path))
            else:
                # Single FFN model
                print(f"Loading single FFN model: {ffn_path.name}")
                self.ffn_models.append(self._load_model(ffn_path))
        
        print(f"Loaded {len(self.ffn_models)} FFN models successfully")
        
        # Extract metadata
        self._extract_metadata()
    
    def _load_model(self, path, function_name=None):
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
                    return ct.models.MLModel(str(path), compute_units=compute_unit, function_name=function_name)
                else:
                    return ct.models.MLModel(str(path), compute_units=compute_unit)
                    
        except RuntimeError as e:
            if "valid manifest does not exist" in str(e):
                print(f"\nError: Could not load compiled model at {path}")
                print("This might be because:")
                print("1. The model is not properly compiled")
                print("2. The model was compiled for a different OS version")
                print("3. The model needs to be recompiled")
                print("\nTry using the .mlpackage version instead, or recompile the model.")
            raise
    
    def _extract_metadata(self):
        """Extract metadata from models."""
        # Default metadata values
        self.metadata = {
            'context_length': 1024,
            'state_length': 1024,
            'batch_size': 64,
            'lut_bits': 0,
            'num_chunks': len(self.ffn_models)
        }
        
        # Try to extract from embedding model first
        if hasattr(self.embedding_model, 'user_defined_metadata'):
            meta = self.embedding_model.user_defined_metadata
            
            # Extract key parameters with defaults
            self.metadata['context_length'] = int(meta.get('com.anemll.context_length', self.metadata['context_length']))
            self.metadata['state_length'] = int(meta.get('com.anemll.state_length', self.metadata['context_length']))
            self.metadata['batch_size'] = int(meta.get('com.anemll.batch_size', self.metadata['batch_size']))
            self.metadata['lut_bits'] = int(meta.get('com.anemll.lut_bits', self.metadata['lut_bits']))
            self.metadata['num_chunks'] = int(meta.get('com.anemll.num_chunks', self.metadata['num_chunks']))
            
            print("\nExtracted Parameters from model metadata:")
            print(f"  Context Length: {self.metadata['context_length']}")
            print(f"  State Length: {self.metadata['state_length']}")
            print(f"  Compiled Batch Size: {self.metadata['batch_size']}")
            print(f"  LUT Bits: {self.metadata['lut_bits']}")
            print(f"  Number of Chunks: {self.metadata['num_chunks']}")
        else:
            print("\nNo metadata found in CoreML model, trying meta.yaml...")
            
            # Try to load from meta.yaml if available
            meta_yaml_path = self.model_path / "meta.yaml"
            if meta_yaml_path.exists():
                try:
                    import yaml
                    with open(meta_yaml_path, 'r') as f:
                        yaml_data = yaml.safe_load(f)
                    
                    # Extract parameters from meta.yaml
                    if 'model_info' in yaml_data and 'parameters' in yaml_data['model_info']:
                        params = yaml_data['model_info']['parameters']
                        self.metadata['context_length'] = int(params.get('context_length', self.metadata['context_length']))
                        self.metadata['batch_size'] = int(params.get('batch_size', self.metadata['batch_size']))
                        self.metadata['num_chunks'] = int(params.get('num_chunks', self.metadata['num_chunks']))
                        
                        print("\nExtracted Parameters from meta.yaml:")
                        print(f"  Context Length: {self.metadata['context_length']}")
                        print(f"  Batch Size: {self.metadata['batch_size']}")
                        print(f"  Number of Chunks: {self.metadata['num_chunks']}")
                    else:
                        print("  No model_info.parameters found in meta.yaml")
                except Exception as e:
                    print(f"  Error loading meta.yaml: {str(e)}")
            else:
                print("  meta.yaml not found in model directory")
                print("  Using default parameters")
        
        return self.metadata
    
    def _initialize_state(self):
        """Initialize the KV cache state and causal mask."""
        context_length = self.metadata['context_length']
        
        # Create KV cache state
        if isinstance(self.ffn_models[0], dict):
            # Use first FFN model's prefill function to create state
            self.kv_cache = self.ffn_models[0]['prefill'].make_state()
            print(f"\nCreated KV cache state for {len(self.ffn_models)} chunks")
        else:
            self.kv_cache = self.ffn_models[0].make_state()
            print("\nCreated KV cache state")
        
        # Initialize causal mask
        self.causal_mask = initialize_causal_mask(context_length)
        
        # Reset position
        self.current_position = 0
    
    def reset_state(self):
        """Reset the model state to start a new sequence."""
        #self._initialize_state()
        self.current_position = 0
    
    def prefill(self, input_ids):
        """
        Run prefill on a sequence of input tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Current position after prefill
        """
        # Reset state for new sequence
        self.reset_state()
        
        # Convert to torch tensor if needed
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.int32)
        
        # Get sequence length
        context_pos = input_ids.shape[1]
        batch_size = self.metadata['batch_size']
        context_length = self.metadata['context_length']
        
        # Use a safe context length with margin to prevent position tracking issues
        safe_context_length = context_length - 64  # Leave a 64-token safety margin
        
        # Enhanced debug output
        if self.debug > 0:
            print(f"\n=== PREFILL DEBUG INFO ===")
            print(f"Input shape: {input_ids.shape}")
            print(f"Input dtype: {input_ids.dtype}")
            print(f"Context position: {context_pos}")
            print(f"Model batch size: {batch_size}")
            print(f"Model context length: {context_length}")
            print(f"Safe context length: {safe_context_length}")
        
        # Critical warning if sequence exceeds context length - always show this
        if context_pos > safe_context_length:
            print(f"WARNING: Input sequence length ({context_pos}) exceeds safe context length ({safe_context_length})")
            print(f"Truncating to {safe_context_length} tokens to prevent inference errors")
            
            # Show token details only in debug mode
            if self.debug > 0:
                # Show beginning tokens
                print(f"First 5 tokens: {input_ids[0, :5].tolist()}")
                # Show end tokens
                print(f"Last 5 tokens: {input_ids[0, -5:].tolist()}")
                
            # Keep most recent tokens up to safe context length
            input_ids = input_ids[:, -safe_context_length:]
            context_pos = safe_context_length
        
        # Process in batches to avoid OOM
        batch_pos = 0
        while batch_pos < context_pos:
            batch_end = min(batch_pos + batch_size, context_pos)
            current_batch_size = batch_end - batch_pos
            
            # Get current batch
            batch_input = input_ids[:, batch_pos:batch_end]
            
            # Debug current batch
            if self.debug > 0:
                print(f"Processing batch {batch_pos}:{batch_end} (size: {current_batch_size})")
                print(f"Batch input shape before padding: {batch_input.shape}")
            
            # Always pad to full batch size for prefill (using F.pad like in chat.py)
            batch_input = F.pad(
                batch_input,
                (0, batch_size - current_batch_size),
                value=0
            )
            
            if self.debug > 0:
                print(f"Batch input shape after padding: {batch_input.shape}")
            
            # Generate position IDs for this batch - FIXED to match chat_full.py implementation
            position_ids = torch.arange(batch_pos, batch_pos + batch_size, dtype=torch.int32)
            
            # Use the pre-initialized causal mask and extract the batch portion
            batch_causal_mask = self.causal_mask[:, :, batch_pos:batch_pos + batch_size, :]
            
            # Run embeddings with proper batch size
            try:
                if self.debug > 0:
                    print(f"Running embedding model for batch {batch_pos}:{batch_end}")
                    
                # Debug embedding inputs
                if self.debug > 1:
                    print(f"Embedding inputs:")
                    print(f"  input_ids: {batch_input.shape} {batch_input.dtype}")
                    print(f"  batch_size: {np.array([batch_size], dtype=np.int32).shape}")
                    
                hidden_states = torch.from_numpy(
                    self.embedding_model.predict({
                        'input_ids': batch_input.numpy(),
                        'batch_size': np.array([batch_size], dtype=np.int32)
                    })['hidden_states']
                )
                
                if self.debug > 0:
                    print(f"Embedding completed successfully, hidden_states shape: {hidden_states.shape}")
                
                # Run through FFN chunks with state
                for i, ffn_model in enumerate(self.ffn_models):
                    if isinstance(ffn_model, dict):
                        inputs = {
                            'hidden_states': hidden_states.numpy(),
                            'position_ids': position_ids.numpy(),
                            'causal_mask': batch_causal_mask.numpy(),
                            'current_pos': np.array([batch_pos], dtype=np.int32)
                        }
                        
                        if self.debug > 0:
                            print(f"Running FFN chunk {i+1}/{len(self.ffn_models)} prefill")
                            # Debug FFN inputs
                            if self.debug > 1:
                                print(f"FFN inputs:")
                                print(f"  hidden_states: {inputs['hidden_states'].shape}")
                                print(f"  position_ids: {inputs['position_ids'].shape}")
                                print(f"  causal_mask: {inputs['causal_mask'].shape}")
                                print(f"  current_pos: {inputs['current_pos'].shape}")
                        
                        output = ffn_model['prefill'].predict(inputs, self.kv_cache)
                        hidden_states = torch.from_numpy(output['output_hidden_states'])
                        
                        if self.debug > 0:
                            print(f"FFN chunk {i+1} prefill completed successfully")
                
                batch_pos = batch_end
                if self.debug > 0:
                    print(f"Batch processed successfully, moving to next batch")
                    
            except Exception as e:
                print(f"Error in prefill at batch position {batch_pos}:{batch_end}:")
                print(f"Error details: {str(e)}")
                
                # Additional error diagnostics
                if self.debug > 0:
                    print("\nInput diagnostics:")
                    print(f"Input shape: {input_ids.shape}")
                    print(f"Current batch shape: {batch_input.shape}")
                    print(f"Model context length: {context_length}")
                    print(f"Model batch size: {batch_size}")
                    import traceback
                    traceback.print_exc()
                raise
        
        # Update current position
        self.current_position = context_pos
        return self.current_position
    
    def predict(self, input_token):
        """
        Generate the next token given the current state.
        
        Args:
            input_token: The most recent token (single token or batch)
            
        Returns:
            Next token ID
        """
        # Convert to torch tensor if needed
        if not isinstance(input_token, torch.Tensor):
            # Ensure it's a 2D tensor [batch_size, 1]
            if isinstance(input_token, (int, np.integer)):
                input_token = torch.tensor([[input_token]], dtype=torch.int32)
            else:
                input_token = torch.tensor([input_token], dtype=torch.int32)
        
        # If input is [batch_size, seq_len] but seq_len > 1, we need just the last token
        if input_token.dim() == 2 and input_token.shape[1] > 1:
            input_token = input_token[:, -1:]
        
        # Ensure shape is [batch_size, 1]
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)
        
        # Get context length from metadata
        context_length = self.metadata['context_length']
        pos = self.current_position
        
        # Safety check: position must be within context bounds
        safe_pos = min(pos, context_length - 2)  # Ensure pos-1 is valid for indexing
        if pos != safe_pos:
            print(f"WARNING: Position {pos} adjusted to {safe_pos} to prevent index errors")
            self.current_position = safe_pos
            pos = safe_pos
        
        # Get current token at position pos-1, like in chat_full.py
        current_token = input_token.to(torch.int32)
        
        try:
            # Forward pass through embedding model for the current token
            # Don't pass batch_size parameter for prediction, matching chat_full.py
            if self.debug > 0:
                print(f"Predicting with token {current_token[0, 0].item()} at position {pos}")
            
            token_embedding = torch.from_numpy(
                self.embedding_model.predict({
                    'input_ids': current_token.numpy()
                })['hidden_states']
            )
            
            # Create masks for the attention mechanism - with safety check
            update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
            # Ensure the position index is valid
            position_idx = max(0, min(pos, context_length-1))  # Write into NEXT slot
            update_mask[0, 0, position_idx, 0] = 1.0
            position_ids = torch.tensor([position_idx], dtype=torch.int32)
            
            # Get causal mask for current position
            single_causal_mask = self.causal_mask[:, :, position_idx:position_idx+1, :]
            
            # Run through FFN chunks with state
            for ffn_model in self.ffn_models:
                if isinstance(ffn_model, dict):
                    inputs = {
                        'hidden_states': token_embedding.numpy(),
                        'update_mask': update_mask.numpy(),
                        'position_ids': position_ids.numpy(),
                        'causal_mask': single_causal_mask.numpy(),
                        'current_pos': position_ids.numpy()
                    }
                    output = ffn_model['infer'].predict(inputs, self.kv_cache)
                    token_embedding = torch.from_numpy(output['output_hidden_states'])
            
            # Run LM head to get logits
            lm_output = self.lm_head_model.predict({'hidden_states': token_embedding.numpy()})
            
            # Combine logits1-8 if they exist
            if 'logits1' in lm_output:
                # Concatenate all logits parts
                logits_parts = []
                for i in range(1, 9):
                    key = f'logits{i}'
                    if key in lm_output:
                        logits_parts.append(torch.from_numpy(lm_output[key]))
                logits = torch.cat(logits_parts, dim=-1)  # Concatenate along vocab dimension
            else:
                # Try output_logits as fallback
                logits = torch.from_numpy(lm_output['output_logits'])
            
            # Get next token (greedy)
            next_token = torch.argmax(logits[0, -1, :]).item()
            
            # Advance pointer once the write succeeded
            self.current_position += 1
            
            return next_token
            
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_logits(self, input_token):
        """
        Get logits for the next token prediction.
        
        Args:
            input_token: The most recent token
            
        Returns:
            Logits tensor [batch_size, vocab_size]
        """
        # Similar to predict but returns full logits
        if not isinstance(input_token, torch.Tensor):
            if isinstance(input_token, (int, np.integer)):
                input_token = torch.tensor([[input_token]], dtype=torch.int32)
            else:
                input_token = torch.tensor([input_token], dtype=torch.int32)
        
        if input_token.dim() == 2 and input_token.shape[1] > 1:
            input_token = input_token[:, -1:]
        
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)
        
        # Get context length from metadata
        context_length = self.metadata['context_length']
        pos = self.current_position
        
        # Get current token
        current_token = input_token.to(torch.int32)
        
        try:
            # Forward pass through embedding model for the current token
            # Don't pass batch_size parameter for prediction
            token_embedding = torch.from_numpy(
                self.embedding_model.predict({
                    'input_ids': current_token.numpy()
                })['hidden_states']
            )
            
            # Create masks
            update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
            # Correctly use pos-1 for position index
            position_idx = max(0, min(pos-1, context_length-1))
            update_mask[0, 0, position_idx, 0] = 1.0
            position_ids = torch.tensor([position_idx], dtype=torch.int32)
            
            # Get causal mask for current position
            single_causal_mask = self.causal_mask[:, :, position_idx:position_idx+1, :]
            
            # Run through FFN chunks with state
            for ffn_model in self.ffn_models:
                if isinstance(ffn_model, dict):
                    inputs = {
                        'hidden_states': token_embedding.numpy(),
                        'update_mask': update_mask.numpy(),
                        'position_ids': position_ids.numpy(),
                        'causal_mask': single_causal_mask.numpy(),
                        'current_pos': position_ids.numpy()
                    }
                    output = ffn_model['infer'].predict(inputs, self.kv_cache)
                    token_embedding = torch.from_numpy(output['output_hidden_states'])
            
            # Run LM head
            lm_output = self.lm_head_model.predict({'hidden_states': token_embedding.numpy()})
            
            # Combine logits1-8 if they exist
            if 'logits1' in lm_output:
                logits_parts = []
                for i in range(1, 9):
                    key = f'logits{i}'
                    if key in lm_output:
                        logits_parts.append(torch.from_numpy(lm_output[key]))
                logits = torch.cat(logits_parts, dim=-1)
            else:
                logits = torch.from_numpy(lm_output['output_logits'])
            
            # Don't update position - this is just for getting logits
            
            return logits
            
        except Exception as e:
            print(f"Error in get_logits: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate(self, input_ids, max_new_tokens=None, stop_tokens=None):
        """
        Generate a sequence of tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            stop_tokens: List of token IDs that stop generation
            
        Returns:
            Generated token IDs
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_tokens
        
        if stop_tokens is None:
            stop_tokens = []
        
        # Run prefill on input sequence
        self.prefill(input_ids)
        
        # Generate tokens
        generated_tokens = []
        try:
            for _ in range(max_new_tokens):
                # Generate next token using our improved predict method
                if not generated_tokens:
                    # First token after context
                    next_token = self.predict(input_ids[:, -1:])
                else:
                    # Subsequent tokens
                    next_token = self.predict(torch.tensor([[generated_tokens[-1]]]))
                
                # Check if generation failed
                if next_token is None:
                    break
                
                # Add to generated tokens
                generated_tokens.append(next_token)
                
                # Check stop condition
                if next_token in stop_tokens:
                    break
                
        except Exception as e:
            print(f"Error in generate: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return generated_tokens

    def compute_logprobs(self, input_token, candidate_tokens=None):
        """
        Compute log probabilities for candidate tokens given the current state.
        
        Args:
            input_token: The most recent token (single token or batch)
            candidate_tokens: Optional list of specific tokens to compute logprobs for
                              If None, returns logprobs for all tokens
            
        Returns:
            Log probabilities tensor for all tokens or specified candidate tokens
            
        WARNING:
            After using compute_logprobs, you MUST call predict() with the target/ground-truth token
            to properly advance the state. compute_logprobs() writes the input token to KV-cache
            but doesn't properly set up the cache for the next token.
        """
        # Similar to predict but returns log probabilities instead of argmax
        if not isinstance(input_token, torch.Tensor):
            if isinstance(input_token, (int, np.integer)):
                input_token = torch.tensor([[input_token]], dtype=torch.int32)
            else:
                input_token = torch.tensor([input_token], dtype=torch.int32)
        
        if input_token.dim() == 2 and input_token.shape[1] > 1:
            input_token = input_token[:, -1:]
        
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)
        
        # Get context length from metadata
        context_length = self.metadata['context_length']
        pos = self.current_position
        
        # Safety check: position must be within context bounds
        safe_pos = min(pos, context_length - 2)  # Ensure pos-1 is valid for indexing
        if pos != safe_pos:
            print(f"WARNING: Position {pos} adjusted to {safe_pos} to prevent index errors")
            self.current_position = safe_pos
            pos = safe_pos
        
        # Get current token at position pos-1, like in chat_full.py
        current_token = input_token.to(torch.int32)
        
        if self.debug > 0:
            print(f"\nComputing logprobs for token {current_token[0, 0].item()} at position {pos}")
        
        try:
            # Forward pass through embedding model for the current token
            # Don't pass batch_size parameter for prediction, matching chat_full.py
            token_embedding = torch.from_numpy(
                self.embedding_model.predict({
                    'input_ids': current_token.numpy()
                })['hidden_states']
            )
            
            # Create masks for the attention mechanism - with safety check
            update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
            # Ensure the position index is valid
            position_idx = max(0, min(pos-1, context_length-1))  # Correctly use pos-1
            update_mask[0, 0, position_idx, 0] = 1.0
            position_ids = torch.tensor([position_idx], dtype=torch.int32)
            
            # Get causal mask for current position
            single_causal_mask = self.causal_mask[:, :, position_idx:position_idx+1, :]
            
            # Run through FFN chunks with state - with error handling for each chunk
            chunk_errors = 0
            for ffn_idx, ffn_model in enumerate(self.ffn_models):
                if isinstance(ffn_model, dict):
                    inputs = {
                        'hidden_states': token_embedding.numpy(),
                        'update_mask': update_mask.numpy(),
                        'position_ids': position_ids.numpy(),
                        'causal_mask': single_causal_mask.numpy(),
                        'current_pos': position_ids.numpy()
                    }
                    
                    try:
                        output = ffn_model['infer'].predict(inputs, self.kv_cache)
                        token_embedding = torch.from_numpy(output['output_hidden_states'])
                    except Exception as e:
                        chunk_errors += 1
                        if self.debug > 0:
                            print(f"Error in FFN chunk {ffn_idx+1}: {str(e)}")
                            
                        # Try alternative approaches if predict fails
                        try:
                            # Option 1: Try without state updating
                            if self.debug > 0:
                                print(f"Trying FFN chunk {ffn_idx+1} without state updating")
                            output = ffn_model['infer'].predict(inputs)
                            token_embedding = torch.from_numpy(output['output_hidden_states'])
                        except Exception as retry_err:
                            if self.debug > 0:
                                print(f"Alternative approach failed: {str(retry_err)}")
                                
                            # If too many chunks fail, abort
                            if chunk_errors > len(self.ffn_models) // 2:
                                print(f"Too many FFN chunk errors ({chunk_errors}/{len(self.ffn_models)}), aborting")
                                return None
            
            # If we've had ANY chunk errors but somehow got here, reset state to be safe
            if chunk_errors > 0:
                print(f"Warning: Had {chunk_errors} FFN chunk errors, but continuing with prediction")
                # Don't advance position to avoid further corruption
                return self._run_lm_head_only(token_embedding)
            
            # Run LM head to get logits (read-only)
            log_probs = self._run_lm_head_only(token_embedding)
            
            # Filter to candidate tokens if specified
            if candidate_tokens is not None:
                if isinstance(candidate_tokens, (list, tuple)):
                    candidate_tokens = torch.tensor(candidate_tokens)
                return log_probs[candidate_tokens]
            
            # Otherwise return all log probs
            return log_probs
            
        except Exception as e:
            print(f"Error in compute_logprobs: {str(e)}")
            if self.debug > 0:
                print(f"Details: pos={pos}, context_length={context_length}")
                print(f"input_token: {current_token}")
            import traceback
            traceback.print_exc()
            return None
            
    def _run_lm_head_only(self, token_embedding):
        """Run only the LM head part to get logits, used as a fallback."""
        try:
            # Run LM head to get logits
            lm_output = self.lm_head_model.predict({'hidden_states': token_embedding.numpy()})
            
            # Combine logits1-8 if they exist
            if 'logits1' in lm_output:
                logits_parts = []
                for i in range(1, 9):
                    key = f'logits{i}'
                    if key in lm_output:
                        logits_parts.append(torch.from_numpy(lm_output[key]))
                logits = torch.cat(logits_parts, dim=-1)
            else:
                logits = torch.from_numpy(lm_output['output_logits'])
            
            # Convert logits to log probabilities
            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
            return log_probs
            
        except Exception as e:
            print(f"Error in LM head: {str(e)}")
            return None


# Test function to verify the implementation
def test_ane_model(model_path, input_text="Hello, world!", max_tokens=5):
    """Test the ANE_Model implementation."""
    try:
        # Try to import tokenizer
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading tokenizer from {model_path}: {str(e)}")
            print("Falling back to Llama tokenizer")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        # Initialize model
        print(f"Initializing ANE_Model from {model_path}")
        model = ANE_Model(model_path)
        
        print(f"\nModel metadata:")
        for key, value in model.metadata.items():
            print(f"  {key}: {value}")
        
        # Tokenize input
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(torch.int32)
        print(f"\nInput text: '{input_text}'")
        print(f"Input tokens: {input_ids[0].tolist()}")
        
        # Run prefill
        print(f"\nRunning prefill...")
        current_pos = model.prefill(input_ids)
        print(f"Prefill completed - current position: {current_pos}")
        
        # Generate tokens
        print(f"\nGenerating tokens...")
        generated_tokens = []
        for i in range(max_tokens):
            if i == 0:
                next_token = model.predict(input_ids[:, -1:])
            else:
                next_token = model.predict(torch.tensor([[generated_tokens[-1]]]))
                
            if next_token is None:
                print(f"Token generation failed")
                break
                
            generated_tokens.append(next_token)
            if model.debug > 0:
                print(f"Generated token {i+1}: {next_token} ('{tokenizer.decode([next_token])}')")
        
        # Decode the full sequence
        full_text = tokenizer.decode(input_ids[0].tolist() + generated_tokens)
        print(f"\nFull text: '{full_text}'")
        
        return True
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_logprobs(model_path, prompt="Hello, my name is", candidates=None):
    """Test the compute_logprobs method with multiple candidate completions."""
    try:
        # Initialize tokenizer
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading tokenizer from {model_path}: {str(e)}")
            print("Falling back to Llama tokenizer")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        # Set default candidates if not provided
        if candidates is None:
            candidates = [" John", " Sarah", " David", " Michael", " Emily"]
        
        # Initialize model
        print(f"Initializing ANE_Model from {model_path}")
        model = ANE_Model(model_path)
        
        # Enable debug mode
        model.debug = 1
        
        print(f"\nModel metadata:")
        for key, value in model.metadata.items():
            print(f"  {key}: {value}")
        
        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.int32)
        print(f"\nPrompt: '{prompt}'")
        print(f"Prompt tokens: {prompt_ids[0].tolist()}")
        
        # Run prefill
        print(f"\nRunning prefill...")
        current_pos = model.prefill(prompt_ids)
        print(f"Prefill completed - current position: {current_pos}")
        
        # Tokenize candidates
        candidate_tokens = []
        candidate_texts = []
        
        print("\nCandidate completions:")
        for candidate in candidates:
            candidate_text = prompt + candidate
            candidate_texts.append(candidate_text)
            
            # Tokenize the full text
            full_ids = tokenizer.encode(candidate_text, return_tensors="pt")[0]
            
            # The first token of the completion is what we want
            completion_token = full_ids[len(prompt_ids[0])].item()
            candidate_tokens.append(completion_token)
            
            print(f"  '{candidate}' -> token: {completion_token} ('{tokenizer.decode([completion_token])}')")
        
        # Get log probabilities for all tokens
        print("\nComputing log probabilities...")
        log_probs = model.compute_logprobs(prompt_ids[:, -1:])
        
        # Print top 10 most likely tokens
        top_tokens = torch.topk(log_probs, 10)
        print("\nTop 10 most likely tokens:")
        for i in range(10):
            token_id = top_tokens.indices[i].item()
            token_logprob = top_tokens.values[i].item()
            token_prob = np.exp(token_logprob)
            token_text = tokenizer.decode([token_id])
            print(f"  {i+1}. Token {token_id}: '{token_text}' - logprob: {token_logprob:.4f}, prob: {token_prob:.6f}")
        
        # Score each candidate completion
        print("\nScoring candidate completions:")
        candidate_scores = []
        for i, (token, text) in enumerate(zip(candidate_tokens, candidate_texts)):
            # Get probability of this completion
            if token < len(log_probs):
                token_logprob = log_probs[token].item()
                token_prob = np.exp(token_logprob)
                print(f"  {i+1}. '{text}' - logprob: {token_logprob:.4f}, prob: {token_prob:.6f}")
                candidate_scores.append((i, token_logprob))
            else:
                print(f"  {i+1}. '{text}' - Token ID {token} out of vocabulary range")
                candidate_scores.append((i, float('-inf')))
        
        # Find most likely completion based on log probability value, not token ID
        sorted_candidates = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        most_likely_idx = sorted_candidates[0][0]
        print(f"\nMost likely completion: '{candidate_texts[most_likely_idx]}'")
        print(f"  logprob: {sorted_candidates[0][1]:.4f}, prob: {np.exp(sorted_candidates[0][1]):.6f}")
        
        return True
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Simple test if module is run directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ANE_Model implementation")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--input", type=str, default="Hello, world!", help="Input text to process")
    parser.add_argument("--max_tokens", type=int, default=5, help="Maximum number of tokens to generate")
    parser.add_argument("--test_logprobs", action="store_true", help="Run log probability test")
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Prompt for logprobs test")
    
    args = parser.parse_args()
    
    if args.test_logprobs:
        success = test_logprobs(args.model, args.prompt)
        if success:
            print("\nLog probability test completed successfully!")
        else:
            print("\nLog probability test failed!")
            sys.exit(1)
    else:
        success = test_ane_model(args.model, args.input, args.max_tokens)
        if success:
            print("\nANE_Model test completed successfully!")
        else:
            print("\nANE_Model test failed!")
            sys.exit(1) 