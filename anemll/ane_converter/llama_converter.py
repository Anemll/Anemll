#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

from .base_converter import BaseConverter
import coremltools as ct
import coremltools.optimize as cto
from coremltools.converters.mil import Builder as mb
import numpy as np
import torch
import os
import gc  # Added import for garbage collection
from ..models.llama_model import (
    LlamaModel, 
    LlamaConfig, 
    LlamaForCausalLM,
    TEST_DEVICE,
    MODEL_DTYPE,
    ENABLE_DEBUG,
    ENABLE_UNIFIED_CACHE,
    STATE_LENGTH,
    CONTEXT_LENGTH
)
import pkg_resources
from .metadata import AddMetadata, get_anemll_version
import argparse
import sys

class LlamaConverter(BaseConverter):
    """Handles LLAMA model conversion to Apple Neural Engine format."""

    def __init__(self, model, context_length=512, state_length=None, lut_bits=4, per_channel=8, batch_size=64, num_chunks=1):
        super().__init__(model)
        self.context_length = context_length
        self.state_length = state_length or context_length
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.lut_bits = lut_bits
        self.per_channel = per_channel
        self.converted_model = None
        self.batch_size = batch_size
        self.num_chunks = num_chunks

    def convert(self, split_part=None):
        """Convert model to CoreML format with optional splitting.
        
        Args:
            split_part: Which part(s) of the model to convert:
                       '1' - embeddings only
                       '2' - transformer FFN only
                       '2_prefill' - transformer prefill mode
                       '3' - LM head only
                       '123' - full model (all components)
                       
        Returns:
            ct.models.MLModel or list[ct.models.MLModel]: Converted model(s)
        """
        if split_part not in ['1', '2', '2_prefill', '3', '123']:
            raise ValueError("split_part must be one of: '1', '2', '2_prefill', '3', '123'")
            
        self.preprocess()
        
        # Handle individual components
        if split_part == '1':
            return self.convert_embeddings(self.model)
        elif split_part == '2':
            return self.convert_FFN(self.model)
        elif split_part == '2_prefill':
            return self.convert_prefill(self.model)
        elif split_part == '3':
            return self.convert_lm_head(self.model, lut_bits=self.lut_bits)
        
        # Handle full model conversion
        elif split_part == '123':
            embeddings_model = self.convert_embeddings(self.model)
            transformer_model = self.convert_FFN(self.model)
            lm_head_model = self.convert_lm_head(self.model, lut_bits=self.lut_bits)
            return [embeddings_model, transformer_model, lm_head_model]
        
        self.postprocess(num_workers=None)

    def GetTransformerStates(model, part=None, prefix="model.model."):
        """Get the transformer states for CoreML conversion"""
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers  # Get total number of layers from config

        if not ENABLE_UNIFIED_CACHE and part:
            # Calculate layer range for this part
            if part.startswith('2D') or part.startswith('prefill_2D'):
                num_layers_this_part = num_layers // 2
            elif part.startswith('2Q'):
                num_layers_this_part = num_layers // 4
            elif part.startswith('2O'):
                num_layers_this_part = num_layers // 8
            else:
                raise ValueError(f"Invalid part {part} for split transformer model")
            
            # Get the group index from the part number
            group_idx = int(part[2]) - 1
            state_name = f"{prefix}kv_cache_0"  # Include prefix to match PyTorch buffer name
            
            print(f"GetTransformerStates part={part} ENABLE_UNIFIED_CACHE={ENABLE_UNIFIED_CACHE} num_layers_this_part={num_layers_this_part} model.config.num_hidden_layers={model.config.num_hidden_layers}")


            # Combined KV cache states per group
            states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(2 * num_layers_this_part,  # Match PyTorch buffer shape exactly: 2 * LAYERS_PER_KVGROUP
                                model.config.num_key_value_heads, 
                                model.config.state_length, 
                                head_dim),
                        dtype=np.float16
                    ),
                    name=state_name  # Use full buffer name from PyTorch model
                )
            ]
            print(f"GetTransformerStates states: StateType name={states[0].name}, shape={states[0].wrapped_type.shape}")
        else:
            # Create states for all layers (unified cache)
            num_layers_this_part = num_layers *2  
            print(f"GetTransformerStates part={part} ENABLE_UNIFIED_CACHE={ENABLE_UNIFIED_CACHE} num_layers_this_part={num_layers_this_part} model.config.num_hidden_layers={model.config.num_hidden_layers}")

            states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(num_layers_this_part, model.config.num_key_value_heads, model.config.state_length, head_dim),
                        dtype=np.float16
                    ),
                    name=f"{prefix}kv_cache_0"  # Only one group for unified cache
                )
            ]
    
        return states


    def convert_to_ane(self, model):
        """Convert LLaMA model to Apple Neural Engine format using CoreMLTools."""
        print("Converting model to ANE format...")
        
        # Create wrapper for tracing
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                if ENABLE_DEBUG:
                    print(f"ModelWrapper initialized with model: {model}")
            
            def forward(self, input_ids,  position_ids, causal_mask, current_pos):
                if ENABLE_DEBUG:
                    print(f"ModelWrapper forward called with input_ids: {input_ids.shape}, position_ids: {position_ids.shape}, causal_mask: {causal_mask.shape}, current_pos: {current_pos.shape}")
                # First get embeddings
                hidden_states = self.model.embed_tokens(input_ids)

                # LlamaModel(forward(self, hidden_states, position_ids=None, causal_mask=None, current_pos=None,
                #start_layer=0, end_layer=None, IN_PREFILL=False):
                hidden_states = self.model.model(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    start_layer=0,
                    end_layer=None,
                    IN_PREFILL=False
                )
                # Then run transformer layers
                #hidden_states = self.model.model(hidden_states,  position_ids, causal_mask, current_pos)
                
                # Finally run through LM head
                if hasattr(self.model, 'lm_head8_1'):  # ENABLE_VACAB_SPLIT8
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)
                    logits = [
                        getattr(self.model, f'lm_head8_{i}')(hidden_states).squeeze(2).transpose(1, 2)
                        for i in range(1, 9)
                    ]
                    return tuple(logits)
                elif hasattr(self.model, 'lm_head2_1'):  # ENABLE_VACAB_SPLIT
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)
                    logits1 = self.model.lm_head2_1(hidden_states).squeeze(2).transpose(1, 2)
                    logits2 = self.model.lm_head2_2(hidden_states).squeeze(2).transpose(1, 2)
                    return logits1, logits2
                elif hasattr(self.model, 'lm_head1'):  # ENABLE_CONV2D
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)
                    logits = self.model.lm_head1(hidden_states).squeeze(2).transpose(1, 2)
                    return logits
                else:  # Linear head
                    return self.model.lm_head(hidden_states)
                
                return hidden_states
        
        # Create wrapper instance
        wrapper = ModelWrapper(model)
        wrapper.eval()
        
        try:
            # Prepare sample inputs for tracing
            print("Preparing sample inputs...")
            sample_input_ids = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)
            sample_position_ids = torch.zeros(1, dtype=torch.int32, device=TEST_DEVICE)  # Fixed: use 1D tensor with int32
            sample_causal_mask = torch.zeros((1, 1, 1, self.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
            sample_current_pos = torch.zeros(1, dtype=torch.int32, device=TEST_DEVICE)  # Fixed: use int32
            
            print("Sample inputs shapes and types:")
            print(f"  input_ids: {sample_input_ids.shape}, {sample_input_ids.dtype}")
            print(f"  position_ids: {sample_position_ids.shape}, {sample_position_ids.dtype}")
            print(f"  causal_mask: {sample_causal_mask.shape}, {sample_causal_mask.dtype}")
            print(f"  current_pos: {sample_current_pos.shape}, {sample_current_pos.dtype}")
            
            # Trace model
            print("Tracing model...")
            with torch.no_grad():
                # First do a test forward pass
                print("Testing forward pass...")
                test_output = wrapper(
                    sample_input_ids,
                    sample_position_ids,
                    sample_causal_mask,
                    sample_current_pos
                )
                print("Forward pass successful")
                
                # Now trace the model
                print("Starting model trace...")
                traced_model = torch.jit.trace(
                    wrapper,
                    (
                        sample_input_ids,
                        sample_position_ids,
                        sample_causal_mask,
                        sample_current_pos
                    )
                )
                print("Model traced successfully...converting")
                
                if ENABLE_DEBUG:
                    print("\nModel inputs:")
                    print(f"  input_ids: {sample_input_ids.shape}, {sample_input_ids.dtype}, device={sample_input_ids.device}")
                    print(f"  position_ids: {sample_position_ids.shape}, {sample_position_ids.dtype}, device={sample_position_ids.device}")
                    print(f"  causal_mask: {sample_causal_mask.shape}, {sample_causal_mask.dtype}, device={sample_causal_mask.device}")
                    print(f"  current_pos: {sample_current_pos.shape}, {sample_current_pos.dtype}, device={sample_current_pos.device}")
                    #print("\nExiting after trace for debug...")
                    #import sys
                    #sys.exit(0)  # Exit after tracing like in r1-min2.py
                
                # Verify the trace
                print("Verifying traced model...")
                traced_output = traced_model(
                    sample_input_ids,
                    sample_position_ids,
                    sample_causal_mask,
                    sample_current_pos
                )
                
                # Check outputs match
                if isinstance(test_output, tuple):
                    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(test_output, traced_output)), \
                        "Traced model outputs don't match original model"
                else:
                    assert torch.allclose(test_output, traced_output, atol=1e-5), \
                        "Traced model output doesn't match original model"
                print("Traced model verification successful")
            
            # Prepare KV cache states
            states = self._get_kv_cache_states(model)
            
            # Prepare outputs based on model configuration
            if hasattr(model, 'lm_head8_1'):  # ENABLE_VACAB_SPLIT8
                outputs = [
                    ct.TensorType(name=f"logits{i}", dtype=np.float16)
                    for i in range(1, 9)
                ]
            elif hasattr(model, 'lm_head2_1'):  # ENABLE_VACAB_SPLIT
                outputs = [
                    ct.TensorType(name="logits1", dtype=np.float16),
                    ct.TensorType(name="logits2", dtype=np.float16),
                ]
            else:
                outputs = [
                    ct.TensorType(name="logits", dtype=np.float16),
                ]
            
            # Convert using CoreML
            print("Converting traced model to CoreML format...")
            mlmodel = ct.convert(
                traced_model,
                inputs=[
                    ct.TensorType(
                        name="input_ids",
                        shape=(1, 1),  # Single token input
                        dtype=np.int32
                    ),

                    ct.TensorType(
                        name="position_ids",
                        shape=(1,),  # Single position ID
                        dtype=np.int32
                    ),
                    ct.TensorType(
                        name="causal_mask",
                        shape=(1, 1, 1, self.context_length),  # Causal mask
                        dtype=np.float16
                    ),
                    ct.TensorType(
                        name="current_pos",
                        shape=(1,),  # Current position
                        dtype=np.int32
                    ),
                ],
                outputs=outputs,
                states=states,
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
                convert_to="mlprogram",
            )
            
            return mlmodel
            
        except Exception as e:
            print(f"Error during model conversion: {str(e)}")
            raise

    def _get_kv_cache_states(self, model):
        """Get KV cache states configuration for unified or split cache."""
        if hasattr(model.model, "kv_cache_0"):  # Unified cache
            print("Using unified KV cache configuration")
            return [
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(
                            2 * model.config.num_hidden_layers,  # Combined K and V caches
                            model.config.num_key_value_heads,
                            self.state_length,
                            self.head_dim
                        ),
                        dtype=np.float16
                    ),
                    name="model.model.kv_cache_0"
                )
            ]
        else:  # Split cache per layer
            print("Using per-layer KV cache configuration")
            states = []
            for i in range(model.config.num_hidden_layers):
                states.append(
                    ct.StateType(
                        wrapped_type=ct.TensorType(
                            shape=(
                                2,  # K and V caches
                                model.config.num_key_value_heads,
                                self.state_length,
                                self.head_dim
                            ),
                            dtype=np.float16
                        ),
                        name=f"model.model.kv_cache_{i}"
                    )
                )
            return states

    def preprocess(self):
        """Preprocessing steps before conversion."""
        print("Preparing model for conversion...")
        
        # Move model to correct device
        print(f"Moving model to device: {TEST_DEVICE}")
        self.model = self.model.to(TEST_DEVICE)
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        
        # Freeze model parameters and disable gradients
        print("Freezing model parameters...")
        self.model.requires_grad_(False)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Ensure all submodules are in eval mode
        def set_eval_and_freeze(module):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        
        self.model.apply(set_eval_and_freeze)
        
        print("Model preprocessing completed")

    def postprocess(self, num_workers=None):
        """Postprocessing steps after conversion.
        
        Args:
            num_workers: Optional number of workers for parallel processing.
                        If None, uses default single worker.
        """
        if self.converted_model is not None and self.lut_bits is not None:
            print(f"Applying LUT quantization with {self.lut_bits} bits and {self.per_channel} channels per group using {num_workers if num_workers else 1} worker(s)...")
            try:
                # Set up quantization config
                config = cto.coreml.OptimizationConfig(
                    global_config=cto.coreml.OpPalettizerConfig(
                        mode="kmeans",
                        nbits=self.lut_bits,
                        granularity="per_grouped_channel",
                        group_size=self.per_channel,
                        num_kmeans_workers=num_workers if num_workers is not None else 1  # Use provided workers or default to 1
                    ),
                )
                
                # Apply quantization in a try-except block
                try:
                    self.converted_model = cto.coreml.palettize_weights(self.converted_model, config)
                    print("LUT quantization completed")
                except ValueError as e:
                    if "Pool not running" in str(e):
                        print("Warning: Multiprocessing pool error, retrying with single process...")
                        # Retry with single process
                        config.global_config.num_kmeans_workers = 1
                        self.converted_model = cto.coreml.palettize_weights(self.converted_model, config)
                        print("LUT quantization completed (single process)")
                    else:
                        raise
            except Exception as e:
                print(f"Warning: LUT quantization failed: {str(e)}")
                print("Continuing with unquantized model...")

    def convert_embeddings(self, model):
        """Convert embeddings layer to CoreML format.
        
        Args:
            model: The PyTorch model containing embeddings
            
        Returns:
            ct.models.MLModel: Converted CoreML model for embeddings
        """
        print("\nConverting embeddings layer...")
        
        class EmbeddingsWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.embed_tokens = model.embed_tokens
                
            def forward(self, input_ids):
                hidden_states = self.embed_tokens(input_ids)
                return hidden_states.to(MODEL_DTYPE)
        
        # Create wrapper and ensure eval mode
        wrapper = EmbeddingsWrapper(model)
        wrapper.eval()
        
        # Create sample input
        sample_input = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)
        
        # Trace model
        print("Tracing embeddings model...")
        traced_model = torch.jit.trace(wrapper, sample_input)
        
        # Define flexible input shapes
        input_shape = ct.EnumeratedShapes(
            shapes=[[1, 1], [1, self.batch_size]],  # Support single token and batch_size tokens
            default=[1, 1]  # Use single token as default
        )
        
        print(f"Converting embeddings model with input shape: {input_shape}")

        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=input_shape,  # Use enumerated shapes instead of fixed shape
                    dtype=np.int32
                )
            ],
            outputs=[
                ct.TensorType(name="hidden_states", dtype=np.float16)
            ],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram"
        )
        
        # Apply LUT quantization if specified
        if self.lut_bits:
            self.converted_model = mlmodel  # Set for postprocess
            self.postprocess(num_workers=8)  # Allow passing num_workers if needed
            mlmodel = self.converted_model
        
        return mlmodel

    def convert_lm_head(self, model, lut_bits=None, output_dir="."):
        """Convert LM head layer to CoreML."""
        print("\nConverting LM head layer...")
        
        class LMHeadWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                if hasattr(model, 'lm_head8_1'):  # 8-way split
                    self.heads = [
                        getattr(model, f'lm_head8_{i}')
                        for i in range(1, 9)
                    ]
                    self.split_mode = '8way'
                elif hasattr(model, 'lm_head2_1'):  # 2-way split
                    self.heads = [model.lm_head2_1, model.lm_head2_2]
                    self.split_mode = '2way'
                elif hasattr(model, 'lm_head1'):  # Single Conv2d
                    self.head = model.lm_head1
                    self.split_mode = 'single'
                else:  # Linear head
                    self.head = model.lm_head
                    self.split_mode = 'linear'
            
            def forward(self, hidden_states):
                # Reshape input for Conv2d if needed
                if self.split_mode != 'linear':
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)
                
                if self.split_mode == '8way':
                    logits = [head(hidden_states).squeeze(2).transpose(1, 2) 
                             for head in self.heads]
                    return tuple(logits)
                elif self.split_mode == '2way':
                    logits1 = self.heads[0](hidden_states).squeeze(2).transpose(1, 2)
                    logits2 = self.heads[1](hidden_states).squeeze(2).transpose(1, 2)
                    return logits1, logits2
                elif self.split_mode == 'single':
                    return self.head(hidden_states).squeeze(2).transpose(1, 2)
                else:  # linear
                    return self.head(hidden_states)
        
        # Create wrapper and ensure eval mode
        wrapper = LMHeadWrapper(model)
        wrapper.eval()
        
        # Create sample input
        sample_input = torch.zeros((1, 1, model.config.hidden_size), 
                                 dtype=MODEL_DTYPE, device=TEST_DEVICE)
        
        # Trace model
        print("Tracing LM head model...")
        traced_model = torch.jit.trace(wrapper, sample_input)
        
        # Define outputs based on head type
        if wrapper.split_mode == '8way':
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16)
                for i in range(1, 9)
            ]
        elif wrapper.split_mode == '2way':
            outputs = [
                ct.TensorType(name="logits1", dtype=np.float16),
                ct.TensorType(name="logits2", dtype=np.float16)
            ]
        else:
            outputs = [ct.TensorType(name="logits", dtype=np.float16)]
        
        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=(1, 1, model.config.hidden_size),
                    dtype=np.float16
                )
            ],
            outputs=outputs,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram"
        )
        
        # Apply LUT quantization if specified
        if lut_bits is not None:
            print(f"Applying LUT quantization with {lut_bits} bits...")
            try:
                # Set up quantization config
                config = cto.coreml.OptimizationConfig(
                    global_config=cto.coreml.OpPalettizerConfig(
                        mode="kmeans",
                        nbits=lut_bits,
                        granularity="per_grouped_channel",
                        group_size=self.per_channel,
                        num_kmeans_workers=8
                    ),
                )
                
                # Apply quantization
                mlmodel = cto.coreml.palettize_weights(mlmodel, config)
                print("LUT quantization completed")
            except Exception as e:
                print(f"Warning: LUT quantization failed: {str(e)}")
                print("Continuing with unquantized model...")
        
        return mlmodel

    def convert_FFN(self, model, chunk_idx=None):
        """Convert Feed-Forward Network layers to CoreML format.
        
        Args:
            model: The model to convert
            chunk_idx: If set, converts only the specified chunk of layers
        """
        print("\nConverting FFN layers...")
        total_layers = model.config.num_hidden_layers
        
        if chunk_idx is not None:
            layers_per_chunk = total_layers // self.num_chunks
            start_layer = chunk_idx * layers_per_chunk
            end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
            print(f"Processing chunk {chunk_idx + 1}/{self.num_chunks}")
            print(f"  Total layers: {total_layers}")
            print(f"  Layers per chunk: {layers_per_chunk}")
            print(f"  This chunk: layers [{start_layer}..{end_layer-1}]")
            if chunk_idx == 0:
                print("  First chunk: includes input layer")
            if chunk_idx == self.num_chunks - 1:
                print("  Last chunk: includes output layer")
        else:
            start_layer = 0
            end_layer = None
            print("Processing all layers at once")
        
        class FFNWrapper(torch.nn.Module):
            def __init__(self, model, start_layer=0, end_layer=None):
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer
                self.states = LlamaConverter.GetTransformerStates(model, part='2', prefix="model.model.")
                
            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                return self.model.model(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    IN_PREFILL=False
                )
        
        try:
            # Create wrapper and ensure eval mode
            wrapper = FFNWrapper(model, start_layer, end_layer)
            wrapper.eval()
            
            # Create sample inputs with correct shapes
            hidden_states = torch.zeros(
                (1, 1, model.config.hidden_size),  # Shape: (batch, seq_len, hidden)
                dtype=torch.float16,device=TEST_DEVICE
            )
            position_ids = torch.zeros((1,), dtype=torch.long, device=TEST_DEVICE)
            causal_mask = torch.full(
                (1, 1, 1, self.context_length),  # Shape: (batch, 1, 1, context_len)
                torch.finfo(MODEL_DTYPE).min,
                dtype=MODEL_DTYPE,device=TEST_DEVICE
            )
            current_pos = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)
            
            # Trace model
            print("Tracing FFN model...")
            traced_model = torch.jit.trace(
                wrapper, 
                (hidden_states, position_ids, causal_mask, current_pos)
            )
            
            # Prepare inputs/outputs for conversion
            inputs = [
                ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),  # (1, 1, context_len)
                ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),      # (1,)
                ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),      # (1, 1, 1, context_len)
                ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),        # (1,)
            ]
            
            outputs = [
                ct.TensorType(name="output_hidden_states", dtype=np.float16)
            ]
            
            # Convert to CoreML
            mlmodel = ct.convert(
                traced_model,
                inputs=inputs,
                outputs=outputs,
                states=wrapper.states,
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
                convert_to="mlprogram"
            )
            
            print("FFN layers conversion completed")
            
            # Apply LUT quantization if specified
            if self.lut_bits:
                self.converted_model = mlmodel
                self.postprocess(num_workers=None)  # Allow passing num_workers if needed
                mlmodel = self.converted_model
            
            return mlmodel
            
        except Exception as e:
            print(f"Error during FFN conversion: {str(e)}")
            raise

    def convert_prefill(self, model, chunk_idx=None):
        """Convert transformer for prefill mode to CoreML format.
        
        Args:
            model: The model to convert
            chunk_idx: If set, converts only the specified chunk of layers
        """
        print("\nConverting transformer prefill mode...")
        total_layers = model.config.num_hidden_layers
        
        if chunk_idx is not None:
            layers_per_chunk = total_layers // self.num_chunks
            start_layer = chunk_idx * layers_per_chunk
            end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
            print(f"Processing chunk {chunk_idx + 1}/{self.num_chunks} (layers {start_layer} to {end_layer-1})")
        else:
            start_layer = 0
            end_layer = None
        
        class PrefillWrapper(torch.nn.Module):
            def __init__(self, model, start_layer=0, end_layer=None):
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer
                self.states = LlamaConverter.GetTransformerStates(model, part='2_prefill', prefix="model.model.")
            
            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                return self.model.model(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    IN_PREFILL=True
                )
        
        try:
            # Create wrapper with layer range if chunking
            wrapper = PrefillWrapper(model, start_layer, end_layer)
            wrapper.eval()
            
            # Always use consistent batch_size input shape for prefill
            # The model will handle output shape changes (returns [:, 0:1, :] for last chunk)
            print(f"Using standard prefill shape: (1, {self.batch_size}, {model.config.hidden_size})")
            
            # Create sample inputs with consistent shapes for prefill
            hidden_states = torch.zeros(
                (1, self.batch_size, model.config.hidden_size),  # Shape: (1, batch_size, hidden)
                dtype=torch.float16, device=TEST_DEVICE
            )
            position_ids = torch.zeros(
                (self.batch_size,),  # Shape: (batch_size,)
                dtype=torch.long, device=TEST_DEVICE
            )
            causal_mask = torch.full(
                (1, 1, self.batch_size, self.context_length),  # Shape: (1, 1, batch_size, context_len)
                torch.finfo(MODEL_DTYPE).min,
                dtype=MODEL_DTYPE, device=TEST_DEVICE
            )
            current_pos = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)  # Shape: (1,)
            
            # Trace model
            print("Tracing prefill model...")
            traced_model = torch.jit.trace(
                wrapper, 
                (hidden_states, position_ids, causal_mask, current_pos)
            )
            
            # Prepare inputs/outputs for conversion
            inputs = [
                ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),    # (1, batch, hidden)
                ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),        # (batch,)
                ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),        # (1, 1, batch, context_len)
                ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),          # (1,)
            ]
            
            outputs = [
                ct.TensorType(name="output_hidden_states", dtype=np.float16)  # Shape will be inferred
            ]
            
            # Convert to CoreML
            mlmodel = ct.convert(
                traced_model,
                inputs=inputs,
                outputs=outputs,
                states=wrapper.states,
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
                convert_to="mlprogram"
            )
            
            print("Prefill mode conversion completed")
            
            # Apply LUT quantization if specified
            if self.lut_bits:
                self.converted_model = mlmodel
                self.postprocess(num_workers=None)  # Allow passing num_workers if needed
                mlmodel = self.converted_model
            
            return mlmodel
            
        except Exception as e:
            print(f"Error during prefill conversion: {str(e)}")
            raise

def parse_args():
    parser = argparse.ArgumentParser(description='Convert LLaMA model to CoreML format')
    
    # Model configuration
    parser.add_argument('--model', type=str, help='Path to model directory (default: ../Meta-Llama-3.2-1B)')
    parser.add_argument('--prefix', type=str, default='llama', help='Prefix for output filenames')
    
    # Conversion options
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for prefill')
    parser.add_argument('--context-length', type=int, default=512, help='Maximum context length')
    parser.add_argument('--lut', type=int, default=None, help='Use LUT quantization with N bits')
    parser.add_argument('--chunk', type=int, default=None, help='Split into N chunks')
    parser.add_argument('--part', type=str, 
                       choices=['1', '2', '2_prefill', '3', 'all'], 
                       default='all',
                       help='Convert specific part (1=embeddings, 2=FFN, 2_prefill=FFN prefill mode, 3=lm_head)')
    parser.add_argument('--output', type=str, default='.',
                      help='Output directory for converted models (default: current directory)')
    
    return parser.parse_args()

def test_conversion(model_path=None, output_path=None, context_length=512, lut_bits=4, 
                   model=None, skip_load_weights=False, split_part='123', 
                   batch_size=64, num_chunks=1, prefix='llama', output_dir='.'):
    """Test conversion of a LLAMA model to ANE format."""
    if model is None:
        print(f"Testing conversion with model from {model_path}")
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")
        
        config = LlamaConfig.from_json(config_path)
        print("Loaded model config:")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  vocab_size: {config.vocab_size}")
        
        # Initialize model
        model = LlamaForCausalLM(config)
        
        # Load weights if available and not skipped
        if os.path.exists(model_path) and not skip_load_weights:
            print("\nLoading pretrained weights...")
            model.load_pretrained_weights(model_path)
        else:
            print("\nSkipping weights loading")
    
    # Create converter with batch_size
    converter = LlamaConverter(
        model=model,
        context_length=context_length,
        lut_bits=lut_bits,
        batch_size=batch_size,
        num_chunks=num_chunks
    )
    
    # Initialize converted_model as None
    converted_model = None
    
    # Handle FFN and prefill conversions (both chunked and non-chunked)
    if split_part in ['2', '2_prefill']:
        converted_models = []
        chunks_to_process = range(num_chunks)
        
        for i in chunks_to_process:
            # Use FFN in filename for mode '2', keep simple prefill for '2_prefill'
            base_name = f'{prefix}_FFN' if split_part == '2' else f'{prefix}_prefill'
            if lut_bits is not None:
                base_name += f'_lut{lut_bits}'
            chunk_output_path = f"{base_name}_chunk_{i+1:02d}of{num_chunks:02d}.mlpackage"
            
            print(f"\nConverting chunk {i+1}/{num_chunks}")
            
            # Clean up before converting next chunk
            gc.collect()
            
            # For single chunk (num_chunks=1), don't pass chunk_idx
            chunk_idx = i if num_chunks > 1 else None
            
            if split_part == '2':
                chunk_model = converter.convert_FFN(model, chunk_idx=i)
            else:  # '2_prefill'
                chunk_model = converter.convert_prefill(model, chunk_idx=i)
                
            if chunk_output_path:
                # Add metadata before saving
                AddMetadata(chunk_model, {
                    'context_length': context_length,
                    'num_chunks': num_chunks,
                    'chunk_no': i+1,
                    'batch_size': batch_size if split_part in ['2_prefill'] else None,
                    'lut_bits': lut_bits,
                    'split_part': split_part
                })
                print(f"Saving chunk to {chunk_output_path}")
                chunk_output_path = os.path.join(output_dir, chunk_output_path)
                chunk_model.save(chunk_output_path)
                
            converted_models.append(chunk_model)
            
            # Clean up after saving
            del chunk_model
            gc.collect()
            
            # Small delay to ensure cleanup
            import time
            time.sleep(1)
            
        converted_model = converted_models
    else:
        # Convert model based on split_part
        if split_part == '1':
            base_name = f'{prefix}_embeddings'
        elif split_part == '3':
            base_name = f'{prefix}_lm_head'
        elif split_part == '123':
            base_name = f'{prefix}_'
        else:
            raise ValueError(f"Invalid split_part: {split_part}")

        if lut_bits is not None:
            base_name += f'_lut{lut_bits}'
        output_path = f"{base_name}.mlpackage"

        print(f"\nConverting model part: {split_part} output_path: {output_path}")
        converted_model = converter.convert(split_part=split_part)

        # Add metadata before saving
        if output_path:
            if isinstance(converted_model, list):
                # Handle multi-part models (123 mode)
                for i, chunk_model in enumerate(converted_model):
                    AddMetadata(chunk_model, {
                        'context_length': context_length,
                        'num_chunks': num_chunks,
                        'chunk_no': i+1,
                        'batch_size': batch_size if split_part in ['2_prefill'] else None,
                        'lut_bits': lut_bits,
                        'split_part': split_part
                    })
                    chunk_output_path = output_path.replace('.mlpackage', f'_{i+1}.mlpackage')
                    print(f"Saving chunk to {chunk_output_path}")
                    chunk_output_path = os.path.join(output_dir, chunk_output_path)
                    chunk_model.save(chunk_output_path)
            else:
                # Handle single model parts
                AddMetadata(converted_model, {
                    'context_length': context_length,
                    'batch_size': batch_size if split_part in ['2_prefill'] else None,
                    'lut_bits': lut_bits,
                    'split_part': split_part
                })
                print(f"Saving model to {output_path}")
                output_path = os.path.join(output_dir, output_path)
                converted_model.save(output_path)

    # Model verification
    if converted_model is not None:
        print("\nModel verification:")
        if isinstance(converted_model, list):
            # For multi-part models, use chunk numbers instead of hardcoded component names
            for i, model in enumerate(converted_model):
                print(f"\nChunk {i+1}:")
                print(f"Input names: {model.input_description}")
                print(f"Output names: {model.output_description}")
        else:
            print(f"Input names: {converted_model.input_description}")
            print(f"Output names: {converted_model.output_description}")

        # Cleanup after verification
        if not isinstance(converted_model, list):
            temp_model = converted_model
            del converted_model
            gc.collect()
            converted_model = temp_model

    return converted_model

def main():
    args = parse_args()
    
    # Set model path
    model_path = args.model if args.model else "../Meta-Llama-3.2-1B"
    
    print(f"\nConverting model from: {model_path}")
    print(f"Output filename prefix: {args.prefix}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    if args.lut:
        print(f"LUT quantization: {args.lut} bits")
    if args.chunk:
        print(f"Splitting into {args.chunk} chunks")
    print(f"Converting part(s): {args.part}")
    
    # Initialize and convert model
    try:
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")
        
        config = LlamaConfig.from_json(config_path)

        config.context_length = args.context_length
        if config.state_length < args.context_length:
            config.state_length = args.context_length
     

        print("\nLoaded model config:")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  vocab_size: {config.vocab_size}")
        
        # Initialize model
        model = LlamaForCausalLM(config)
        
        # Load weights
        print("\nLoading pretrained weights...")
        model.load_pretrained_weights(model_path)
        
        # Create output directory if needed
        os.makedirs(args.output, exist_ok=True)
        
        # Pass output directory to test_conversion
        test_conversion(
            model=model,
            split_part=args.part,
            prefix=args.prefix,
            context_length=args.context_length,
            lut_bits=args.lut,
            batch_size=args.batch_size,
            num_chunks=args.chunk,
            output_dir=args.output
        )
            
    except Exception as e:
        print(f"\nError during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# to RUN
# python -m anemll.ane_converter.llama_converter