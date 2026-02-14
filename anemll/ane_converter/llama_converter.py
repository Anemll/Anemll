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
import warnings
try:
    from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
except Exception:  # pragma: no cover - sklearn optional
    SklearnConvergenceWarning = None

if SklearnConvergenceWarning is not None:
    warnings.filterwarnings("ignore", category=SklearnConvergenceWarning)
warnings.filterwarnings("ignore", message="Number of distinct clusters .* smaller than n_clusters")
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
from .metadata import AddMetadata, get_anemll_version
import argparse
import sys

class LlamaConverter(BaseConverter):
    """Handles LLAMA model conversion to Apple Neural Engine format."""

    def __init__(
        self,
        model,
        context_length=512,
        state_length=None,
        lut_bits=4,
        per_channel=8,
        batch_size=64,
        num_chunks=1,
        argmax_in_model=False,
        lut_embeddings_bits=None,
        lut_embeddings_per_channel=8,
        lut_lmhead_bits=None,
        lut_lmhead_per_channel=8,
    ):
        super().__init__(model)
        self.context_length = context_length
        self.state_length = state_length or context_length
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.lut_bits = lut_bits
        self.per_channel = per_channel
        self.converted_model = None
        self.batch_size = batch_size
        self.num_chunks = num_chunks
        self.argmax_in_model = argmax_in_model
        self.lut_embeddings_bits = lut_embeddings_bits
        self.lut_embeddings_per_channel = lut_embeddings_per_channel
        self.lut_lmhead_bits = lut_lmhead_bits
        self.lut_lmhead_per_channel = lut_lmhead_per_channel

    def convert(self, split_part=None):
        """Convert model to CoreML format with optional splitting.

        Args:
            split_part: Which part(s) of the model to convert:
                       '1' - embeddings only
                       '2' - transformer FFN only
                       '2_prefill' - transformer prefill mode
                       '3' - LM head only
                       '123' - full model (all components)
                       'monolithic' - single file with embed+FFN+lmhead (inference mode)
                       'monolithic_prefill' - single file with embed+FFN+lmhead (prefill mode)

        Returns:
            ct.models.MLModel or list[ct.models.MLModel]: Converted model(s)
        """
        if split_part not in ['1', '2', '2_prefill', '3', '123', 'monolithic', 'monolithic_prefill']:
            raise ValueError("split_part must be one of: '1', '2', '2_prefill', '3', '123', 'monolithic', 'monolithic_prefill'")
            
        self.preprocess()
        
        # Handle individual components
        if split_part == '1':
            return self.convert_embeddings(self.model)
        elif split_part == '2':
            return self.convert_FFN(self.model)
        elif split_part == '2_prefill':
            return self.convert_prefill(self.model)
        elif split_part == '3':
            return self.convert_lm_head(
                self.model,
                lut_bits=self.lut_bits,
                argmax_in_model=self.argmax_in_model,
            )
        elif split_part == 'monolithic':
            return self.convert_monolithic(
                self.model,
                is_prefill=False,
                argmax_in_model=self.argmax_in_model,
            )
        elif split_part == 'monolithic_prefill':
            # Prefill is cache-building only; keep logits path for compatibility.
            return self.convert_monolithic(
                self.model,
                is_prefill=True,
                argmax_in_model=False,
            )

        # Handle full model conversion
        elif split_part == '123':
            embeddings_model = self.convert_embeddings(self.model)
            transformer_model = self.convert_FFN(self.model)
            lm_head_model = self.convert_lm_head(
                self.model,
                lut_bits=self.lut_bits,
                argmax_in_model=self.argmax_in_model,
            )
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
                self._reset_kv_cache_buffers(wrapper)
                traced_model = torch.jit.trace(
                    wrapper,
                    (
                        sample_input_ids,
                        sample_position_ids,
                        sample_causal_mask,
                        sample_current_pos
                    )
                )
                self._reset_kv_cache_buffers(wrapper)
                self._reset_kv_cache_buffers(traced_model)
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

    @staticmethod
    def _make_palettizer_config(nbits, per_channel, num_workers):
        """Build an OpPalettizerConfig for the given bit-width / granularity."""
        if per_channel <= 0:
            return cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=nbits,
                granularity="per_tensor",
                num_kmeans_workers=num_workers if num_workers is not None else 1,
            )
        return cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=nbits,
            granularity="per_grouped_channel",
            group_size=per_channel,
            num_kmeans_workers=num_workers if num_workers is not None else 1,
        )

    def postprocess(self, num_workers=None):
        """Postprocessing steps after conversion.

        Supports per-component LUT overrides for monolithic models via
        lut_embeddings_bits / lut_lmhead_bits.

        Args:
            num_workers: Optional number of workers for parallel processing.
                        If None, uses default single worker.
        """
        if self.converted_model is not None and self.lut_bits is not None:
            # Check if using per-tensor quantization (per_channel <= 0 means per-tensor)
            use_per_tensor = self.per_channel <= 0
            if use_per_tensor:
                print(f"Applying LUT quantization with {self.lut_bits} bits using PER-TENSOR granularity with {num_workers if num_workers else 1} worker(s)...")
            else:
                print(f"Applying LUT quantization with {self.lut_bits} bits and {self.per_channel} channels per group using {num_workers if num_workers else 1} worker(s)...")
            try:
                # Suppress sklearn ConvergenceWarning during quantization
                with warnings.catch_warnings():
                    if SklearnConvergenceWarning is not None:
                        warnings.simplefilter('ignore', SklearnConvergenceWarning)
                    warnings.simplefilter('ignore', UserWarning)

                    # Default (FFN) quantization config
                    global_cfg = self._make_palettizer_config(
                        self.lut_bits, self.per_channel, num_workers
                    )

                    # Per-component overrides for monolithic models
                    op_name_configs = {}
                    has_overrides = (
                        self.lut_embeddings_bits is not None
                        or self.lut_lmhead_bits is not None
                    )

                    if has_overrides:
                        prog = self.converted_model._mil_program  # noqa: SLF001
                        for fn_name in prog.functions:
                            fn = prog.functions[fn_name]
                            for op in fn.operations:
                                op_name = op.name or ""
                                if self.lut_embeddings_bits is not None and "embed_tokens" in op_name:
                                    op_name_configs[op_name] = self._make_palettizer_config(
                                        self.lut_embeddings_bits,
                                        self.lut_embeddings_per_channel,
                                        num_workers,
                                    )
                                elif self.lut_lmhead_bits is not None and "lm_head" in op_name:
                                    op_name_configs[op_name] = self._make_palettizer_config(
                                        self.lut_lmhead_bits,
                                        self.lut_lmhead_per_channel,
                                        num_workers,
                                    )

                        if op_name_configs:
                            embed_count = sum(1 for k in op_name_configs if "embed_tokens" in k)
                            lmhead_count = sum(1 for k in op_name_configs if "lm_head" in k)
                            if self.lut_embeddings_bits is not None:
                                print(f"  Embeddings: {self.lut_embeddings_bits}-bit LUT (per_channel={self.lut_embeddings_per_channel}) -> {embed_count} ops")
                            if self.lut_lmhead_bits is not None:
                                print(f"  LM head: {self.lut_lmhead_bits}-bit LUT (per_channel={self.lut_lmhead_per_channel}) -> {lmhead_count} ops")

                    config = cto.coreml.OptimizationConfig(
                        global_config=global_cfg,
                        op_name_configs=op_name_configs if op_name_configs else None,
                    )

                    # Apply quantization in a try-except block
                    try:
                        self.converted_model = cto.coreml.palettize_weights(self.converted_model, config)
                        print("LUT quantization completed")
                    except ValueError as e:
                        if "Pool not running" in str(e):
                            print("Warning: Multiprocessing pool error, retrying with single process...")
                            config.global_config.num_kmeans_workers = 1
                            self.converted_model = cto.coreml.palettize_weights(self.converted_model, config)
                            print("LUT quantization completed (single process)")
                        else:
                            raise
            except Exception as e:
                print(f"Warning: LUT quantization failed: {str(e)}")
                print("Continuing with unquantized model...")

    @staticmethod
    def _reset_kv_cache_buffers(module):
        """Clear mutable KV-cache buffers to avoid trace side-effects in state dict checks."""
        with torch.no_grad():
            for name, buffer in module.named_buffers():
                if "kv_cache_" in name:
                    buffer.zero_()

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

    def convert_lm_head(
        self, model, lut_bits=None, output_dir=".", argmax_in_model: bool = False
    ):
        """Convert LM head layer to CoreML."""
        print("\nConverting LM head layer...")
        
        class LMHeadWrapper(torch.nn.Module):
            def __init__(self, model, argmax_mode: bool = False):
                super().__init__()
                self.argmax_mode = argmax_mode
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
                    logits = [head(hidden_states).squeeze(2).transpose(1, 2) for head in self.heads]
                elif self.split_mode == '2way':
                    logits = [
                        self.heads[0](hidden_states).squeeze(2).transpose(1, 2),
                        self.heads[1](hidden_states).squeeze(2).transpose(1, 2),
                    ]
                elif self.split_mode == 'single':
                    logits = [self.head(hidden_states).squeeze(2).transpose(1, 2)]
                else:  # linear
                    logits = [self.head(hidden_states)]

                if self.argmax_mode:
                    all_idx = []
                    all_val = []
                    for chunk_logits in logits:
                        chunk_argmax = torch.argmax(chunk_logits, dim=-1, keepdim=True)
                        chunk_max_val = torch.gather(chunk_logits, -1, chunk_argmax)
                        all_idx.append(chunk_argmax.to(torch.int32))
                        all_val.append(chunk_max_val)
                    argmax_idx = torch.cat(all_idx, dim=-1).squeeze(0).squeeze(0)
                    argmax_val = torch.cat(all_val, dim=-1).squeeze(0).squeeze(0)
                    return (argmax_idx, argmax_val)

                if self.split_mode == '8way':
                    return tuple(logits)
                if self.split_mode == '2way':
                    return logits[0], logits[1]
                return logits[0]
        
        # Create wrapper and ensure eval mode
        wrapper = LMHeadWrapper(model, argmax_mode=argmax_in_model)
        wrapper.eval()
        
        # Create sample input
        sample_input = torch.zeros((1, 1, model.config.hidden_size), 
                                 dtype=MODEL_DTYPE, device=TEST_DEVICE)
        
        # Trace model
        print("Tracing LM head model...")
        traced_model = torch.jit.trace(wrapper, sample_input)
        
        # Define outputs based on head type
        if argmax_in_model:
            outputs = [
                ct.TensorType(name="argmax_idx", dtype=np.int32),
                ct.TensorType(name="argmax_val", dtype=np.float16),
            ]
        elif wrapper.split_mode == '8way':
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
            base, rem = divmod(total_layers, self.num_chunks)
            start_layer = chunk_idx * base + min(chunk_idx, rem)
            end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
            print(f"Processing chunk {chunk_idx + 1}/{self.num_chunks}")
            print(f"  Total layers: {total_layers}")
            print(f"  Layers per chunk: {base} (+1 for first {rem} chunks)")
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
            self._reset_kv_cache_buffers(wrapper)
            traced_model = torch.jit.trace(
                wrapper, 
                (hidden_states, position_ids, causal_mask, current_pos)
            )
            self._reset_kv_cache_buffers(wrapper)
            self._reset_kv_cache_buffers(traced_model)
            
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
            base, rem = divmod(total_layers, self.num_chunks)
            start_layer = chunk_idx * base + min(chunk_idx, rem)
            end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
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
            self._reset_kv_cache_buffers(wrapper)
            traced_model = torch.jit.trace(
                wrapper, 
                (hidden_states, position_ids, causal_mask, current_pos)
            )
            self._reset_kv_cache_buffers(wrapper)
            self._reset_kv_cache_buffers(traced_model)
            
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

    def convert_monolithic(
        self, model, is_prefill: bool = False, argmax_in_model: bool = False
    ):
        """Convert full model (embeddings + FFN + LM head) to single CoreML model.

        This creates a monolithic model that takes input_ids and returns logits,
        combining all components into a single file for simpler deployment.

        Args:
            model: The LLaMA model to convert
            is_prefill: If True, convert for prefill mode (batch processing)
                       If False, convert for inference mode (single token)

        Returns:
            ct.models.MLModel: Monolithic CoreML model
        """
        mode_str = "prefill" if is_prefill else "inference"
        print(f"\nConverting monolithic model for {mode_str} mode...")

        class MonolithicWrapper(torch.nn.Module):
            """Wrapper combining embeddings + transformer + LM head."""

            def __init__(
                self,
                model,
                context_length: int,
                is_prefill: bool,
                argmax_in_model: bool = False,
            ) -> None:
                super().__init__()
                self.model = model
                self.context_length = context_length
                self.is_prefill = is_prefill
                self.argmax_in_model = argmax_in_model

                # Determine LM head mode
                if hasattr(model, "lm_head8_1"):
                    self.lm_head_mode = "8"
                    self.lm_heads = [
                        getattr(model, f"lm_head8_{i}") for i in range(1, 9)
                    ]
                elif hasattr(model, "lm_head2_1"):
                    self.lm_head_mode = "2"
                    self.lm_heads = [model.lm_head2_1, model.lm_head2_2]
                elif hasattr(model, "lm_head1"):
                    self.lm_head_mode = "1"
                    self.lm_head = model.lm_head1
                else:
                    self.lm_head_mode = "linear"
                    self.lm_head = model.lm_head

            def forward(
                self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                causal_mask: torch.Tensor,
                current_pos: torch.Tensor,
            ) -> tuple:
                # Step 1: Embeddings
                hidden_states = self.model.embed_tokens(input_ids)
                hidden_states = hidden_states.to(MODEL_DTYPE)

                # Step 2: Transformer layers
                hidden_states = self.model.model(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    start_layer=0,
                    end_layer=None,
                    IN_PREFILL=self.is_prefill,
                )

                # Step 3: LM Head
                if self.lm_head_mode != "linear":
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)

                if self.lm_head_mode in ("8", "2"):
                    logits_list = [
                        h(hidden_states).squeeze(2).transpose(1, 2)
                        for h in self.lm_heads
                    ]
                elif self.lm_head_mode == "1":
                    logits_list = [self.lm_head(hidden_states).squeeze(2).transpose(1, 2)]
                else:
                    logits_list = [self.lm_head(hidden_states)]

                if self.argmax_in_model and not self.is_prefill:
                    all_idx = []
                    all_val = []
                    for logits in logits_list:
                        chunk_argmax = torch.argmax(logits, dim=-1, keepdim=True)
                        chunk_max_val = torch.gather(logits, -1, chunk_argmax)
                        all_idx.append(chunk_argmax.to(torch.int32))
                        all_val.append(chunk_max_val)
                    argmax_idx = torch.cat(all_idx, dim=-1).squeeze(0).squeeze(0)
                    argmax_val = torch.cat(all_val, dim=-1).squeeze(0).squeeze(0)
                    return (argmax_idx, argmax_val)

                return tuple(logits_list)

        wrapper = MonolithicWrapper(
            model,
            self.context_length,
            is_prefill,
            argmax_in_model=argmax_in_model,
        )
        wrapper.eval()

        # Ensure no gradients
        for param in wrapper.parameters():
            param.requires_grad = False

        argmax_str = ", argmax_in_model=True" if (argmax_in_model and not is_prefill) else ""
        print(f"Monolithic wrapper created (LM head mode: {wrapper.lm_head_mode}{argmax_str})")

        # Prepare inputs based on mode
        if is_prefill:
            # Prefill mode: batch processing
            sample_input_ids = torch.zeros(
                (1, self.batch_size), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_position_ids = torch.zeros(
                (self.batch_size,), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_causal_mask = torch.zeros(
                (1, 1, self.batch_size, self.context_length),
                dtype=torch.float16,
                device=TEST_DEVICE,
            )
        else:
            # Inference mode: single token
            sample_input_ids = torch.zeros(
                (1, 1), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_position_ids = torch.zeros(
                (1,), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_causal_mask = torch.zeros(
                (1, 1, 1, self.context_length),
                dtype=torch.float16,
                device=TEST_DEVICE,
            )

        sample_current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        print(f"Sample inputs ({mode_str} mode):")
        print(f"  input_ids: {sample_input_ids.shape}")
        print(f"  position_ids: {sample_position_ids.shape}")
        print(f"  causal_mask: {sample_causal_mask.shape}")
        print(f"  current_pos: {sample_current_pos.shape}")

        # Trace model
        print("Tracing monolithic model...")
        self._reset_kv_cache_buffers(wrapper)
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper,
                (
                    sample_input_ids,
                    sample_position_ids,
                    sample_causal_mask,
                    sample_current_pos,
                ),
            )
        self._reset_kv_cache_buffers(wrapper)
        self._reset_kv_cache_buffers(traced)
        print("Tracing completed!")

        # Define outputs based on LM head mode
        if argmax_in_model and not is_prefill:
            outputs = [
                ct.TensorType(name="argmax_idx", dtype=np.int32),
                ct.TensorType(name="argmax_val", dtype=np.float16),
            ]
        elif wrapper.lm_head_mode == "8":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16)
                for i in range(1, 9)
            ]
        elif wrapper.lm_head_mode == "2":
            outputs = [
                ct.TensorType(name="logits1", dtype=np.float16),
                ct.TensorType(name="logits2", dtype=np.float16),
            ]
        else:
            outputs = [ct.TensorType(name="logits", dtype=np.float16)]

        # Convert to CoreML
        print("Starting CoreML conversion...")
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="input_ids", shape=sample_input_ids.shape, dtype=np.int32
                ),
                ct.TensorType(
                    name="position_ids", shape=sample_position_ids.shape, dtype=np.int32
                ),
                ct.TensorType(
                    name="causal_mask", shape=sample_causal_mask.shape, dtype=np.float16
                ),
                ct.TensorType(
                    name="current_pos", shape=sample_current_pos.shape, dtype=np.int32
                ),
            ],
            outputs=outputs,
            states=LlamaConverter.GetTransformerStates(model, part=None, prefix="model.model."),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        print(f"CoreML conversion for monolithic {mode_str} completed!")

        # Apply LUT quantization if specified
        if self.lut_bits:
            print(f"Applying LUT quantization ({self.lut_bits} bits)...")
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)
            mlmodel = self.converted_model

        return mlmodel


def parse_lut_arg(lut_value):
    """Parse LUT argument that can be 'bits', 'bits,per_channel', or 'bits,0' for per-tensor.

    Args:
        lut_value: String value from command line (e.g., '6', '6,4', '4,0' for per-tensor)

    Returns:
        tuple: (lut_bits, per_channel) where:
               - per_channel > 0 means per_grouped_channel quantization
               - per_channel <= 0 means per_tensor quantization
               - per_channel defaults to 8 if not specified
    """
    if lut_value is None:
        return None, 8

    if isinstance(lut_value, int):
        return lut_value, 8  # Default per_channel value

    lut_str = str(lut_value).strip().lower()

    # Handle "none" as no LUT quantization
    if lut_str in ('none', 'no', 'false', ''):
        return None, 8

    if ',' in lut_str:
        parts = lut_str.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid LUT format: {lut_value}. Expected 'bits' or 'bits,per_channel'")
        try:
            lut_bits = int(parts[0])
            per_channel_str = parts[1].strip().lower()
            # Allow "tensor" or "t" or "0" for per-tensor quantization
            if per_channel_str in ('tensor', 't', '0'):
                per_channel = 0
            else:
                per_channel = int(parts[1])
            return lut_bits, per_channel
        except ValueError:
            raise ValueError(f"Invalid LUT format: {lut_value}. Expected 'bits' or 'bits,per_channel'")
    else:
        try:
            lut_bits = int(lut_str)
            return lut_bits, 8  # Default per_channel value
        except ValueError:
            raise ValueError(f"Invalid LUT bits value: {lut_value}")

def parse_args():
    parser = argparse.ArgumentParser(description='Convert LLaMA model to CoreML format')

    # Model configuration
    parser.add_argument('--model', type=str, help='Path to model directory (default: ../Meta-Llama-3.2-1B)')
    parser.add_argument('--prefix', type=str, default='llama', help='Prefix for output filenames')

    # Conversion options
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for prefill')
    parser.add_argument('--context-length', type=int, default=512, help='Maximum context length')
    parser.add_argument('--lut', type=str, default=None, help='Use LUT quantization with N bits, optionally specify per_channel as "bits,per_channel" (e.g., "6,4"). Default per_channel is 8')
    parser.add_argument('--chunk', type=int, default=None, help='Split into N chunks')
    parser.add_argument('--dynamic-prefill-slice', action='store_true',
                       help='Use dynamic slicing for prefill KV writes (default ON for meta generation; no-op here)')
    parser.add_argument('--static-prefill-slice', action='store_true',
                       help='Disable dynamic slicing for prefill KV writes (no-op here)')
    parser.add_argument('--part', type=str,
                       choices=['1', '2', '2_prefill', '3', 'all', 'monolithic', 'monolithic_prefill'],
                       default='all',
                       help='Convert specific part (1=embeddings, 2=FFN, 2_prefill=FFN prefill mode, 3=lm_head, monolithic=single file with embed+FFN+lmhead)')
    parser.add_argument('--output', type=str, default='.',
                      help='Output directory for converted models (default: current directory)')
    parser.add_argument(
        '--argmax',
        action='store_true',
        help='Compute argmax inside LM head for part 3 / monolithic inference',
    )
    parser.add_argument(
        '--lut-embeddings',
        type=str,
        default=None,
        help="Override LUT for embeddings in monolithic models. Same format as --lut. "
             "If not set, uses --lut value.",
    )
    parser.add_argument(
        '--lut-lmhead',
        type=str,
        default=None,
        help="Override LUT for LM head in monolithic models. Same format as --lut. "
             "If not set, uses --lut value.",
    )

    return parser.parse_args()

def test_conversion(model_path=None, output_path=None, context_length=512, lut_bits=4,
                   model=None, skip_load_weights=False, split_part='123',
                   batch_size=64, num_chunks=1, prefix='llama', output_dir='.',
                   per_channel=8, argmax_in_model: bool = False,
                   lut_embeddings_bits=None, lut_embeddings_per_channel=8,
                   lut_lmhead_bits=None, lut_lmhead_per_channel=8):
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
    
    # Create converter with batch_size and per_channel
    converter = LlamaConverter(
        model=model,
        context_length=context_length,
        lut_bits=lut_bits,
        batch_size=batch_size,
        num_chunks=num_chunks,
        per_channel=per_channel,
        argmax_in_model=argmax_in_model,
        lut_embeddings_bits=lut_embeddings_bits,
        lut_embeddings_per_channel=lut_embeddings_per_channel,
        lut_lmhead_bits=lut_lmhead_bits,
        lut_lmhead_per_channel=lut_lmhead_per_channel,
    )

    vocab_size_meta = int(getattr(model.config, "vocab_size", 0)) if model is not None else None
    lm_head_chunk_sizes_meta = None
    if hasattr(model, "lm_head8_1"):
        lm_head_chunk_sizes_meta = [
            int(getattr(model, f"lm_head8_{i}").out_channels) for i in range(1, 9)
        ]
    elif hasattr(model, "lm_head2_1"):
        lm_head_chunk_sizes_meta = [
            int(model.lm_head2_1.out_channels),
            int(model.lm_head2_2.out_channels),
        ]
    elif hasattr(model, "lm_head1"):
        lm_head_chunk_sizes_meta = [int(model.lm_head1.out_channels)]
    elif hasattr(model, "lm_head"):
        lm_head_chunk_sizes_meta = [int(model.lm_head.out_features)]
    
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
                    'split_part': split_part,
                    'argmax_in_model': argmax_in_model if split_part in ['3', 'monolithic'] else None,
                    'vocab_size': vocab_size_meta if split_part in ['3', 'monolithic'] else None,
                    'lm_head_chunk_sizes': lm_head_chunk_sizes_meta if split_part in ['3', 'monolithic'] else None,
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
    elif split_part in ['monolithic', 'monolithic_prefill']:
        # Handle monolithic model conversion
        base_name = f'{prefix}_monolithic'
        if split_part == 'monolithic_prefill':
            base_name += '_prefill'
        if lut_bits is not None:
            base_name += f'_lut{lut_bits}'
        output_path = f"{base_name}.mlpackage"

        print(f"\nConverting monolithic model: {split_part} output_path: {output_path}")
        converted_model = converter.convert(split_part=split_part)

        # Add metadata and save
        AddMetadata(converted_model, {
            'context_length': context_length,
            'batch_size': batch_size if split_part == 'monolithic_prefill' else None,
            'lut_bits': lut_bits,
            'split_part': split_part,
            'argmax_in_model': argmax_in_model if split_part in ['monolithic'] else None,
            'vocab_size': vocab_size_meta if split_part in ['3', 'monolithic'] else None,
            'lm_head_chunk_sizes': lm_head_chunk_sizes_meta if split_part in ['3', 'monolithic'] else None,
        })
        print(f"Saving monolithic model to {output_path}")
        output_path = os.path.join(output_dir, output_path)
        converted_model.save(output_path)

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
                        'split_part': split_part,
                        'argmax_in_model': argmax_in_model if split_part in ['3', 'monolithic'] else None,
                        'vocab_size': vocab_size_meta if split_part in ['3', 'monolithic'] else None,
                        'lm_head_chunk_sizes': lm_head_chunk_sizes_meta if split_part in ['3', 'monolithic'] else None,
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
                    'split_part': split_part,
                    'argmax_in_model': argmax_in_model if split_part in ['3', 'monolithic'] else None,
                    'vocab_size': vocab_size_meta if split_part in ['3', 'monolithic'] else None,
                    'lm_head_chunk_sizes': lm_head_chunk_sizes_meta if split_part in ['3', 'monolithic'] else None,
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

    # Parse LUT argument
    lut_bits, per_channel = parse_lut_arg(args.lut)

    # Parse per-component LUT overrides
    lut_embeddings_bits, lut_embeddings_per_channel = parse_lut_arg(args.lut_embeddings)
    lut_lmhead_bits, lut_lmhead_per_channel = parse_lut_arg(args.lut_lmhead)

    # Set model path
    model_path = args.model if args.model else "../Meta-Llama-3.2-1B"

    print(f"\nConverting model from: {model_path}")
    print(f"Output filename prefix: {args.prefix}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    if lut_bits:
        print(f"LUT quantization: {lut_bits} bits, per_channel group size: {per_channel}")
    if lut_embeddings_bits is not None:
        print(f"LUT embeddings override: {lut_embeddings_bits} bits, per_channel={lut_embeddings_per_channel}")
    if lut_lmhead_bits is not None:
        print(f"LUT lm_head override: {lut_lmhead_bits} bits, per_channel={lut_lmhead_per_channel}")
    if args.chunk:
        print(f"Splitting into {args.chunk} chunks")
    if args.argmax:
        print("Argmax in model: enabled")
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
            lut_bits=lut_bits,
            batch_size=args.batch_size,
            num_chunks=args.chunk,
            output_dir=args.output,
            per_channel=per_channel,
            argmax_in_model=args.argmax,
            lut_embeddings_bits=lut_embeddings_bits,
            lut_embeddings_per_channel=lut_embeddings_per_channel,
            lut_lmhead_bits=lut_lmhead_bits,
            lut_lmhead_per_channel=lut_lmhead_per_channel,
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
