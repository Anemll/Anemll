"""Converter for Qwen 3 models.

This module provides a lightweight converter that mirrors the
:class:`LlamaConverter` behaviour for Qwen models without inheriting from
it. Only the pieces required for the unit tests are implemented."""

from __future__ import annotations

import argparse
import os
from typing import Optional, List

import numpy as np
import torch
import coremltools as ct
import coremltools.optimize as cto

from .environment import require_coreml

from .base_converter import BaseConverter
from .metadata import AddMetadata, ModelPart
from ..models.qwen_model import (
    QwenForCausalLM,
    QwenConfig,
    MODEL_DTYPE,
    TEST_DEVICE,
    CONTEXT_LENGTH,
)


class QwenConverter(BaseConverter):
    """Handle conversion of Qwen 3 models to Core ML."""

    model_cls = QwenForCausalLM

    def __init__(
        self,
        model: QwenForCausalLM,
        context_length: int = CONTEXT_LENGTH,
        batch_size: int = 64,
        lut_bits: int | None = 4,
        per_channel: int = 8,
        num_chunks: int = 1,
    ) -> None:
        super().__init__(model)
        self.context_length = context_length
        self.batch_size = batch_size
        self.lut_bits = lut_bits
        self.per_channel = per_channel
        self.head_dim = (
            model.model.config.hidden_size // model.model.config.num_attention_heads
        )
        self.converted_model = None
        self.num_chunks = num_chunks

    @staticmethod
    def GetTransformerStates(model, part=None, prefix="model.model."):
        """Get the transformer states for CoreML conversion"""
        head_dim = getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        )
        num_layers = (
            model.config.num_hidden_layers
        )  # Get total number of layers from config

        # For unified cache
        num_layers_this_part = num_layers * 2
        print(
            f"GetTransformerStates part={part} num_layers_this_part={num_layers_this_part} model.config.num_hidden_layers={model.config.num_hidden_layers}"
        )
        print(f"Using head_dim={head_dim} from config")

        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(
                        num_layers_this_part,
                        model.config.num_key_value_heads,
                        model.config.state_length,
                        head_dim,
                    ),
                    dtype=np.float16,
                ),
                name=f"{prefix}kv_cache_0",  # Only one group for unified cache
            )
        ]
        return states

    def postprocess(self, num_workers=None):
        """Apply LUT quantization if configured.

        Args:
            num_workers: Optional number of workers for parallel processing.
                        If None, uses default single worker.
        """
        if self.converted_model is not None and self.lut_bits is not None:
            print(
                f"Applying LUT quantization with {self.lut_bits} bits and {self.per_channel} channels per group using {num_workers if num_workers else 1} worker(s)..."
            )
            try:
                # Set up quantization config
                config = cto.coreml.OptimizationConfig(
                    global_config=cto.coreml.OpPalettizerConfig(
                        mode="kmeans",
                        nbits=self.lut_bits,
                        granularity="per_grouped_channel",
                        group_size=self.per_channel,
                        num_kmeans_workers=(
                            num_workers if num_workers is not None else 1
                        ),  # Use provided workers or default to 1
                    ),
                )

                # Apply quantization
                self.converted_model = cto.coreml.palettize_weights(
                    self.converted_model, config
                )
                print("✅ LUT quantization completed successfully")

            except Exception as e:
                print(f"❌ LUT quantization failed: {str(e)}")
                print("Continuing without quantization...")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(
        self, part: str = "full"
    ) -> ct.models.MLModel | List[ct.models.MLModel]:
        """Convert the wrapped model to CoreML format.

        Args:
            part: Which part of the model to convert:
                 "full" - complete model (default)
                 "prefill" - prefill mode for initial sequence processing
                 "embeddings" - embeddings only (input_ids -> hidden_states)

        Returns:
            ct.models.MLModel: Converted model
        """
        print(f"QwenConverter.convert() called with part={part}")
        require_coreml()
        print("Calling preprocess()...")
        self.preprocess()

        if part in ("full", "all", "123"):
            print("Converting full model...")
            mlmodel = self.convert_to_coreml(self.model)
        elif part in ("embeddings", "1"):
            print("Converting embeddings...")
            mlmodel = self.convert_part_1(self.model)
        elif part in ("prefill", "2_prefill"):
            print("Converting prefill...")
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2_prefill(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2_prefill(self.model)
        elif part == "2":
            print("Converting FFN...")
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2(self.model)
        elif part == "3":
            print("Converting LM head...")
            mlmodel = self.convert_part_3(self.model)
        else:
            raise ValueError(f"Unsupported part: {part}")

        print("Calling postprocess()...")
        self.postprocess()
        print("QwenConverter.convert() completed")
        return mlmodel

    def convert_to_coreml(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert the entire model to CoreML."""
        require_coreml()
        print("Creating wrapper model...")

        class Wrapper(torch.nn.Module):
            def __init__(self, model: QwenForCausalLM, context_length: int) -> None:
                super().__init__()
                self.model = model
                self.context_length = context_length

            def forward(
                self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                causal_mask: torch.Tensor,
                current_pos: torch.Tensor,
                update_mask: torch.Tensor,
            ) -> torch.Tensor:
                # Fixed window approach: return full logits, extract position on Python side
                return self.model(
                    input_ids=input_ids,
                    update_mask=update_mask,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    IN_PREFILL=False,
                )

        wrapper = Wrapper(model, self.context_length)
        wrapper.eval()
        print("Wrapper model created and set to eval mode")

        print("Preparing model inputs for tracing...")
        # Use single token approach for KV cache compatibility
        sample_input_ids = torch.zeros(
            (1, 1), dtype=torch.int32, device=TEST_DEVICE
        )  # [1, 1] - single token
        sample_position_ids = torch.zeros(
            (1,), dtype=torch.int32, device=TEST_DEVICE
        )  # [1] - single position
        sample_causal_mask = torch.zeros(
            (1, 1, 1, self.context_length), dtype=torch.float16, device=TEST_DEVICE
        )  # [1, 1, 1, context_length]
        sample_current_pos = torch.zeros(
            (1,), dtype=torch.int32, device=TEST_DEVICE
        )  # [1] - current position
        sample_update_mask = torch.zeros(
            (1, 1, self.context_length, 1), dtype=torch.float16, device=TEST_DEVICE
        )  # [1, 1, context_length, 1]
        print("Sample inputs created (Single Token)")
        print(f"sample_input_ids shape: {sample_input_ids.shape}")
        print(f"sample_position_ids shape: {sample_position_ids.shape}")
        print(f"sample_causal_mask shape: {sample_causal_mask.shape}")
        print(f"sample_current_pos shape: {sample_current_pos.shape}")
        print(f"sample_update_mask shape: {sample_update_mask.shape}")

        print("Starting torch.jit.trace...")
        traced = torch.jit.trace(
            wrapper,
            (
                sample_input_ids,
                sample_position_ids,
                sample_causal_mask,
                sample_current_pos,
                sample_update_mask,
            ),
        )
        print("torch.jit.trace completed!")

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
                ct.TensorType(
                    name="update_mask", shape=sample_update_mask.shape, dtype=np.float16
                ),
            ],
            outputs=[
                ct.TensorType(name="logits1", dtype=np.float16),
                ct.TensorType(name="logits2", dtype=np.float16),
                ct.TensorType(name="logits3", dtype=np.float16),
                ct.TensorType(name="logits4", dtype=np.float16),
                ct.TensorType(name="logits5", dtype=np.float16),
                ct.TensorType(name="logits6", dtype=np.float16),
                ct.TensorType(name="logits7", dtype=np.float16),
                ct.TensorType(name="logits8", dtype=np.float16),
                ct.TensorType(name="logits9", dtype=np.float16),
                ct.TensorType(name="logits10", dtype=np.float16),
                ct.TensorType(name="logits11", dtype=np.float16),
                ct.TensorType(name="logits12", dtype=np.float16),
                ct.TensorType(name="logits13", dtype=np.float16),
                ct.TensorType(name="logits14", dtype=np.float16),
                ct.TensorType(name="logits15", dtype=np.float16),
                ct.TensorType(name="logits16", dtype=np.float16),
            ],
            states=self.GetTransformerStates(model, part=None, prefix="model.model."),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        print("CoreML conversion completed!")

        # Apply LUT quantization if specified
        if self.lut_bits:
            self.converted_model = mlmodel  # Set for postprocess
            self.postprocess(num_workers=8)  # Allow passing num_workers if needed
            mlmodel = self.converted_model

        return mlmodel

    # --------------------------------------------------------------
    # Part-based conversion helpers
    # --------------------------------------------------------------
    def convert_part_1(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert embeddings layer only."""
        require_coreml()
        return self.convert_embeddings(model)

    def convert_part_3(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert LM head only."""
        require_coreml()

        class LMHeadWrapper(torch.nn.Module):
            def __init__(self, model: QwenForCausalLM) -> None:
                super().__init__()
                if hasattr(model, "lm_head16_1"):
                    self.heads = [
                        getattr(model, f"lm_head16_{i}") for i in range(1, 17)
                    ]
                    self.mode = "16"
                elif hasattr(model, "lm_head8_1"):
                    self.heads = [getattr(model, f"lm_head8_{i}") for i in range(1, 9)]
                    self.mode = "8"
                elif hasattr(model, "lm_head2_1"):
                    self.heads = [model.lm_head2_1, model.lm_head2_2]
                    self.mode = "2"
                elif hasattr(model, "lm_head1"):
                    self.head = model.lm_head1
                    self.mode = "1"
                else:
                    self.head = model.lm_head
                    self.mode = "linear"

            def forward(self, hidden_states: torch.Tensor):
                if self.mode != "linear":
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)

                if self.mode == "16":
                    return tuple(
                        h(hidden_states).squeeze(2).transpose(1, 2) for h in self.heads
                    )
                if self.mode == "8":
                    return tuple(
                        h(hidden_states).squeeze(2).transpose(1, 2) for h in self.heads
                    )
                if self.mode == "2":
                    logits1 = self.heads[0](hidden_states).squeeze(2).transpose(1, 2)
                    logits2 = self.heads[1](hidden_states).squeeze(2).transpose(1, 2)
                    return logits1, logits2
                if self.mode == "1":
                    return self.head(hidden_states).squeeze(2).transpose(1, 2)
                return self.head(hidden_states)

        wrapper = LMHeadWrapper(model)
        wrapper.eval()
        
        # Ensure no gradients
        for param in wrapper.parameters():
            param.requires_grad = False

        sample_input = torch.zeros(
            (1, 1, model.config.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE
        )
        
        # Trace with no_grad context
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, sample_input)

        if getattr(wrapper, "mode") == "16":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 17)
            ]
        elif getattr(wrapper, "mode") == "8":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 9)
            ]
        elif getattr(wrapper, "mode") == "2":
            outputs = [
                ct.TensorType(name="logits1", dtype=np.float16),
                ct.TensorType(name="logits2", dtype=np.float16),
            ]
        else:
            outputs = [ct.TensorType(name="logits", dtype=np.float16)]

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="hidden_states", shape=sample_input.shape, dtype=np.float16
                )
            ],
            outputs=outputs,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)
            mlmodel = self.converted_model

        return mlmodel

    def convert_part_2(
        self, model: QwenForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        """Convert transformer layers for generation (FFN)."""
        require_coreml()
        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            layers_per_chunk = total_layers // total_chunks
            start_layer = chunk_idx * layers_per_chunk
            end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
        else:
            start_layer = 0
            end_layer = None

        class FFNWrapper(torch.nn.Module):
            def __init__(self, model: QwenForCausalLM) -> None:
                super().__init__()
                self.model = model  # Use QwenForCausalLM as root
                self.states = QwenConverter.GetTransformerStates(
                    model, part="2", prefix="model.model."
                )

            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                rotary = self.model.model.get_rotary_embeddings_s(current_pos)
                out = self.model.model.process_layers(
                    hidden_states,
                    position_ids,
                    causal_mask,
                    current_pos,
                    rotary,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    IN_PREFILL=False,
                )
                out = self.model.model.norm(out)
                return out

        wrapper = FFNWrapper(model)
        wrapper.eval()

        hidden_states = torch.zeros(
            (1, 1, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE
        )
        position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        causal_mask = torch.zeros(
            (1, 1, 1, self.context_length), dtype=torch.float16, device=TEST_DEVICE
        )
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        traced = torch.jit.trace(
            wrapper, (hidden_states, position_ids, causal_mask, current_pos)
        )

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="hidden_states", shape=hidden_states.shape, dtype=np.float16
                ),
                ct.TensorType(
                    name="position_ids", shape=position_ids.shape, dtype=np.int32
                ),
                ct.TensorType(
                    name="causal_mask", shape=causal_mask.shape, dtype=np.float16
                ),
                ct.TensorType(
                    name="current_pos", shape=current_pos.shape, dtype=np.int32
                ),
            ],
            outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
            states=self.GetTransformerStates(model, part=None, prefix="model.model."),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        if self.lut_bits:
            self.converted_model = mlmodel
            # WORKAROUND: CoreMLTools has a known bug where LUT quantization fails with multiple workers
            # when processing chunked models. The second chunk quantization fails with "Pool not running".
            # Setting workers to None (single-threaded) avoids this issue.
            # TODO: File bug report with Apple CoreMLTools team about multi-worker quantization failure on chunked models
            num_workers = None if total_chunks > 1 else 8
            self.postprocess(num_workers=num_workers)
            mlmodel = self.converted_model

        return mlmodel

    def convert_part_2_prefill(
        self, model: QwenForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        """Convert transformer layers for prefill mode."""
        require_coreml()
        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            layers_per_chunk = total_layers // total_chunks
            start_layer = chunk_idx * layers_per_chunk
            end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
        else:
            start_layer = 0
            end_layer = None

        class PrefillWrapper(torch.nn.Module):
            def __init__(self, model: QwenForCausalLM, start_layer=0, end_layer=None):
                super().__init__()
                self.model = model  # Use QwenForCausalLM as root
                self.start_layer = start_layer
                self.end_layer = end_layer
                self.states = QwenConverter.GetTransformerStates(
                    model, part="2_prefill", prefix="model.model."
                )

            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                rotary = self.model.model.get_rotary_embedding_prefill(position_ids)
                out = self.model.model.process_layers(
                    hidden_states,
                    position_ids,
                    causal_mask,
                    current_pos,
                    rotary,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    IN_PREFILL=True,
                )

                # Skip normalization for prefill - data not used, only KV cache is updated!
                # This follows the LLAMA pattern and avoids unnecessary computation
                if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                    print("Skipping final normalization for prefill, data not used!")
                    # Return only first token to minimize memory usage
                    return out[:, 0:1, :]
                
                return out

        wrapper = PrefillWrapper(model, start_layer, end_layer)
        wrapper.eval()

        # Check if this is the last chunk in a multi-chunk model
        is_last_chunk = (chunk_idx == total_chunks - 1)
        
        hidden_states = torch.zeros(
            (1, self.batch_size, model.config.hidden_size),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )
        position_ids = torch.zeros(
            (self.batch_size,), dtype=torch.int32, device=TEST_DEVICE
        )
        causal_mask = torch.zeros(
            (1, 1, self.batch_size, self.context_length),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        traced = torch.jit.trace(
            wrapper, (hidden_states, position_ids, causal_mask, current_pos)
        )

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="hidden_states", shape=hidden_states.shape, dtype=np.float16
                ),
                ct.TensorType(
                    name="position_ids", shape=position_ids.shape, dtype=np.int32
                ),
                ct.TensorType(
                    name="causal_mask", shape=causal_mask.shape, dtype=np.float16
                ),
                ct.TensorType(
                    name="current_pos", shape=current_pos.shape, dtype=np.int32
                ),
            ],
            outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
            states=wrapper.states,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        if self.lut_bits:
            self.converted_model = mlmodel
            # WORKAROUND: CoreMLTools has a known bug where LUT quantization fails with multiple workers
            # when processing chunked models. The second chunk quantization fails with "Pool not running".
            # Setting workers to None (single-threaded) avoids this issue.
            # TODO: File bug report with Apple CoreMLTools team about multi-worker quantization failure on chunked models
            num_workers = None if total_chunks > 1 else 8
            self.postprocess(num_workers=num_workers)
            mlmodel = self.converted_model

        return mlmodel

    def convert_prefill(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert Qwen model to CoreML format for prefill mode.

        Args:
            model: The Qwen model to convert

        Returns:
            ct.models.MLModel: Converted model for prefill processing
        """
        require_coreml()
        print("Converting Qwen model for prefill mode...")

        class PrefillWrapper(torch.nn.Module):
            def __init__(
                self, model: QwenForCausalLM, context_length: int, batch_size: int
            ) -> None:
                super().__init__()
                self.model = model
                self.context_length = context_length
                self.batch_size = batch_size

            def forward(
                self,
                hidden_states: torch.Tensor,
                position_ids: torch.Tensor,
                causal_mask: torch.Tensor,
                current_pos: torch.Tensor,
            ) -> torch.Tensor:
                # Prefill mode: only process transformer layers, skip embeddings and LM head
                # This updates KV cache state without generating logits
                return self.model.forward_prefill(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                )

        wrapper = PrefillWrapper(model, self.context_length, self.batch_size)
        wrapper.eval()
        print("Prefill wrapper model created and set to eval mode")

        print("Preparing prefill model inputs for tracing...")
        # Use batch_size for prefill mode (multiple tokens at once)
        # Input is hidden_states instead of input_ids (skip embeddings)
        sample_hidden_states = torch.zeros(
            (1, self.batch_size, model.config.hidden_size),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )  # [1, batch_size, hidden_size]
        sample_position_ids = torch.zeros(
            (self.batch_size,), dtype=torch.int32, device=TEST_DEVICE
        )  # [batch_size]
        sample_causal_mask = torch.zeros(
            (1, 1, self.batch_size, self.context_length),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )  # [1, 1, batch_size, context_length]
        sample_current_pos = torch.zeros(
            (1,), dtype=torch.int32, device=TEST_DEVICE
        )  # [1] - current position

        print("Prefill sample inputs created")
        print(f"sample_hidden_states shape: {sample_hidden_states.shape}")
        print(f"sample_position_ids shape: {sample_position_ids.shape}")
        print(f"sample_causal_mask shape: {sample_causal_mask.shape}")
        print(f"sample_current_pos shape: {sample_current_pos.shape}")

        print("Starting torch.jit.trace for prefill...")
        traced = torch.jit.trace(
            wrapper,
            (
                sample_hidden_states,
                sample_position_ids,
                sample_causal_mask,
                sample_current_pos,
            ),
        )
        print("torch.jit.trace for prefill completed!")

        print("Starting CoreML conversion for prefill...")
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=sample_hidden_states.shape,
                    dtype=np.float16,
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
            outputs=[
                ct.TensorType(
                    name="output_hidden_states", dtype=np.float16
                ),  # Only output hidden states, no logits
            ],
            states=self.GetTransformerStates(model, part=None, prefix="model.model."),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        print("CoreML conversion for prefill completed!")

        # Apply LUT quantization if specified
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)  # Allow passing num_workers if needed
            mlmodel = self.converted_model

        return mlmodel

    def convert_embeddings(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert embeddings layer to CoreML format.

        Args:
            model: The Qwen model containing embeddings

        Returns:
            ct.models.MLModel: Converted CoreML model for embeddings
        """
        require_coreml()
        print("\nConverting Qwen embeddings layer...")

        class EmbeddingsWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.embed_tokens = model.model.embed_tokens

            def forward(self, input_ids):
                hidden_states = self.embed_tokens(input_ids)
                return hidden_states.to(MODEL_DTYPE)

        # Create wrapper and ensure eval mode
        wrapper = EmbeddingsWrapper(model)
        wrapper.eval()

        # Create sample input for tracing
        sample_input = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)

        # Trace model
        print("Tracing embeddings model...")
        traced_model = torch.jit.trace(wrapper, sample_input)

        # Define flexible input shapes for both single token and batch processing
        input_shape = ct.EnumeratedShapes(
            shapes=[
                [1, 1],
                [1, self.batch_size],
            ],  # Support single token and batch_size tokens
            default=[1, 1],  # Use single token as default
        )

        print(f"Converting embeddings model with input shape: {input_shape}")

        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=input_shape,  # Use enumerated shapes for flexibility
                    dtype=np.int32,
                )
            ],
            outputs=[ct.TensorType(name="hidden_states", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        print("Embeddings conversion completed")

        # Apply LUT quantization if specified
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)  # Allow passing num_workers if needed
            mlmodel = self.converted_model

        return mlmodel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the converter."""

    parser = argparse.ArgumentParser(description="Convert Qwen model to CoreML format")

    parser.add_argument(
        "--model",
        type=str,
        help="Path to model directory (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="qwen",
        help="Prefix for output filenames",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prefill",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH,
        help="Maximum context length",
    )
    parser.add_argument(
        "--lut",
        type=int,
        default=None,
        help="Use LUT quantization with N bits",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Split FFN/prefill into N chunks",
    )
    parser.add_argument(
        "--part",
        type=str,
        choices=["1", "2", "2_prefill", "3", "all", "full", "prefill", "embeddings"],
        default="all",
        help="Model part to convert",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory for converted models",
    )

    return parser.parse_args()


def test_conversion(
    model: Optional[QwenForCausalLM] = None,
    model_path: Optional[str] = None,
    prefix: str = "qwen",
    context_length: int = CONTEXT_LENGTH,
    lut_bits: Optional[int] = None,
    batch_size: int = 64,
    output_dir: str = ".",
    part: str = "full",
    num_chunks: int = 1,
) -> ct.models.MLModel | List[ct.models.MLModel]:
    """Convert a Qwen model and save the result.

    Args:
        model: Pre-loaded Qwen model (optional)
        model_path: Path to model directory
        prefix: Model name prefix
        context_length: Context length for conversion
        lut_bits: LUT quantization bits
        batch_size: Batch size for conversion
        output_dir: Output directory
        part: Part to convert ("full" or "prefill")
    """
    print(
        f"test_conversion called with model_path={model_path}, prefix={prefix}, part={part}"
    )

    if model is None:
        if model_path is None:
            raise ValueError("model_path must be provided if model is None")

        config_path = os.path.join(model_path, "config.json")
        print(f"Looking for config at: {config_path}")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")

        print("Loading config...")
        config = QwenConfig.from_json(config_path)
        print(
            f"Config loaded: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}"
        )

        # Update config to match conversion parameters
        config.context_length = context_length
        config.state_length = max(
            config.state_length, context_length
        )  # Ensure state_length is at least context_length
        print(
            f"Updated config: context_length={config.context_length}, state_length={config.state_length}"
        )

        print("Creating model...")
        model = QwenForCausalLM(config, enable_coreml=True)
        print("Loading pretrained weights...")
        model.load_pretrained_weights(model_path)
        print("Model loaded successfully!")
        
        # Ensure model is in eval mode and gradients are disabled
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("Model set to eval mode and gradients disabled")

    print("Creating converter...")
    converter = QwenConverter(
        model=model,
        context_length=context_length,
        batch_size=batch_size,
        lut_bits=lut_bits,
        num_chunks=num_chunks,
    )

    print("Starting conversion...")
    mlmodel = converter.convert(part=part)
    print("Conversion completed!")

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(mlmodel, list):
        models = mlmodel
    else:
        models = [mlmodel]

    for i, m in enumerate(models):
        AddMetadata(
            m,
            {
                "context_length": context_length,
                "batch_size": batch_size if part in ["2_prefill", "prefill"] else None,
                "lut_bits": lut_bits,
                "num_chunks": num_chunks if part in ["2", "2_prefill"] else None,
                "chunk_no": i + 1 if part in ["2", "2_prefill"] else None,
                "split_part": (
                    ModelPart.FULL.value if part in ["full", "all", "123"] else part
                ),
            },
        )
        fname = f"{prefix}"
        if part in ["1", "embeddings"]:
            fname += "_embeddings"
        elif part in ["3"]:
            fname += "_lm_head"
        elif part in ["2", "2_prefill"]:
            base = "FFN" if part == "2" else "prefill"
            fname += f"_{base}"
            if lut_bits is not None:
                fname += f"_lut{lut_bits}"
            fname += f"_chunk_{i+1:02d}of{num_chunks:02d}"
        if part in ["full", "all", "123"]:
            fname += ""
        if part not in ["2", "2_prefill"]:
            if lut_bits is not None and part != "1":
                fname += f"_lut{lut_bits}"
            fname += ".mlpackage"
        else:
            fname += ".mlpackage"
        out_path = os.path.join(output_dir, fname)
        print(f"Saving model to: {out_path}")
        m.save(out_path)

    return mlmodel


def main() -> None:
    print("Starting qwen_converter main()...")
    args = parse_args()
    print(f"Parsed args: {args}")

    model_path = args.model if args.model else "Qwen/Qwen3-0.6B"

    print(f"\nConverting model from: {model_path}")
    print(f"Output filename prefix: {args.prefix}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    if args.lut:
        print(f"LUT quantization: {args.lut} bits")
    if args.chunk:
        print(f"Splitting into {args.chunk} chunks")
    print(f"Converting part(s): {args.part}")

    # Map legacy part names to numeric equivalents
    part_map = {"full": "all", "embeddings": "1", "prefill": "2_prefill"}
    part = part_map.get(args.part, args.part)

    try:
        print("\nCalling test_conversion()...")
        result = test_conversion(
            model_path=model_path,
            prefix=args.prefix,
            context_length=args.context_length,
            lut_bits=args.lut,
            batch_size=args.batch_size,
            output_dir=args.output,
            part=part,
            num_chunks=args.chunk or 1,
        )
        print(f"Conversion completed successfully! Result: {type(result)}")
    except Exception as e:  # pragma: no cover - CLI tool
        print(f"\nError during conversion: {str(e)}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
