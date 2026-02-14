"""Converter for Gemma3 models.

This module provides a lightweight converter that mirrors the
:class:`LlamaConverter` behaviour for Gemma3 models without inheriting from
it. Supports Gemma3 architecture with its unique features:
- Interleaved sliding window (512) and full attention at layers 6, 12, 18
- Dual RoPE bases (1e6 for global, 10k for local layers)
- Per-head Q/K normalization
- Large vocabulary (262,144 tokens) with 16-way LM head splitting
- GEGLU activation (GELU with tanh approximation)
"""

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
from ..models.gemma3_model import (
    Gemma3ForCausalLM,
    Gemma3Config,
    MODEL_DTYPE,
    TEST_DEVICE,
    CONTEXT_LENGTH,
    ENABLE_SPLIT_CACHE,
)


class Gemma3Converter(BaseConverter):
    """Handle conversion of Gemma3 270M models to Core ML."""

    model_cls = Gemma3ForCausalLM

    def __init__(
        self,
        model: Gemma3ForCausalLM,
        context_length: int = CONTEXT_LENGTH,
        batch_size: int = 64,
        lut_bits: int | None = 4,
        per_channel: int = 8,
        num_chunks: int = 1,
        argmax_in_model: bool = False,
        attention_size: int | None = None,
        fp16_scaled: bool = False,
        prefill_dynamic_slice: bool = True,
        lut_embeddings_bits: int | None = None,
        lut_embeddings_per_channel: int = 8,
        lut_lmhead_bits: int | None = None,
        lut_lmhead_per_channel: int = 8,
    ) -> None:
        super().__init__(model)
        self.context_length = context_length
        self.batch_size = batch_size
        self.lut_bits = lut_bits
        self.per_channel = per_channel
        # Per-component LUT overrides for monolithic models.
        # When set, these override lut_bits for the respective component.
        self.lut_embeddings_bits = lut_embeddings_bits
        self.lut_embeddings_per_channel = lut_embeddings_per_channel
        self.lut_lmhead_bits = lut_lmhead_bits
        self.lut_lmhead_per_channel = lut_lmhead_per_channel
        self.head_dim = (
            model.model.config.hidden_size // model.model.config.num_attention_heads
        )
        self.converted_model = None
        self.num_chunks = num_chunks
        self.argmax_in_model = argmax_in_model
        self.fp16_scaled = fp16_scaled  # If True, exclude scaled tensors from LUT quantization
        self.prefill_dynamic_slice = prefill_dynamic_slice
        # Propagate to model config for prefill KV write behavior
        setattr(self.model.model.config, "prefill_dynamic_slice", prefill_dynamic_slice)
        # attention_size controls the attention computation span
        # For Gemma3, must be at least sliding_window for local attention to work
        if attention_size is not None:
            self.attention_size = attention_size
        else:
            # Default to max of context_length and sliding_window
            self.attention_size = max(context_length, model.model.config.sliding_window)

    def load_weights_from_hf(self, hf_model_path: str) -> bool:
        """Load weights from Hugging Face model and transform them for ANEMLL.

        Args:
            hf_model_path: Path to Hugging Face model or model name

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading weights from Hugging Face model: {hf_model_path}")

            # Load HF model in float32 first to avoid bfloat16->float16 overflow
            # Gemma3 was trained in bfloat16 which has a larger dynamic range than float16
            from transformers import AutoModelForCausalLM
            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_path,
                torch_dtype=torch.float32,  # Load in float32 to preserve precision
                device_map='cpu',
                trust_remote_code=True
            )
            hf_state_dict = hf_model.state_dict()
            
            print(f"Loaded {len(hf_state_dict)} weights from HF model")

            # Get ANEMLL state dict
            anemll_state_dict = self.model.state_dict()
            print(f"ANEMLL model has {len(anemll_state_dict)} weights")

            # Helper function to safely convert weights to float16 with clamping
            def safe_to_fp16(tensor: torch.Tensor, name: str = "") -> torch.Tensor:
                """Convert tensor to float16 with clamping to avoid Inf/NaN.

                Bfloat16 has range ±3.4e38 while float16 has range ±65504.
                Values outside float16 range must be clamped to avoid overflow.
                """
                # First check for NaN/Inf in source tensor
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    nan_count = torch.isnan(tensor).sum().item()
                    inf_count = torch.isinf(tensor).sum().item()
                    print(f"  ⚠️  Warning: {name} has {nan_count} NaN and {inf_count} Inf values, replacing with 0")
                    tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor),
                                        torch.zeros_like(tensor), tensor)

                # Clamp to float16 safe range before conversion
                FP16_MAX = 65504.0
                tensor = tensor.clamp(-FP16_MAX, FP16_MAX)

                return tensor.to(dtype=torch.float16, device="cpu")

            # Track loading statistics
            loaded_count = 0
            skipped_count = 0
            transformed_count = 0

            # Direct mappings (no shape transformation needed)
            direct_mappings = [
                'model.embed_tokens.weight',
                'model.norm.weight',
            ]
            
            # Layer-specific mappings
            for layer_idx in range(self.model.config.num_hidden_layers):
                direct_mappings.extend([
                    f'model.layers.{layer_idx}.input_layernorm.weight',
                    f'model.layers.{layer_idx}.post_attention_layernorm.weight',
                    f'model.layers.{layer_idx}.pre_feedforward_layernorm.weight',
                    f'model.layers.{layer_idx}.post_feedforward_layernorm.weight',
                    f'model.layers.{layer_idx}.self_attn.q_norm.weight',
                    f'model.layers.{layer_idx}.self_attn.k_norm.weight',
                ])
            
            # Load direct mappings
            for hf_key in direct_mappings:
                if hf_key in hf_state_dict and hf_key in anemll_state_dict:
                    # Convert to correct dtype and device with safe clamping
                    hf_weight = hf_state_dict[hf_key]
                    anemll_weight = safe_to_fp16(hf_weight.clone(), hf_key)
                    anemll_state_dict[hf_key] = anemll_weight
                    loaded_count += 1
                    print(f"  ✅ Direct copy: {hf_key}")
                elif hf_key in hf_state_dict:
                    print(f"  ⚠️  HF key not in ANEMLL: {hf_key}")
                    skipped_count += 1
                else:
                    print(f"  ⚠️  ANEMLL key not in HF: {hf_key}")
                    skipped_count += 1
            
            # Transform and load attention weights (Linear -> Conv2d)
            attention_mappings = [
                ('q_proj', 'q_proj'),
                ('k_proj', 'k_proj'), 
                ('v_proj', 'v_proj'),
                ('o_proj', 'o_proj'),
            ]
            
            for layer_idx in range(self.model.config.num_hidden_layers):
                for hf_suffix, anemll_suffix in attention_mappings:
                    hf_key = f'model.layers.{layer_idx}.self_attn.{hf_suffix}.weight'
                    anemll_key = f'model.layers.{layer_idx}.self_attn.{anemll_suffix}.weight'

                    if hf_key in hf_state_dict and anemll_key in anemll_state_dict:
                        # Transform from Linear [out, in] to Conv2d [out, in, 1, 1]
                        hf_weight = hf_state_dict[hf_key]
                        # Safe conversion with clamping, then reshape for Conv2d
                        safe_weight = safe_to_fp16(hf_weight.clone(), hf_key)
                        transformed_weight = safe_weight.view(safe_weight.shape[0], safe_weight.shape[1], 1, 1)
                        anemll_state_dict[anemll_key] = transformed_weight
                        transformed_count += 1
                        print(f"  ✅ Transformed attention: {hf_key} -> {anemll_key}")
            
            # Transform and load MLP weights (Linear -> Conv2d)
            mlp_mappings = [
                ('gate_proj', 'gate_proj'),
                ('up_proj', 'up_proj'),
                ('down_proj', 'down_proj'),
            ]
            
            for layer_idx in range(self.model.config.num_hidden_layers):
                for hf_suffix, anemll_suffix in mlp_mappings:
                    hf_key = f'model.layers.{layer_idx}.mlp.{hf_suffix}.weight'
                    anemll_key = f'model.layers.{layer_idx}.mlp.{anemll_suffix}.weight'

                    if hf_key in hf_state_dict and anemll_key in anemll_state_dict:
                        # Transform from Linear [out, in] to Conv2d [out, in, 1, 1]
                        hf_weight = hf_state_dict[hf_key]
                        # Safe conversion with clamping, then reshape for Conv2d
                        safe_weight = safe_to_fp16(hf_weight.clone(), hf_key)
                        transformed_weight = safe_weight.view(safe_weight.shape[0], safe_weight.shape[1], 1, 1)
                        anemll_state_dict[anemll_key] = transformed_weight
                        transformed_count += 1
                        print(f"  ✅ Transformed MLP: {hf_key} -> {anemll_key}")
            
            # Handle LM head splitting
            if 'lm_head.weight' in hf_state_dict:
                hf_lm_head = hf_state_dict['lm_head.weight']  # [262144, 640]
                vocab_size = hf_lm_head.shape[0]
                hidden_size = hf_lm_head.shape[1]
                split_size = vocab_size // 16

                print(f"  📦 Splitting LM head: {hf_lm_head.shape} -> 16 × [{split_size}, {hidden_size}, 1, 1]")

                for i in range(16):
                    start_idx = i * split_size
                    end_idx = start_idx + split_size if i < 15 else vocab_size

                    anemll_key = f'lm_head16_{i+1}.weight'
                    if anemll_key in anemll_state_dict:
                        # Extract slice and transform to Conv2d with safe conversion
                        slice_weight = hf_lm_head[start_idx:end_idx, :]
                        safe_weight = safe_to_fp16(slice_weight.clone(), f"lm_head_split_{i+1}")
                        transformed_weight = safe_weight.view(safe_weight.shape[0], safe_weight.shape[1], 1, 1)
                        anemll_state_dict[anemll_key] = transformed_weight
                        loaded_count += 1
                        print(f"  ✅ Split LM head {i+1}: {slice_weight.shape} -> {transformed_weight.shape}")
            
            # Note: pre_feedforward_layernorm and post_feedforward_layernorm are now implemented
            # and handled in the direct mappings above
            
            # Load the transformed weights into the model
            missing_keys, unexpected_keys = self.model.load_state_dict(anemll_state_dict, strict=False)
            
            # Force dtype conversion for all parameters to match MODEL_DTYPE
            print("  🔄 Converting all parameters to float16...")
            for name, param in self.model.named_parameters():
                if param.dtype != torch.float16:
                    param.data = param.data.to(torch.float16)
                    print(f"    ✅ Converted {name}: {param.dtype}")
            
            # Filter out expected missing keys
            expected_missing = ['kv_cache_0']  # KV cache buffer is initialized separately
            missing_keys = [k for k in missing_keys if k not in expected_missing]
            
            if missing_keys:
                print(f"  ⚠️  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"  ⚠️  Unexpected keys: {unexpected_keys}")
            
            print(f"\n📊 Weight Loading Summary:")
            print(f"  ✅ Direct copies: {loaded_count}")
            print(f"  🔄 Transformations: {transformed_count}")
            print(f"  ⏭️  Skipped: {skipped_count}")
            print(f"  ❌ Missing: {len(missing_keys)}")
            print(f"  ⚠️  Unexpected: {len(unexpected_keys)}")
            
            success = len(missing_keys) == 0
            if success:
                print("✅ Weight loading completed successfully!")
            else:
                print("❌ Weight loading completed with missing keys")
            
            return success
            
        except Exception as e:
            print(f"❌ Error loading weights: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def GetTransformerStates(model, part=None, prefix="model.model.", start_layer=None, end_layer=None):
        """Get the transformer states for CoreML conversion.

        For split cache mode (Gemma3 local/global attention):
        - Returns TWO state tensors: kv_cache_local and kv_cache_global
        - kv_cache_local: For sliding window layers (15 layers), size = sliding_window
        - kv_cache_global: For full attention layers (3 layers), size = state_length

        For unified cache mode (legacy):
        - Returns ONE state tensor: kv_cache_0

        Args:
            model: The model to get states from
            part: Part identifier (legacy, unused)
            prefix: Prefix for state tensor names
            start_layer: Start layer index for chunk-aware states (optional)
            end_layer: End layer index for chunk-aware states (optional)
        """
        head_dim = getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        )

        # Check if split cache is enabled
        use_split_cache = getattr(model.config, 'use_split_cache', True)

        if use_split_cache:
            # Split cache mode for Gemma3 local/global attention
            # Count layers by type (total in model)
            num_global_layers = sum(1 for t in model.config.layer_types
                                   if t == "full_attention")
            num_local_layers = model.config.num_hidden_layers - num_global_layers

            sliding_window = model.config.sliding_window  # 512

            # Log which layer types are in this chunk (for debugging)
            if start_layer is not None:
                if end_layer is None:
                    end_layer = model.config.num_hidden_layers
                chunk_layers = range(start_layer, end_layer)
                chunk_layer_types = [model.config.layer_types[i] for i in chunk_layers]
                has_global = "full_attention" in chunk_layer_types
                has_local = "sliding_attention" in chunk_layer_types
                print(f"GetTransformerStates: Chunk layers {start_layer}-{end_layer-1}")
                print(f"  Has local layers: {has_local}, Has global layers: {has_global}")
                print(f"  (Both caches declared for all chunks - state is shared)")

            print(f"GetTransformerStates: SPLIT CACHE mode")
            print(f"  {num_local_layers} local layers -> kv_cache_local [{2*num_local_layers}, {model.config.num_key_value_heads}, {sliding_window}, {head_dim}]")
            print(f"  {num_global_layers} global layers -> kv_cache_global [{2*num_global_layers}, {model.config.num_key_value_heads}, {model.config.state_length}, {head_dim}]")

            # Always declare BOTH state types for all chunks, even if chunk doesn't
            # have that layer type. This is required because:
            # 1. State is shared across all chunks - they must have same interface
            # 2. Chunks without global layers pass through global cache unchanged
            # 3. CoreML requires consistent state declarations
            states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(
                            2 * num_local_layers,  # K and V for local layers
                            model.config.num_key_value_heads,
                            sliding_window,  # sliding_window positions
                            head_dim,
                        ),
                        dtype=np.float16,
                    ),
                    name=f"{prefix}kv_cache_local",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(
                            2 * num_global_layers,  # K and V for global layers
                            model.config.num_key_value_heads,
                            model.config.state_length,  # Full context length
                            head_dim,
                        ),
                        dtype=np.float16,
                    ),
                    name=f"{prefix}kv_cache_global",
                ),
            ]
        else:
            # Legacy unified cache mode
            num_layers = model.config.num_hidden_layers
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

    @staticmethod
    def _make_palettizer_config(nbits, per_channel, num_workers, weight_threshold):
        """Build an OpPalettizerConfig for the given bit-width / granularity."""
        if per_channel <= 0:
            return cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=nbits,
                granularity="per_tensor",
                num_kmeans_workers=num_workers if num_workers is not None else 1,
                weight_threshold=weight_threshold,
            )
        return cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=nbits,
            granularity="per_grouped_channel",
            group_size=per_channel,
            num_kmeans_workers=num_workers if num_workers is not None else 1,
            weight_threshold=weight_threshold,
        )

    def postprocess(self, num_workers=None):
        """Apply LUT quantization if configured.

        Supports per-component LUT overrides for monolithic models via
        lut_embeddings_bits / lut_lmhead_bits.  When set, embedding and/or
        LM head ops get a different quantization config from the FFN default.

        Args:
            num_workers: Optional number of workers for parallel processing.
                        If None, uses default single worker.
        """
        import warnings

        if self.converted_model is not None and self.lut_bits is not None:
            # Check if using per-tensor quantization (per_channel <= 0 means per-tensor)
            use_per_tensor = self.per_channel <= 0
            if use_per_tensor:
                print(
                    f"Applying LUT quantization with {self.lut_bits} bits using PER-TENSOR granularity with {num_workers if num_workers else 1} worker(s)..."
                )
            else:
                print(
                    f"Applying LUT quantization with {self.lut_bits} bits and {self.per_channel} channels per group using {num_workers if num_workers else 1} worker(s)..."
                )
            try:
                # Suppress sklearn warnings during quantization (common with edge cases)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    # Exclude small weights from LUT quantization (norms, etc.)
                    weight_threshold = 8192  # 8KB - excludes norms (~5KB), keeps projections
                    print(f"  Using weight_threshold={weight_threshold} bytes (excludes norm weights)")

                    # Default (FFN) quantization config
                    global_cfg = self._make_palettizer_config(
                        self.lut_bits, self.per_channel, num_workers, weight_threshold
                    )

                    # Per-component overrides for monolithic models
                    op_name_configs: dict = {}
                    has_overrides = (
                        self.lut_embeddings_bits is not None
                        or self.lut_lmhead_bits is not None
                    )

                    if has_overrides:
                        # Scan ops to find embedding / lm_head weights by name pattern
                        prog = self.converted_model._mil_program  # noqa: SLF001
                        for fn_name in prog.functions:
                            fn = prog.functions[fn_name]
                            for op in fn.operations:
                                op_name = op.name or ""
                                # Embeddings: ops whose name contains 'embed_tokens'
                                if self.lut_embeddings_bits is not None and "embed_tokens" in op_name:
                                    cfg = self._make_palettizer_config(
                                        self.lut_embeddings_bits,
                                        self.lut_embeddings_per_channel,
                                        num_workers,
                                        weight_threshold,
                                    )
                                    op_name_configs[op_name] = cfg
                                # LM head: ops whose name contains 'lm_head'
                                elif self.lut_lmhead_bits is not None and "lm_head" in op_name:
                                    cfg = self._make_palettizer_config(
                                        self.lut_lmhead_bits,
                                        self.lut_lmhead_per_channel,
                                        num_workers,
                                        weight_threshold,
                                    )
                                    op_name_configs[op_name] = cfg

                        if op_name_configs:
                            embed_count = sum(1 for k in op_name_configs if "embed_tokens" in k)
                            lmhead_count = sum(1 for k in op_name_configs if "lm_head" in k)
                            if self.lut_embeddings_bits is not None:
                                print(
                                    f"  Embeddings: {self.lut_embeddings_bits}-bit LUT "
                                    f"(per_channel={self.lut_embeddings_per_channel}) "
                                    f"-> {embed_count} ops"
                                )
                            if self.lut_lmhead_bits is not None:
                                print(
                                    f"  LM head: {self.lut_lmhead_bits}-bit LUT "
                                    f"(per_channel={self.lut_lmhead_per_channel}) "
                                    f"-> {lmhead_count} ops"
                                )

                    config = cto.coreml.OptimizationConfig(
                        global_config=global_cfg,
                        op_name_configs=op_name_configs if op_name_configs else None,
                    )

                    # Apply quantization
                    self.converted_model = cto.coreml.palettize_weights(
                        self.converted_model, config
                    )
                print("✅ LUT quantization completed successfully")

            except Exception as e:
                print(f"❌ LUT quantization failed: {str(e)}")
                print("Continuing without quantization...")

    @staticmethod
    def _reset_kv_cache_buffers(module: torch.nn.Module | torch.jit.ScriptModule) -> None:
        """Clear mutable KV-cache buffers to avoid trace side-effects in state dict checks."""
        with torch.no_grad():
            for name, buffer in module.named_buffers():
                if "kv_cache" in name:
                    buffer.zero_()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(
        self, part: str = "full", chunk_no: int | None = None
    ) -> ct.models.MLModel | List[ct.models.MLModel]:
        """Convert the wrapped model to CoreML format.

        Args:
            part: Which part of the model to convert:
                 "full" - complete model (default)
                 "prefill" - prefill mode for initial sequence processing
                 "embeddings" - embeddings only (input_ids -> hidden_states)
            chunk_no: If set (1-based), convert only this chunk for chunked parts.

        Returns:
            ct.models.MLModel: Converted model
        """
        print(f"Gemma3Converter.convert() called with part={part}")
        require_coreml()
        print("Calling preprocess()...")
        self.preprocess()

        if chunk_no is not None:
            chunk_parts = {"2", "2_rotate", "2_prefill", "2_prefill_rotate", "prefill"}
            if part not in chunk_parts:
                raise ValueError(f"--chunk-no is only valid for chunked parts: {sorted(chunk_parts)}")
            if self.num_chunks <= 1:
                raise ValueError("--chunk-no requires --chunk > 1")
            if chunk_no < 1 or chunk_no > self.num_chunks:
                raise ValueError(f"--chunk-no must be between 1 and {self.num_chunks}")

        if part in ("full", "all", "123"):
            print("Converting full model...")
            mlmodel = self.convert_to_coreml(self.model)
        elif part in ("embeddings", "1"):
            print("Converting embeddings...")
            mlmodel = self.convert_part_1(self.model)
        elif part in ("prefill", "2_prefill"):
            print("Converting prefill (fill mode)...")
            if self.num_chunks > 1:
                if chunk_no is not None:
                    i = chunk_no - 1
                    mlmodel = self.convert_part_2_prefill(self.model, i, self.num_chunks, force_rotation=False)
                else:
                    mlmodel = [
                        self.convert_part_2_prefill(self.model, i, self.num_chunks, force_rotation=False)
                        for i in range(self.num_chunks)
                    ]
            else:
                mlmodel = self.convert_part_2_prefill(self.model, force_rotation=False)
        elif part == "2":
            print("Converting FFN (infer - fill mode)...")
            if self.num_chunks > 1:
                if chunk_no is not None:
                    i = chunk_no - 1
                    mlmodel = self.convert_part_2(self.model, i, self.num_chunks, force_rotation=False)
                else:
                    mlmodel = [
                        self.convert_part_2(self.model, i, self.num_chunks, force_rotation=False)
                        for i in range(self.num_chunks)
                    ]
            else:
                mlmodel = self.convert_part_2(self.model, force_rotation=False)
        elif part == "2_rotate":
            print("Converting FFN (infer_rotate - rotation mode)...")
            if self.num_chunks > 1:
                if chunk_no is not None:
                    i = chunk_no - 1
                    mlmodel = self.convert_part_2(self.model, i, self.num_chunks, force_rotation=True)
                else:
                    mlmodel = [
                        self.convert_part_2(self.model, i, self.num_chunks, force_rotation=True)
                        for i in range(self.num_chunks)
                    ]
            else:
                mlmodel = self.convert_part_2(self.model, force_rotation=True)
        elif part == "2_prefill_rotate":
            print("Converting prefill (prefill_rotate - rotation mode)...")
            if self.num_chunks > 1:
                if chunk_no is not None:
                    i = chunk_no - 1
                    mlmodel = self.convert_part_2_prefill(self.model, i, self.num_chunks, force_rotation=True)
                else:
                    mlmodel = [
                        self.convert_part_2_prefill(self.model, i, self.num_chunks, force_rotation=True)
                        for i in range(self.num_chunks)
                    ]
            else:
                mlmodel = self.convert_part_2_prefill(self.model, force_rotation=True)
        elif part == "3":
            print("Converting LM head...")
            mlmodel = self.convert_part_3(self.model, argmax_in_model=self.argmax_in_model)
        elif part == "monolithic":
            print("Converting monolithic model (infer - fill mode)...")
            mlmodel = self.convert_monolithic(self.model, is_prefill=False, argmax_in_model=self.argmax_in_model, force_rotation=False)
        elif part == "monolithic_rotate":
            print("Converting monolithic model (infer_rotate - rotation mode)...")
            mlmodel = self.convert_monolithic(self.model, is_prefill=False, argmax_in_model=self.argmax_in_model, force_rotation=True)
        elif part == "monolithic_prefill":
            print("Converting monolithic model (prefill)...")
            # Note: prefill should NEVER have argmax - it only fills the KV cache
            mlmodel = self.convert_monolithic(self.model, is_prefill=True, argmax_in_model=False)
        elif part == "monolithic_prefill_rotate":
            print("Converting monolithic model (prefill_rotate - rotation mode)...")
            # Note: prefill should NEVER have argmax - it only fills the KV cache
            mlmodel = self.convert_monolithic(self.model, is_prefill=True, argmax_in_model=False, force_rotation=True)
        else:
            raise ValueError(f"Unsupported part: {part}")

        print("Calling postprocess()...")
        self.postprocess()
        print("Gemma3Converter.convert() completed")
        return mlmodel

    def convert_to_coreml(self, model: Gemma3ForCausalLM) -> ct.models.MLModel:
        """Convert the entire model to CoreML."""
        require_coreml()
        print("Creating wrapper model...")

        class Wrapper(torch.nn.Module):
            def __init__(self, model: Gemma3ForCausalLM, context_length: int) -> None:
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
            (1, 1, 1, self.attention_size), dtype=torch.float16, device=TEST_DEVICE
        )  # [1, 1, 1, attention_size] - smaller window = faster attention
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

        self._reset_kv_cache_buffers(wrapper)
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
        self._reset_kv_cache_buffers(wrapper)
        self._reset_kv_cache_buffers(traced)
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
            # Use single-threaded for full model to avoid multiprocessing issues
            self.postprocess(num_workers=None)
            mlmodel = self.converted_model

        return mlmodel

    def convert_monolithic(
        self, model: Gemma3ForCausalLM, is_prefill: bool = False,
        argmax_in_model: bool = False, force_rotation: bool = None
    ) -> ct.models.MLModel:
        """Convert full model (embeddings + FFN + LM head) to a single CoreML model.

        Args:
            model: The Gemma3 model to convert
            is_prefill: If True, convert for prefill mode (batch processing)
                       If False, convert for inference mode (single token)
            argmax_in_model: If True, compute argmax per LM head chunk inside the model.
                            Outputs argmax_idx[16] and argmax_val[16] instead of 16 logits tensors.
                            This reduces output from 262K values to 32 values (2 tensors).
            force_rotation: Controls local cache update behavior during tracing:
                           None - use conditional (NOT recommended for CoreML, leads to incomplete tracing)
                           False - always use fill mode (for infer function, positions < sliding_window)
                           True - always use rotate mode (for infer_rotate function, positions >= sliding_window)

        Returns:
            ct.models.MLModel: Monolithic CoreML model
        """
        require_coreml()
        if is_prefill and force_rotation:
            mode_str = "prefill_rotate (prefill with cache rotation)"
        elif is_prefill:
            mode_str = "prefill"
        elif force_rotation:
            mode_str = "infer_rotate (cache rotation enabled)"
        else:
            mode_str = "infer (cache fill mode)"
        print(f"\nConverting monolithic model for {mode_str}...")

        # Set force_rotation_mode on model config before tracing
        if force_rotation is not None:
            model.model.config.force_rotation_mode = force_rotation
            print(f"  force_rotation_mode = {force_rotation}")

        # Disable dynamic prefill slicing when tracing rotation variants
        original_prefill_dynamic_slice = getattr(model.model.config, "prefill_dynamic_slice", False)
        effective_dynamic_slice = self.prefill_dynamic_slice and not bool(force_rotation)
        if effective_dynamic_slice != original_prefill_dynamic_slice:
            model.model.config.prefill_dynamic_slice = effective_dynamic_slice

        use_update_mask = is_prefill and not force_rotation and not self.prefill_dynamic_slice

        if is_prefill and not force_rotation:
            if use_update_mask:
                class MonolithicWrapper(torch.nn.Module):
                    """Wrapper combining embeddings + transformer + LM head (prefill with update_mask)."""

                    def __init__(
                        self, model: Gemma3ForCausalLM, context_length: int, is_prefill: bool,
                        argmax_in_model: bool = False, is_prefill_rotate: bool = False
                    ) -> None:
                        super().__init__()
                        self.model = model
                        self.context_length = context_length
                        self.is_prefill = is_prefill
                        self.is_prefill_rotate = is_prefill_rotate
                        self.argmax_in_model = argmax_in_model

                        # Determine LM head mode
                        if hasattr(model, "lm_head16_1"):
                            self.lm_head_mode = "16"
                            self.lm_heads = [
                                getattr(model, f"lm_head16_{i}") for i in range(1, 17)
                            ]
                            self.chunk_size = 16384  # 262144 / 16
                        elif hasattr(model, "lm_head8_1"):
                            self.lm_head_mode = "8"
                            self.lm_heads = [
                                getattr(model, f"lm_head8_{i}") for i in range(1, 9)
                            ]
                            self.chunk_size = 32768  # 262144 / 8
                        elif hasattr(model, "lm_head2_1"):
                            self.lm_head_mode = "2"
                            self.lm_heads = [model.lm_head2_1, model.lm_head2_2]
                            self.chunk_size = 131072  # 262144 / 2
                        elif hasattr(model, "lm_head1"):
                            self.lm_head_mode = "1"
                            self.lm_head = model.lm_head1
                            self.chunk_size = 262144
                        else:
                            self.lm_head_mode = "linear"
                            self.lm_head = model.lm_head
                            self.chunk_size = 262144

                    def forward(
                        self,
                        input_ids: torch.Tensor,
                        position_ids: torch.Tensor,
                        causal_mask: torch.Tensor,
                        current_pos: torch.Tensor,
                        update_mask: torch.Tensor,
                    ) -> tuple:
                        # Step 1: Embeddings (with Gemma3 scaling)
                        hidden_states = self.model.model.embed_tokens(input_ids)
                        hidden_states = hidden_states * self.model.model.embedding_scale
                        hidden_states = hidden_states.to(MODEL_DTYPE)

                        # Step 2: Transformer layers (RoPE handled inside process_layers)
                        hidden_states = self.model.model.process_layers(
                            hidden_states,
                            position_ids,
                            causal_mask,
                            current_pos,
                            start_layer=0,
                            end_layer=None,
                            IN_PREFILL=self.is_prefill,
                            IN_PREFILL_ROTATE=self.is_prefill_rotate,
                            update_mask=update_mask,
                        )

                        # Apply final normalization
                        hidden_states = self.model.model.norm(hidden_states)

                        # Prefill output: return only the first token to minimize output size
                        return hidden_states[:, 0:1, :]
            else:
                class MonolithicWrapper(torch.nn.Module):
                    """Wrapper combining embeddings + transformer + LM head (prefill, dynamic slice)."""

                    def __init__(
                        self, model: Gemma3ForCausalLM, context_length: int, is_prefill: bool,
                        argmax_in_model: bool = False, is_prefill_rotate: bool = False
                    ) -> None:
                        super().__init__()
                        self.model = model
                        self.context_length = context_length
                        self.is_prefill = is_prefill
                        self.is_prefill_rotate = is_prefill_rotate
                        self.argmax_in_model = argmax_in_model

                        # Determine LM head mode
                        if hasattr(model, "lm_head16_1"):
                            self.lm_head_mode = "16"
                            self.lm_heads = [
                                getattr(model, f"lm_head16_{i}") for i in range(1, 17)
                            ]
                            self.chunk_size = 16384  # 262144 / 16
                        elif hasattr(model, "lm_head8_1"):
                            self.lm_head_mode = "8"
                            self.lm_heads = [
                                getattr(model, f"lm_head8_{i}") for i in range(1, 9)
                            ]
                            self.chunk_size = 32768  # 262144 / 8
                        elif hasattr(model, "lm_head2_1"):
                            self.lm_head_mode = "2"
                            self.lm_heads = [model.lm_head2_1, model.lm_head2_2]
                            self.chunk_size = 131072  # 262144 / 2
                        elif hasattr(model, "lm_head1"):
                            self.lm_head_mode = "1"
                            self.lm_head = model.lm_head1
                            self.chunk_size = 262144
                        else:
                            self.lm_head_mode = "linear"
                            self.lm_head = model.lm_head
                            self.chunk_size = 262144

                    def forward(
                        self,
                        input_ids: torch.Tensor,
                        position_ids: torch.Tensor,
                        causal_mask: torch.Tensor,
                        current_pos: torch.Tensor,
                    ) -> tuple:
                        # Step 1: Embeddings (with Gemma3 scaling)
                        hidden_states = self.model.model.embed_tokens(input_ids)
                        hidden_states = hidden_states * self.model.model.embedding_scale
                        hidden_states = hidden_states.to(MODEL_DTYPE)

                        # Step 2: Transformer layers (RoPE handled inside process_layers)
                        hidden_states = self.model.model.process_layers(
                            hidden_states,
                            position_ids,
                            causal_mask,
                            current_pos,
                            start_layer=0,
                            end_layer=None,
                            IN_PREFILL=self.is_prefill,
                            IN_PREFILL_ROTATE=self.is_prefill_rotate,
                        )

                        # Apply final normalization
                        hidden_states = self.model.model.norm(hidden_states)

                        # Prefill output: return only the first token to minimize output size
                        return hidden_states[:, 0:1, :]
        else:
            class MonolithicWrapper(torch.nn.Module):
                """Wrapper combining embeddings + transformer + LM head."""

                def __init__(
                    self, model: Gemma3ForCausalLM, context_length: int, is_prefill: bool,
                    argmax_in_model: bool = False, is_prefill_rotate: bool = False
                ) -> None:
                    super().__init__()
                    self.model = model
                    self.context_length = context_length
                    self.is_prefill = is_prefill
                    self.is_prefill_rotate = is_prefill_rotate
                    self.argmax_in_model = argmax_in_model

                    # Determine LM head mode
                    if hasattr(model, "lm_head16_1"):
                        self.lm_head_mode = "16"
                        self.lm_heads = [
                            getattr(model, f"lm_head16_{i}") for i in range(1, 17)
                        ]
                        self.chunk_size = 16384  # 262144 / 16
                    elif hasattr(model, "lm_head8_1"):
                        self.lm_head_mode = "8"
                        self.lm_heads = [
                            getattr(model, f"lm_head8_{i}") for i in range(1, 9)
                        ]
                        self.chunk_size = 32768  # 262144 / 8
                    elif hasattr(model, "lm_head2_1"):
                        self.lm_head_mode = "2"
                        self.lm_heads = [model.lm_head2_1, model.lm_head2_2]
                        self.chunk_size = 131072  # 262144 / 2
                    elif hasattr(model, "lm_head1"):
                        self.lm_head_mode = "1"
                        self.lm_head = model.lm_head1
                        self.chunk_size = 262144
                    else:
                        self.lm_head_mode = "linear"
                        self.lm_head = model.lm_head
                        self.chunk_size = 262144

                def forward(
                    self,
                    input_ids: torch.Tensor,
                    position_ids: torch.Tensor,
                    causal_mask: torch.Tensor,
                    current_pos: torch.Tensor,
                ) -> tuple:
                    # Step 1: Embeddings (with Gemma3 scaling)
                    hidden_states = self.model.model.embed_tokens(input_ids)
                    hidden_states = hidden_states * self.model.model.embedding_scale
                    hidden_states = hidden_states.to(MODEL_DTYPE)

                    # Step 2: Transformer layers (RoPE handled inside process_layers)
                    hidden_states = self.model.model.process_layers(
                        hidden_states,
                        position_ids,
                        causal_mask,
                        current_pos,
                        start_layer=0,
                        end_layer=None,
                        IN_PREFILL=self.is_prefill,
                        IN_PREFILL_ROTATE=self.is_prefill_rotate,
                    )

                    # Apply final normalization
                    hidden_states = self.model.model.norm(hidden_states)

                    # Prefill output: return only hidden states, skip LM head
                    if self.is_prefill:
                        return hidden_states[:, 0:1, :]

                    # Step 3: LM Head
                    if self.lm_head_mode != "linear":
                        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)

                    # Compute logits for each chunk
                    if self.lm_head_mode in ("16", "8", "2"):
                        logits_list = [
                            h(hidden_states).squeeze(2).transpose(1, 2)
                            for h in self.lm_heads
                        ]
                    elif self.lm_head_mode == "1":
                        logits_list = [self.lm_head(hidden_states).squeeze(2).transpose(1, 2)]
                    else:
                        logits_list = [self.lm_head(hidden_states)]

                    # If argmax_in_model, compute argmax per chunk and return 2 tensors
                    # NOTE: We return LOCAL indices (0 to chunk_size-1), not global indices.
                    # The global offset is computed on the Python/Swift side as:
                    #   global_idx = local_idx + (best_chunk * chunk_size)
                    # This avoids baking constants into the CoreML model.
                    # NOTE: Using int16 for ANE compatibility - ANE doesn't support int32 for argmax.
                    # Local indices (0 to 16383) fit in int16 (max 32767).
                    if self.argmax_in_model:
                        all_idx = []
                        all_val = []
                        for i, logits in enumerate(logits_list):
                            # logits shape: [1, 1, chunk_size] for inference mode
                            # Get argmax index within chunk (0 to chunk_size-1)
                            chunk_argmax = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1, 1], int64
                            # Cast to int32 for CoreML compatibility (int16 not supported for outputs)
                            # Local indices 0-16383 fit easily in int32
                            local_idx = chunk_argmax.to(torch.int32)  # [1, 1, 1]
                            # Get max value
                            max_val = torch.max(logits, dim=-1, keepdim=True).values  # [1, 1, 1]
                            all_idx.append(local_idx)
                            all_val.append(max_val)
                        # Concatenate along last dim: [1, 1, num_chunks], then squeeze to [num_chunks]
                        argmax_idx = torch.cat(all_idx, dim=-1).squeeze(0).squeeze(0)  # [num_chunks], int32 (LOCAL indices)
                        argmax_val = torch.cat(all_val, dim=-1).squeeze(0).squeeze(0)  # [num_chunks], fp16
                        return (argmax_idx, argmax_val)
                    else:
                        return tuple(logits_list)

        # Determine if this is prefill with rotation (prefill_rotate function)
        is_prefill_rotate = is_prefill and force_rotation == True
        wrapper = MonolithicWrapper(model, self.context_length, is_prefill, argmax_in_model, is_prefill_rotate)
        wrapper.eval()

        for param in wrapper.parameters():
            param.requires_grad = False

        argmax_str = ", argmax_in_model=True" if argmax_in_model else ""
        print(f"Monolithic wrapper created (LM head mode: {wrapper.lm_head_mode}{argmax_str})")

        # Determine mask size based on cache mode
        # For split cache: mask must be large enough for BOTH local (sliding_window) and global (state_length) layers
        # For unified cache: mask sized to attention_size
        use_split_cache = getattr(model.model.config, 'use_split_cache', ENABLE_SPLIT_CACHE)
        if use_split_cache:
            # For split cache, mask must cover the larger of sliding_window and state_length
            # Local layers need sliding_window, global layers need state_length
            mask_size = max(model.model.config.sliding_window, model.model.config.state_length)
            print(f"Split cache mode: mask size = {mask_size} (max of sliding_window and state_length)")
            print(f"  Local cache: {model.model.config.sliding_window} (sliding window)")
            print(f"  Global cache: {model.model.config.state_length} (full context)")
        else:
            mask_size = self.attention_size
            print(f"Unified cache mode: mask size = {mask_size}")

        use_update_mask = is_prefill and not force_rotation and not self.prefill_dynamic_slice

        # Prepare inputs based on mode
        if is_prefill:
            sample_input_ids = torch.zeros(
                (1, self.batch_size), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_position_ids = torch.zeros(
                (self.batch_size,), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_causal_mask = torch.zeros(
                (1, 1, self.batch_size, mask_size),
                dtype=torch.float16,
                device=TEST_DEVICE,
            )
            sample_update_mask = None
            if use_update_mask:
                sample_update_mask = torch.zeros(
                    (1, 1, mask_size, self.batch_size),
                    dtype=torch.float16,
                    device=TEST_DEVICE,
                )
        else:
            sample_input_ids = torch.zeros(
                (1, 1), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_position_ids = torch.zeros(
                (1,), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_causal_mask = torch.zeros(
                (1, 1, 1, mask_size),
                dtype=torch.float16,
                device=TEST_DEVICE,
            )
            sample_update_mask = None

        sample_current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        print(f"Sample inputs ({mode_str} mode):")
        print(f"  input_ids: {sample_input_ids.shape}")
        print(f"  position_ids: {sample_position_ids.shape}")
        print(f"  causal_mask: {sample_causal_mask.shape}")
        print(f"  current_pos: {sample_current_pos.shape}")
        if sample_update_mask is not None:
            print(f"  update_mask: {sample_update_mask.shape}")

        print("Tracing monolithic model...")
        self._reset_kv_cache_buffers(wrapper)
        with torch.no_grad():
            if sample_update_mask is not None:
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
            else:
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

        # Determine number of chunks based on LM head mode
        if wrapper.lm_head_mode == "16":
            num_chunks = 16
        elif wrapper.lm_head_mode == "8":
            num_chunks = 8
        elif wrapper.lm_head_mode == "2":
            num_chunks = 2
        else:
            num_chunks = 1

        # Build output specifications
        if is_prefill:
            # Prefill models only update KV cache; output hidden state for API compatibility.
            outputs = [ct.TensorType(name="output_hidden_states", dtype=np.float16)]
        elif argmax_in_model:
            # Output 2 tensors: argmax_idx[num_chunks] and argmax_val[num_chunks]
            # Note: shape is inferred automatically by coremltools
            # Using int32 for CoreML compatibility (int16 is not supported for outputs)
            outputs = [
                ct.TensorType(name="argmax_idx", dtype=np.int32),
                ct.TensorType(name="argmax_val", dtype=np.float16)
            ]
            print(f"Outputs: argmax_idx[{num_chunks}] (int32) + argmax_val[{num_chunks}] (fp16) - reduced from {num_chunks * wrapper.chunk_size} logits")
        else:
            # Original logits outputs
            if num_chunks == 1:
                outputs = [ct.TensorType(name="logits", dtype=np.float16)]
            else:
                outputs = [
                    ct.TensorType(name=f"logits{i}", dtype=np.float16)
                    for i in range(1, num_chunks + 1)
                ]

        print("Starting CoreML conversion...")
        inputs = [
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
        ]
        if sample_update_mask is not None:
            inputs.append(
                ct.TensorType(
                    name="update_mask", shape=sample_update_mask.shape, dtype=np.float16
                )
            )

        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=outputs,
            states=self.GetTransformerStates(model, part=None, prefix="model.model."),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        print(f"CoreML conversion for monolithic {mode_str} completed!")

        if self.lut_bits:
            print(f"Applying LUT quantization ({self.lut_bits} bits)...")
            self.converted_model = mlmodel
            # Use single-threaded quantization for monolithic models to avoid
            # multiprocessing pool hanging issues with large models on macOS
            self.postprocess(num_workers=None)
            mlmodel = self.converted_model

        # Restore original prefill_dynamic_slice
        if getattr(model.model.config, "prefill_dynamic_slice", None) != original_prefill_dynamic_slice:
            model.model.config.prefill_dynamic_slice = original_prefill_dynamic_slice

        return mlmodel

    # --------------------------------------------------------------
    # Part-based conversion helpers
    # --------------------------------------------------------------
    def convert_part_1(self, model: Gemma3ForCausalLM) -> ct.models.MLModel:
        """Convert embeddings layer only."""
        require_coreml()
        return self.convert_embeddings(model)

    def convert_part_3(self, model: Gemma3ForCausalLM, argmax_in_model: bool = False) -> ct.models.MLModel:
        """Convert LM head only.

        Args:
            model: The Gemma3 model
            argmax_in_model: If True, compute argmax per chunk inside the model.
                            Outputs argmax_idx[num_chunks] and argmax_val[num_chunks]
                            instead of num_chunks logits tensors.
        """
        require_coreml()

        class LMHeadWrapper(torch.nn.Module):
            def __init__(self, model: Gemma3ForCausalLM, argmax_mode: bool = False) -> None:
                super().__init__()
                self.argmax_mode = argmax_mode
                if hasattr(model, "lm_head16_1"):
                    self.heads = [
                        getattr(model, f"lm_head16_{i}") for i in range(1, 17)
                    ]
                    self.mode = "16"
                    self.num_chunks = 16
                    self.chunk_size = 16384  # 262144 / 16
                elif hasattr(model, "lm_head8_1"):
                    self.heads = [getattr(model, f"lm_head8_{i}") for i in range(1, 9)]
                    self.mode = "8"
                    self.num_chunks = 8
                    self.chunk_size = 32768  # 262144 / 8
                elif hasattr(model, "lm_head2_1"):
                    self.heads = [model.lm_head2_1, model.lm_head2_2]
                    self.mode = "2"
                    self.num_chunks = 2
                    self.chunk_size = 131072  # 262144 / 2
                elif hasattr(model, "lm_head1"):
                    self.head = model.lm_head1
                    self.mode = "1"
                    self.num_chunks = 1
                    self.chunk_size = 262144
                else:
                    self.head = model.lm_head
                    self.mode = "linear"
                    self.num_chunks = 1
                    self.chunk_size = 262144

            def forward(self, hidden_states: torch.Tensor):
                if self.mode != "linear":
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)

                # Compute logits for each chunk
                if self.mode in ("16", "8", "2"):
                    logits_list = [
                        h(hidden_states).squeeze(2).transpose(1, 2) for h in self.heads
                    ]
                elif self.mode == "1":
                    logits_list = [self.head(hidden_states).squeeze(2).transpose(1, 2)]
                else:
                    logits_list = [self.head(hidden_states)]

                # If argmax_mode, compute argmax per chunk and return 2 tensors
                # NOTE: We return LOCAL indices (0 to chunk_size-1), not global indices.
                # The Swift/Python inference code must add (chunk_idx * chunk_size) to get global index.
                # NOTE: Using int32 for ANE compatibility - ANE doesn't support int64 for argmax.
                if self.argmax_mode:
                    all_idx = []
                    all_val = []
                    for logits in logits_list:
                        # logits: [1, 1, chunk_size]
                        # Get argmax index within chunk (0 to chunk_size-1)
                        chunk_argmax = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1, 1], int64
                        chunk_max_val = torch.gather(logits, -1, chunk_argmax)  # [1, 1, 1], fp16
                        # Convert to int32 for ANE compatibility
                        local_idx = chunk_argmax.to(torch.int32)  # [1, 1, 1]
                        all_idx.append(local_idx)
                        all_val.append(chunk_max_val)
                    # Stack into single tensors
                    argmax_idx = torch.cat(all_idx, dim=-1).squeeze(0).squeeze(0)  # [num_chunks], int32 (LOCAL indices)
                    argmax_val = torch.cat(all_val, dim=-1).squeeze(0).squeeze(0)  # [num_chunks], fp16
                    return (argmax_idx, argmax_val)
                else:
                    # Return logits as tuple
                    return tuple(logits_list)

        wrapper = LMHeadWrapper(model, argmax_mode=argmax_in_model)
        wrapper.eval()

        # Ensure no gradients
        for param in wrapper.parameters():
            param.requires_grad = False

        argmax_str = ", argmax_in_model=True" if argmax_in_model else ""
        print(f"LM head wrapper created (mode: {wrapper.mode}, chunks: {wrapper.num_chunks}{argmax_str})")

        sample_input = torch.zeros(
            (1, 1, model.config.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE
        )

        # Trace with no_grad context
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, sample_input)

        if argmax_in_model:
            # Output 2 tensors: argmax_idx[num_chunks] and argmax_val[num_chunks]
            outputs = [
                ct.TensorType(name="argmax_idx", dtype=np.int32),
                ct.TensorType(name="argmax_val", dtype=np.float16)
            ]
            print(f"Outputs: argmax_idx[{wrapper.num_chunks}] (int32) + argmax_val[{wrapper.num_chunks}] (fp16)")
        elif wrapper.mode == "16":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 17)
            ]
        elif wrapper.mode == "8":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 9)
            ]
        elif wrapper.mode == "2":
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
            # Use single-threaded for LM head (large vocab) to avoid multiprocessing issues
            self.postprocess(num_workers=None)
            mlmodel = self.converted_model

        return mlmodel

    def convert_part_2(
        self, model: Gemma3ForCausalLM, chunk_idx: int = 0, total_chunks: int = 1,
        force_rotation: bool = False
    ) -> ct.models.MLModel:
        """Convert transformer layers for generation (FFN).

        Args:
            model: The Gemma3 model to convert
            chunk_idx: Index of the chunk (0-based)
            total_chunks: Total number of chunks
            force_rotation: If True, force rotation mode for local cache updates.
                          False = fill mode (positions < sliding_window)
                          True = rotate mode (positions >= sliding_window)
        """
        require_coreml()
        mode_str = "infer_rotate (rotation mode)" if force_rotation else "infer (fill mode)"
        print(f"\nConverting chunked FFN for {mode_str}...")

        # Set force_rotation_mode on model config before tracing
        model.model.config.force_rotation_mode = force_rotation
        print(f"  force_rotation_mode = {force_rotation}")

        # Disable dynamic prefill slicing when tracing rotation variants
        original_prefill_dynamic_slice = getattr(model.model.config, "prefill_dynamic_slice", False)
        effective_dynamic_slice = self.prefill_dynamic_slice and not bool(force_rotation)
        if effective_dynamic_slice != original_prefill_dynamic_slice:
            model.model.config.prefill_dynamic_slice = effective_dynamic_slice

        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            # Balanced distribution: first `rem` chunks get one extra layer
            # Same algorithm as llama_converter.py and qwen_converter.py
            base = total_layers // total_chunks
            rem = total_layers % total_chunks
            start_layer = chunk_idx * base + min(chunk_idx, rem)
            end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
            print(f"  Chunk {chunk_idx + 1}/{total_chunks}: layers {start_layer}-{end_layer - 1}")
        else:
            start_layer = 0
            end_layer = None

        class FFNWrapper(torch.nn.Module):
            def __init__(self, model: Gemma3ForCausalLM, start_layer: int, end_layer: int) -> None:
                super().__init__()
                self.model = model  # Use Gemma3ForCausalLM as root
                self.start_layer = start_layer
                self.end_layer = end_layer
                self.states = Gemma3Converter.GetTransformerStates(
                    model, part="2", prefix="model.model.",
                    start_layer=start_layer, end_layer=end_layer
                )

            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                # RoPE is now retrieved per-layer inside process_layers
                # force_rotation_mode is read from config inside process_layers
                out = self.model.model.process_layers(
                    hidden_states,
                    position_ids,
                    causal_mask,
                    current_pos,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    IN_PREFILL=False,
                )
                # Only apply final norm if this is the last chunk
                if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                    out = self.model.model.norm(out)

                # CRITICAL: Touch both caches to mark them as "used" for CoreML
                # Chunks without global layers still need to declare kv_cache_global
                # but CoreML errors on unused state inputs. This adds 0 to output
                # but forces CoreML to track both states as used.
                if hasattr(self.model.model, 'kv_cache_local') and hasattr(self.model.model, 'kv_cache_global'):
                    dummy_local = self.model.model.kv_cache_local[0, 0, 0, 0] * 0.0
                    dummy_global = self.model.model.kv_cache_global[0, 0, 0, 0] * 0.0
                    out = out + (dummy_local + dummy_global).view(1, 1, 1)

                return out

        wrapper = FFNWrapper(model, start_layer, end_layer)
        wrapper.eval()

        hidden_states = torch.zeros(
            (1, 1, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE
        )
        position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        # Use attention_size for mask size
        causal_mask = torch.zeros(
            (1, 1, 1, self.attention_size), dtype=torch.float16, device=TEST_DEVICE
        )
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        self._reset_kv_cache_buffers(wrapper)
        traced = torch.jit.trace(
            wrapper, (hidden_states, position_ids, causal_mask, current_pos)
        )
        self._reset_kv_cache_buffers(wrapper)
        self._reset_kv_cache_buffers(traced)

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
            states=self.GetTransformerStates(model, part=None, prefix="model.model.",
                                             start_layer=start_layer, end_layer=end_layer),
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

        # Restore original prefill_dynamic_slice
        if getattr(model.model.config, "prefill_dynamic_slice", None) != original_prefill_dynamic_slice:
            model.model.config.prefill_dynamic_slice = original_prefill_dynamic_slice

        return mlmodel

    def convert_part_2_prefill(
        self, model: Gemma3ForCausalLM, chunk_idx: int = 0, total_chunks: int = 1,
        force_rotation: bool = False
    ) -> ct.models.MLModel:
        """Convert transformer layers for prefill mode.

        Args:
            model: The Gemma3 model to convert
            chunk_idx: Index of the chunk (0-based)
            total_chunks: Total number of chunks
            force_rotation: If True, force rotation mode for local cache updates.
                          False = fill mode (prefill, positions < sliding_window)
                          True = rotate mode (prefill_rotate, positions >= sliding_window)
        """
        require_coreml()
        mode_str = "prefill_rotate (rotation mode)" if force_rotation else "prefill (fill mode)"
        print(f"\nConverting chunked prefill for {mode_str}...")

        # Set force_rotation_mode on model config before tracing
        model.model.config.force_rotation_mode = force_rotation
        print(f"  force_rotation_mode = {force_rotation}")

        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            # Balanced distribution: first `rem` chunks get one extra layer
            # Same algorithm as llama_converter.py and qwen_converter.py
            base = total_layers // total_chunks
            rem = total_layers % total_chunks
            start_layer = chunk_idx * base + min(chunk_idx, rem)
            end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
            print(f"  Prefill chunk {chunk_idx + 1}/{total_chunks}: layers {start_layer}-{end_layer - 1}")
        else:
            start_layer = 0
            end_layer = None

        # Determine if this is prefill with rotation
        is_prefill_rotate = force_rotation

        use_update_mask = (not force_rotation) and (not self.prefill_dynamic_slice)

        if use_update_mask:
            class PrefillWrapper(torch.nn.Module):
                def __init__(self, model: Gemma3ForCausalLM, start_layer=0, end_layer=None,
                            is_prefill_rotate=False):
                    super().__init__()
                    self.model = model  # Use Gemma3ForCausalLM as root
                    self.start_layer = start_layer
                    self.end_layer = end_layer
                    self.is_prefill_rotate = is_prefill_rotate
                    self.states = Gemma3Converter.GetTransformerStates(
                        model, part="2_prefill", prefix="model.model.",
                        start_layer=start_layer, end_layer=end_layer
                    )

                def forward(self, hidden_states, position_ids, causal_mask, current_pos, update_mask):
                    # RoPE is now retrieved per-layer inside process_layers
                    # force_rotation_mode is read from config inside process_layers
                    out = self.model.model.process_layers(
                        hidden_states,
                        position_ids,
                        causal_mask,
                        current_pos,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        IN_PREFILL=True,
                        IN_PREFILL_ROTATE=self.is_prefill_rotate,
                        update_mask=update_mask,
                    )

                    # CRITICAL: Touch both caches to mark them as "used" for CoreML
                    # Chunks without global layers still need to declare kv_cache_global
                    # but CoreML errors on unused state inputs. This adds 0 to output
                    # but forces CoreML to track both states as used.
                    if hasattr(self.model.model, 'kv_cache_local') and hasattr(self.model.model, 'kv_cache_global'):
                        dummy_local = self.model.model.kv_cache_local[0, 0, 0, 0] * 0.0
                        dummy_global = self.model.model.kv_cache_global[0, 0, 0, 0] * 0.0
                        out = out + (dummy_local + dummy_global).view(1, 1, 1)

                    # Skip normalization for prefill - data not used, only KV cache is updated!
                    # This follows the LLAMA pattern and avoids unnecessary computation
                    if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                        print("Skipping final normalization for prefill, data not used!")
                        # Return only first token to minimize memory usage
                        return out[:, 0:1, :]

                    return out
        else:
            class PrefillWrapper(torch.nn.Module):
                def __init__(self, model: Gemma3ForCausalLM, start_layer=0, end_layer=None,
                            is_prefill_rotate=False):
                    super().__init__()
                    self.model = model  # Use Gemma3ForCausalLM as root
                    self.start_layer = start_layer
                    self.end_layer = end_layer
                    self.is_prefill_rotate = is_prefill_rotate
                    self.states = Gemma3Converter.GetTransformerStates(
                        model, part="2_prefill", prefix="model.model.",
                        start_layer=start_layer, end_layer=end_layer
                    )

                def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                    # RoPE is now retrieved per-layer inside process_layers
                    # force_rotation_mode is read from config inside process_layers
                    out = self.model.model.process_layers(
                        hidden_states,
                        position_ids,
                        causal_mask,
                        current_pos,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        IN_PREFILL=True,
                        IN_PREFILL_ROTATE=self.is_prefill_rotate,
                    )

                    # CRITICAL: Touch both caches to mark them as "used" for CoreML
                    # Chunks without global layers still need to declare kv_cache_global
                    # but CoreML errors on unused state inputs. This adds 0 to output
                    # but forces CoreML to track both states as used.
                    if hasattr(self.model.model, 'kv_cache_local') and hasattr(self.model.model, 'kv_cache_global'):
                        dummy_local = self.model.model.kv_cache_local[0, 0, 0, 0] * 0.0
                        dummy_global = self.model.model.kv_cache_global[0, 0, 0, 0] * 0.0
                        out = out + (dummy_local + dummy_global).view(1, 1, 1)

                    # Skip normalization for prefill - data not used, only KV cache is updated!
                    # This follows the LLAMA pattern and avoids unnecessary computation
                    if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                        print("Skipping final normalization for prefill, data not used!")
                        # Return only first token to minimize memory usage
                        return out[:, 0:1, :]

                    return out

        wrapper = PrefillWrapper(model, start_layer, end_layer, is_prefill_rotate)
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
        # Use attention_size for prefill mask size
        causal_mask = torch.zeros(
            (1, 1, self.batch_size, self.attention_size),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        update_mask = None
        if use_update_mask:
            mask_size = self.attention_size
            update_mask = torch.zeros(
                (1, 1, mask_size, self.batch_size),
                dtype=torch.float16,
                device=TEST_DEVICE,
            )

        self._reset_kv_cache_buffers(wrapper)
        if update_mask is not None:
            traced = torch.jit.trace(
                wrapper, (hidden_states, position_ids, causal_mask, current_pos, update_mask)
            )
        else:
            traced = torch.jit.trace(
                wrapper, (hidden_states, position_ids, causal_mask, current_pos)
            )
        self._reset_kv_cache_buffers(wrapper)
        self._reset_kv_cache_buffers(traced)

        inputs = [
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
        ]
        if update_mask is not None:
            inputs.append(
                ct.TensorType(
                    name="update_mask", shape=update_mask.shape, dtype=np.float16
                )
            )

        mlmodel = ct.convert(
            traced,
            inputs=inputs,
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

    def convert_prefill(self, model: Gemma3ForCausalLM) -> ct.models.MLModel:
        """Convert Gemma3 model to CoreML format for prefill mode.

        Args:
            model: The Gemma3 model to convert

        Returns:
            ct.models.MLModel: Converted model for prefill processing
        """
        require_coreml()
        print("Converting Gemma3 model for prefill mode...")

        use_update_mask = not self.prefill_dynamic_slice

        if use_update_mask:
            class PrefillWrapper(torch.nn.Module):
                def __init__(
                    self, model: Gemma3ForCausalLM, context_length: int, batch_size: int
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
                    update_mask: torch.Tensor,
                ) -> torch.Tensor:
                    # Prefill mode: only process transformer layers, skip embeddings and LM head
                    # This updates KV cache state without generating logits
                    return self.model.forward_prefill(
                        hidden_states=hidden_states,
                        position_ids=position_ids,
                        causal_mask=causal_mask,
                        current_pos=current_pos,
                        update_mask=update_mask,
                    )
        else:
            class PrefillWrapper(torch.nn.Module):
                def __init__(
                    self, model: Gemma3ForCausalLM, context_length: int, batch_size: int
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
            (1, 1, self.batch_size, self.attention_size),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )  # [1, 1, batch_size, attention_size] - smaller = faster attention
        sample_current_pos = torch.zeros(
            (1,), dtype=torch.int32, device=TEST_DEVICE
        )  # [1] - current position
        sample_update_mask = None
        if use_update_mask:
            sample_update_mask = torch.zeros(
                (1, 1, self.attention_size, self.batch_size),
                dtype=torch.float16,
                device=TEST_DEVICE,
            )  # [1, 1, attention_size, batch_size]

        print("Prefill sample inputs created")
        print(f"sample_hidden_states shape: {sample_hidden_states.shape}")
        print(f"sample_position_ids shape: {sample_position_ids.shape}")
        print(f"sample_causal_mask shape: {sample_causal_mask.shape}")
        print(f"sample_current_pos shape: {sample_current_pos.shape}")
        if sample_update_mask is not None:
            print(f"sample_update_mask shape: {sample_update_mask.shape}")

        print("Starting torch.jit.trace for prefill...")
        self._reset_kv_cache_buffers(wrapper)
        if sample_update_mask is not None:
            traced = torch.jit.trace(
                wrapper,
                (
                    sample_hidden_states,
                    sample_position_ids,
                    sample_causal_mask,
                    sample_current_pos,
                    sample_update_mask,
                ),
            )
        else:
            traced = torch.jit.trace(
                wrapper,
                (
                    sample_hidden_states,
                    sample_position_ids,
                    sample_causal_mask,
                    sample_current_pos,
                ),
            )
        self._reset_kv_cache_buffers(wrapper)
        self._reset_kv_cache_buffers(traced)
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
                *( [ct.TensorType(
                    name="update_mask", shape=sample_update_mask.shape, dtype=np.float16
                )] if sample_update_mask is not None else [] ),
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
            # Use single-threaded for full prefill to avoid multiprocessing issues
            self.postprocess(num_workers=None)
            mlmodel = self.converted_model

        return mlmodel

    def convert_embeddings(self, model: Gemma3ForCausalLM) -> ct.models.MLModel:
        """Convert embeddings layer to CoreML format.

        Args:
            model: The Gemma3 model containing embeddings

        Returns:
            ct.models.MLModel: Converted CoreML model for embeddings
        """
        require_coreml()
        print("\nConverting Gemma3 embeddings layer...")

        class EmbeddingsWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.embed_tokens = model.model.embed_tokens
                # Gemma3 scales embeddings by sqrt(hidden_size)
                self.embedding_scale = model.model.embedding_scale

            def forward(self, input_ids):
                hidden_states = self.embed_tokens(input_ids)
                # Apply Gemma3 embedding scaling (sqrt(hidden_size))
                hidden_states = hidden_states * self.embedding_scale
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
        # IMPORTANT: Skip LUT for embeddings when FP16 scaling is applied
        # The op_name_configs regex patterns don't work for embeddings-only models
        # because CoreML assigns generic names (op_0, op_1) instead of preserving
        # the PyTorch module names. LUT quantization on embeddings causes the
        # gather operation to return constant values regardless of input token.
        if self.lut_bits and not self.fp16_scaled:
            self.converted_model = mlmodel
            # Use single-threaded for embeddings (large vocab) to avoid multiprocessing issues
            self.postprocess(num_workers=None)
            mlmodel = self.converted_model
        elif self.lut_bits and self.fp16_scaled:
            print("  ⚠️  Skipping LUT quantization for embeddings (FP16-scaled model)")
            print("      Embeddings will remain in FP16 to preserve lookup accuracy")

        return mlmodel


def parse_lut_arg(lut_value: str | int | None) -> tuple[int | None, int]:
    """Parse LUT argument that can be 'bits', 'bits,per_channel', or 'bits,0' for per-tensor.

    Args:
        lut_value: String or int value from command line (e.g., '6', '6,4', '4,0' for per-tensor)

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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the converter."""

    parser = argparse.ArgumentParser(description="Convert Gemma3 model to CoreML format")

    parser.add_argument(
        "--model",
        type=str,
        help="Path to model directory (default: google/gemma-3n-E2B-it)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="gemma3",
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
        "--state-length",
        type=int,
        default=None,
        help="KV cache size for global attention layers. If not specified, uses context_length. "
             "For split cache, this controls the size of the global cache buffer.",
    )
    parser.add_argument(
        "--lut",
        type=str,
        default=None,
        help="Use LUT quantization with N bits. Format: 'bits' or 'bits,per_channel' "
             "(e.g., '4', '4,8', '4,0' for per-tensor). Default per_channel is 8.",
    )
    parser.add_argument(
        "--lut-embeddings",
        type=str,
        default=None,
        help="Override LUT for embeddings in monolithic models. Same format as --lut. "
             "If not set, uses --lut value.",
    )
    parser.add_argument(
        "--lut-lmhead",
        type=str,
        default=None,
        help="Override LUT for LM head in monolithic models. Same format as --lut. "
             "If not set, uses --lut value.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Split FFN/prefill into N chunks",
    )
    parser.add_argument(
        "--chunk-no",
        type=int,
        default=None,
        help="Convert only this chunk number (1-based) for chunked parts",
    )
    parser.add_argument(
        "--dynamic-prefill-slice",
        action="store_true",
        help="Use dynamic slicing for prefill KV writes (default ON).",
    )
    parser.add_argument(
        "--static-prefill-slice",
        action="store_true",
        help="Disable dynamic slicing for prefill KV writes (uses update_mask where applicable).",
    )
    parser.add_argument(
        "--single-cache",
        action="store_true",
        help="Use a single unified KV cache instead of split local/global caches (Gemma3 only).",
    )
    parser.add_argument(
        "--part",
        type=str,
        choices=["1", "2", "2_rotate", "2_prefill", "2_prefill_rotate", "3", "all", "full", "prefill", "embeddings", "monolithic", "monolithic_rotate", "monolithic_prefill", "monolithic_prefill_rotate"],
        default="all",
        help="Model part to convert",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory for converted models",
    )
    parser.add_argument(
        "--argmax",
        action="store_true",
        help="Compute argmax inside model for monolithic conversion. "
             "Outputs (argmax_idx, argmax_val) pairs instead of full logits.",
    )
    parser.add_argument(
        "--attention-size",
        type=int,
        default=None,
        help="Attention computation size (default: context_length). "
             "Controls KV cache size and causal mask dimension. "
             "Smaller values = smaller attention matrix = faster inference.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip ANE minimum state size check (256). Use for testing smaller KV cache sizes.",
    )
    parser.add_argument(
        "--decorate",
        action="store_true",
        help="Add CTX and ATT values to output filename (e.g., gemma3_monolithic_CTX512_ATT64.mlpackage)",
    )
    parser.add_argument(
        "--fp16-scale",
        type=str,
        default=None,
        help="FP16 residual stream scaling factor to prevent overflow. "
             "Use 'auto' to compute based on model, or specify a value (e.g., 0.1875). "
             "Recommended: 0.48 for 270M, 0.82 for 1B, 0.1875 for 4B QAT. "
             "See docs/GEMMA3_FP16_SCALING.md for details.",
    )
    parser.add_argument(
        "--clamp",
        type=float,
        default=None,
        help="Enable runtime residual clamping at the specified value (e.g., 55000). "
             "This adds clamp ops to the CoreML model to prevent FP16 overflow. "
             "Alternative to --fp16-scale for models that need runtime overflow protection.",
    )

    return parser.parse_args()


# Recommended FP16 scaling factors for Gemma3 models
# Based on residual stream analysis - see docs/GEMMA3_FP16_SCALING.md
GEMMA3_SCALING_FACTORS = {
    # Model identifier patterns -> recommended alpha
    "gemma-3-270m": 0.48,      # Peak 104,162 (1.6x FP16 max)
    "gemma-3-1b": 0.82,        # Peak 61,040 (0.93x FP16 max)
    "gemma-3-4b-it-qat": 0.1875,  # Peak 292,969 (4.5x FP16 max), use 3/16
    "gemma-3-4b": 0.5,         # Estimate for non-QAT 4B
    "gemma-3n-E2B": 0.5,       # Estimate, needs verification
    "gemma-3n-E4B": 0.5,       # Estimate, needs verification
}


def _get_auto_scale_factor(model_path: str) -> Optional[float]:
    """Get recommended scaling factor based on model path/name."""
    model_lower = model_path.lower()

    # Check for known model patterns
    for pattern, alpha in GEMMA3_SCALING_FACTORS.items():
        if pattern.lower() in model_lower:
            return alpha

    # Default: no scaling for unknown models
    return None


def _apply_fp16_scaling(
    model: "Gemma3ForCausalLM",
    fp16_scale: str,
    model_path: str,
) -> Optional[float]:
    """Apply FP16 residual stream scaling to prevent activation overflow.

    This implements the weight-only transformation from GEMMA3_FP16_SCALING.md:
    1. Scale embedding weights by α
    2. Transform post-norm weights: w_new = α * (1 + w_old) - 1

    This shrinks the residual stream without changing model behavior (RMSNorm
    is scale-invariant, and the final norm cancels the global scale).

    Args:
        model: The Gemma3 model with loaded weights
        fp16_scale: Either "auto" or a float value like "0.1875"
        model_path: Model path for auto-detection

    Returns:
        The alpha value applied, or None if no scaling was applied
    """
    # Determine alpha value
    if fp16_scale.lower() == "auto":
        alpha = _get_auto_scale_factor(model_path)
        if alpha is None:
            print(f"  ⚠️  Could not auto-detect scaling factor for: {model_path}")
            print(f"      Known models: {list(GEMMA3_SCALING_FACTORS.keys())}")
            print(f"      Skipping FP16 scaling. Use --fp16-scale <value> to specify manually.")
            return None
        print(f"  Auto-detected FP16 scale factor: α = {alpha}")
    else:
        try:
            alpha = float(fp16_scale)
        except ValueError:
            print(f"  ⚠️  Invalid fp16_scale value: {fp16_scale}")
            return None

    if alpha <= 0 or alpha > 1:
        print(f"  ⚠️  Alpha must be in (0, 1], got: {alpha}")
        return None

    print(f"  Applying FP16 residual stream scaling with α = {alpha}")

    # Get model components
    # Handle both text-only and multimodal model structures
    if hasattr(model, 'model'):
        if hasattr(model.model, 'language_model'):
            # Multimodal model (e.g., Gemma3n)
            embed = model.model.language_model.embed_tokens if hasattr(model.model.language_model, 'embed_tokens') else None
            layers = model.model.language_model.layers if hasattr(model.model.language_model, 'layers') else None
        else:
            # Text-only model
            embed = model.model.embed_tokens if hasattr(model.model, 'embed_tokens') else None
            layers = model.model.layers if hasattr(model.model, 'layers') else None
    else:
        embed = None
        layers = None

    if embed is None or layers is None:
        print(f"  ⚠️  Could not find model components for scaling")
        return None

    # 1. Scale embedding weights
    with torch.no_grad():
        if hasattr(embed, 'weight'):
            embed.weight.mul_(alpha)
            print(f"    ✅ Scaled embed_tokens.weight by {alpha}")

    # 2. Transform post-norm weights
    # Gemma3 RMSNorm uses: output = norm(x) * (1 + weight)
    # To scale the gain by α: w_new = α * (1 + w_old) - 1
    scaled_layers = 0
    for layer_idx, layer in enumerate(layers):
        for norm_name in ['post_attention_layernorm', 'post_feedforward_layernorm']:
            if hasattr(layer, norm_name):
                norm = getattr(layer, norm_name)
                if hasattr(norm, 'weight'):
                    with torch.no_grad():
                        norm.weight.data = alpha * (1 + norm.weight.data) - 1
                    scaled_layers += 1

    print(f"    ✅ Transformed {scaled_layers} post-norm weights across {len(layers)} layers")

    return alpha


def test_conversion(
    model: Optional[Gemma3ForCausalLM] = None,
    model_path: Optional[str] = None,
    prefix: str = "gemma3",
    context_length: int = CONTEXT_LENGTH,
    state_length: Optional[int] = None,
    lut_bits: Optional[int] = None,
    per_channel: int = 8,
    batch_size: int = 64,
    output_dir: str = ".",
    part: str = "full",
    num_chunks: int = 1,
    argmax_in_model: bool = False,
    attention_size: Optional[int] = None,
    force: bool = False,
    decorate: bool = False,
    fp16_scale: Optional[str] = None,
    clamp: Optional[float] = None,
    prefill_dynamic_slice: bool = False,
    single_cache: bool = False,
    chunk_no: Optional[int] = None,
    lut_embeddings_bits: Optional[int] = None,
    lut_embeddings_per_channel: int = 8,
    lut_lmhead_bits: Optional[int] = None,
    lut_lmhead_per_channel: int = 8,
) -> ct.models.MLModel | List[ct.models.MLModel]:
    """Convert a Gemma3 model and save the result.

    Args:
        model: Pre-loaded Gemma3 model (optional)
        model_path: Path to model directory
        prefix: Model name prefix
        context_length: Context length for conversion
        state_length: KV cache size for global attention layers (default: context_length).
                      For split cache, this controls the size of the global cache buffer.
        lut_bits: LUT quantization bits
        per_channel: Group size for per_grouped_channel quantization.
                     Use 0 for per-tensor quantization.
        batch_size: Batch size for conversion
        output_dir: Output directory
        part: Part to convert ("full" or "prefill")
        argmax_in_model: If True, compute argmax inside model for monolithic conversion
        attention_size: Attention computation window size (default: context_length).
                          Controls how many KV positions each token attends to.
    """
    # Early validation: Check output directory can be created/accessed
    # This prevents wasting time on conversion only to fail at save time
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Test write access by creating a temporary file
        test_file = os.path.join(output_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"Output directory validated: {output_dir}")
    except PermissionError:
        raise RuntimeError(
            f"Permission denied: Cannot write to output directory '{output_dir}'. "
            f"Please check permissions or choose a different location."
        )
    except OSError as e:
        raise RuntimeError(
            f"Cannot create/access output directory '{output_dir}': {e}. "
            f"Please ensure the path is valid and the parent directory exists."
        )

    # Validate minimum state size for ANE
    # ANE requires minimum context_length (KV cache/state size) of 256
    # attention_size only controls causal mask dimension (can be smaller)
    MIN_ANE_STATE_SIZE = 256
    if context_length < MIN_ANE_STATE_SIZE:
        if force:
            print(f"WARNING: context_length={context_length} is below ANE minimum of {MIN_ANE_STATE_SIZE}. "
                  f"Using --force to override. Model may not run on ANE.")
        else:
            raise ValueError(
                f"ANE requires minimum context_length (state size) of {MIN_ANE_STATE_SIZE}. "
                f"Got context_length={context_length}. "
                f"Use --context-length >= {MIN_ANE_STATE_SIZE}, or --force to override."
            )

    # attention_size must be <= context_length
    if attention_size is not None and attention_size > context_length:
        raise ValueError(
            f"attention_size ({attention_size}) cannot exceed context_length ({context_length}). "
            f"attention_size controls how many KV positions to attend to."
        )

    print(
        f"test_conversion called with model_path={model_path}, prefix={prefix}, part={part}"
    )

    if single_cache and part in [
        "2_rotate",
        "2_prefill_rotate",
        "monolithic_rotate",
        "monolithic_prefill_rotate",
    ]:
        raise ValueError(
            "single-cache is not supported with rotate/prefill_rotate conversions. "
            "Use context_length <= sliding_window or disable rotation conversions."
        )

    # Track if FP16 scaling was applied (to exclude scaled tensors from LUT quantization)
    fp16_scaled = False

    if model is None:
        if model_path is None:
            raise ValueError("model_path must be provided if model is None")

        config_path = os.path.join(model_path, "config.json")
        print(f"Looking for config at: {config_path}")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")

        print("Loading config...")
        config = Gemma3Config.from_json(config_path)
        print(
            f"Config loaded: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}"
        )

        # Update config to match conversion parameters
        # For Gemma3, context_length must be >= sliding_window for local attention to work
        if context_length < config.sliding_window:
            print(f"  ⚠️  context_length ({context_length}) < sliding_window ({config.sliding_window})")
            print(f"      Auto-adjusting context_length to {config.sliding_window}")
            context_length = config.sliding_window
        config.context_length = context_length
        # state_length controls global KV cache size (for split cache: global attention layers)
        # If not specified, defaults to context_length
        config.state_length = state_length if state_length is not None else context_length
        # Set attention_size for ANE optimization (separate from Gemma3's sliding_window)
        # sliding_window stays unchanged - it's Gemma3's architectural feature
        # attention_size controls causal_mask dimension - must be at least sliding_window
        # for local attention layers to work properly
        if attention_size is not None:
            config.attention_size = attention_size
        else:
            # For Gemma3, attention_size must be >= sliding_window for local attention
            config.attention_size = max(context_length, config.sliding_window)
        # Set batch_size for prefill operations (needed for prefill_rotate tracing)
        config.batch_size = batch_size
        print(
            f"Updated config: context_length={config.context_length}, state_length={config.state_length}, attention_size={config.attention_size}, sliding_window={config.sliding_window}, batch_size={config.batch_size}"
        )

        if single_cache and context_length > config.sliding_window:
            print(
                f"WARNING: single-cache with context_length {context_length} > sliding_window {config.sliding_window}. "
                "Rotation is unsupported in unified cache mode; use <= sliding_window or split cache."
            )

        # Enable residual clamping if specified (alternative to FP16 scaling)
        if clamp is not None:
            config.enable_residual_clamp = True
            config.residual_clamp_value = clamp
            print(f"Enabling residual clamping at {clamp}")

        if single_cache:
            config.single_cache = True
            config.use_split_cache = False
            print("Single-cache mode: using unified KV cache (split cache disabled)")

        print("Creating model...")
        model = Gemma3ForCausalLM(config, enable_coreml=True)
        print("Loading pretrained weights...")
        model.load_pretrained_weights(model_path)
        print("Model loaded successfully!")

        # Apply FP16 residual stream scaling if specified
        if fp16_scale is not None:
            alpha = _apply_fp16_scaling(model, fp16_scale, model_path)
            if alpha is not None:
                print(f"Applied FP16 residual scaling with α = {alpha}")
                fp16_scaled = True  # Mark that scaling was applied

        # Ensure model is in eval mode and gradients are disabled
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("Model set to eval mode and gradients disabled")

    print("Creating converter...")
    converter = Gemma3Converter(
        model=model,
        context_length=context_length,
        batch_size=batch_size,
        lut_bits=lut_bits,
        per_channel=per_channel,
        num_chunks=num_chunks,
        argmax_in_model=argmax_in_model,
        attention_size=attention_size,
        fp16_scaled=fp16_scaled,  # Exclude scaled tensors from LUT quantization
        prefill_dynamic_slice=prefill_dynamic_slice,
        lut_embeddings_bits=lut_embeddings_bits,
        lut_embeddings_per_channel=lut_embeddings_per_channel,
        lut_lmhead_bits=lut_lmhead_bits,
        lut_lmhead_per_channel=lut_lmhead_per_channel,
    )

    print("Starting conversion...")
    mlmodel = converter.convert(part=part, chunk_no=chunk_no)
    print("Conversion completed!")

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(mlmodel, list):
        models = mlmodel
    else:
        models = [mlmodel]

    effective_attention_size = attention_size if attention_size is not None else context_length
    for i, m in enumerate(models):
        chunk_no_value = None
        if part in ["2", "2_prefill"]:
            chunk_no_value = chunk_no if chunk_no is not None else (i + 1)
        AddMetadata(
            m,
            {
                "context_length": context_length,
                "batch_size": batch_size if part in ["2_prefill", "prefill", "monolithic_prefill"] else None,
                "lut_bits": lut_bits,
                "num_chunks": num_chunks if part in ["2", "2_prefill"] else None,
                "chunk_no": chunk_no_value,
                "split_part": (
                    ModelPart.FULL.value if part in ["full", "all", "123"] else part
                ),
                "attention_size": effective_attention_size,
                "single_cache": single_cache,
            },
        )
        fname = f"{prefix}"
        if part in ["1", "embeddings"]:
            fname += "_embeddings"
        elif part in ["3"]:
            fname += "_lm_head"
        elif part == "monolithic":
            fname += "_monolithic"
            if decorate:
                att_size = attention_size if attention_size is not None else context_length
                if lut_bits is not None:
                    fname += f"_LUT{lut_bits}_CTX{context_length}_ATT{att_size}"
                else:
                    fname += f"_FP16_CTX{context_length}_ATT{att_size}"
        elif part == "monolithic_rotate":
            fname += "_monolithic_rotate"
            if decorate:
                att_size = attention_size if attention_size is not None else context_length
                if lut_bits is not None:
                    fname += f"_LUT{lut_bits}_CTX{context_length}_ATT{att_size}"
                else:
                    fname += f"_FP16_CTX{context_length}_ATT{att_size}"
        elif part == "monolithic_prefill":
            fname += "_monolithic_prefill"
            if decorate:
                att_size = attention_size if attention_size is not None else context_length
                if lut_bits is not None:
                    fname += f"_LUT{lut_bits}_CTX{context_length}_ATT{att_size}"
                else:
                    fname += f"_FP16_CTX{context_length}_ATT{att_size}"
        elif part == "monolithic_prefill_rotate":
            fname += "_monolithic_prefill_rotate"
            if decorate:
                att_size = attention_size if attention_size is not None else context_length
                if lut_bits is not None:
                    fname += f"_LUT{lut_bits}_CTX{context_length}_ATT{att_size}"
                else:
                    fname += f"_FP16_CTX{context_length}_ATT{att_size}"
        elif part in ["2", "2_rotate", "2_prefill", "2_prefill_rotate"]:
            # Map part to base name:
            # 2 -> FFN (infer, fill mode)
            # 2_rotate -> FFN_rotate (infer_rotate, rotation mode)
            # 2_prefill -> prefill (prefill, fill mode)
            # 2_prefill_rotate -> prefill_rotate (prefill_rotate, rotation mode)
            if part == "2":
                base = "FFN"
            elif part == "2_rotate":
                base = "FFN_rotate"
            elif part == "2_prefill":
                base = "prefill"
            else:  # 2_prefill_rotate
                base = "prefill_rotate"
            fname += f"_{base}"
            if lut_bits is not None:
                fname += f"_lut{lut_bits}"
            chunk_id = chunk_no if chunk_no is not None else (i + 1)
            fname += f"_chunk_{chunk_id:02d}of{num_chunks:02d}"
        if part in ["full", "all", "123"]:
            fname += ""
        if part not in ["2", "2_rotate", "2_prefill", "2_prefill_rotate"]:
            # Skip lut suffix if already included in decoration
            if lut_bits is not None and not (decorate and part in ["monolithic", "monolithic_rotate", "monolithic_prefill", "monolithic_prefill_rotate"]):
                fname += f"_lut{lut_bits}"
            fname += ".mlpackage"
        else:
            fname += ".mlpackage"
        out_path = os.path.join(output_dir, fname)
        print(f"Saving model to: {out_path}")
        m.save(out_path)

    return mlmodel


def main() -> None:
    print("Starting gemma3_converter main()...")
    args = parse_args()
    print(f"Parsed args: {args}")

    # Dynamic prefill slice defaults to ON; allow --static-prefill-slice to disable.
    prefill_dynamic_slice = True
    if getattr(args, "static_prefill_slice", False):
        prefill_dynamic_slice = False
    elif getattr(args, "dynamic_prefill_slice", False):
        prefill_dynamic_slice = True

    single_cache = bool(getattr(args, "single_cache", False))

    model_path = args.model if args.model else "google/gemma-3n-E2B-it"

    print(f"\nConverting model from: {model_path}")
    print(f"Output filename prefix: {args.prefix}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    if args.lut:
        print(f"LUT quantization: {args.lut} bits")
    if args.chunk:
        print(f"Splitting into {args.chunk} chunks")
    if args.fp16_scale:
        print(f"FP16 scaling: {args.fp16_scale}")
    print(f"Converting part(s): {args.part}")

    # Map legacy part names to numeric equivalents
    part_map = {"full": "all", "embeddings": "1", "prefill": "2_prefill"}
    part = part_map.get(args.part, args.part)

    # Parse LUT argument to extract bits and per_channel
    lut_bits, per_channel = parse_lut_arg(args.lut)
    if lut_bits is not None:
        if per_channel <= 0:
            print(f"LUT quantization: {lut_bits} bits, per-tensor granularity")
        else:
            print(f"LUT quantization: {lut_bits} bits, per_channel group size: {per_channel}")

    # Parse per-component LUT overrides
    lut_embeddings_bits, lut_embeddings_per_channel = parse_lut_arg(args.lut_embeddings)
    lut_lmhead_bits, lut_lmhead_per_channel = parse_lut_arg(args.lut_lmhead)
    if lut_embeddings_bits is not None:
        print(f"LUT embeddings override: {lut_embeddings_bits} bits, per_channel={lut_embeddings_per_channel}")
    if lut_lmhead_bits is not None:
        print(f"LUT lm_head override: {lut_lmhead_bits} bits, per_channel={lut_lmhead_per_channel}")

    try:
        print("\nCalling test_conversion()...")
        result = test_conversion(
            model_path=model_path,
            prefix=args.prefix,
            context_length=args.context_length,
            state_length=args.state_length,
            lut_bits=lut_bits,
            per_channel=per_channel,
            batch_size=args.batch_size,
            output_dir=args.output,
            part=part,
            num_chunks=args.chunk or 1,
            argmax_in_model=args.argmax,
            attention_size=args.attention_size,
            force=args.force,
            decorate=args.decorate,
            fp16_scale=args.fp16_scale,
            clamp=args.clamp,
            prefill_dynamic_slice=prefill_dynamic_slice,
            single_cache=single_cache,
            chunk_no=args.chunk_no,
            lut_embeddings_bits=lut_embeddings_bits,
            lut_embeddings_per_channel=lut_embeddings_per_channel,
            lut_lmhead_bits=lut_lmhead_bits,
            lut_lmhead_per_channel=lut_lmhead_per_channel,
        )
        print(f"Conversion completed successfully! Result: {type(result)}")
    except Exception as e:  # pragma: no cover - CLI tool
        print(f"\nError during conversion: {str(e)}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
