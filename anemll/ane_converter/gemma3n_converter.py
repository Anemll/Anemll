# Copyright (c) 2025 ANEMLL
# Licensed under the MIT License
# 
# Gemma3n model converter for Apple Neural Engine acceleration
# Converts Google's Gemma3n models to CoreML format with ANE optimizations

import os
import torch
import torch.nn as nn
import coremltools as ct
import coremltools.optimize as cto
from coremltools.converters.mil import Builder as mb
import numpy as np
from typing import Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoConfig
import json
import shutil

from anemll.ane_converter.base_converter import BaseConverter
from anemll.ane_converter.environment import require_coreml

# Enable CoreML mode before importing model classes
import anemll.models.gemma3n_model
anemll.models.gemma3n_model.ENABLE_COREML = True

from anemll.models.gemma3n_model import (
    Gemma3nModel, Gemma3nLaurelBlock,
    Gemma3nAttention, Gemma3nFFN, Gemma3nRMSNorm, Gemma3nConfig,
    MODEL_DTYPE, TEST_DEVICE
)


class Gemma3nConverter(BaseConverter):
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        context_length: int = 512,
        batch_size: int = 1,
        lut2: Optional[int] = None,
        lut3: Optional[int] = None,
        lut_per_channel: int = 8,
        lut_num_workers: int = 1,
        lut_scope: str = "all",
        lut_include: Optional[str] = None,
        lut_exclude: Optional[str] = None,
        lut_report: bool = False,
        chunk_size: int = 4,
        enable_laurel: bool = True,
        enable_per_layer_embeddings: bool = True,
        text_only_mode: bool = True,
        disable_sparsity: bool = False
    ):
        super().__init__(model_path)
        self.model_type = "gemma3n"
        self.model_path = model_path
        self.output_dir = output_dir
        self.context_length = context_length
        self.batch_size = batch_size
        self.lut2 = lut2
        self.lut3 = lut3
        self.lut_per_channel = lut_per_channel
        self.lut_num_workers = lut_num_workers
        self.lut_scope = lut_scope
        self.lut_include = lut_include
        self.lut_exclude = lut_exclude
        self.lut_report = lut_report
        self.chunk_size = chunk_size
        self.enable_laurel = enable_laurel
        self.enable_per_layer_embeddings = enable_per_layer_embeddings
        self.text_only_mode = text_only_mode
        self.disable_sparsity = disable_sparsity

        # Load config
        hf_config = AutoConfig.from_pretrained(model_path)
        self.config = Gemma3nConfig.from_pretrained_config(hf_config)
        if self.disable_sparsity:
            self.config.activation_sparsity_pattern = [0.0] * self.config.num_hidden_layers

        # Update config with conversion parameters
        self.config.context_length = context_length
        self.config.state_length = max(context_length, self.config.state_length)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def _apply_lut(self, mlmodel: ct.models.MLModel, bits: Optional[int]) -> ct.models.MLModel:
        if not bits:
            return mlmodel
        print(
            f"Applying LUT quantization with {bits} bits and {self.lut_per_channel} channels per group using {self.lut_num_workers} worker(s)..."
        )
        op_config = cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=bits,
            granularity="per_grouped_channel",
            group_size=self.lut_per_channel,
            num_kmeans_workers=self.lut_num_workers,
        )
        if self.lut_scope == "none":
            return mlmodel

        if self.lut_include or self.lut_exclude:
            import re
            from coremltools.optimize.coreml import get_weights_metadata

            include_re = re.compile(self.lut_include) if self.lut_include else None
            exclude_re = re.compile(self.lut_exclude) if self.lut_exclude else None

            meta = get_weights_metadata(mlmodel, weight_threshold=0)
            selected = []
            for name in meta.keys():
                if include_re and not include_re.search(name):
                    continue
                if exclude_re and exclude_re.search(name):
                    continue
                selected.append(name)

            if not selected:
                print("⚠️ LUT include/exclude patterns matched no weights. Skipping palettization.")
                return mlmodel

            if self.lut_report:
                matched = len(selected)
                total = len(meta)
                print(f"🔎 LUT report: matched {matched}/{total} weights for palettization")

            config = cto.coreml.OptimizationConfig(op_name_configs={name: op_config for name in selected})
        else:
            if self.lut_scope == "all":
                config = cto.coreml.OptimizationConfig(global_config=op_config)
            elif self.lut_scope in ("linear", "conv"):
                config = cto.coreml.OptimizationConfig(op_type_configs={self.lut_scope: op_config})
            else:
                raise ValueError(f"Unsupported lut_scope: {self.lut_scope}")
        mlmodel = cto.coreml.palettize_weights(mlmodel, config)
        if self.lut_report:
            try:
                from coremltools.optimize.coreml._post_training_quantization import get_weights_metadata
                meta = get_weights_metadata(mlmodel, weight_threshold=0)
                palettized = [k for k, v in meta.items() if v.unique_values <= (2 ** bits)]
                print(f"✅ LUT report: {len(palettized)}/{len(meta)} weights have <= {2 ** bits} unique values")
            except Exception as e:
                print(f"⚠️ LUT report failed to compute unique values: {e}")
        return mlmodel

    def _maybe_report_weight_bin_size(self, output_path: str) -> None:
        if not self.lut_report:
            return
        weight_path = os.path.join(
            output_path, "Data", "com.apple.CoreML", "weights", "weight.bin"
        )
        if os.path.isfile(weight_path):
            size_bytes = os.path.getsize(weight_path)
            size_mb = size_bytes / (1024 * 1024)
            print(f"📦 weight.bin size: {size_mb:.1f} MB ({weight_path})")
            return
        if os.path.isfile(output_path):
            size_bytes = os.path.getsize(output_path)
            size_mb = size_bytes / (1024 * 1024)
            print(f"📦 model file size: {size_mb:.1f} MB ({output_path})")
            return
        print(f"⚠️ LUT report: weight.bin not found at {weight_path}")

    def _get_kv_cache_states(self, prefix: str = "model.") -> list:
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        # Must be FP16 for ANE compatibility - persistent buffers are FP16 only
        return [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(
                        2 * self.config.num_hidden_layers,
                        self.config.num_key_value_heads,
                        self.config.state_length,
                        head_dim,
                    ),
                    dtype=np.float16,
                ),
                name=f"{prefix}kv_cache_0",
            )
        ]

    def _get_chunk_range(self, chunk_idx: int, total_chunks: int) -> tuple[int, int]:
        """Compute layer start/end for a chunk, distributing remainders to early chunks."""
        total_layers = self.config.num_hidden_layers
        if total_chunks <= 0:
            raise ValueError("total_chunks must be > 0")
        base = total_layers // total_chunks
        rem = total_layers % total_chunks
        if chunk_idx < rem:
            start = chunk_idx * (base + 1)
            end = start + (base + 1)
        else:
            start = rem * (base + 1) + (chunk_idx - rem) * base
            end = start + base
        return start, end
        
    def _is_multimodal_weight(self, key: str) -> bool:
        """Check if a weight belongs to multimodal components (vision/audio)"""
        multimodal_patterns = [
            # Vision encoder patterns
            "vision_encoder", "vision_adapter", "vision_projection",
            "visual_encoder", "visual_adapter", "visual_projection",
            "image_encoder", "image_adapter", "image_projection",
            "mobilenet", "vit", "clip",
            
            # Audio encoder patterns
            "audio_encoder", "audio_adapter", "audio_projection",
            "speech_encoder", "usm", "whisper",
            "audio_tower", "speech_tower",
            
            # Multimodal fusion patterns
            "multimodal_projector", "mm_projector", "multi_modal_projector",
            "vision_tower", "audio_tower", "modality_adapter",
            "cross_modal", "fusion_layer",
            
            # Gemma3n specific multimodal patterns
            "soft_tokens", "modality_tokens", "adapter_tokens",
            "video_encoder", "frame_encoder",
            ]
        
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in multimodal_patterns)
    
    def _filter_text_only_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Filter state dict to keep only text-related weights"""
        filtered_dict = {}
        multimodal_keys = []
        
        for key, value in state_dict.items():
            if self._is_multimodal_weight(key):
                multimodal_keys.append(key)
            else:
                filtered_dict[key] = value
        
        if multimodal_keys:
            print(f"Filtering {len(multimodal_keys)} multimodal weights for text-only mode:")
            for key in multimodal_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(multimodal_keys) > 5:
                print(f"  ... and {len(multimodal_keys) - 5} more")
        
        return filtered_dict
    
    def _load_model_weights(self) -> Dict[str, torch.Tensor]:
        """Load model weights from safetensors files (proven method from test files)"""
        try:
            # Check if this is a local path with safetensors
            if os.path.exists(self.model_path):
                safetensor_files = [
                    "model-00001-of-00003.safetensors",
                    "model-00002-of-00003.safetensors", 
                    "model-00003-of-00003.safetensors"
                    ]
                
                all_weights = {}
                files_found = 0
                
                # Load from safetensors files (proven working method)
                for filename in safetensor_files:
                    filepath = os.path.join(self.model_path, filename)
                    if os.path.exists(filepath):
                        print(f"  📁 Loading from {filename}...")
                        from safetensors import safe_open
                        with safe_open(filepath, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                # Keep all text weights
                                if key.startswith("model.language_model"):
                                    tensor = f.get_tensor(key)
                                    all_weights[key] = tensor.to(torch.float32)
                                # Also keep the root-level lm_head if it exists
                                elif key == "lm_head.weight":
                                    tensor = f.get_tensor(key)
                                    all_weights[key] = tensor.to(torch.float32)
                        files_found += 1
                
                if files_found > 0:
                    print(f"✅ Loaded weights from {files_found} safetensor files ({len(all_weights)} tensors)")
                    return all_weights
                
                # Fallback to pytorch_model.bin
                pytorch_bin = os.path.join(self.model_path, "pytorch_model.bin")
                if os.path.exists(pytorch_bin):
                    print(f"  📁 Loading from pytorch_model.bin...")
                    return torch.load(pytorch_bin, map_location="cpu")
            
            # Try loading from HuggingFace with safetensors preference
            print(f"Loading model from HuggingFace: {self.model_path}")
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    torch_dtype=torch.float32,
                    use_safetensors=True
                )
                return model.state_dict()
            except Exception as hf_error:
                print(f"HuggingFace loading failed: {hf_error}")
                raise
                
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise
        
    def convert_embeddings(self):
        """Convert embeddings (Part 1) with optional per-layer projections"""
        print("Converting embeddings (Part 1)...")
        
        class EmbeddingModel(nn.Module):
            def __init__(self, config, embed_weights):
                super().__init__()
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
                self.embed_tokens.weight.data = embed_weights
                
            def forward(self, input_ids):
                embeds = self.embed_tokens(input_ids)
                # Reshape for Conv2d [batch, seq, hidden] -> [batch, hidden, seq, 1]
                embeds = embeds.transpose(1, 2).unsqueeze(-1)
                return embeds
        
        # Load weights and filter for text-only mode
        state_dict = self._load_model_weights()
        if self.text_only_mode:
            state_dict = self._filter_text_only_weights(state_dict)
        
        # Use correct Gemma3n embedding weight key
        embed_weights = state_dict.get("model.language_model.embed_tokens.weight")
        if embed_weights is None:
            # Fallback keys for other model formats
            embed_weights = state_dict.get("model.embed_tokens.weight", state_dict.get("embed_tokens.weight"))
        
        if embed_weights is None:
            raise ValueError("Could not find embedding weights in state dict. Available keys: " + 
                           str(list(state_dict.keys())[:10]))
        
        # Note: Per-layer embeddings are handled by the main model's weight loading
        # No need to extract them separately here as they're already integrated
        
        # Create model
        model = EmbeddingModel(self.config, embed_weights)
        model.eval()
        
        # Convert to CoreML (enumerated shapes: single-token + prefill length)
        example_input = torch.randint(0, self.config.vocab_size, (1, 1))
        traced_model = torch.jit.trace(model, example_input)

        input_shape = ct.EnumeratedShapes(
            shapes=[[1, 1], [1, self.batch_size]],
            default=[1, 1],
        )

        inputs = [
            ct.TensorType(
                name="input_ids",
                shape=input_shape,
                dtype=np.int32
            )
        ]
        
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=[ct.TensorType(name="embeddings")],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            convert_to="mlprogram"
        )
        
        # Apply LUT quantization if specified
        if self.lut2:
            mlmodel = self._apply_lut(mlmodel, self.lut2)
        
        # Save model
        output_path = os.path.join(self.output_dir, "gemma3n_embeddings.mlpackage")
        mlmodel.save(output_path)
        print(f"Embeddings saved to {output_path}")
        self._maybe_report_weight_bin_size(output_path)
        
    def convert_ffn(self, chunk_idx: int, total_chunks: int):
        """Convert FFN layers (Part 2) with LAUREL blocks"""
        print(f"Converting FFN chunk {chunk_idx}/{total_chunks} (Part 2)...")
        
        # Enable CoreML mode for conversion
        import anemll.models.gemma3n_model as gemma3n_mod
        original_coreml = anemll.models.gemma3n_model.ENABLE_COREML
        anemll.models.gemma3n_model.ENABLE_COREML = True
        
        try:
            # Calculate which layers belong to this chunk
            start_layer, end_layer = self._get_chunk_range(chunk_idx, total_chunks)
            
            class FFNChunkModel(nn.Module):
                def __init__(self, config, layers, enable_laurel):
                    super().__init__()
                    self.config = config
                    self.enable_laurel = enable_laurel
                    self.layers = nn.ModuleList(layers)
                
                def forward(self, hidden_states, causal_mask):
                    conv2d_input = False
                    if hidden_states.dim() == 4:
                        conv2d_input = True
                        hidden_states = hidden_states.squeeze(-1).transpose(1, 2)

                    for layer in self.layers:
                        if self.enable_laurel:
                            # Full LAUREL block processing (includes attention + MLP + residuals)
                            hidden_states, _ = layer(hidden_states, attention_mask=causal_mask)
                        else:
                            # Standard attention + MLP + residuals (LAUREL disabled)
                            residual = hidden_states
                            attn_input = layer.input_layernorm(hidden_states)
                            attn_output, _ = layer.attention(
                                attn_input,
                                attention_mask=causal_mask,
                                position_ids=None,
                                past_key_value=None,
                            )
                            attn_output = layer.post_attention_layernorm(attn_output)
                            hidden_states = residual + attn_output

                            ffn_input = layer.pre_feedforward_layernorm(hidden_states)
                            ffn_output = layer.ffn(ffn_input)
                            ffn_output = layer.post_feedforward_layernorm(ffn_output)
                            hidden_states = hidden_states + ffn_output

                    if conv2d_input:
                        hidden_states = hidden_states.transpose(1, 2).unsqueeze(-1)
                    return hidden_states
            
            # Load state dict and filter for text-only mode
            state_dict = self._load_model_weights()
            if self.text_only_mode:
                state_dict = self._filter_text_only_weights(state_dict)
        
            # Create layers for this chunk
            layers = []
            for i in range(start_layer, end_layer):
                layer = Gemma3nLaurelBlock(self.config, i)
                
                # Load weights for this layer
                layer_prefix = f"model.layers.{i}."
                layer_state_dict = {}
                
                for key, value in state_dict.items():
                    if key.startswith(layer_prefix):
                        new_key = key[len(layer_prefix):]
                        # Convert linear weights to Conv2d format
                        if any(proj in new_key for proj in ["proj", "low_rank"]):
                            value = value.unsqueeze(-1).unsqueeze(-1)
                        layer_state_dict[new_key] = value
                        
                layer.load_state_dict(layer_state_dict, strict=False)
                layers.append(layer)
            
            # Create chunk model
            model = FFNChunkModel(self.config, layers, self.enable_laurel)
            model.eval()
            
            # Convert to CoreML - use float16 for consistency with KV cache
            example_input = torch.randn(self.batch_size, self.config.hidden_size, self.context_length, 1, dtype=torch.float16)
            example_mask = torch.zeros((self.batch_size, 1, self.context_length, self.context_length), dtype=torch.float16)

            traced_model = torch.jit.trace(model, (example_input, example_mask))

            mlmodel = ct.convert(
                traced_model,
                inputs=[
                    ct.TensorType(
                        name="hidden_states",
                        shape=(self.batch_size, self.config.hidden_size, self.context_length, 1),
                        dtype=np.float16
                    ),
                    ct.TensorType(
                        name="causal_mask",
                        shape=(self.batch_size, 1, self.context_length, self.context_length),
                        dtype=np.float16
                    ),
                    ],
                outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                convert_to="mlprogram"
            )
            
            # Apply LUT quantization if specified
            if self.lut2:
                mlmodel = self._apply_lut(mlmodel, self.lut2)
            
            # Save model
            output_path = os.path.join(
                self.output_dir, 
                f"gemma3n_FFN_chunk_{chunk_idx:02d}of{total_chunks:02d}.mlpackage"
            )
            mlmodel.save(output_path)
            print(f"FFN chunk saved to {output_path}")
            self._maybe_report_weight_bin_size(output_path)
        finally:
            # Restore original ENABLE_COREML value
            anemll.models.gemma3n_model.ENABLE_COREML = original_coreml
        
    def convert_attention_prefill(self):
        """Convert attention layers for prefill (Part 2_prefill)"""
        print("Converting attention for prefill...")
        
        class AttentionPrefillModel(nn.Module):
            def __init__(self, config, attention_layers):
                super().__init__()
                self.config = config
                self.layers = nn.ModuleList(attention_layers)
                
            def forward(self, hidden_states, attention_mask=None):
                all_hidden_states = []
                for layer in self.layers:
                    if hidden_states.dim() == 4:
                        hidden_states = hidden_states.squeeze(-1).transpose(1, 2)
                        hidden_states = layer.input_layernorm(hidden_states)
                        hidden_states, _ = layer.attention(hidden_states, attention_mask)
                        hidden_states = layer.post_attention_layernorm(hidden_states)
                        hidden_states = hidden_states.transpose(1, 2).unsqueeze(-1)
                    else:
                        hidden_states = layer.input_layernorm(hidden_states)
                        hidden_states, _ = layer.attention(hidden_states, attention_mask)
                        hidden_states = layer.post_attention_layernorm(hidden_states)
                    all_hidden_states.append(hidden_states)
                return torch.stack(all_hidden_states, dim=1)
        
        # Load state dict and filter for text-only mode
        state_dict = self._load_model_weights()
        if self.text_only_mode:
            state_dict = self._filter_text_only_weights(state_dict)
        
        # Create attention layers
        attention_layers = []
        for i in range(self.config.num_hidden_layers):
            layer = Gemma3nLaurelBlock(self.config, i)
            
            # Load only attention-related weights
            layer_prefix = f"model.layers.{i}."
            attention_state_dict = {}
            
            for key, value in state_dict.items():
                if key.startswith(layer_prefix) and ("attention" in key or "norm" in key):
                    new_key = key[len(layer_prefix):]
                    if "proj" in new_key:
                        value = value.unsqueeze(-1).unsqueeze(-1)
                    attention_state_dict[new_key] = value
                    
            layer.load_state_dict(attention_state_dict, strict=False)
            attention_layers.append(layer)
        
            # Create model
        model = AttentionPrefillModel(self.config, attention_layers)
        model.eval()
        
        # Convert to CoreML
        example_input = torch.randn(self.batch_size, self.config.hidden_size, self.context_length, 1)
        traced_model = torch.jit.trace(model, example_input)
        
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=(self.batch_size, self.config.hidden_size, self.context_length, 1),
                    dtype=np.float32
                )
                ],
            outputs=[ct.TensorType(name="attention_outputs")],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            convert_to="mlprogram"
        )
        
        # Save model
        output_path = os.path.join(self.output_dir, "gemma3n_attention_prefill.mlpackage")
        mlmodel.save(output_path)
        print(f"Attention prefill saved to {output_path}")
        self._maybe_report_weight_bin_size(output_path)
        
    def convert_lm_head(self, vocab_split_factor=16):
        """Convert LM head (Part 3) with vocabulary splitting for memory efficiency
        
        Note: Returns split logits for memory efficiency. Concatenation and softcapping
        should be handled outside the CoreML model in the inference pipeline.
        """
        print(f"Converting LM head (Part 3) with {vocab_split_factor}-way vocabulary split...")
        
        # Enable CoreML mode for conversion
        import anemll.models.gemma3n_model as gemma3n_mod
        original_coreml = anemll.models.gemma3n_model.ENABLE_COREML
        anemll.models.gemma3n_model.ENABLE_COREML = True
        
        try:
            class LMHeadSplitModel(nn.Module):
                def __init__(self, config, norm_weight, lm_head_weight, split_factor=16):
                    super().__init__()
                    self.norm = Gemma3nRMSNorm(config.hidden_size, config.rms_norm_eps)
                    self.norm.weight.data = norm_weight.float()
                    
                    self.split_factor = split_factor
                    self.vocab_size = config.vocab_size
                    
                    # Calculate split sizes (distribute remainder evenly among first splits)
                    vocab_split = self.vocab_size // split_factor
                    vocab_remainder = self.vocab_size % split_factor
                    self.split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(split_factor)]
                    
                    print(f"📊 Vocabulary split: {self.vocab_size} tokens → {split_factor} heads")
                    print(f"   Split sizes: {self.split_sizes[:4]}...{self.split_sizes[-2:]} (showing first 4 and last 2)")
                    
                    # Create split heads
                    self.lm_heads = nn.ModuleList()
                    splits = torch.split(lm_head_weight, self.split_sizes, dim=0)
                    
                    for i, (split_size, split_weight) in enumerate(zip(self.split_sizes, splits)):
                        head = nn.Conv2d(config.hidden_size, split_size, 1, bias=False)
                        head.weight.data = split_weight.float()
                        self.lm_heads.append(head)
                        if i < 3 or i >= split_factor - 2:  # Show first 3 and last 2
                            print(f"   Head {i+1}: {split_weight.shape} → {split_size} tokens")
                    
                def forward(self, hidden_states):
                    # Apply norm
                    hidden_states = self.norm(hidden_states)
                    
                    # Reshape for Conv2d: (batch, seq, hidden) -> (batch, hidden, seq, 1)
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(-1)
                    
                    # Process each split head and return separate outputs
                    logits_splits = []
                    for i, head in enumerate(self.lm_heads):
                        logits_split = head(hidden_states).squeeze(-1).transpose(1, 2)  # (batch, vocab_split, seq)
                        logits_splits.append(logits_split)
                    
                    # Return tuple of split logits for CoreML
                    return tuple(logits_splits)
        
            # Load weights
            state_dict = self._load_model_weights()
            
            # Get final norm weight (Gemma3n pattern)
            norm_weight = state_dict.get("model.language_model.norm.weight")
            if norm_weight is None:
                # Fallback patterns
                norm_weight = state_dict.get("model.norm.weight", state_dict.get("norm.weight"))
        
            if norm_weight is None:
                raise ValueError("Could not find final norm weights in state dict")
        
            # Get LM head weight (Gemma3n uses tied embeddings)
            lm_head_weight = None
            lm_head_candidates = [
            "lm_head.weight", 
            "model.language_model.lm_head.weight", 
            "model.lm_head.weight"
            ]
        
            for lm_key in lm_head_candidates:
                if lm_key in state_dict:
                    lm_head_weight = state_dict[lm_key]
                    print(f"✅ Found LM head weight: {lm_key}")
                    break
        
            if lm_head_weight is None:
                # Use tied embeddings (common pattern for Gemma3n)
                embed_weight = state_dict.get("model.language_model.embed_tokens.weight")
                if embed_weight is not None:
                    print("🔄 Using tied embeddings as LM head (Gemma3n pattern)")
                    lm_head_weight = embed_weight
                else:
                    raise ValueError("Could not find LM head weights or embeddings for tied weights")
        
            # Reshape for Conv2d format
            lm_head_weight = lm_head_weight.unsqueeze(-1).unsqueeze(-1)
            print(f"📐 LM head weight shape: {lm_head_weight.shape}")
        
            # Create model
            print("🔧 Creating LM head split model...")
            model = LMHeadSplitModel(self.config, norm_weight, lm_head_weight, vocab_split_factor)
            model.eval()
        
            # Convert to CoreML
            print("🔄 Creating example input for tracing...")
            # LM head only processes single token: (batch=1, seq=1, hidden_size)
            example_input = torch.randn(1, 1, self.config.hidden_size)
            print(f"📐 Example input shape: {example_input.shape}")
        
            print("🔄 Tracing model (memory-efficient with split vocab...)") 
            traced_model = torch.jit.trace(model, example_input)
            print("✅ Model tracing completed")
            
            # Create output types for each split
            output_types = []
            for i in range(vocab_split_factor):
                output_types.append(ct.TensorType(name=f"logits_split_{i+1}"))
            
            print(f"🔄 Converting to CoreML with {vocab_split_factor} output heads...") 
            mlmodel = ct.convert(
                traced_model,
                inputs=[
                    ct.TensorType(
                        name="hidden_states",
                        shape=(1, 1, self.config.hidden_size),
                        dtype=np.float16
                    )
                    ],
                outputs=output_types,
                #compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
                convert_to="mlprogram"
            )
            print("✅ CoreML conversion completed")
            
            # Apply LUT quantization if specified
            if self.lut3:
                mlmodel = self._apply_lut(mlmodel, self.lut3)

            # Save model
            output_path = os.path.join(self.output_dir, "gemma3n_lm_head.mlpackage")
            mlmodel.save(output_path)
            print(f"LM head saved to {output_path}")
            self._maybe_report_weight_bin_size(output_path)
        finally:
            # Restore original ENABLE_COREML value
            anemll.models.gemma3n_model.ENABLE_COREML = original_coreml
        
    def copy_tokenizer(self):
        """Copy tokenizer files to output directory"""
        print("Copying tokenizer files...")
        
        tokenizer_files = [
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "tokenizer.model",  # For SentencePiece tokenizers
            ]
        
        for file in tokenizer_files:
            src = os.path.join(self.model_path, file)
            if os.path.exists(src):
                dst = os.path.join(self.output_dir, file)
                shutil.copy2(src, dst)
                print(f"Copied {file}")
                
    def create_meta_config(self):
        """Create meta.yaml configuration file"""
        print("Creating meta.yaml configuration...")
        
        meta_config = {
            "model_type": "gemma3n",
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": self.config.num_key_value_heads,
            "intermediate_size": self.config.intermediate_size,
            "rms_norm_eps": self.config.rms_norm_eps,
            "rope_theta": self.config.rope_theta,
            "max_position_embeddings": self.config.max_position_embeddings,
            "sliding_window": getattr(self.config, "sliding_window", None),
            "final_logit_softcapping": getattr(self.config, "final_logit_softcapping", 30.0),
            "chunk_size": self.chunk_size,
            "enable_laurel": self.enable_laurel,
            "enable_per_layer_embeddings": self.enable_per_layer_embeddings,
            "lut": self.lut2 if self.lut2 == self.lut3 else None,
            "lut2": self.lut2,
            "lut3": self.lut3,
            "lut_per_channel": self.lut_per_channel,
            "lut_scope": self.lut_scope,
            "lut_include": self.lut_include,
            "lut_exclude": self.lut_exclude,
            # Gemma3n specific
            "low_rank_dim": getattr(self.config, "low_rank_dim", 256),
            "activation_topk": getattr(self.config, "activation_topk", None),
            # LM Head splitting for memory efficiency
            "vocab_split_factor": 16,
            "vocab_split_note": "LM head uses 16-way vocabulary splitting. Inference pipeline must concatenate split logits and apply softcapping outside CoreML.",
            # Model paths
            "embeddings_path": "gemma3n_embeddings.mlpackage",
            "lm_head_path": "gemma3n_lm_head.mlpackage",
            "attention_prefill_path": "gemma3n_attention_prefill.mlpackage",
            "infer_init_path": "gemma3n_infer_init.mlpackage",
            "combine_streams_path": "gemma3n_combine_streams.mlpackage",
            "infer_path": "gemma3n_infer.mlpackage",
            "infer_chunks": [
                f"gemma3n_infer_chunk_{i:02d}of{self.chunk_size:02d}.mlpackage"
                for i in range(self.chunk_size)
            ],
            "ffn_chunks": [
                f"gemma3n_FFN_chunk_{i:02d}of{self.chunk_size:02d}.mlpackage"
                for i in range(self.chunk_size)
                ]
        }
        
        # Save as YAML
        import yaml
        meta_path = os.path.join(self.output_dir, "meta.yaml")
        with open(meta_path, 'w') as f:
            yaml.dump(meta_config, f, default_flow_style=False)
        print(f"Meta configuration saved to {meta_path}")

    def convert_infer_init(self):
        """Convert Gemma3n infer init (input_ids -> hidden_states + per_layer_inputs)."""
        print("Converting Gemma3n infer init model...")

        class InferInitWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                inputs_embeds, per_layer_inputs = self.model._compute_inputs_and_per_layer(input_ids)
                hidden_states = self.model._init_hidden_states(inputs_embeds)
                return hidden_states, per_layer_inputs

        model = Gemma3nModel(self.config)
        model.load_weights(self.model_path, config=self.config)
        model.to(device=TEST_DEVICE, dtype=MODEL_DTYPE)
        model.eval()
        # Avoid dynamic branches in CoreML tracing for sliding-window rotation.
        # Rotation-specific models can set this to True before tracing.
        model.config.force_rotation_mode = False

        wrapper = InferInitWrapper(model)
        wrapper.eval()

        sample_input_ids = torch.zeros((1, 1), dtype=torch.int32)
        traced_model = torch.jit.trace(wrapper, (sample_input_ids,))

        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="hidden_states", dtype=np.float16),
                ct.TensorType(name="per_layer_inputs", dtype=np.float16),
            ],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        output_path = os.path.join(self.output_dir, "gemma3n_infer_init.mlpackage")
        mlmodel.save(output_path)
        print(f"Infer init model saved to {output_path}")
        self._maybe_report_weight_bin_size(output_path)

    def convert_combine_streams(self):
        """Convert Gemma3n combine-streams model (4-stream -> single stream)."""
        print("Converting Gemma3n combine-streams model...")

        class CombineStreamsWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, hidden_states):
                return self.model._combine_streams(hidden_states)

        model = Gemma3nModel(self.config)
        model.load_weights(self.model_path, config=self.config)
        model.to(device=TEST_DEVICE, dtype=MODEL_DTYPE)
        model.eval()

        wrapper = CombineStreamsWrapper(model)
        wrapper.eval()

        sample_hidden_states = torch.zeros((4, 1, 1, self.config.hidden_size), dtype=MODEL_DTYPE)
        traced_model = torch.jit.trace(wrapper, (sample_hidden_states,))

        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=(4, 1, 1, self.config.hidden_size),
                    dtype=np.float16,
                )
            ],
            outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        output_path = os.path.join(self.output_dir, "gemma3n_combine_streams.mlpackage")
        mlmodel.save(output_path)
        print(f"Combine-streams model saved to {output_path}")
        self._maybe_report_weight_bin_size(output_path)

    def _convert_infer_chunk(self, chunk_idx: int, total_chunks: int, rotate: bool) -> None:
        """Convert a stateful single-token infer model with KV cache (chunked)."""
        label = "infer_rotate" if rotate else "infer"
        print(f"Converting stateful Gemma3n {label} chunk {chunk_idx}/{total_chunks}...")

        start_layer, end_layer = self._get_chunk_range(chunk_idx, total_chunks)
        print(f"  Layers: {start_layer}..{end_layer-1}")

        class Gemma3nInferChunkWrapper(nn.Module):
            def __init__(self, model, start_layer, end_layer):
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer

            def forward(self, hidden_states, per_layer_inputs, causal_mask, current_pos, position_one_hot, rotary_cos_local, rotary_sin_local, rotary_cos_global, rotary_sin_global):
                # position_one_hot and rotary embeddings must be passed as inputs for CoreML
                # because comparisons and tensor indexing with dynamic values become constants during conversion
                hidden_states = self.model.process_layers(
                    hidden_states,
                    per_layer_inputs,
                    causal_mask,
                    current_pos,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    prefill=False,
                    position_one_hot=position_one_hot,
                    rotary_cos_local=rotary_cos_local,
                    rotary_sin_local=rotary_sin_local,
                    rotary_cos_global=rotary_cos_global,
                    rotary_sin_global=rotary_sin_global,
                )
                return hidden_states

        model = Gemma3nModel(self.config)
        model.load_weights(self.model_path, config=self.config)
        model.to(device=TEST_DEVICE, dtype=MODEL_DTYPE)
        model.eval()
        model.config.force_rotation_mode = True if rotate else False

        wrapper = Gemma3nInferChunkWrapper(model, start_layer, end_layer)
        wrapper.eval()

        sample_hidden_states = torch.zeros((4, 1, 1, self.config.hidden_size), dtype=MODEL_DTYPE)
        sample_per_layer_inputs = torch.zeros(
            (1, 1, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input),
            dtype=MODEL_DTYPE,
        )
        sample_causal_mask = torch.zeros(
            (1, 1, self.context_length, self.context_length), dtype=torch.float16
        )
        sample_current_pos = torch.zeros((1,), dtype=torch.int32)
        # Position one-hot: [1, 1, seq_len, 1] with 1.0 at current_pos
        sample_position_one_hot = torch.zeros((1, 1, self.context_length, 1), dtype=torch.float16)
        sample_position_one_hot[0, 0, 0, 0] = 1.0  # Sample with position 0
        # Rotary embeddings: [1, 1, head_dim] for both local and global
        head_dim = self.config.head_dim
        sample_rotary_cos_local = torch.zeros((1, 1, head_dim), dtype=torch.float16)
        sample_rotary_sin_local = torch.zeros((1, 1, head_dim), dtype=torch.float16)
        sample_rotary_cos_global = torch.zeros((1, 1, head_dim), dtype=torch.float16)
        sample_rotary_sin_global = torch.zeros((1, 1, head_dim), dtype=torch.float16)

        traced_model = torch.jit.trace(
            wrapper,
            (sample_hidden_states, sample_per_layer_inputs, sample_causal_mask, sample_current_pos,
             sample_position_one_hot, sample_rotary_cos_local, sample_rotary_sin_local,
             sample_rotary_cos_global, sample_rotary_sin_global),
        )

        outputs = [
            ct.TensorType(name="output_hidden_states", dtype=np.float16)
        ]

        states = self._get_kv_cache_states(prefix="model.")

        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=(4, 1, 1, self.config.hidden_size),
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="per_layer_inputs",
                    shape=(1, 1, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input),
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="causal_mask",
                    shape=(1, 1, self.context_length, self.context_length),
                    dtype=np.float16,
                ),
                ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
                ct.TensorType(
                    name="position_one_hot",
                    shape=(1, 1, self.context_length, 1),
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="rotary_cos_local",
                    shape=(1, 1, head_dim),
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="rotary_sin_local",
                    shape=(1, 1, head_dim),
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="rotary_cos_global",
                    shape=(1, 1, head_dim),
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="rotary_sin_global",
                    shape=(1, 1, head_dim),
                    dtype=np.float16,
                ),
            ],
            outputs=outputs,
            states=states,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        # Apply LUT quantization if specified (same as FFN)
        if self.lut2:
            mlmodel = self._apply_lut(mlmodel, self.lut2)

        if total_chunks > 1:
            output_path = os.path.join(
                self.output_dir,
                f"gemma3n_{label}_chunk_{chunk_idx:02d}of{total_chunks:02d}.mlpackage",
            )
        else:
            output_path = os.path.join(self.output_dir, f"gemma3n_{label}.mlpackage")
        mlmodel.save(output_path)
        print(f"Infer model saved to {output_path}")
        self._maybe_report_weight_bin_size(output_path)

    def convert_infer(self, chunk_idx: int = 0, total_chunks: int = 1):
        self._convert_infer_chunk(chunk_idx, total_chunks, rotate=False)

    def convert_infer_rotate(self, chunk_idx: int = 0, total_chunks: int = 1):
        self._convert_infer_chunk(chunk_idx, total_chunks, rotate=True)

    def _convert_prefill_chunk(self, chunk_idx: int, total_chunks: int, rotate: bool) -> None:
        """Convert a stateful prefill model (chunked)."""
        label = "prefill_rotate" if rotate else "prefill"
        print(f"Converting Gemma3n {label} chunk {chunk_idx}/{total_chunks}...")

        start_layer, end_layer = self._get_chunk_range(chunk_idx, total_chunks)
        print(f"  Layers: {start_layer}..{end_layer-1}")

        class Gemma3nPrefillChunkWrapper(nn.Module):
            def __init__(self, model, start_layer, end_layer):
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer

            def forward(self, hidden_states, per_layer_inputs, causal_mask, current_pos, position_ids, rotary_cos_local, rotary_sin_local, rotary_cos_global, rotary_sin_global):
                hidden_states = self.model.process_layers(
                    hidden_states,
                    per_layer_inputs,
                    causal_mask,
                    current_pos,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    prefill=True,
                    position_ids=position_ids,
                    rotary_cos_local=rotary_cos_local,
                    rotary_sin_local=rotary_sin_local,
                    rotary_cos_global=rotary_cos_global,
                    rotary_sin_global=rotary_sin_global,
                )
                return hidden_states

        model = Gemma3nModel(self.config)
        model.load_weights(self.model_path, config=self.config)
        model.to(device=TEST_DEVICE, dtype=MODEL_DTYPE)
        model.eval()
        model.config.force_rotation_mode = True if rotate else False

        wrapper = Gemma3nPrefillChunkWrapper(model, start_layer, end_layer)
        wrapper.eval()

        prefill_len = max(1, int(self.batch_size))
        sample_hidden_states = torch.zeros((4, 1, prefill_len, self.config.hidden_size), dtype=MODEL_DTYPE)
        sample_per_layer_inputs = torch.zeros(
            (1, prefill_len, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input),
            dtype=MODEL_DTYPE,
        )
        sample_causal_mask = torch.zeros((1, 1, self.context_length, self.context_length), dtype=torch.float16)
        sample_current_pos = torch.zeros((1,), dtype=torch.int32)
        sample_position_ids = torch.arange(prefill_len, dtype=torch.int32).view(1, -1)
        head_dim = self.config.head_dim
        sample_rotary_cos_local = torch.zeros((1, prefill_len, head_dim), dtype=torch.float16)
        sample_rotary_sin_local = torch.zeros((1, prefill_len, head_dim), dtype=torch.float16)
        sample_rotary_cos_global = torch.zeros((1, prefill_len, head_dim), dtype=torch.float16)
        sample_rotary_sin_global = torch.zeros((1, prefill_len, head_dim), dtype=torch.float16)

        traced_model = torch.jit.trace(
            wrapper,
            (
                sample_hidden_states,
                sample_per_layer_inputs,
                sample_causal_mask,
                sample_current_pos,
                sample_position_ids,
                sample_rotary_cos_local,
                sample_rotary_sin_local,
                sample_rotary_cos_global,
                sample_rotary_sin_global,
            ),
        )

        outputs = [ct.TensorType(name="output_hidden_states", dtype=np.float16)]
        states = self._get_kv_cache_states(prefix="model.")

        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=(4, 1, prefill_len, self.config.hidden_size),
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="per_layer_inputs",
                    shape=(1, prefill_len, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input),
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="causal_mask",
                    shape=(1, 1, self.context_length, self.context_length),
                    dtype=np.float16,
                ),
                ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
                ct.TensorType(
                    name="position_ids",
                    shape=(1, prefill_len),
                    dtype=np.int32,
                ),
                ct.TensorType(
                    name="rotary_cos_local",
                    shape=(1, prefill_len, head_dim),
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="rotary_sin_local",
                    shape=(1, prefill_len, head_dim),
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="rotary_cos_global",
                    shape=(1, prefill_len, head_dim),
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="rotary_sin_global",
                    shape=(1, prefill_len, head_dim),
                    dtype=np.float16,
                ),
            ],
            outputs=outputs,
            states=states,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        if self.lut2:
            mlmodel = self._apply_lut(mlmodel, self.lut2)

        if total_chunks > 1:
            output_path = os.path.join(
                self.output_dir,
                f"gemma3n_{label}_chunk_{chunk_idx:02d}of{total_chunks:02d}.mlpackage",
            )
        else:
            output_path = os.path.join(self.output_dir, f"gemma3n_{label}.mlpackage")
        mlmodel.save(output_path)
        print(f"Prefill model saved to {output_path}")
        self._maybe_report_weight_bin_size(output_path)

    def convert_prefill(self, chunk_idx: int = 0, total_chunks: int = 1):
        self._convert_prefill_chunk(chunk_idx, total_chunks, rotate=False)

    def convert_prefill_rotate(self, chunk_idx: int = 0, total_chunks: int = 1):
        self._convert_prefill_chunk(chunk_idx, total_chunks, rotate=True)
        
    def convert(self):
        """Run full conversion pipeline"""
        print(f"Starting Gemma3n conversion from {self.model_path} to {self.output_dir}")
        
        # Step 1: Convert embeddings
        self.convert_embeddings()
        
        # Step 2: Convert LM head
        self.convert_lm_head()
        
        # Step 3: Convert FFN chunks
        for chunk_idx in range(self.chunk_size):
            self.convert_ffn(chunk_idx, self.chunk_size)
            
        # Step 4: Convert attention prefill
        self.convert_attention_prefill()

        # Step 5: Convert infer init (input ids -> hidden states + per-layer inputs)
        self.convert_infer_init()

        # Step 6: Convert stateful infer (KV cache) chunks
        for chunk_idx in range(self.chunk_size):
            self.convert_infer(chunk_idx, self.chunk_size)

        # Step 7: Convert combine-streams model
        self.convert_combine_streams()

        # Step 8: Copy tokenizer
        self.copy_tokenizer()

        # Step 9: Create meta configuration
        self.create_meta_config()
        
        print("Conversion complete!")
        print(f"Models saved to: {self.output_dir}")
        print("\nTo test the converted model, run:")
        print(f"python tests/chat.py --meta {os.path.join(self.output_dir, 'meta.yaml')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Gemma3n model to ANEMLL format")
    parser.add_argument("--model", required=True, help="Path to HuggingFace Gemma3n model")
    parser.add_argument("--output", required=True, help="Output directory for converted models")
    parser.add_argument("--context", type=int, default=512, help="Context length")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--lut2", type=int, help="LUT quantization for Part 2")
    parser.add_argument("--lut3", type=int, help="LUT quantization for Part 3")
    parser.add_argument("--chunk", type=int, default=4, help="Number of FFN chunks")
    parser.add_argument("--disable-laurel", action="store_true", help="Disable LAUREL blocks")
    parser.add_argument("--disable-per-layer-embeddings", action="store_true", help="Disable per-layer embeddings")
    parser.add_argument("--enable-multimodal", action="store_true", help="Enable multimodal weights (default: text-only mode)")
    parser.add_argument("--disable-sparsity", action="store_true", help="Disable activation sparsity (conversion-friendly)")
    
    args = parser.parse_args()
    
    converter = Gemma3nConverter(
        model_path=args.model,
        output_dir=args.output,
        context_length=args.context,
        batch_size=args.batch,
        lut2=args.lut2,
        lut3=args.lut3,
        chunk_size=args.chunk,
        enable_laurel=not args.disable_laurel,
        enable_per_layer_embeddings=not args.disable_per_layer_embeddings,
        text_only_mode=not args.enable_multimodal,
        disable_sparsity=args.disable_sparsity,
    )
    
    converter.convert()
