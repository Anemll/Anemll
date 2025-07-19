# Copyright (c) 2025 ANEMLL
# Licensed under the MIT License
# 
# Gemma3n model converter for Apple Neural Engine acceleration
# Converts Google's Gemma3n models to CoreML format with ANE optimizations

import os
import torch
import torch.nn as nn
import coremltools as ct
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
    Gemma3nAttention, Gemma3nFFN, Gemma3nRMSNorm, Gemma3nConfig
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
        chunk_size: int = 2,
        enable_laurel: bool = True,
        enable_per_layer_embeddings: bool = True,
        text_only_mode: bool = True
    ):
        super().__init__(model_path)
        self.model_type = "gemma3n"
        self.model_path = model_path
        self.output_dir = output_dir
        self.context_length = context_length
        self.batch_size = batch_size
        self.lut2 = lut2
        self.lut3 = lut3
        self.chunk_size = chunk_size
        self.enable_laurel = enable_laurel
        self.enable_per_layer_embeddings = enable_per_layer_embeddings
        self.text_only_mode = text_only_mode
        
        # Load config
        hf_config = AutoConfig.from_pretrained(model_path)
        self.config = Gemma3nConfig.from_pretrained_config(hf_config)
        
        # Update config with conversion parameters
        self.config.context_length = context_length
        self.config.state_length = max(context_length, self.config.state_length)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        # Convert to CoreML
        example_input = torch.randint(0, self.config.vocab_size, (self.batch_size, self.context_length))
        traced_model = torch.jit.trace(model, example_input)
        
        inputs = [
            ct.TensorType(
                name="input_ids",
                shape=(self.batch_size, self.context_length),
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
            from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig
            config = OptimizationConfig(
                global_config=OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    weight_threshold=self.lut2
                )
            )
            mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config)
        
        # Save model
        output_path = os.path.join(self.output_dir, "gemma3n_embeddings.mlpackage")
        mlmodel.save(output_path)
        print(f"Embeddings saved to {output_path}")
        
    def convert_ffn(self, chunk_idx: int, total_chunks: int):
        """Convert FFN layers (Part 2) with LAUREL blocks"""
        print(f"Converting FFN chunk {chunk_idx}/{total_chunks} (Part 2)...")
        
        # Enable CoreML mode for conversion
        import anemll.models.gemma3n_model as gemma3n_mod
        original_coreml = anemll.models.gemma3n_model.ENABLE_COREML
        anemll.models.gemma3n_model.ENABLE_COREML = True
        
        try:
            # Calculate which layers belong to this chunk
            layers_per_chunk = self.config.num_hidden_layers // total_chunks
            start_layer = chunk_idx * layers_per_chunk
            end_layer = start_layer + layers_per_chunk if chunk_idx < total_chunks - 1 else self.config.num_hidden_layers
            
            class FFNChunkModel(nn.Module):
                def __init__(self, config, layers, enable_laurel):
                    super().__init__()
                    self.config = config
                    self.enable_laurel = enable_laurel
                    self.layers = nn.ModuleList(layers)
                
                def forward(self, hidden_states):
                    for layer in self.layers:
                        if self.enable_laurel:
                            # Full LAUREL block processing
                            hidden_states, _ = layer(hidden_states)
                        else:
                            # Just FFN processing
                            residual = hidden_states
                            hidden_states = layer.post_attention_layernorm(hidden_states)
                            hidden_states = layer.ffn(hidden_states)
                            hidden_states = residual + hidden_states
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
            
            # Convert to CoreML
            #example_input = torch.randn(self.batch_size, self.config.hidden_size, self.context_length, 1)
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
                outputs=[ct.TensorType(name="output_hidden_states")],
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                convert_to="mlprogram"
            )
            
            # Apply LUT quantization if specified
            if self.lut2:
                from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig
                config = OptimizationConfig(
                    global_config=OpLinearQuantizerConfig(
                        mode="linear_symmetric",
                        weight_threshold=self.lut2
                    )
                )
                mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config)
            
            # Save model
            output_path = os.path.join(
                self.output_dir, 
                f"gemma3n_FFN_chunk_{chunk_idx:02d}of{total_chunks:02d}.mlpackage"
            )
            mlmodel.save(output_path)
            print(f"FFN chunk saved to {output_path}")
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
                    hidden_states = layer.attention_norm(hidden_states)
                    hidden_states, _ = layer.attention(hidden_states, attention_mask)
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
                from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig
                config = OptimizationConfig(
                    global_config=OpLinearQuantizerConfig(
                        mode="linear_symmetric",
                        weight_threshold=self.lut3
                    )
                )
                mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config)
            
                # Save model
                output_path = os.path.join(self.output_dir, "gemma3n_lm_head.mlpackage")
                mlmodel.save(output_path)
                print(f"LM head saved to {output_path}")
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
            "lut2": self.lut2,
            "lut3": self.lut3,
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
        
        # Step 5: Copy tokenizer
        self.copy_tokenizer()
        
        # Step 6: Create meta configuration
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
    parser.add_argument("--chunk", type=int, default=2, help="Number of FFN chunks")
    parser.add_argument("--disable-laurel", action="store_true", help="Disable LAUREL blocks")
    parser.add_argument("--disable-per-layer-embeddings", action="store_true", help="Disable per-layer embeddings")
    parser.add_argument("--enable-multimodal", action="store_true", help="Enable multimodal weights (default: text-only mode)")
    
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
        text_only_mode=not args.enable_multimodal
    )
    
    converter.convert()