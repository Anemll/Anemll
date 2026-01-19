#!/usr/bin/env python3
"""
V2 Trace-Based CoreML Conversion

Converts ANEMLL V2 checkpoints using factored computation:
  y = Σₖ gₖ · (aₖ ⊙ (Q @ (bₖ ⊙ x)))

Key differences from V1:
- Q (snapped weights) stored directly in Conv2d weight (NOT baked with scales)
- Factored per-rank computation preserves structure
- No folding bypass patches needed
- Post-conversion palettization compresses Q weights

LoRA Support:
- Auto-detects LoRA weights in checkpoint (lora_A, lora_B keys)
- LoRA adds low-rank update: y = quant_forward(x) + (x @ lora_A.T @ lora_B.T) * scaling
- Supports MLP-only, MLP+attention, or no LoRA via config/flags
- Use --no-lora to disable even if present in checkpoint

Supports both prefill and decoder modes.

Usage:
    python tests/dev/convert_v2_trace.py \
        --checkpoint /path/to/model_snapped.pt \
        --model Qwen/Qwen3-0.6B \
        --context 512 \
        --output /path/to/output \
        --mode both \
        --palettize 4

    # With LoRA (auto-detected)
    python tests/dev/convert_v2_trace.py \
        --checkpoint /path/to/checkpoint_with_lora.pt \
        --model Qwen/Qwen3-0.6B \
        --output /path/to/output

    # Disable LoRA
    python tests/dev/convert_v2_trace.py \
        --checkpoint /path/to/checkpoint_with_lora.pt \
        --no-lora \
        --output /path/to/output
"""

import os
import sys
import argparse
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class FactoredConv2d(nn.Module):
    """Conv2d with V2 factored scale computation and optional LoRA.

    Implements: y = Σₖ gₖ · (aₖ ⊙ (Q @ (bₖ ⊙ x))) + LoRA(x)

    Where:
    - Q: snapped weights [out, in] stored in weight buffer
    - scale_A: [out, rank] direction vectors
    - scale_B: [rank, in] direction vectors
    - rank_magnitude: [rank] per-rank magnitudes (gₖ)

    LoRA (optional):
    - lora_A: [lora_r, in] input projection
    - lora_B: [out, lora_r] output projection
    - lora_scaling: alpha / lora_r
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 lora_r: int = 0, lora_alpha: float = None, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Q: snapped weights [out, in, 1, 1] - will be palettized
        self.register_buffer('weight', torch.zeros(out_features, in_features, 1, 1, dtype=dtype))

        # Factored scales
        self.register_buffer('scale_A', torch.zeros(out_features, rank, dtype=dtype))
        self.register_buffer('scale_B', torch.zeros(rank, in_features, dtype=dtype))
        self.register_buffer('rank_magnitude', torch.ones(rank, dtype=dtype))

        # LoRA parameters (optional)
        self.lora_r = lora_r
        if lora_r > 0:
            if lora_alpha is None:
                lora_alpha = float(lora_r)
            self.lora_scaling = lora_alpha / lora_r
            self.register_buffer('lora_A', torch.zeros(lora_r, in_features, dtype=dtype))
            self.register_buffer('lora_B', torch.zeros(out_features, lora_r, dtype=dtype))
        else:
            self.lora_scaling = 0.0
            self.register_buffer('lora_A', None)
            self.register_buffer('lora_B', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """V2 factored forward pass.

        Args:
            x: Input [batch, channels, height, width] in Conv2d format
               or [batch, seq, features] in Linear format

        Returns:
            Output with same batch dimensions
        """
        # Get Q without spatial dims
        Q = self.weight.squeeze(-1).squeeze(-1)  # [out, in]

        # Handle Conv2d input format [B, C, H, W] -> treat as [B*H, W, C]
        if x.dim() == 4:
            B, C, H, W = x.shape
            # Reshape to [B, H*W, C] for factored computation
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            y_flat = self._factored_forward(x_flat, Q)
            # Reshape back to [B, out, H, W]
            y = y_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            return y
        else:
            # Already in [batch, seq, features] format
            return self._factored_forward(x, Q)

    def _factored_forward(self, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """Core factored computation: y = Σₖ gₖ · (aₖ ⊙ (Q @ (bₖ ⊙ x)))

        Vectorized implementation (no Python loops) for CoreML compatibility.

        Args:
            x: [batch, seq, in_features]
            Q: [out_features, in_features]

        Returns:
            y: [batch, seq, out_features]
        """
        # Vectorized computation to avoid Python loops (CoreML can't convert loops)
        # x: [batch, seq, in]
        # B: [rank, in]
        # A: [out, rank]
        # g: [rank]
        # Q: [out, in]

        # Step 1: Scale x by all B rows simultaneously
        # x_expanded: [batch, seq, 1, in]
        # B: [rank, in] -> [1, 1, rank, in]
        # x_scaled: [batch, seq, rank, in]
        x_expanded = x.unsqueeze(-2)  # [batch, seq, 1, in]
        B_expanded = self.scale_B.unsqueeze(0).unsqueeze(0)  # [1, 1, rank, in]
        x_scaled = x_expanded * B_expanded  # [batch, seq, rank, in]

        # Step 2: Apply Q to each rank's scaled input
        # x_scaled @ Q.T: [batch, seq, rank, in] @ [in, out] -> [batch, seq, rank, out]
        y_all = x_scaled @ Q.T  # [batch, seq, rank, out]

        # Step 3: Scale by A and g (combined)
        # A: [out, rank], g: [rank]
        # A * g: [out, rank] (broadcast g across out dimension)
        # Transpose to [rank, out] for proper broadcasting with y_all
        A_g = (self.scale_A * self.rank_magnitude.unsqueeze(0)).T  # [rank, out]
        A_g = A_g.unsqueeze(0).unsqueeze(0)  # [1, 1, rank, out]

        y_scaled = y_all * A_g  # [batch, seq, rank, out]

        # Step 4: Sum over rank dimension
        y = y_scaled.sum(dim=-2)  # [batch, seq, out]

        # Step 5: Add LoRA contribution if enabled
        # y += (x @ lora_A.T @ lora_B.T) * scaling
        if self.lora_r > 0 and self.lora_A is not None:
            # x: [batch, seq, in]
            # lora_A: [lora_r, in] -> x @ lora_A.T: [batch, seq, lora_r]
            # lora_B: [out, lora_r] -> hidden @ lora_B.T: [batch, seq, out]
            hidden = x @ self.lora_A.T  # [batch, seq, lora_r]
            y = y + (hidden @ self.lora_B.T) * self.lora_scaling

        return y


def load_v2_factored_weights(model: nn.Module, checkpoint_path: str,
                            enable_lora: bool = True, verbose: bool = True) -> dict:
    """Load V2 checkpoint into FactoredConv2d layers AND other model weights.

    Uses _Q buffer directly (already contains lut[_indices] with 16 unique values).
    Also loads embeddings, norms, and other non-projection weights.

    V2 checkpoint structure:
    - _Q: [out, in] - snapped weights (16 unique values from lut[_indices])
    - scale_A: [out, rank] - output directions (with rank_magnitude baked in)
    - scale_B: [rank, in] - input directions
    - rank_magnitude: [rank] - per-rank magnitudes (all 1s after snap, baked into scale_A)

    LoRA (optional):
    - lora_A: [lora_r, in] - input projection
    - lora_B: [out, lora_r] - output projection

    Args:
        model: Model with FactoredConv2d layers
        checkpoint_path: Path to V2 checkpoint
        enable_lora: Whether to load LoRA weights if present
        verbose: Print loading info

    Returns:
        Dict with loading info: {'layers': int, 'lora_layers': int, 'has_lora': bool}
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Verify V2 format (has _Q buffers)
    has_Q = any('._Q' in k for k in state_dict.keys())
    has_scale_A = any('.scale_A' in k for k in state_dict.keys())

    # Detect LoRA
    lora_keys = [k for k in state_dict.keys() if 'lora_' in k]
    has_lora = len(lora_keys) > 0

    if not (has_Q and has_scale_A):
        raise ValueError("Checkpoint is not V2 format (missing _Q or scale_A)")

    if verbose:
        print(f"Loading V2 checkpoint (using _Q directly): {len(state_dict)} keys")
        if has_lora:
            print(f"  Detected LoRA weights: {len(lora_keys)} tensors")
            if enable_lora:
                print(f"  LoRA: ENABLED (will be included in model)")
            else:
                print(f"  LoRA: DISABLED via --no-lora (will be skipped)")
        else:
            print(f"  No LoRA weights detected in checkpoint")

    # First, load non-projection weights (embeddings, norms, etc.)
    proj_patterns = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    other_loaded = 0

    model_state = model.state_dict()
    for key, value in state_dict.items():
        # Skip projection-related keys (handled separately)
        if any(p in key for p in proj_patterns):
            continue
        # Skip quantization-specific keys
        if '._indices' in key or '.lut' in key or '.scale_A' in key or '.scale_B' in key or '.rank_magnitude' in key:
            continue

        # Try to load this key into the model
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key].copy_(value.to(model_state[key].dtype))
                other_loaded += 1
            elif verbose:
                print(f"  Shape mismatch for {key}: model={model_state[key].shape}, ckpt={value.shape}")
        # Try with 'model.' prefix
        elif f'model.{key}' in model_state:
            full_key = f'model.{key}'
            if model_state[full_key].shape == value.shape:
                model_state[full_key].copy_(value.to(model_state[full_key].dtype))
                other_loaded += 1

    # Load state dict back
    model.load_state_dict(model_state, strict=False)

    if verbose:
        print(f"  Loaded {other_loaded} non-projection weights (embeddings, norms, etc.)")

    proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    loaded = 0
    lora_loaded = 0

    for name, module in model.named_modules():
        if not isinstance(module, FactoredConv2d):
            continue
        if not any(proj in name for proj in proj_names):
            continue

        # Try key formats
        for base_key in [name, name.replace('model.model.', 'model.'),
                        name.replace('model.', ''), f'model.{name}']:
            q_key = f'{base_key}._Q'
            a_key = f'{base_key}.scale_A'
            b_key = f'{base_key}.scale_B'
            g_key = f'{base_key}.rank_magnitude'

            if q_key in state_dict:
                # Use _Q directly - already contains lut[_indices] with 16 unique values
                Q = state_dict[q_key].to(torch.float16)
                unique_count = len(Q.unique())

                # Store Q in weight [out, in] -> [out, in, 1, 1]
                module.weight.data.copy_(Q.view(Q.shape[0], Q.shape[1], 1, 1))

                if a_key in state_dict:
                    module.scale_A.data.copy_(state_dict[a_key].to(torch.float16))
                if b_key in state_dict:
                    module.scale_B.data.copy_(state_dict[b_key].to(torch.float16))
                if g_key in state_dict:
                    module.rank_magnitude.data.copy_(state_dict[g_key].to(torch.float16))

                # Load LoRA weights if present and enabled
                lora_a_key = f'{base_key}.lora_A'
                lora_b_key = f'{base_key}.lora_B'
                has_layer_lora = lora_a_key in state_dict and lora_b_key in state_dict

                if has_layer_lora and enable_lora and has_lora:
                    lora_A = state_dict[lora_a_key].to(torch.float16)
                    lora_B = state_dict[lora_b_key].to(torch.float16)

                    # Check if module has LoRA enabled (lora_r > 0)
                    if module.lora_r > 0 and module.lora_A is not None:
                        module.lora_A.data.copy_(lora_A)
                        module.lora_B.data.copy_(lora_B)
                        lora_loaded += 1

                loaded += 1
                if verbose and loaded <= 5:
                    lora_info = f", LoRA r={module.lora_r}" if (has_layer_lora and enable_lora and module.lora_r > 0) else ""
                    print(f"  {base_key}: _Q{list(Q.shape)}, {unique_count} unique values{lora_info}")
                break

    if verbose:
        if loaded > 5:
            print(f"  ...and {loaded - 5} more")
        print(f"  Total: {loaded} factored layers")
        if lora_loaded > 0:
            print(f"  LoRA: {lora_loaded} layers loaded")

    return {
        'layers': loaded,
        'lora_layers': lora_loaded,
        'has_lora': has_lora and enable_lora,
        'state_dict': state_dict,
    }


def replace_conv_with_factored(model: nn.Module, checkpoint_path: str,
                               enable_lora: bool = True, verbose: bool = True) -> dict:
    """Replace Conv2d layers with FactoredConv2d using V2 checkpoint info.

    Detects rank and LoRA from checkpoint and creates appropriate FactoredConv2d modules.

    LoRA modes (from config.json):
    - mlp_only=True: Only add LoRA to MLP layers (gate/up/down_proj)
    - mlp_only=False: Add LoRA to both MLP and attention layers
    - skip_k_proj=True: Skip LoRA on k_proj layers

    Returns:
        Dict with: {'replaced': int, 'lora_layers': int, 'lora_r': int}
    """
    import json

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load config for LoRA settings
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

    # LoRA config
    mlp_only = config.get('mlp_only', False)
    skip_k_proj = config.get('skip_k_proj', True)
    recovery_r = config.get('recovery_r', 0)  # LoRA rank from config

    proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    attn_proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']  # For mlp_only check

    # Build replacement map
    replacements = []
    lora_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.kernel_size != (1, 1):
            continue
        if not any(proj in name for proj in proj_names):
            continue

        # Find checkpoint key and get rank
        for base_key in [name, name.replace('model.model.', 'model.'),
                        name.replace('model.', ''), f'model.{name}']:
            a_key = f'{base_key}.scale_A'
            if a_key in state_dict:
                rank = state_dict[a_key].shape[1]

                # Determine LoRA rank for this layer
                lora_r = 0
                if enable_lora:
                    # Check if LoRA weights exist for this layer
                    lora_a_key = f'{base_key}.lora_A'
                    if lora_a_key in state_dict:
                        # Determine if we should enable LoRA based on config
                        is_attn = any(p in name for p in attn_proj_names)
                        is_k_proj = 'k_proj' in name

                        # Apply LoRA mode rules
                        should_enable = True
                        if mlp_only and is_attn:
                            should_enable = False  # Skip attention if mlp_only
                        if skip_k_proj and is_k_proj:
                            should_enable = False  # Skip k_proj if configured

                        if should_enable:
                            lora_r = state_dict[lora_a_key].shape[0]  # [lora_r, in]
                            lora_count += 1

                replacements.append((name, module, rank, lora_r))
                break

    # Replace modules
    for name, old_module, rank, lora_r in replacements:
        new_module = FactoredConv2d(
            in_features=old_module.in_channels,
            out_features=old_module.out_channels,
            rank=rank,
            lora_r=lora_r,
            dtype=old_module.weight.dtype,
        )

        # Navigate to parent and replace
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    if verbose:
        print(f"Replaced {len(replacements)} Conv2d -> FactoredConv2d")
        if lora_count > 0:
            print(f"  LoRA enabled on {lora_count} layers (mlp_only={mlp_only}, skip_k_proj={skip_k_proj})")

    return {
        'replaced': len(replacements),
        'lora_layers': lora_count,
        'lora_r': recovery_r,
        'mlp_only': mlp_only,
        'skip_k_proj': skip_k_proj,
    }


def convert_decoder(model, context_length: int, batch_size: int = 64,
                   chunk_idx: int = 0, total_chunks: int = 1):
    """Convert model for decoder mode using factored computation."""
    import coremltools as ct
    from anemll.models.qwen_model import TEST_DEVICE
    from anemll.ane_converter.qwen_converter import QwenConverter

    total_layers = model.config.num_hidden_layers
    if total_chunks > 1:
        layers_per_chunk = total_layers // total_chunks
        start_layer = chunk_idx * layers_per_chunk
        end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
    else:
        start_layer, end_layer = 0, None

    class DecoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, hidden_states, position_ids, causal_mask, current_pos):
            rotary = self.model.model.get_rotary_embeddings_s(current_pos)
            out = self.model.model.process_layers(
                hidden_states, position_ids, causal_mask, current_pos, rotary,
                start_layer=start_layer, end_layer=end_layer, IN_PREFILL=False)
            return self.model.model.norm(out)

    wrapper = DecoderWrapper(model)
    wrapper.eval()

    # Sample inputs
    hs = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    mask = torch.zeros((1, 1, 1, context_length), dtype=torch.float16, device=TEST_DEVICE)
    cur = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

    print("Tracing decoder (V2 factored)...")
    traced = torch.jit.trace(wrapper, (hs, pos, mask, cur))

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hs.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=pos.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=cur.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=QwenConverter.GetTransformerStates(model, part=None, prefix="model.model."),
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    print("Conversion complete!")
    return mlmodel


def convert_prefill(model, context_length: int, batch_size: int = 64,
                   chunk_idx: int = 0, total_chunks: int = 1):
    """Convert model for prefill mode using factored computation."""
    import coremltools as ct
    from anemll.models.qwen_model import TEST_DEVICE
    from anemll.ane_converter.qwen_converter import QwenConverter

    total_layers = model.config.num_hidden_layers
    if total_chunks > 1:
        layers_per_chunk = total_layers // total_chunks
        start_layer = chunk_idx * layers_per_chunk
        end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
    else:
        start_layer, end_layer = 0, None

    class PrefillWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, hidden_states, position_ids, causal_mask, current_pos):
            rotary = self.model.model.get_rotary_embedding_prefill(position_ids)
            out = self.model.model.process_layers(
                hidden_states, position_ids, causal_mask, current_pos, rotary,
                start_layer=start_layer, end_layer=end_layer, IN_PREFILL=True)
            return self.model.model.norm(out)

    wrapper = PrefillWrapper(model)
    wrapper.eval()

    # Sample inputs for prefill
    hs = torch.zeros((1, batch_size, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    pos = torch.arange(batch_size, dtype=torch.int32, device=TEST_DEVICE)
    mask = torch.zeros((1, 1, batch_size, context_length), dtype=torch.float16, device=TEST_DEVICE)
    cur = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

    print("Tracing prefill (V2 factored)...")
    traced = torch.jit.trace(wrapper, (hs, pos, mask, cur))

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hs.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=pos.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=cur.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=QwenConverter.GetTransformerStates(model, part=None, prefix="model.model."),
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    print("Conversion complete!")
    return mlmodel


def apply_palettization(mlmodel, mlp_bits: int = 4, verbose: bool = True):
    """Apply palettization using mode="unique" which auto-skips incompatible weights.

    Since Q weights are already snapped to LUT values with exactly 2^bits unique values,
    mode="unique" will palettize them correctly. Scale factors with many unique values
    will be automatically skipped with a warning.

    Args:
        mlmodel: CoreML model to palettize
        mlp_bits: Expected bits for Q weights (for logging only)
        verbose: Print progress
    """
    import coremltools.optimize as cto

    if verbose:
        print(f"Applying palettization (mode=unique, expecting {mlp_bits}-bit Q weights)...")
        print("  Note: Weights with >256 unique values will be skipped automatically")

    # Simple global config with mode="unique"
    # - Q weights with 16 unique values -> 4-bit palettization
    # - Scale factors with many unique values -> skipped automatically
    config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpPalettizerConfig(
            mode="unique",
            granularity="per_tensor"
        )
    )

    palettized = cto.coreml.palettize_weights(mlmodel, config)

    if verbose:
        print("  Palettization complete!")

    return palettized


def main():
    parser = argparse.ArgumentParser(description="V2 Trace-Based Conversion (Factored)")
    parser.add_argument("--checkpoint", required=True, help="V2 checkpoint path")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model")
    parser.add_argument("--context", type=int, default=512, help="Context length")
    parser.add_argument("--batch", type=int, default=64, help="Prefill batch size")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--mode", default="both", choices=["decoder", "prefill", "both"])
    parser.add_argument("--mlp-bits", type=int, default=4, help="LUT bits for MLP (gate/up/down)")
    parser.add_argument("--attn-bits", type=int, default=4, help="LUT bits for attention (q/k/v/o)")
    parser.add_argument("--no-palettize", action="store_true", help="Skip palettization")
    parser.add_argument("--prefix", default="qwen", help="Model prefix")
    parser.add_argument("--chunk", type=int, default=1, help="Number of chunks")

    # LoRA options
    parser.add_argument("--no-lora", action="store_true",
                        help="Disable LoRA even if present in checkpoint")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    enable_lora = not args.no_lora

    print("=" * 60)
    print("V2 Trace-Based Conversion (Factored)")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {args.mode}")
    if not args.no_palettize:
        print(f"Palettization: MLP={args.mlp_bits}-bit, Attn={args.attn_bits}-bit")
    else:
        print("Palettization: disabled")
    print(f"LoRA: {'auto-detect' if enable_lora else 'disabled'}")
    print()

    # Find HF model
    cache_pattern = os.path.expanduser(f'~/.cache/huggingface/hub/models--{args.model.replace("/", "--")}/snapshots/*')
    cached = glob.glob(cache_pattern)
    model_path = cached[0] if cached else None

    if not model_path:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(args.model)

    # Create model
    from anemll.models.qwen_model import QwenForCausalLM, QwenConfig

    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    config.context_length = args.context
    config.state_length = args.context

    print(f"Creating model: layers={config.num_hidden_layers}")
    model = QwenForCausalLM(config, enable_coreml=True)
    model.eval()

    # Replace Conv2d with FactoredConv2d (with LoRA if present and enabled)
    print("\nReplacing Conv2d with FactoredConv2d...")
    replace_info = replace_conv_with_factored(model, args.checkpoint,
                                               enable_lora=enable_lora, verbose=True)

    # Load V2 weights (and LoRA weights if enabled)
    print("\nLoading V2 factored weights (Q stored directly, not baked)...")
    load_info = load_v2_factored_weights(model, args.checkpoint,
                                          enable_lora=enable_lora, verbose=True)

    # Print LoRA summary
    if load_info['has_lora']:
        print(f"\n  LoRA Summary: {load_info['lora_layers']} layers with LoRA")
        print(f"  Mode: mlp_only={replace_info['mlp_only']}, skip_k_proj={replace_info['skip_k_proj']}")

    # Convert
    for chunk_idx in range(args.chunk):
        if args.mode in ["decoder", "both"]:
            print(f"\n--- Decoder (chunk {chunk_idx + 1}/{args.chunk}) ---")
            ml = convert_decoder(model, args.context, args.batch, chunk_idx, args.chunk)

            if not args.no_palettize:
                ml = apply_palettization(ml, args.mlp_bits)

            suffix = f"_lut{args.mlp_bits}" if not args.no_palettize else ""
            path = os.path.join(args.output, f"{args.prefix}_FFN{suffix}_chunk_{chunk_idx+1:02d}of{args.chunk:02d}.mlpackage")
            print(f"Saving: {path}")
            ml.save(path)

        if args.mode in ["prefill", "both"]:
            print(f"\n--- Prefill (chunk {chunk_idx + 1}/{args.chunk}) ---")
            ml = convert_prefill(model, args.context, args.batch, chunk_idx, args.chunk)

            if not args.no_palettize:
                ml = apply_palettization(ml, args.mlp_bits)

            suffix = f"_lut{args.mlp_bits}" if not args.no_palettize else ""
            path = os.path.join(args.output, f"{args.prefix}_prefill{suffix}_chunk_{chunk_idx+1:02d}of{args.chunk:02d}.mlpackage")
            print(f"Saving: {path}")
            ml.save(path)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
