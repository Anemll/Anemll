#!/usr/bin/env python3
"""Test loading and inference with ANEMLL-QUANT-1 Qwen model using REFACTORED pipeline.

This script tests the FACTORED approach that avoids materializing A @ B:
    Original: y = x @ (Q * (A @ B)).T           # Materializes [out, in] scales
    Factored: y = sum_k(A[:, k] * (x * B[k]) @ Q.T)  # No [out, in] intermediate

Benefits:
- No A @ B matmul (avoids [out_features, in_features] intermediate)
- Same numerical result as original approach
- Tests the factored computation before CoreML conversion

LoRA Support:
- Auto-detects LoRA weights in checkpoint (lora_A, lora_B keys)
- Supports separate LoRA adapter files via --lora flag
- Can disable LoRA with --no-lora even if present in checkpoint
- LoRA adds low-rank update: y = quant_forward(x) + (x @ lora_A.T @ lora_B.T) * scaling

Usage:
    # Basic test
    python tests/dev/test_qwenAQ1_load_rf.py --checkpoint ~/Downloads/q2_pt_good1/snapped_lut/model_state_dict.pt

    # Compare with original (non-refactored)
    python tests/dev/test_qwenAQ1_load_rf.py --checkpoint ~/Downloads/q2_pt_good1/snapped_lut/model_state_dict.pt --compare

    # Interactive mode
    python tests/dev/test_qwenAQ1_load_rf.py --checkpoint ~/Downloads/q2_pt_good1/snapped_lut/model_state_dict.pt --interactive

    # With LoRA adapter (auto-detected in checkpoint)
    python tests/dev/test_qwenAQ1_load_rf.py --checkpoint full_checkpoint_with_lora.pt -i

    # With separate LoRA file
    python tests/dev/test_qwenAQ1_load_rf.py --checkpoint base_checkpoint.pt --lora lora_adapter.pt -i

    # Disable LoRA (even if present in checkpoint)
    python tests/dev/test_qwenAQ1_load_rf.py --checkpoint checkpoint_with_lora.pt --no-lora -i
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# ANEMLL QUANT CONFIG
# =============================================================================

from dataclasses import dataclass

@dataclass
class AnemllQuantConfig:
    """Configuration for Anemll-style groupwise LUT quantization."""
    lut_size: int = 16
    group_size: int = 128
    scale_rank: int = 4
    lut_include_zero: bool = False
    learnable_lut: bool = False

    @property
    def lut_bits(self) -> int:
        return int(math.ceil(math.log2(self.lut_size)))


# =============================================================================
# ANEMLL QAT LINEAR - REFACTORED (avoids A @ B materialization)
# =============================================================================

def make_lut(lut_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create monotonic LUT in [-1, 1]."""
    return torch.linspace(-1.0, 1.0, steps=lut_size, device=device, dtype=dtype)


class AnemllQATLinearRefactored(nn.Module):
    """Linear layer with Anemll-style groupwise LUT quantization - REFACTORED.

    Uses factored computation to avoid materializing [out_features, in_features] scales:
        Original: y = x @ (Q * (A @ B)).T
        Factored: y = sum_k(A[:, k] * (x * B[k]) @ Q.T)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: AnemllQuantConfig = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or AnemllQuantConfig()

        # Compute group dimensions
        self.group_size = self.config.group_size
        self.pad = (-in_features) % self.group_size
        self.padded_in = in_features + self.pad
        self.num_groups = self.padded_in // self.group_size

        # Scale rank
        self.max_rank = min(out_features, self.padded_in)
        self.scale_rank = min(self.config.scale_rank, self.max_rank) if self.config.scale_rank > 0 else 0
        self.use_low_rank = self.scale_rank > 0 and self.scale_rank < self.max_rank

        # Base weights (Q in the factored formula)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Scale parameters
        if self.use_low_rank:
            self.scale_A = nn.Parameter(torch.empty(out_features, self.scale_rank))
            self.scale_B = nn.Parameter(torch.empty(self.scale_rank, self.padded_in))
            # V2: per-rank magnitude (gₖ) - initialized to ones
            self.rank_magnitude = nn.Parameter(torch.ones(self.scale_rank))
        else:
            self.register_parameter("scale_A", None)
            self.register_parameter("scale_B", None)
            self.register_parameter("rank_magnitude", None)
            self.full_scales = nn.Parameter(torch.empty(out_features, self.padded_in))

        # LUT
        lut = make_lut(self.config.lut_size, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("lut", lut)

        # _Q buffer for snapped weights (used by V2 checkpoints)
        # V2 checkpoints store snapped weights in _Q, not in weight
        self.register_buffer("_Q", None)

        # Snapped mode: None, 'lut', or 'baked'
        self.snapped_mode = None
        self.lut_bits = self.config.lut_bits

        # Flag to use factored computation (can be disabled for comparison)
        self.use_factored = True

        # LoRA (Low-Rank Adaptation) parameters - initialized as disabled
        self.lora_r = 0
        self.lora_alpha = 0.0
        self.lora_dropout = 0.0
        self.lora_A = None
        self.lora_B = None
        self.scaling = 0.0
        self.lora_drop = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        self._init_scales_from_weight()

    @torch.no_grad()
    def _init_scales_from_weight(self):
        """Initialize scale parameters from weight statistics."""
        w = self.weight.float()
        if self.pad > 0:
            w = F.pad(w, (0, self.pad))

        grouped = w.view(self.out_features, self.num_groups, self.group_size)
        scales_per_group = grouped.abs().amax(dim=2).clamp(min=1e-8)
        scales_per_weight = scales_per_group.repeat_interleave(self.group_size, dim=1)

        if self.use_low_rank:
            u, s, vh = torch.linalg.svd(scales_per_weight, full_matrices=False)
            r = self.scale_rank
            self.scale_A.data = (u[:, :r] * s[:r]).to(self.weight.dtype)
            self.scale_B.data = vh[:r, :].to(self.weight.dtype)
        else:
            self.full_scales.data = scales_per_weight.to(self.weight.dtype)

    def get_scales(self) -> torch.Tensor:
        """Get the per-weight scale matrix [out_features, padded_in]."""
        if self.use_low_rank:
            return (self.scale_A @ self.scale_B).clamp(min=1e-8)
        else:
            return self.full_scales.clamp(min=1e-8)

    def enable_lora(self, r: int, alpha: float = None, dropout: float = 0.0):
        """Enable LoRA (Low-Rank Adaptation) on this layer.

        LoRA adds a low-rank update to the quantized output:
            y = quant_forward(x) + (x @ lora_A.T @ lora_B.T) * scaling

        Args:
            r: LoRA rank
            alpha: LoRA alpha (default: r, so scaling = 1.0)
            dropout: Dropout rate for LoRA (default: 0.0)
        """
        if alpha is None:
            alpha = float(r)

        self.lora_r = int(r)
        self.lora_alpha = float(alpha)
        self.lora_dropout = float(dropout)
        self.scaling = alpha / r

        # Create LoRA parameters
        device = self.weight.device
        dtype = self.weight.dtype

        self.lora_A = nn.Parameter(
            torch.zeros(self.lora_r, self.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.lora_r, device=device, dtype=dtype)
        )
        self.lora_drop = nn.Dropout(p=dropout) if dropout > 0 else None

        # Standard LoRA initialization: A ~ N(0, 0.02), B = 0
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    def disable_lora(self):
        """Disable LoRA on this layer."""
        self.lora_r = 0
        self.lora_alpha = 0.0
        self.lora_dropout = 0.0
        self.scaling = 0.0
        self.lora_A = None
        self.lora_B = None
        self.lora_drop = None

    def forward_factored(self, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """Factored forward: y = sum_k(A[:, k] * (x * B[k]) @ Q.T)

        This avoids materializing the [out_features, in_features] scale matrix.

        Bounding options (controlled by self.bound_mode):
        - 'none': No bounding (raw A, B)
        - 'clip': Clip A and B to [-max, +max]
        - 'tanh': A = Amax * tanh(A_raw), B = Bmax * tanh(B_raw)
                  Guarantees |S_ij| <= r * Amax * Bmax

        Args:
            x: Input tensor [batch, seq, in_features]
            Q: Base weights [out_features, in_features] (already scaled by LUT if snapped)

        Returns:
            y: Output tensor [batch, seq, out_features]
        """
        if not self.use_low_rank:
            # Fall back to materialized for full-rank scales
            scales = self.full_scales.clamp(min=1e-8)
            if self.pad > 0:
                scales = scales[:, :self.in_features]
            W_eff = Q * scales
            return F.linear(x, W_eff, None)

        # Factored computation
        # scale_A: [out_features, rank]
        # scale_B: [rank, padded_in] -> trim to [rank, in_features]
        A_raw = self.scale_A  # [out, rank]
        B_raw = self.scale_B[:, :self.in_features]  # [rank, in]

        rank = A_raw.shape[1]

        # Bounding parameters
        A_max = getattr(self, 'clip_A_max', 1.0)
        B_max = getattr(self, 'clip_B_max', 1.0)
        bound_mode = getattr(self, 'bound_mode', 'tanh')

        # Apply bounding to A and B
        if bound_mode == 'tanh':
            # Symmetric tanh: guarantees |A| <= Amax, |B| <= Bmax
            # Then |S_ij| = |sum_k A_ik B_kj| <= r * Amax * Bmax
            A = A_max * torch.tanh(A_raw)
            B = B_max * torch.tanh(B_raw)
        elif bound_mode == 'clip':
            # Clip to [-max, +max]
            A = A_raw.clamp(-A_max, A_max)
            B = B_raw.clamp(-B_max, B_max)
        elif bound_mode == 'abs':
            # Use absolute values - ensures A@B >= 0 (like clamp(min=0))
            # This matches the original's .clamp(min=1e-8) behavior
            A = A_raw.abs()
            B = B_raw.abs()
        elif bound_mode == 'relu':
            # ReLU on A and B - only keep positive values
            A = torch.relu(A_raw)
            B = torch.relu(B_raw)
        else:  # 'none'
            A = A_raw
            B = B_raw

        # V2 formula: y = Σₖ gₖ · (aₖ ⊙ (Q @ (bₖ ⊙ x)))
        # gₖ = rank_magnitude[k] (per-rank magnitude scalar)
        # aₖ = A[:, k] (direction column)
        # bₖ = B[k, :] (direction row)
        # x: [batch, seq, in]

        y = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)

        # Get rank magnitudes (gₖ) - V2 style
        # For V1 checkpoints without rank_magnitude, default to 1.0
        g = self.rank_magnitude if self.rank_magnitude is not None else torch.ones(rank, device=x.device, dtype=x.dtype)
        g = g.to(x.dtype)

        for k in range(rank):
            # gₖ: per-rank magnitude scalar
            g_k = g[k]

            # bₖ: [in_features] - direction row
            B_k = B[k]  # [in]

            # x_scaled = bₖ ⊙ x: [batch, seq, in] * [in] -> [batch, seq, in]
            x_scaled = x * B_k

            # y_k = Q @ x_scaled: [batch, seq, in] @ [in, out] -> [batch, seq, out]
            y_k = x_scaled @ Q.T

            # aₖ: [out_features] - direction column
            A_k = A[:, k]  # [out]

            # y += gₖ · aₖ ⊙ y_k: scalar * [out] * [batch, seq, out] -> [batch, seq, out]
            y = y + (g_k * A_k) * y_k

        return y

    def forward_materialized(self, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """Materialized forward (original approach for comparison)."""
        scales = self.get_scales()
        if self.pad > 0:
            scales = scales[:, :self.in_features]
        W_eff = Q * scales
        return F.linear(x, W_eff, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get base weights Q
        snapped_mode = getattr(self, 'snapped_mode', None)

        # Check if we have _Q buffer from V2 checkpoint
        if self._Q is not None:
            # V2 checkpoint: use _Q which contains snapped LUT values
            Q = self._Q
        elif snapped_mode == 'lut':
            # In LUT mode, weight contains LUT[indices] values (base Q)
            Q = self.weight
        elif snapped_mode == 'baked':
            # Baked mode: weights are already final (Q * scales baked in)
            # Can't use factored approach here
            return F.linear(x, self.weight.to(x.dtype),
                          self.bias.to(x.dtype) if self.bias is not None else None)
        else:
            Q = self.weight

        Q = Q.to(x.dtype)

        # Choose computation method
        if self.use_factored and self.use_low_rank:
            y = self.forward_factored(x, Q)
        else:
            y = self.forward_materialized(x, Q)

        # Add bias
        if self.bias is not None:
            y = y + self.bias.to(x.dtype)

        # Add LoRA contribution if enabled
        # y += (x @ lora_A.T @ lora_B.T) * scaling
        if self.lora_r > 0 and self.lora_A is not None:
            x_d = self.lora_drop(x) if self.lora_drop is not None else x
            lora_A = self.lora_A.to(x.dtype)
            lora_B = self.lora_B.to(x.dtype)
            # In-place addition for memory efficiency
            hidden = x_d @ lora_A.T  # [*, lora_r]
            y = y + (hidden @ lora_B.T) * self.scaling

        return y

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: AnemllQuantConfig = None) -> "AnemllQATLinearRefactored":
        """Create AnemllQATLinearRefactored from existing nn.Linear."""
        config = config or AnemllQuantConfig()
        new = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
        )
        with torch.no_grad():
            new.weight.copy_(linear.weight)
            if linear.bias is not None:
                new.bias.copy_(linear.bias)
        new._init_scales_from_weight()
        new = new.to(device=linear.weight.device, dtype=linear.weight.dtype)
        return new

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.scale_rank}, factored={self.use_factored}"


# =============================================================================
# MODEL REPLACEMENT UTILITY
# =============================================================================

def replace_linear_with_anemll(
    model: nn.Module,
    mlp_config: AnemllQuantConfig,
    attn_config: AnemllQuantConfig = None,
    quantize_attn: bool = True,
    verbose: bool = True,
) -> int:
    """Replace MLP and attention linears with AnemllQATLinearRefactored."""
    import re

    mlp_pattern = re.compile(r'\.mlp\.(gate_proj|up_proj|down_proj)$')
    attn_pattern = re.compile(r'\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$')

    if attn_config is None:
        attn_config = mlp_config

    replacements = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, AnemllQATLinearRefactored):
            continue

        is_mlp = mlp_pattern.search(name)
        is_attn = attn_pattern.search(name)

        if is_mlp:
            cfg = mlp_config
        elif is_attn and quantize_attn:
            cfg = attn_config
        else:
            continue

        new_module = AnemllQATLinearRefactored.from_linear(module, config=cfg)

        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr = name

        replacements.append((parent, attr, new_module, name))

    for parent, attr, new_module, name in replacements:
        setattr(parent, attr, new_module)
        if verbose:
            print(f'  [replaced] {name}')

    if verbose:
        print(f'\nReplaced {len(replacements)} layers with REFACTORED linear')

    return len(replacements)


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device = None,
    verbose: bool = True,
) -> dict:
    """Load checkpoint into model with V1 and V2 format support."""
    if device is None:
        device = next(model.parameters()).device

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Keep a reference to the original state dict for LoRA detection
    # (we modify 'state' in place below, so save it first)
    original_state_dict = state.copy()

    # Detect V2 format
    has_q_buffers = any('._Q' in k for k in state.keys())
    has_rank_magnitude = any('.rank_magnitude' in k for k in state.keys())
    is_v2 = has_q_buffers and has_rank_magnitude

    # Detect LoRA weights early
    lora_keys = [k for k in state.keys() if 'lora_' in k]
    has_lora = len(lora_keys) > 0

    if verbose:
        if is_v2:
            print(f"  Detected V2 checkpoint format (_Q + rank_magnitude)")
        else:
            print(f"  Using V1 checkpoint format")
        if has_lora:
            print(f"  Detected LoRA weights in checkpoint ({len(lora_keys)} tensors)")
        else:
            print(f"  No LoRA weights detected in checkpoint")

    # Build key mapping to handle prefix differences
    # Model might use "model.model.layers" but checkpoint uses "model.layers"
    model_keys = set(k for k, _ in model.named_parameters())
    model_keys.update(k for k, _ in model.named_buffers())

    # Try to find the correct prefix mapping
    def find_module_key(module_name, suffix, state_dict):
        """Find the checkpoint key that matches module.suffix with various prefixes."""
        candidates = [
            f"{module_name}.{suffix}",
            f"model.{module_name}.{suffix}",
            module_name.replace('model.model.', 'model.') + f".{suffix}",
            module_name.replace('model.', '') + f".{suffix}",
        ]
        for key in candidates:
            if key in state_dict:
                return key
        return None

    # Load V2 parameters manually for each AnemllQATLinearRefactored module
    q_loaded = 0
    mag_loaded = 0
    scale_a_loaded = 0
    scale_b_loaded = 0

    for name, module in model.named_modules():
        if type(module).__name__ != 'AnemllQATLinearRefactored':
            continue

        # Load _Q buffer (V2: snapped weights)
        q_key = find_module_key(name, '_Q', state)
        if q_key:
            module._Q = state[q_key].to(device)
            q_loaded += 1
            del state[q_key]

        # Load rank_magnitude (V2: per-rank magnitudes gₖ)
        mag_key = find_module_key(name, 'rank_magnitude', state)
        if mag_key and module.rank_magnitude is not None:
            with torch.no_grad():
                module.rank_magnitude.copy_(state[mag_key].to(device))
            mag_loaded += 1
            del state[mag_key]

        # Load scale_A
        a_key = find_module_key(name, 'scale_A', state)
        if a_key and module.scale_A is not None:
            with torch.no_grad():
                module.scale_A.copy_(state[a_key].to(device))
            scale_a_loaded += 1
            del state[a_key]

        # Load scale_B
        b_key = find_module_key(name, 'scale_B', state)
        if b_key and module.scale_B is not None:
            with torch.no_grad():
                module.scale_B.copy_(state[b_key].to(device))
            scale_b_loaded += 1
            del state[b_key]

        # For V2, also try to load .weight (frozen original weights) but we don't need it
        # since we use _Q directly. Remove from state to avoid unexpected key warnings.
        weight_key = find_module_key(name, 'weight', state)
        if weight_key and is_v2:
            del state[weight_key]

        # Remove other V2-specific keys we don't need
        for suffix in ['_indices', 'lut']:
            key = find_module_key(name, suffix, state)
            if key:
                del state[key]

    if verbose:
        if q_loaded > 0:
            print(f"  Loaded {q_loaded} _Q buffers (V2 snapped weights)")
        if mag_loaded > 0:
            print(f"  Loaded {mag_loaded} rank_magnitude (V2 gₖ)")
        if scale_a_loaded > 0:
            print(f"  Loaded {scale_a_loaded} scale_A, {scale_b_loaded} scale_B")

    # Load remaining parameters (embeddings, norms, etc.)
    result = model.load_state_dict(state, strict=False)

    if verbose:
        print(f"Loaded checkpoint: {checkpoint_path}")
        if result.missing_keys:
            # Filter out expected missing keys for V2
            missing = [k for k in result.missing_keys if not any(
                x in k for x in ['.weight', '.lut', '.full_scales']
            )] if is_v2 else result.missing_keys
            if missing:
                print(f"  Missing keys: {len(missing)}")
                if len(missing) <= 5:
                    for k in missing:
                        print(f"    - {k}")
        if result.unexpected_keys:
            print(f"  Unexpected keys: {len(result.unexpected_keys)}")
        if not result.missing_keys and not result.unexpected_keys:
            print(f"  All keys matched")

    # Load config.json to set snapped_mode
    config = None
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    # Determine snapped_mode
    snapped_mode = None
    if config:
        snapped_mode = config.get('snapped_mode')
        if snapped_mode is None:
            # V2 format always uses 'lut' mode when _Q is present
            if is_v2 or config.get('version') == 'v2':
                snapped_mode = 'lut'
            elif config.get('snapped'):
                snapped_mode = 'baked' if config.get('snap_bake_scales') else 'lut'
    elif is_v2:
        # No config but V2 checkpoint detected
        snapped_mode = 'lut'

    # Set snapped_mode and lut_bits on all layers
    attn_proj_names = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
    mlp_lut_bits = 4  # default
    attn_lut_bits = 4  # default
    if config:
        mlp_lut_bits = config.get('lut_bits', int(math.log2(config.get('lut_size', 16))))
        attn_lut_bits = config.get('attn_lut_bits', mlp_lut_bits)

    # Determine bound_mode from config
    # V2 with force_positive_scales=False needs 'none' to preserve sign info
    force_positive = True  # V1 default
    if config:
        force_positive = config.get('force_positive_scales', True)
    bound_mode = 'abs' if force_positive else 'none'

    count = 0
    for name, m in model.named_modules():
        if type(m).__name__ == 'AnemllQATLinearRefactored':
            is_attn = any(p in name for p in attn_proj_names)
            m.snapped_mode = snapped_mode
            m.lut_bits = attn_lut_bits if is_attn else mlp_lut_bits
            m.bound_mode = bound_mode
            count += 1

    if verbose:
        print(f"  Set snapped_mode='{snapped_mode}', bound_mode='{bound_mode}' on {count} layers")
        print(f"  MLP lut_bits={mlp_lut_bits}, attn lut_bits={attn_lut_bits}")

    return {
        'missing_keys': result.missing_keys,
        'unexpected_keys': result.unexpected_keys,
        'config': config,
        'is_v2': is_v2,
        'state_dict': original_state_dict,  # For LoRA detection
    }


# =============================================================================
# COMPARISON TEST
# =============================================================================

def test_factored_vs_materialized(model, tokenizer, device, verbose=True):
    """Compare factored vs materialized outputs to verify correctness."""
    if verbose:
        print("\n" + "=" * 60)
        print("Testing: Factored vs Materialized computation")
        print("=" * 60)

    # Create a test input
    test_text = "Hello, world!"
    inputs = tokenizer(test_text, return_tensors='pt').to(device)

    # Collect all refactored layers
    layers = []
    for name, m in model.named_modules():
        if type(m).__name__ == 'AnemllQATLinearRefactored':
            layers.append((name, m))

    if not layers:
        print("No AnemllQATLinearRefactored layers found!")
        return False

    # Test a few layers
    max_diff_overall = 0.0
    test_count = min(5, len(layers))

    for name, layer in layers[:test_count]:
        if not layer.use_low_rank:
            continue

        # Get input shape for this layer
        # Use a dummy forward pass to get intermediate activations
        # For simplicity, test with random input matching layer dimensions
        batch, seq = 1, 8
        x = torch.randn(batch, seq, layer.in_features, device=device, dtype=torch.float32)

        # Get base Q weights
        Q = layer.weight.to(x.dtype)

        # Compute both ways
        layer.use_factored = True
        y_factored = layer.forward_factored(x, Q)

        y_materialized = layer.forward_materialized(x, Q)

        # Compare
        max_diff = (y_factored - y_materialized).abs().max().item()
        mean_diff = (y_factored - y_materialized).abs().mean().item()
        max_diff_overall = max(max_diff_overall, max_diff)

        if verbose:
            print(f"  {name}:")
            print(f"    max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e}")

    # Verify
    tolerance = 1e-4  # FP32 tolerance
    if max_diff_overall < tolerance:
        if verbose:
            print(f"\n✓ PASS: max_diff={max_diff_overall:.2e} < {tolerance:.0e}")
        return True
    else:
        if verbose:
            print(f"\n✗ FAIL: max_diff={max_diff_overall:.2e} >= {tolerance:.0e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Test ANEMLL-QUANT-1 model with REFACTORED pipeline')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='Base model ID (default: Qwen/Qwen3-0.6B)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt to test')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Max new tokens (default: 512)')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature (default: 0.6)')
    parser.add_argument('--no-think', action='store_true',
                        help='Disable Qwen3 thinking mode (passes enable_thinking=False to chat template)')

    # Comparison test
    parser.add_argument('--compare', action='store_true',
                        help='Compare factored vs materialized computation')
    parser.add_argument('--no-factored', action='store_true',
                        help='Disable factored computation (use materialized)')

    # Quantization config
    parser.add_argument('--lut-bits', type=int, default=2,
                        help='LUT bits for MLP (default: 2)')
    parser.add_argument('--attn-lut-bits', type=int, default=4,
                        help='LUT bits for attention (default: 4)')
    parser.add_argument('--group-size', type=int, default=16,
                        help='Group size (default: 16)')
    parser.add_argument('--scale-rank', type=int, default=32,
                        help='Scale rank for MLP (default: 32)')
    parser.add_argument('--attn-scale-rank', type=int, default=8,
                        help='Scale rank for attention (default: 8)')

    # Bounding for factored computation
    parser.add_argument('--bound-mode', type=str, default='auto',
                        choices=['auto', 'none', 'clip', 'tanh', 'abs', 'relu'],
                        help='Bounding mode: auto (from config), none, clip, tanh, abs, relu (default: auto)')
    parser.add_argument('--A-max', type=float, default=1.0,
                        help='Max absolute value for A scale factors (default: 1.0)')
    parser.add_argument('--B-max', type=float, default=1.0,
                        help='Max absolute value for B scale factors (default: 1.0)')

    # LoRA (Low-Rank Adaptation) options
    parser.add_argument('--lora', type=str, default=None,
                        help='Path to separate LoRA adapter checkpoint (optional)')
    parser.add_argument('--lora-r', type=int, default=8,
                        help='LoRA rank (default: 8, must match saved adapter)')
    parser.add_argument('--no-lora', action='store_true',
                        help='Disable LoRA even if present in checkpoint')

    return parser.parse_args()


def load_model(args):
    """Load model with QAT layers and checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        dtype = torch.bfloat16
    else:
        device = torch.device('cpu')
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}")

    # Auto-detect config from checkpoint directory
    checkpoint_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            ckpt_config = json.load(f)
        print(f"Found config.json in checkpoint directory:")
        print(f"  {ckpt_config}")
        # Override defaults with config values (if not explicitly set by user)
        if args.lut_bits == 2 and 'lut_bits' in ckpt_config:
            args.lut_bits = ckpt_config['lut_bits']
        if args.attn_lut_bits == 4 and 'attn_lut_bits' in ckpt_config:
            args.attn_lut_bits = ckpt_config['attn_lut_bits']
        if args.scale_rank == 32 and 'scale_rank' in ckpt_config:
            args.scale_rank = ckpt_config['scale_rank']
        if args.attn_scale_rank == 8 and 'attn_scale_rank' in ckpt_config:
            args.attn_scale_rank = ckpt_config['attn_scale_rank']
        if args.group_size == 16 and 'group_size' in ckpt_config:
            args.group_size = ckpt_config['group_size']
        print(f"Using: lut_bits={args.lut_bits}, attn_lut_bits={args.attn_lut_bits}, "
              f"scale_rank={args.scale_rank}, attn_scale_rank={args.attn_scale_rank}")

    print(f"Loading base model: {args.model_id}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Replace with QAT layers
    print(f"Replacing linears with REFACTORED QAT (q{args.lut_bits}_a{args.attn_lut_bits})...")

    mlp_config = AnemllQuantConfig(
        lut_size=2**args.lut_bits,
        group_size=args.group_size,
        scale_rank=args.scale_rank,
    )
    attn_config = AnemllQuantConfig(
        lut_size=2**args.attn_lut_bits,
        group_size=args.group_size,
        scale_rank=args.attn_scale_rank,
    )

    replace_linear_with_anemll(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=True,
        verbose=False,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_result = load_checkpoint(model, args.checkpoint, device='cpu', verbose=True)

    # Analyze scale ranges
    A_mins, A_maxs, B_mins, B_maxs = [], [], [], []
    A_tanh_mins, A_tanh_maxs, B_tanh_mins, B_tanh_maxs = [], [], [], []
    ranks = []
    for m in model.modules():
        if type(m).__name__ == 'AnemllQATLinearRefactored' and m.use_low_rank:
            A_mins.append(m.scale_A.min().item())
            A_maxs.append(m.scale_A.max().item())
            B_mins.append(m.scale_B.min().item())
            B_maxs.append(m.scale_B.max().item())
            # Also compute tanh-bounded values
            A_tanh = torch.tanh(m.scale_A)
            B_tanh = torch.tanh(m.scale_B)
            A_tanh_mins.append(A_tanh.min().item())
            A_tanh_maxs.append(A_tanh.max().item())
            B_tanh_mins.append(B_tanh.min().item())
            B_tanh_maxs.append(B_tanh.max().item())
            ranks.append(m.scale_A.shape[1])
    if A_mins:
        print(f"  Scale A raw range: [{min(A_mins):.4f}, {max(A_maxs):.4f}]")
        print(f"  Scale B raw range: [{min(B_mins):.4f}, {max(B_maxs):.4f}]")
        print(f"  Scale A tanh range: [{min(A_tanh_mins):.4f}, {max(A_tanh_maxs):.4f}]")
        print(f"  Scale B tanh range: [{min(B_tanh_mins):.4f}, {max(B_tanh_maxs):.4f}]")
        print(f"  Ranks: {set(ranks)}")

    # Set factored mode and bounding parameters
    use_factored = not args.no_factored

    # Determine bound_mode: 'auto' reads from config, otherwise use arg
    if args.bound_mode == 'auto':
        # Auto-detect from config: V2 with force_positive_scales=False needs 'none'
        config = load_result.get('config', {})
        force_positive = config.get('force_positive_scales', True) if config else True
        bound_mode = 'abs' if force_positive else 'none'
        print(f"  Auto-detected bound_mode='{bound_mode}' from config (force_positive_scales={force_positive})")
    else:
        bound_mode = args.bound_mode

    A_max = args.A_max
    B_max = args.B_max
    count = 0
    for m in model.modules():
        if type(m).__name__ == 'AnemllQATLinearRefactored':
            m.use_factored = use_factored
            m.bound_mode = bound_mode
            m.clip_A_max = A_max  # Used by both clip and tanh modes
            m.clip_B_max = B_max
            count += 1
    print(f"  Set use_factored={use_factored} on {count} layers")
    print(f"  Bound mode: {bound_mode}")
    if bound_mode == 'abs':
        print(f"  A = |A|, B = |B| -> A@B always >= 0 (matches clamp(min=0))")
    elif bound_mode == 'relu':
        print(f"  A = relu(A), B = relu(B) -> only positive contributions")
    elif bound_mode == 'clip':
        print(f"  A_max={A_max}, B_max={B_max}")
    elif bound_mode == 'tanh':
        print(f"  A_max={A_max}, B_max={B_max}")
        rank = args.scale_rank
        max_S = rank * A_max * B_max
        print(f"  Max |S_ij| bound: {rank} * {A_max} * {B_max} = {max_S:.1f}")

    # LoRA (Low-Rank Adaptation) handling
    has_lora = False
    if not args.no_lora:
        # Get the state dict for LoRA detection
        state_dict = load_result.get('state_dict', {})
        config = load_result.get('config', {})

        # Check for LoRA weights
        lora_state = None
        lora_keys = []

        if args.lora:
            # Explicit --lora flag: load from separate file
            lora_path = os.path.expanduser(args.lora)
            if os.path.exists(lora_path):
                print(f"\nLoading LoRA adapter: {lora_path}")
                lora_state = torch.load(lora_path, map_location='cpu')
                if isinstance(lora_state, dict) and 'model_state_dict' in lora_state:
                    lora_state = lora_state['model_state_dict']
                lora_keys = [k for k in lora_state if 'lora_' in k]
            else:
                print(f"Warning: LoRA file not found: {lora_path}")
        else:
            # Check if checkpoint contains LoRA weights (auto-detect)
            lora_keys = [k for k in state_dict if 'lora_' in k]
            if lora_keys:
                print(f"\nDetected LoRA weights in checkpoint ({len(lora_keys)} keys)")
                lora_state = state_dict

        if lora_keys and lora_state:
            # Get LoRA rank from config if available (CLI arg takes priority)
            lora_r = args.lora_r
            if config.get('recovery_r') and args.lora_r == 8:  # 8 is default
                lora_r = config.get('recovery_r')
                print(f"  Using recovery_r={lora_r} from config")

            # Get LoRA settings from config
            mlp_only = config.get('mlp_only', False)
            skip_k_proj = config.get('skip_k_proj', True)

            # Enable LoRA on all matching layers
            lora_count = 0
            for name, m in model.named_modules():
                if type(m).__name__ != 'AnemllQATLinearRefactored':
                    continue

                # Skip k_proj if configured
                if skip_k_proj and 'k_proj' in name:
                    continue

                # Skip attention layers if mlp_only
                if mlp_only and any(a in name for a in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                    continue

                # Enable LoRA on this layer
                m.enable_lora(r=lora_r)
                lora_count += 1

            print(f"  Enabled LoRA (r={lora_r}) on {lora_count} layers")

            # Load LoRA weights
            lora_only = {k: lora_state[k] for k in lora_keys}
            missing, unexpected = model.load_state_dict(lora_only, strict=False)
            source = "separate file" if args.lora else "checkpoint"
            print(f"  Loaded {len(lora_keys)} LoRA tensors (from {source})")
            if missing:
                missing_lora = [k for k in missing if 'lora_' in k]
                if missing_lora:
                    print(f"  Warning: {len(missing_lora)} LoRA keys not loaded")

            has_lora = True
    else:
        print("\n  LoRA disabled via --no-lora")

    # Move to device
    model.to(device)
    model.eval()

    # Print summary
    print("\n" + "=" * 50)
    print("Model Configuration Summary")
    print("=" * 50)
    print(f"  Model ID:       {args.model_id}")
    print(f"  Device:         {device}")
    print(f"  LUT bits:       {args.lut_bits} (MLP), {args.attn_lut_bits} (Attn)")
    print(f"  Scale rank:     {args.scale_rank} (MLP), {args.attn_scale_rank} (Attn)")
    print(f"  Bound mode:     {bound_mode}")
    print(f"  LoRA:           {'Enabled (r=' + str(args.lora_r) + ')' if has_lora else 'Disabled'}")
    print("=" * 50)

    print("\nModel ready!\n")
    return model, tokenizer, device


def generate(model, tokenizer, device, prompt, args):
    """Generate response for a prompt."""
    messages = [{'role': 'user', 'content': prompt}]

    template_kwargs = {
        'tokenize': False,
        'add_generation_prompt': True,
    }
    if args.no_think:
        template_kwargs['enable_thinking'] = False

    text = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer(text, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=False
    )

    response = response.replace('<|im_end|>', '').strip()
    response = response.replace('<think>\n<think>', '<think>')

    return response


def run_default_prompts(model, tokenizer, device, args):
    """Run default test prompts."""
    prompts = [
        'What is the capital of France?',
        'What is Apple Neural Engine?',
        'Explain quantum mechanics briefly.',
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        response = generate(model, tokenizer, device, prompt, args)
        print(f"Response: {response}")
        print('-' * 60)


def run_interactive(model, tokenizer, device, args):
    """Interactive prompt loop."""
    print("Interactive mode. Type 'q' or 'quit' to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ('q', 'quit', 'exit'):
            print("Bye!")
            break

        response = generate(model, tokenizer, device, prompt, args)
        print(f"\nAssistant: {response}\n")


def main():
    args = parse_args()

    # Expand checkpoint path
    checkpoint_path = os.path.expanduser(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    args.checkpoint = checkpoint_path

    print("=" * 60)
    print("ANEMLL-QUANT-1 Test (REFACTORED Pipeline)")
    print("=" * 60)
    print("\nUsing FACTORED computation: y = sum_k(A[:, k] * (x * B[k]) @ Q.T)")
    print("This avoids materializing the [out, in] scale matrix.\n")

    # Load model
    model, tokenizer, device = load_model(args)

    # Comparison test
    if args.compare:
        success = test_factored_vs_materialized(model, tokenizer, device)
        if not success:
            print("\nWARNING: Factored computation differs from materialized!")
            sys.exit(1)

    # Run inference
    if args.prompt:
        print(f"Prompt: {args.prompt}")
        response = generate(model, tokenizer, device, args.prompt, args)
        print(f"Response: {response}")
    elif args.interactive:
        run_interactive(model, tokenizer, device, args)
    else:
        run_default_prompts(model, tokenizer, device, args)

    print("\n" + "=" * 60)
    print("Test Complete (REFACTORED Pipeline)")
    print("=" * 60)


if __name__ == '__main__':
    main()
