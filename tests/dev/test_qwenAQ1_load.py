#!/usr/bin/env python3
"""Test loading and inference with ANEMLL-QUANT-1 Qwen model.

This script tests loading snapped ANEMLL checkpoints using the HF-based approach:
- Load HF model with AutoModelForCausalLM
- Replace linears with AnemllQATLinear
- Load checkpoint (sets snapped_mode from config.json)
- Call freeze_for_inference() to cache quantized weights

Usage:
    # Basic test
    python tests/dev/test_qwenAQ1_load.py --checkpoint ~/Downloads/q2_pt_good1/snapped_lut/model_state_dict.pt

    # Custom prompt
    python tests/dev/test_qwenAQ1_load.py --checkpoint ~/Downloads/q2_pt_good1/snapped_lut/model_state_dict.pt --prompt "Hello!"

    # Interactive mode
    python tests/dev/test_qwenAQ1_load.py --checkpoint ~/Downloads/q2_pt_good1/snapped_lut/model_state_dict.pt --interactive
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
# ANEMLL QAT LINEAR (simplified version for inference)
# =============================================================================

def make_lut(lut_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create monotonic LUT in [-1, 1]."""
    return torch.linspace(-1.0, 1.0, steps=lut_size, device=device, dtype=dtype)


class AnemllQATLinear(nn.Module):
    """Linear layer with Anemll-style groupwise LUT quantization."""

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

        # Base weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Scale parameters
        if self.use_low_rank:
            self.scale_A = nn.Parameter(torch.empty(out_features, self.scale_rank))
            self.scale_B = nn.Parameter(torch.empty(self.scale_rank, self.padded_in))
        else:
            self.register_parameter("scale_A", None)
            self.register_parameter("scale_B", None)
            self.full_scales = nn.Parameter(torch.empty(out_features, self.padded_in))

        # LUT
        lut = make_lut(self.config.lut_size, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("lut", lut)

        # Snapped mode: None, 'lut', or 'baked'
        self.snapped_mode = None
        self.lut_bits = self.config.lut_bits

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use cached weights if available
        cached = getattr(self, '_cached_weight_q', None)
        if cached is not None:
            w_q = cached
        elif getattr(self, 'snapped_mode', None) == 'lut':
            # LUT[idx] mode: multiply by scale
            scales = self.get_scales()
            if self.pad > 0:
                w_padded = F.pad(self.weight, (0, self.pad))
                w_q = (w_padded * scales)[:, :self.in_features]
            else:
                w_q = self.weight * scales[:, :self.in_features]
        elif getattr(self, 'snapped_mode', None) == 'baked':
            # Baked mode: weights are already final
            w_q = self.weight
        else:
            w_q = self.weight

        w_q = w_q.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_q, bias)

    @torch.no_grad()
    def freeze_for_inference(self):
        """Precompute and cache quantized weights for fast inference."""
        snapped_mode = getattr(self, 'snapped_mode', None)
        if snapped_mode == 'lut':
            scales = self.get_scales()
            if self.pad > 0:
                w_padded = F.pad(self.weight, (0, self.pad))
                w_q = (w_padded * scales)[:, :self.in_features]
            else:
                w_q = self.weight * scales[:, :self.in_features]
        elif snapped_mode == 'baked':
            w_q = self.weight
        else:
            w_q = self.weight

        # Store as buffer - handle if already exists
        w_q_cached = w_q.detach().clone()
        if hasattr(self, '_cached_weight_q') and self._cached_weight_q is not None:
            self._cached_weight_q = w_q_cached
        elif '_cached_weight_q' in self._buffers:
            self._buffers['_cached_weight_q'] = w_q_cached
        else:
            self.register_buffer('_cached_weight_q', w_q_cached)

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: AnemllQuantConfig = None) -> "AnemllQATLinear":
        """Create AnemllQATLinear from existing nn.Linear."""
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
        return f"in={self.in_features}, out={self.out_features}, rank={self.scale_rank}"


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
    """Replace MLP and attention linears with AnemllQATLinear."""
    import re

    mlp_pattern = re.compile(r'\.mlp\.(gate_proj|up_proj|down_proj)$')
    attn_pattern = re.compile(r'\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$')

    if attn_config is None:
        attn_config = mlp_config

    replacements = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, AnemllQATLinear):
            continue

        is_mlp = mlp_pattern.search(name)
        is_attn = attn_pattern.search(name)

        if is_mlp:
            cfg = mlp_config
        elif is_attn and quantize_attn:
            cfg = attn_config
        else:
            continue

        new_module = AnemllQATLinear.from_linear(module, config=cfg)

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
        print(f'\nReplaced {len(replacements)} layers')

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
    """Load checkpoint into model."""
    if device is None:
        device = next(model.parameters()).device

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    result = model.load_state_dict(state, strict=False)

    if verbose:
        print(f"Loaded checkpoint: {checkpoint_path}")
        if result.missing_keys:
            print(f"  Missing keys: {len(result.missing_keys)}")
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
        snapped_mode = config.get('snapped_mode')
        if snapped_mode is None and config.get('snapped'):
            snapped_mode = 'baked' if config.get('snap_bake_scales') else 'lut'

        # Set snapped_mode and lut_bits on all layers
        attn_proj_names = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
        mlp_lut_bits = int(math.log2(config.get('lut_size', 16)))
        attn_lut_bits = int(math.log2(config.get('attn_lut_size', config.get('lut_size', 16))))

        count = 0
        for name, m in model.named_modules():
            if type(m).__name__ == 'AnemllQATLinear':
                # Determine lut_bits from loaded LUT or config
                if hasattr(m, 'lut') and m.lut is not None:
                    lut_bits = int(math.log2(m.lut.numel()))
                else:
                    is_attn = any(p in name for p in attn_proj_names)
                    lut_bits = attn_lut_bits if is_attn else mlp_lut_bits

                m.snapped_mode = snapped_mode
                m.lut_bits = lut_bits
                count += 1

        if verbose:
            print(f"  Set snapped_mode='{snapped_mode}' on {count} layers")
            print(f"  MLP lut_bits={mlp_lut_bits}, attn lut_bits={attn_lut_bits}")

    return {'missing_keys': result.missing_keys, 'unexpected_keys': result.unexpected_keys, 'config': config}


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Test ANEMLL-QUANT-1 model inference')
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
    parser.add_argument('--no-thinking', action='store_true',
                        help='Disable thinking mode')

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

    return parser.parse_args()


def load_model(args):
    """Load model with QAT layers and checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Detect device - temporarily force CPU due to MPS attention shape issues
    # if torch.backends.mps.is_available():
    #     device = torch.device('mps')
    #     dtype = torch.float32
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
    print(f"Replacing linears (q{args.lut_bits}_a{args.attn_lut_bits})...")

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
    load_checkpoint(model, args.checkpoint, device='cpu', verbose=True)

    # Move to device
    model.to(device)
    model.eval()

    # Freeze for inference
    print("Freezing for inference...")
    count = 0
    for m in model.modules():
        if type(m).__name__ == 'AnemllQATLinear':
            m.freeze_for_inference()
            count += 1
    print(f"  Frozen {count} layers")

    print("Model ready!\n")
    return model, tokenizer, device


def generate(model, tokenizer, device, prompt, args):
    """Generate response for a prompt."""
    messages = [{'role': 'user', 'content': prompt}]

    template_kwargs = {
        'tokenize': False,
        'add_generation_prompt': True,
    }
    if not args.no_thinking:
        template_kwargs['enable_thinking'] = True

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
    print("ANEMLL-QUANT-1 Test")
    print("=" * 60)

    # Load model
    model, tokenizer, device = load_model(args)

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
    print("Test Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
