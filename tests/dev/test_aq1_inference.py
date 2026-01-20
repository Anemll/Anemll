#!/usr/bin/env python3
"""Simple inference test for Qwen with AQ1 (ANEMLL-QUANT-1) quantization.

Compares PyTorch inference with optional CoreML/ANE inference.

Usage:
    # Basic PyTorch inference
    python tests/dev/test_aq1_inference.py ~/Downloads/snapped_step1800.pt --prompt "History of England and UK"

    # With custom config
    python tests/dev/test_aq1_inference.py ~/Downloads/snapped_step1800.pt --prompt "What is AI?" --config q4_r32

    # Skip thinking tokens (like /no-think)
    python tests/dev/test_aq1_inference.py ~/Downloads/snapped_step1800.pt --prompt "Hello" --no-think

    # Compare with CoreML/ANE
    python tests/dev/test_aq1_inference.py ~/Downloads/snapped_step1800.pt --prompt "Hello" --compare-ane

    # Verbose mode
    python tests/dev/test_aq1_inference.py ~/Downloads/snapped_step1800.pt --prompt "Hello" -v
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from anemll.models.qwen_model import (
    QwenConfig, QwenForCausalLM,
    MODEL_DTYPE, TEST_DEVICE, STATE_LENGTH
)


# =============================================================================
# CONFIG PRESETS
# =============================================================================

CONFIG_PRESETS = {
    "q2_r32": {"lut_bits": 2, "mlp_scale_rank": 32, "attn_scale_rank": 8},
    "q4_r32": {"lut_bits": 4, "mlp_scale_rank": 32, "attn_scale_rank": 8},
    "q4_r64": {"lut_bits": 4, "mlp_scale_rank": 64, "attn_scale_rank": 16},
    "auto": None,  # Auto-detect from checkpoint
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_tokenizer(model_id: str):
    """Load tokenizer from HuggingFace."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def get_model_path(model_id: str) -> str:
    """Get local path for HF model."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, local_files_only=False)


def create_causal_mask(seq_len: int, state_length: int, dtype=MODEL_DTYPE, current_pos: int = 0):
    """Create a causal attention mask."""
    mask = torch.zeros((1, 1, seq_len, state_length), dtype=dtype)
    for i in range(seq_len):
        actual_pos = current_pos + i
        mask[0, 0, i, actual_pos + 1:] = float('-inf')
    return mask


def detect_checkpoint_config(checkpoint_path: str) -> dict:
    """Auto-detect quantization config from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    config = {
        "lut_bits": 4,  # Default
        "mlp_scale_rank": 32,
        "attn_scale_rank": 8,
    }

    # Detect from scale_A shapes
    for key in state_dict.keys():
        if '.scale_A' in key:
            shape = state_dict[key].shape
            if 'mlp' in key or 'gate_proj' in key or 'up_proj' in key or 'down_proj' in key:
                config["mlp_scale_rank"] = shape[1]
            elif 'attn' in key or 'q_proj' in key or 'k_proj' in key or 'v_proj' in key:
                config["attn_scale_rank"] = shape[1]
            break

    # Detect LUT bits from weight values
    for key in state_dict.keys():
        if key.endswith('.weight') and ('q_proj' in key or 'gate_proj' in key):
            unique_vals = torch.unique(state_dict[key]).numel()
            if unique_vals <= 4:
                config["lut_bits"] = 2
            elif unique_vals <= 16:
                config["lut_bits"] = 4
            else:
                config["lut_bits"] = 8
            break

    return config


def load_model_with_aq1_weights(
    model_id: str,
    checkpoint_path: str,
    context_length: int = 512,
    state_length: int = 512,
    verbose: bool = False,
) -> tuple:
    """Load Qwen model with AQ1 quantized weights (baked).

    Supports both V1 and V2 checkpoint formats:
    - V1: snapped weights in `.weight`, scales = A @ B
    - V2: snapped weights in `._Q`, scales = (A * rank_magnitude) @ B
    """

    # Get HF model path for config
    model_path = get_model_path(model_id)
    config_path = os.path.join(model_path, "config.json")

    with open(config_path) as f:
        config_dict = json.load(f)
    config_dict['context_length'] = context_length
    config_dict['state_length'] = state_length
    config = QwenConfig(**config_dict)

    if verbose:
        print(f"Model config: {config.num_hidden_layers} layers, {config.hidden_size} hidden")

    # Create model
    model = QwenForCausalLM(config)

    # Load HF weights first (for non-quantized layers like embed_tokens, norms)
    model.load_pretrained_weights(model_path)

    # Load AQ1 checkpoint
    if verbose:
        print(f"Loading AQ1 checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # Detect V2 format
    is_v2 = any('._Q' in k for k in state_dict.keys())
    if verbose:
        print(f"Checkpoint format: {'V2 (with _Q buffers)' if is_v2 else 'V1'}")

    baked_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.kernel_size != (1, 1):
            continue

        # Try different key formats
        base_keys = [
            name,
            name.replace('model.model.', 'model.'),
            name.replace('model.', ''),
            f'model.{name}',
        ]

        for base_key in base_keys:
            # V2 format: use _Q for snapped weights
            q_key = f'{base_key}._Q'
            weight_key = f'{base_key}.weight'
            scale_a_key = f'{base_key}.scale_A'
            scale_b_key = f'{base_key}.scale_B'
            mag_key = f'{base_key}.rank_magnitude'

            # Check if this is a quantized layer
            has_scales = scale_a_key in state_dict and scale_b_key in state_dict
            if not has_scales:
                continue

            # Get snapped weights - prefer _Q (V2) over weight (V1)
            if q_key in state_dict:
                snapped = state_dict[q_key].to(torch.float32)
            elif weight_key in state_dict:
                snapped = state_dict[weight_key].to(torch.float32)
            else:
                continue

            scale_A = state_dict[scale_a_key].to(torch.float32)
            scale_B = state_dict[scale_b_key].to(torch.float32)

            # V2 format: incorporate rank_magnitude into scale_A
            if mag_key in state_dict:
                rank_magnitude = state_dict[mag_key].to(torch.float32)
                scale_A = scale_A * rank_magnitude.unsqueeze(0)

            # Compute baked weight: snapped * (A @ B)
            scales = (scale_A @ scale_B).clamp(min=1e-8)
            if snapped.dim() == 4:
                snapped = snapped.squeeze(-1).squeeze(-1)
            baked = snapped * scales
            baked_4d = baked.view(baked.shape[0], baked.shape[1], 1, 1)

            # Load into module
            with torch.no_grad():
                module.weight.data.copy_(baked_4d.to(module.weight.dtype))

            baked_count += 1
            if verbose and baked_count <= 3:
                print(f"  Baked {base_key}: snapped{list(snapped.shape)} * scales -> weight")
            break

    if verbose:
        print(f"  Baked {baked_count} projection layers with AQ1 quantization")

    # Load non-quantized weights from checkpoint (embed_tokens, norms, lm_head)
    loaded_other = 0
    for key, value in state_dict.items():
        # Skip quantization-related keys
        if any(x in key for x in ['.scale_A', '.scale_B', '.lut', '.rank_magnitude', '._Q', '_baked_flag']):
            continue
        # Skip projection layer weights (already baked)
        if any(proj in key for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
            continue

        parts = key.split('.')
        try:
            param = model
            for part in parts:
                if part.isdigit():
                    param = param[int(part)]
                else:
                    param = getattr(param, part)

            if isinstance(param, nn.Parameter) and param.shape == value.shape:
                param.data.copy_(value.to(param.dtype))
                loaded_other += 1
        except (AttributeError, IndexError, KeyError):
            pass

    if verbose:
        print(f"  Loaded {loaded_other} non-projection weights from checkpoint")

    # Load tokenizer
    tokenizer = load_tokenizer(model_id)

    return model, tokenizer, config


@torch.no_grad()
def generate(
    model: QwenForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    no_think: bool = False,
    verbose: bool = False,
    stream: bool = True,
) -> str:
    """Generate text using the model."""

    # Apply chat template - use enable_thinking=False for no-think mode (official Qwen3 approach)
    messages = [{"role": "user", "content": prompt}]
    template_kwargs = {
        "tokenize": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    if no_think:
        template_kwargs["enable_thinking"] = False

    input_ids = tokenizer.apply_chat_template(messages, **template_kwargs)

    batch_size, seq_len = input_ids.shape
    state_length = model.config.state_length

    # Reset KV cache
    if hasattr(model.model, 'kv_cache_0'):
        model.model.kv_cache_0.zero_()

    model.eval()

    # Prefill phase
    if verbose:
        print(f"Prefilling {seq_len} tokens...")

    t_prefill = time.time()
    position_ids = torch.arange(seq_len)
    causal_mask = create_causal_mask(seq_len, state_length)
    update_mask = torch.ones((batch_size, seq_len), dtype=MODEL_DTYPE)

    logits = model(
        input_ids=input_ids,
        update_mask=update_mask,
        position_ids=position_ids,
        causal_mask=causal_mask,
        current_pos=0,
        IN_PREFILL=True,
    )

    prefill_time = time.time() - t_prefill

    # Get first token
    if logits.dim() == 3:
        next_logits = logits[:, -1, :]
    else:
        next_logits = logits

    if temperature > 0:
        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = next_logits.argmax(dim=-1, keepdim=True)

    generated_tokens = [next_token.item()]

    if verbose:
        print(f"Prefill: {prefill_time:.3f}s")

    if stream:
        token_str = tokenizer.decode([next_token.item()], skip_special_tokens=True)
        print(token_str, end="", flush=True)

    # Decode phase
    current_pos = seq_len
    t_decode = time.time()

    for _ in range(max_new_tokens - 1):
        if current_pos >= state_length:
            if verbose:
                print(f"\nReached state_length limit ({state_length})")
            break

        # Check for EOS
        if next_token.item() in [151643, 151644, 151645]:  # Qwen EOS tokens
            break

        single_mask = create_causal_mask(1, state_length, current_pos=current_pos)
        single_update = torch.ones((batch_size, 1), dtype=MODEL_DTYPE)

        logits = model(
            input_ids=next_token,
            update_mask=single_update,
            position_ids=torch.tensor([current_pos]),
            causal_mask=single_mask,
            current_pos=current_pos,
            IN_PREFILL=False,
        )

        if logits.dim() == 3:
            next_logits = logits[:, -1, :]
        else:
            next_logits = logits

        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        generated_tokens.append(next_token.item())
        current_pos += 1

        if stream:
            token_str = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(token_str, end="", flush=True)

    decode_time = time.time() - t_decode
    tokens_per_sec = len(generated_tokens) / decode_time if decode_time > 0 else 0

    if stream:
        print()  # Newline

    if verbose:
        print(f"Decode: {len(generated_tokens)} tokens in {decode_time:.3f}s ({tokens_per_sec:.1f} tok/s)")

    # Decode full response
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    response = response.replace('<|im_end|>', '').strip()

    return response


def compare_with_ane(
    model: QwenForCausalLM,
    tokenizer,
    prompt: str,
    coreml_model_path: Optional[str] = None,
    verbose: bool = False,
) -> tuple:
    """Compare PyTorch inference with CoreML/ANE inference.

    Returns (pytorch_response, coreml_response, match_percentage)
    """
    print("\n" + "=" * 60)
    print("Comparing PyTorch vs CoreML/ANE Inference")
    print("=" * 60)

    # PyTorch inference
    print("\n[PyTorch]")
    t0 = time.time()
    pytorch_response = generate(model, tokenizer, prompt, verbose=verbose, stream=True)
    pytorch_time = time.time() - t0

    # CoreML inference
    if coreml_model_path is None:
        print("\n[CoreML/ANE] No CoreML model path provided. Skipping ANE comparison.")
        return pytorch_response, None, None

    try:
        import coremltools as ct

        print(f"\n[CoreML/ANE] Loading model from {coreml_model_path}...")
        # This would require the full CoreML inference pipeline
        # For now, we just show the PyTorch result
        print("CoreML inference not yet implemented in this script.")
        print("Use the converted .mlpackage models with chat.py for ANE inference.")

        return pytorch_response, None, None

    except ImportError:
        print("\n[CoreML/ANE] coremltools not available. Install with: pip install coremltools")
        return pytorch_response, None, None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Simple inference test for Qwen with AQ1 quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('checkpoint', type=str,
                        help='Path to AQ1 checkpoint .pt file')
    parser.add_argument('--prompt', '-p', type=str, default='What is the capital of France?',
                        help='Prompt for generation')
    parser.add_argument('--model-id', '-m', type=str, default='Qwen/Qwen3-0.6B',
                        help='HuggingFace model ID')
    parser.add_argument('--config', '-c', type=str, default='auto',
                        choices=list(CONFIG_PRESETS.keys()),
                        help='Quantization config preset (default: auto-detect)')
    parser.add_argument('--max-tokens', '-n', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', '-t', type=float, default=0.0,
                        help='Sampling temperature (0 for greedy)')
    parser.add_argument('--context-length', type=int, default=512,
                        help='Context length')
    parser.add_argument('--state-length', type=int, default=512,
                        help='KV cache state length')
    parser.add_argument('--no-think', action='store_true',
                        help='Disable thinking mode (uses enable_thinking=False)')
    parser.add_argument('--compare-ane', action='store_true',
                        help='Compare with CoreML/ANE inference')
    parser.add_argument('--coreml-model', type=str, default=None,
                        help='Path to CoreML model for ANE comparison')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--no-stream', action='store_true',
                        help='Disable streaming output')

    args = parser.parse_args()

    # Expand checkpoint path
    checkpoint_path = os.path.expanduser(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("=" * 60)
    print("ANEMLL AQ1 Inference Test")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model ID: {args.model_id}")
    print(f"Prompt: {args.prompt}")
    if args.no_think:
        print(f"Mode: No-think (skip thinking tokens)")

    # Detect or use preset config
    if args.config == 'auto':
        quant_config = detect_checkpoint_config(checkpoint_path)
        print(f"Auto-detected config: {quant_config}")
    else:
        quant_config = CONFIG_PRESETS[args.config]
        print(f"Using config preset: {args.config}")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    model, tokenizer, config = load_model_with_aq1_weights(
        model_id=args.model_id,
        checkpoint_path=checkpoint_path,
        context_length=args.context_length,
        state_length=args.state_length,
        verbose=args.verbose,
    )
    print(f"Model loaded in {time.time() - t0:.2f}s")

    # Generate
    print("\n" + "-" * 60)
    print("Response:")
    print("-" * 60)

    if args.compare_ane:
        pytorch_response, coreml_response, match_pct = compare_with_ane(
            model, tokenizer, args.prompt,
            coreml_model_path=args.coreml_model,
            verbose=args.verbose,
        )
    else:
        response = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            no_think=args.no_think,
            verbose=args.verbose,
            stream=not args.no_stream,
        )

        if args.no_stream:
            print(response)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
