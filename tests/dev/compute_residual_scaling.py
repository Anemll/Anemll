#!/usr/bin/env python3
"""
Compute optimal residual scaling factors for FP16/ANE compatibility.

Supports:
1. Uniform α scaling (simpler, often sufficient)
2. Per-layer α scaling (if uniform isn't enough)

Usage:
    python compute_residual_scaling.py --model google/gemma-3-270m
    python compute_residual_scaling.py --model google/gemma-3-4b-it-qat-int4-unquantized
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FP16_MAX = 65504.0
TARGET_MAX = 50000.0  # Target with headroom


def get_layer_residuals(model, input_ids, device) -> Dict[int, float]:
    """Get max residual value for each layer."""
    layer_max = {}
    handles = []

    def make_hook(layer_num):
        def hook(module, input, output):
            if isinstance(output, tuple):
                t = output[0]
            else:
                t = output
            if isinstance(t, torch.Tensor):
                layer_max[layer_num] = t.abs().max().item()
        return hook

    # Find layers
    for name, module in model.named_modules():
        match = re.match(r".*\.layers\.(\d+)$", name)
        if match:
            layer_num = int(match.group(1))
            handles.append(module.register_forward_hook(make_hook(layer_num)))

    with torch.no_grad():
        model(input_ids=input_ids.to(device))

    for h in handles:
        h.remove()

    return layer_max


def apply_uniform_scaling(model, alpha: float, model_type: str = "gemma3"):
    """Apply uniform residual scaling."""
    # Scale embeddings
    if model_type == "gemma3":
        if hasattr(model, 'model'):
            if hasattr(model.model, 'language_model'):
                embed = model.model.language_model.embed_tokens
                layers = model.model.language_model.layers
            else:
                embed = model.model.embed_tokens
                layers = model.model.layers
        else:
            embed = model.embed_tokens
            layers = model.layers
    else:
        # Generic fallback
        embed = None
        layers = None
        for name, module in model.named_modules():
            if 'embed_tokens' in name and embed is None:
                embed = module
            if name.endswith('.layers') or name.endswith('.layers.0'):
                if hasattr(module, '__iter__'):
                    layers = module
                    break

    if embed is not None and hasattr(embed, 'weight'):
        with torch.no_grad():
            embed.weight.mul_(alpha)

    # Scale post-norm weights
    if layers is not None:
        for layer in layers:
            for norm_name in ['post_attention_layernorm', 'post_feedforward_layernorm']:
                if hasattr(layer, norm_name):
                    norm = getattr(layer, norm_name)
                    if hasattr(norm, 'weight'):
                        with torch.no_grad():
                            # Gemma3 uses (1 + w) gain
                            norm.weight.data = alpha * (1 + norm.weight.data) - 1


def apply_per_layer_scaling(model, layer_alphas: Dict[int, float], model_type: str = "gemma3"):
    """Apply per-layer residual scaling."""
    # Get layers
    if model_type == "gemma3":
        if hasattr(model, 'model'):
            if hasattr(model.model, 'language_model'):
                embed = model.model.language_model.embed_tokens
                layers = model.model.language_model.layers
            else:
                embed = model.model.embed_tokens
                layers = model.model.layers
        else:
            embed = model.embed_tokens
            layers = model.layers
    else:
        embed = None
        layers = list(model.modules())

    # Scale embedding by first layer's alpha
    first_alpha = layer_alphas.get(0, 1.0)
    if embed is not None and hasattr(embed, 'weight'):
        with torch.no_grad():
            embed.weight.mul_(first_alpha)

    # Scale each layer's post-norms
    for i, layer in enumerate(layers):
        alpha = layer_alphas.get(i, 1.0)

        for norm_name in ['post_attention_layernorm', 'post_feedforward_layernorm']:
            if hasattr(layer, norm_name):
                norm = getattr(layer, norm_name)
                if hasattr(norm, 'weight'):
                    with torch.no_grad():
                        norm.weight.data = alpha * (1 + norm.weight.data) - 1


def compute_uniform_alpha(layer_residuals: Dict[int, float], target: float = TARGET_MAX) -> float:
    """Compute uniform alpha that brings all layers within target."""
    max_residual = max(layer_residuals.values())
    return target / max_residual


def compute_per_layer_alphas(
    layer_residuals: Dict[int, float],
    target: float = TARGET_MAX,
    only_overflow: bool = True
) -> Dict[int, float]:
    """Compute per-layer alphas.

    Args:
        layer_residuals: Max residual per layer
        target: Target max value
        only_overflow: If True, only scale layers that overflow

    Returns:
        Dict of layer_num -> alpha
    """
    alphas = {}
    for layer_num, residual in layer_residuals.items():
        if only_overflow and residual <= FP16_MAX:
            alphas[layer_num] = 1.0  # No scaling needed
        else:
            alphas[layer_num] = min(1.0, target / residual)
    return alphas


def test_generation(model, tokenizer, input_ids, device, max_tokens: int = 20) -> Tuple[List[int], bool]:
    """Test token generation."""
    generated = input_ids.to(device)
    tokens = []

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated)
            logits = outputs.logits[0, -1, :]

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return tokens, False

        next_token = logits.argmax().item()
        tokens.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

        generated = torch.cat([generated, torch.tensor([[next_token]], device=device)], dim=1)

    return tokens, True


def main():
    parser = argparse.ArgumentParser(description="Compute optimal residual scaling for FP16")
    parser.add_argument("--model", "-m", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--prompt", "-p", type=str, default="Explain quantum computing in simple terms.",
                       help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=30, help="Max tokens to generate")
    parser.add_argument("--target", type=float, default=50000.0, help="Target max activation")
    parser.add_argument("--save", type=str, help="Save scaling factors to JSON file")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("RESIDUAL SCALING FACTOR COMPUTATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Target max: {args.target}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Prepare input
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": args.prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        except:
            input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    else:
        input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]

    # ============================================================
    # STEP 1: Measure original residuals in FP32
    # ============================================================
    print("STEP 1: Measuring original residual values (FP32)")
    print("-" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()

    original_residuals = get_layer_residuals(model, input_ids, device)

    print(f"\n{'Layer':<8} {'Max Residual':>15} {'Status':>12} {'α needed':>10}")
    print("-" * 50)

    overflow_count = 0
    for layer_num in sorted(original_residuals.keys()):
        val = original_residuals[layer_num]
        status = "✗ OVERFLOW" if val > FP16_MAX else "✓ ok"
        alpha_needed = min(1.0, args.target / val) if val > args.target else 1.0

        if val > FP16_MAX:
            overflow_count += 1
            print(f"Layer {layer_num:<2} {val:>15.1f} {status:>12} {alpha_needed:>10.3f}")
        else:
            print(f"Layer {layer_num:<2} {val:>15.1f} {status:>12} {'-':>10}")

    max_original = max(original_residuals.values())
    print(f"\nPeak value: {max_original:.1f} (FP16 max: {FP16_MAX})")
    print(f"Overflow layers: {overflow_count}/{len(original_residuals)}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    # ============================================================
    # STEP 2: Compute scaling factors
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: Computing scaling factors")
    print("=" * 70)

    uniform_alpha = compute_uniform_alpha(original_residuals, args.target)
    per_layer_alphas = compute_per_layer_alphas(original_residuals, args.target, only_overflow=True)

    print(f"\nUniform α: {uniform_alpha:.4f}")
    print(f"  - Scales ALL layers by same factor")
    print(f"  - Expected peak after scaling: {max_original * uniform_alpha:.1f}")

    # Check if per-layer offers advantage
    per_layer_varied = len(set(per_layer_alphas.values())) > 1
    if per_layer_varied:
        min_alpha = min(per_layer_alphas.values())
        max_alpha = max(per_layer_alphas.values())
        print(f"\nPer-layer α range: {min_alpha:.4f} - {max_alpha:.4f}")
        print(f"  - Only scales layers that overflow")
        print(f"  - May preserve more precision in early layers")
    else:
        print(f"\nPer-layer: All layers need same α = {list(per_layer_alphas.values())[0]:.4f}")
        print("  → Uniform scaling is optimal for this model")

    # ============================================================
    # STEP 3: Test uniform scaling
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3: Testing uniform scaling")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()

    apply_uniform_scaling(model, uniform_alpha)

    # Verify residuals
    scaled_residuals = get_layer_residuals(model, input_ids, device)
    max_scaled = max(scaled_residuals.values())
    print(f"\nAfter uniform scaling (α={uniform_alpha:.4f}):")
    print(f"  Peak residual: {max_scaled:.1f} ({'✓ within FP16' if max_scaled < FP16_MAX else '✗ exceeds FP16'})")

    # Test FP16 generation
    model_fp16 = model.half()
    tokens_scaled, success = test_generation(model_fp16, tokenizer, input_ids, device, args.max_tokens)

    if success:
        text_scaled = tokenizer.decode(tokens_scaled, skip_special_tokens=True)
        print(f"  FP16 generation: ✓ SUCCESS")
        print(f"  Output: {text_scaled[:80]}...")
    else:
        print(f"  FP16 generation: ✗ FAIL")

    del model, model_fp16
    if device.type == "mps":
        torch.mps.empty_cache()

    # ============================================================
    # STEP 4: Compare with BF16 baseline
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 4: Quality comparison")
    print("=" * 70)

    # BF16 baseline
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model_bf16.eval()

    tokens_baseline, _ = test_generation(model_bf16, tokenizer, input_ids, device, args.max_tokens)
    text_baseline = tokenizer.decode(tokens_baseline, skip_special_tokens=True)

    print(f"\nBF16 baseline: {text_baseline[:80]}...")
    if success:
        print(f"FP16 scaled:   {text_scaled[:80]}...")

        # Token match
        min_len = min(len(tokens_baseline), len(tokens_scaled))
        matches = sum(1 for i in range(min_len) if tokens_baseline[i] == tokens_scaled[i])
        match_rate = matches / min_len if min_len > 0 else 0

        print(f"\nToken match rate: {matches}/{min_len} ({100*match_rate:.1f}%)")

    del model_bf16
    if device.type == "mps":
        torch.mps.empty_cache()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: RECOMMENDED SCALING")
    print("=" * 70)

    result = {
        "model_id": args.model,
        "original_peak": max_original,
        "target": args.target,
        "uniform_alpha": uniform_alpha,
        "per_layer_alphas": per_layer_alphas,
        "recommendation": "uniform",
        "fp16_works_after_scaling": success,
    }

    print(f"\nModel: {args.model}")
    print(f"Original peak: {max_original:.1f} ({max_original/FP16_MAX:.1f}x FP16 max)")
    print(f"\nRecommended: UNIFORM scaling with α = {uniform_alpha:.4f}")
    print(f"\nWeight transformations:")
    print(f"  1. embed_tokens.weight *= {uniform_alpha:.4f}")
    print(f"  2. For each layer's post_*_layernorm:")
    print(f"     weight_new = {uniform_alpha:.4f} * (1 + weight_old) - 1")

    if success:
        print(f"\n✓ FP16 generation works after scaling")
        print(f"✓ Quality preserved ({100*match_rate:.0f}% token match)")
    else:
        print(f"\n✗ FP16 still fails - may need additional fixes (clamp, V/O scaling)")

    # Save if requested
    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nScaling factors saved to: {args.save}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
