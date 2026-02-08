#!/usr/bin/env python3
"""
Compute optimal residual scaling factors for FP16/ANE compatibility - Qwen3 version.

Measures per-layer residual activations to check if Qwen3 models need
FP16 scaling like Gemma3 models do.

Usage:
    python compute_residual_scaling_qwen.py --model Qwen/Qwen3-1.7B
    python compute_residual_scaling_qwen.py --model Qwen/Qwen3-0.6B
    python compute_residual_scaling_qwen.py --model Qwen/Qwen3-4B
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


def get_layer_residuals(model, input_ids, device) -> Dict[int, Dict[str, float]]:
    """Get max residual value for each layer and sub-component."""
    layer_stats = defaultdict(dict)
    handles = []

    def make_hook(layer_num, component_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                t = output[0]
            else:
                t = output
            if isinstance(t, torch.Tensor):
                layer_stats[layer_num][component_name] = t.abs().max().item()
        return hook

    # Find layers and components
    for name, module in model.named_modules():
        # Match layer outputs (full decoder layer)
        match = re.match(r".*\.layers\.(\d+)$", name)
        if match:
            layer_num = int(match.group(1))
            handles.append(module.register_forward_hook(make_hook(layer_num, "layer_output")))

        # Match attention output
        match = re.match(r".*\.layers\.(\d+)\.self_attn$", name)
        if match:
            layer_num = int(match.group(1))
            handles.append(module.register_forward_hook(make_hook(layer_num, "attn_output")))

        # Match MLP output
        match = re.match(r".*\.layers\.(\d+)\.mlp$", name)
        if match:
            layer_num = int(match.group(1))
            handles.append(module.register_forward_hook(make_hook(layer_num, "mlp_output")))

        # Match input/post norms (Qwen uses input_layernorm and post_attention_layernorm)
        match = re.match(r".*\.layers\.(\d+)\.input_layernorm$", name)
        if match:
            layer_num = int(match.group(1))
            handles.append(module.register_forward_hook(make_hook(layer_num, "input_norm")))

        match = re.match(r".*\.layers\.(\d+)\.post_attention_layernorm$", name)
        if match:
            layer_num = int(match.group(1))
            handles.append(module.register_forward_hook(make_hook(layer_num, "post_attn_norm")))

    with torch.no_grad():
        model(input_ids=input_ids.to(device))

    for h in handles:
        h.remove()

    return dict(layer_stats)


def get_embedding_stats(model, input_ids, device) -> Dict[str, float]:
    """Get embedding output statistics."""
    stats = {}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                stats[name] = output.abs().max().item()
        return hook

    for name, module in model.named_modules():
        if 'embed_tokens' in name:
            handles.append(module.register_forward_hook(make_hook("embed_tokens")))
            break

    with torch.no_grad():
        model(input_ids=input_ids.to(device))

    for h in handles:
        h.remove()

    return stats


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
    parser = argparse.ArgumentParser(description="Compute optimal residual scaling for FP16 - Qwen3")
    parser.add_argument("--model", "-m", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--prompt", "-p", type=str, default="Explain quantum computing in simple terms.",
                       help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=30, help="Max tokens to generate")
    parser.add_argument("--target", type=float, default=50000.0, help="Target max activation")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("--detailed", action="store_true", help="Show per-component breakdown")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("QWEN3 RESIDUAL SCALING ANALYSIS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"FP16 max: {FP16_MAX}")
    print(f"Target max: {args.target}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Prepare input with chat template
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": args.prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        except Exception as e:
            print(f"Chat template failed: {e}, using raw prompt")
            input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    else:
        input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]

    print(f"Input tokens: {input_ids.shape[1]}")
    print()

    # ============================================================
    # STEP 1: Measure residuals in FP32
    # ============================================================
    print("STEP 1: Measuring residual values (FP32)")
    print("-" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()

    # Get embedding stats
    embed_stats = get_embedding_stats(model, input_ids, device)
    print(f"Embedding output max: {embed_stats.get('embed_tokens', 0):.1f}")

    # Get layer residuals
    layer_stats = get_layer_residuals(model, input_ids, device)

    print(f"\n{'Layer':<8} {'Layer Output':>14} {'Attn':>12} {'MLP':>12} {'Status':>12}")
    print("-" * 62)

    overflow_count = 0
    max_residual = 0
    first_overflow_layer = None

    for layer_num in sorted(layer_stats.keys()):
        stats = layer_stats[layer_num]
        layer_out = stats.get("layer_output", 0)
        attn_out = stats.get("attn_output", 0)
        mlp_out = stats.get("mlp_output", 0)

        max_residual = max(max_residual, layer_out)

        if layer_out > FP16_MAX:
            status = "OVERFLOW"
            overflow_count += 1
            if first_overflow_layer is None:
                first_overflow_layer = layer_num
        elif layer_out > args.target:
            status = "WARNING"
        else:
            status = "ok"

        print(f"Layer {layer_num:<2} {layer_out:>14.1f} {attn_out:>12.1f} {mlp_out:>12.1f} {status:>12}")

    print("-" * 62)
    print(f"\nPeak layer output: {max_residual:.1f}")
    print(f"FP16 max: {FP16_MAX}")
    print(f"Ratio to FP16 max: {max_residual / FP16_MAX:.2f}x")

    # ============================================================
    # STEP 2: Analysis and recommendations
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: FP16 Compatibility Analysis")
    print("=" * 70)

    needs_scaling = max_residual > FP16_MAX
    close_to_limit = max_residual > (FP16_MAX * 0.8)  # Within 80% of limit

    if needs_scaling:
        alpha = args.target / max_residual
        print(f"\n*** FP16 OVERFLOW DETECTED ***")
        print(f"First overflow at layer: {first_overflow_layer}")
        print(f"Overflow layers: {overflow_count}/{len(layer_stats)}")
        print(f"\nRecommended scaling factor (alpha): {alpha:.4f}")
        print(f"  This would reduce peak from {max_residual:.1f} to {max_residual * alpha:.1f}")
    elif close_to_limit:
        alpha = args.target / max_residual
        print(f"\n*** CLOSE TO FP16 LIMIT ***")
        print(f"Peak is {max_residual / FP16_MAX:.1%} of FP16 max")
        print(f"Consider scaling for safety headroom")
        print(f"Suggested alpha for headroom: {alpha:.4f}")
    else:
        alpha = 1.0
        print(f"\n*** FP16 COMPATIBLE ***")
        print(f"Peak ({max_residual:.1f}) is well within FP16 range ({FP16_MAX})")
        print(f"No --fp16-scale needed for this model")

    # ============================================================
    # STEP 3: Test FP16 generation
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3: FP16 Generation Test")
    print("=" * 70)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    # Test FP16 directly
    print("\nLoading model in FP16...")
    try:
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device)
        model_fp16.eval()

        tokens_fp16, fp16_success = test_generation(model_fp16, tokenizer, input_ids, device, args.max_tokens)

        if fp16_success:
            text_fp16 = tokenizer.decode(tokens_fp16, skip_special_tokens=True)
            print(f"FP16 generation: SUCCESS")
            print(f"Output: {text_fp16[:100]}...")
        else:
            print(f"FP16 generation: FAILED (NaN/Inf detected)")
            text_fp16 = ""

        del model_fp16
        if device.type == "mps":
            torch.mps.empty_cache()

    except Exception as e:
        print(f"FP16 test failed: {e}")
        fp16_success = False
        text_fp16 = ""

    # ============================================================
    # STEP 4: BF16 baseline comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 4: BF16 Baseline Comparison")
    print("=" * 70)

    print("\nLoading model in BF16...")
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model_bf16.eval()

    tokens_bf16, bf16_success = test_generation(model_bf16, tokenizer, input_ids, device, args.max_tokens)

    if bf16_success:
        text_bf16 = tokenizer.decode(tokens_bf16, skip_special_tokens=True)
        print(f"BF16 generation: SUCCESS")
        print(f"Output: {text_bf16[:100]}...")
    else:
        print(f"BF16 generation: FAILED")
        text_bf16 = ""

    # Compare tokens
    if fp16_success and bf16_success:
        min_len = min(len(tokens_fp16), len(tokens_bf16))
        if min_len > 0:
            matches = sum(1 for i in range(min_len) if tokens_fp16[i] == tokens_bf16[i])
            match_rate = matches / min_len
            print(f"\nFP16 vs BF16 token match: {matches}/{min_len} ({100*match_rate:.1f}%)")
        else:
            match_rate = 0
    else:
        match_rate = 0 if not fp16_success else 1.0

    del model_bf16
    if device.type == "mps":
        torch.mps.empty_cache()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    result = {
        "model_id": args.model,
        "peak_residual": max_residual,
        "fp16_max": FP16_MAX,
        "ratio_to_fp16_max": max_residual / FP16_MAX,
        "needs_scaling": needs_scaling,
        "close_to_limit": close_to_limit,
        "recommended_alpha": alpha if (needs_scaling or close_to_limit) else None,
        "first_overflow_layer": first_overflow_layer,
        "overflow_layers": overflow_count,
        "total_layers": len(layer_stats),
        "fp16_generation_works": fp16_success,
        "bf16_generation_works": bf16_success,
        "fp16_bf16_match_rate": match_rate,
        "layer_stats": {k: v for k, v in layer_stats.items()},
    }

    print(f"\nModel: {args.model}")
    print(f"Peak residual: {max_residual:.1f} ({max_residual/FP16_MAX:.2f}x FP16 max)")

    if needs_scaling:
        print(f"\n*** SCALING REQUIRED ***")
        print(f"Recommended --fp16-scale: {alpha:.4f}")
        print(f"First overflow: Layer {first_overflow_layer}")
    elif close_to_limit:
        print(f"\n*** SCALING RECOMMENDED (for safety) ***")
        print(f"Suggested --fp16-scale: {alpha:.4f}")
    else:
        print(f"\n*** NO SCALING NEEDED ***")
        print(f"Model is FP16 compatible out of the box")

    print(f"\nFP16 works: {'Yes' if fp16_success else 'No'}")
    print(f"BF16 works: {'Yes' if bf16_success else 'No'}")
    if fp16_success and bf16_success:
        print(f"Token match rate: {100*match_rate:.1f}%")

    # Save if requested
    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to: {args.save}")

    return 0 if (not needs_scaling or fp16_success) else 1


if __name__ == "__main__":
    sys.exit(main())
