#!/usr/bin/env python3
"""
Test FP16 scaling on Gemma3 models.

Tests different scaling configurations:
1. BF16 baseline (reference)
2. Embed-only FP16 with scaling
3. Full FP16 scaling (embed + post-norms)

Usage:
    # Embed only
    python tests/dev/test_gemma3_embed_fp16_only.py \
        --model google/gemma-3-4b-it-qat-int4-unquantized \
        --alpha 0.1875 --mode embed-only

    # Full scaling (embed + post-norms)
    python tests/dev/test_gemma3_embed_fp16_only.py \
        --model google/gemma-3-4b-it-qat-int4-unquantized \
        --alpha 0.1875 --mode full
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def analyze_weights(model_id: str):
    """Analyze the 69 scaled tensors before any transformation."""
    print(f"\nLoading model for weight analysis: {model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Get model components
    if hasattr(model.model, 'language_model'):
        embed = model.model.language_model.embed_tokens
        layers = model.model.language_model.layers
    else:
        embed = model.model.embed_tokens
        layers = model.model.layers

    print("\n" + "="*90)
    print("WEIGHT ANALYSIS (before scaling)")
    print("="*90)

    results = []

    # 1. Embedding weights
    w = embed.weight.data.float()
    stats = {
        'name': 'embed_tokens.weight',
        'shape': tuple(w.shape),
        'min': w.min().item(),
        'max': w.max().item(),
        'abs_max': w.abs().max().item(),
        'mean': w.mean().item(),
        'std': w.std().item(),
        'zeros': (w == 0).sum().item(),
        'unique_approx': min(w.numel(), len(torch.unique(w[:1000].flatten()))),  # Sample
    }
    results.append(stats)

    # 2. Post-norm weights
    for layer_idx, layer in enumerate(layers):
        for norm_name in ['post_attention_layernorm', 'post_feedforward_layernorm']:
            if hasattr(layer, norm_name):
                norm = getattr(layer, norm_name)
                if hasattr(norm, 'weight'):
                    w = norm.weight.data.float()
                    stats = {
                        'name': f'layers.{layer_idx}.{norm_name}.weight',
                        'shape': tuple(w.shape),
                        'min': w.min().item(),
                        'max': w.max().item(),
                        'abs_max': w.abs().max().item(),
                        'mean': w.mean().item(),
                        'std': w.std().item(),
                        'zeros': (w == 0).sum().item(),
                        'unique_approx': len(torch.unique(w)),
                    }
                    results.append(stats)

    # Print summary table
    print(f"\n{'Tensor':<55} {'Shape':<15} {'Min':>10} {'Max':>10} {'AbsMax':>10} {'Mean':>10} {'Std':>8}")
    print("-"*120)

    # Embed
    s = results[0]
    print(f"{s['name']:<55} {str(s['shape']):<15} {s['min']:>10.4f} {s['max']:>10.4f} {s['abs_max']:>10.2f} {s['mean']:>10.6f} {s['std']:>8.4f}")

    # Post-norms (sample)
    print("\nPost-norm weights (first 5 layers):")
    for s in results[1:11]:  # First 10 post-norms (5 layers x 2)
        print(f"  {s['name']:<53} {str(s['shape']):<15} {s['min']:>10.4f} {s['max']:>10.4f} {s['abs_max']:>10.4f} {s['mean']:>10.6f} {s['std']:>8.4f}")

    # Aggregate stats for post-norms
    post_norms = results[1:]
    if post_norms:
        all_mins = [s['min'] for s in post_norms]
        all_maxs = [s['max'] for s in post_norms]
        all_abs_maxs = [s['abs_max'] for s in post_norms]
        all_means = [s['mean'] for s in post_norms]
        all_stds = [s['std'] for s in post_norms]

        print(f"\nPost-norm aggregate ({len(post_norms)} tensors):")
        print(f"  Min range:     [{min(all_mins):.4f}, {max(all_mins):.4f}]")
        print(f"  Max range:     [{min(all_maxs):.4f}, {max(all_maxs):.4f}]")
        print(f"  AbsMax range:  [{min(all_abs_maxs):.4f}, {max(all_abs_maxs):.4f}]")
        print(f"  Mean range:    [{min(all_means):.6f}, {max(all_means):.6f}]")
        print(f"  Std range:     [{min(all_stds):.4f}, {max(all_stds):.4f}]")

    # Check for QAT indicators
    print("\n" + "-"*90)
    print("QAT INDICATORS:")
    print("-"*90)

    # QAT typically shows:
    # - Clustered/quantized values (few unique values)
    # - Specific value ranges matching quantization grid
    embed_unique = results[0]['unique_approx']
    print(f"  Embed unique values (sample): {embed_unique}")

    # Check if post-norm weights look quantized
    post_norm_uniques = [s['unique_approx'] for s in post_norms[:10]]
    print(f"  Post-norm unique values: {post_norm_uniques}")

    # Histogram of embedding values
    embed_w = embed.weight.data.float().flatten()
    hist, bins = np.histogram(embed_w.numpy(), bins=50)
    peak_bin = np.argmax(hist)
    print(f"  Embed histogram peak at: [{bins[peak_bin]:.4f}, {bins[peak_bin+1]:.4f}] ({hist[peak_bin]} values)")

    del model
    return results


def load_model_bf16(model_id: str):
    """Load model in BF16 for baseline."""
    print(f"Loading BF16 baseline: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_model_embed_fp16(model_id: str, alpha: float):
    """Load model with only embed_tokens in FP16 (scaled)."""
    print(f"Loading model with embed FP16 only, α={alpha}")

    # Load in BF16 first
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Get embedding layer
    if hasattr(model.model, 'language_model'):
        embed = model.model.language_model.embed_tokens
    else:
        embed = model.model.embed_tokens

    # Scale and convert to FP16
    with torch.no_grad():
        original_max = embed.weight.abs().max().item()
        embed.weight.mul_(alpha)
        scaled_max = embed.weight.abs().max().item()
        embed.weight.data = embed.weight.data.to(torch.float16)
        fp16_max = embed.weight.abs().max().item()

    print(f"  embed_tokens.weight:")
    print(f"    Original max: {original_max:.2f}")
    print(f"    After α={alpha}: {scaled_max:.2f}")
    print(f"    After FP16 cast: {fp16_max:.2f}")
    print(f"    FP16 safe: {fp16_max < 65504}")

    model.eval()
    return model


def load_model_full_fp16(model_id: str, alpha: float):
    """Load model with full FP16 scaling (embed + post-norms)."""
    print(f"Loading model with FULL FP16 scaling, α={alpha}")

    # Load in BF16 first
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Get model components
    if hasattr(model.model, 'language_model'):
        embed = model.model.language_model.embed_tokens
        layers = model.model.language_model.layers
    else:
        embed = model.model.embed_tokens
        layers = model.model.layers

    scaled_tensors = []

    # 1. Scale embedding weights and convert to FP16
    with torch.no_grad():
        original_max = embed.weight.abs().max().item()
        embed.weight.mul_(alpha)
        embed.weight.data = embed.weight.data.to(torch.float16)
        scaled_tensors.append(("embed_tokens.weight", original_max, embed.weight.abs().max().item()))

    # 2. Transform post-norm weights: w_new = α * (1 + w_old) - 1
    # And convert to FP16
    for layer_idx, layer in enumerate(layers):
        for norm_name in ['post_attention_layernorm', 'post_feedforward_layernorm']:
            if hasattr(layer, norm_name):
                norm = getattr(layer, norm_name)
                if hasattr(norm, 'weight'):
                    with torch.no_grad():
                        original_max = norm.weight.abs().max().item()
                        # Gemma3 uses (1 + w) gain, so transform is: w_new = α * (1 + w_old) - 1
                        norm.weight.data = alpha * (1 + norm.weight.data) - 1
                        norm.weight.data = norm.weight.data.to(torch.float16)
                        new_max = norm.weight.abs().max().item()
                        scaled_tensors.append((
                            f"layers.{layer_idx}.{norm_name}.weight",
                            original_max, new_max
                        ))

    print(f"\n  Scaled {len(scaled_tensors)} tensors to FP16:")
    print(f"  {'Tensor':<50} {'Orig Max':>10} {'New Max':>10} {'Safe':>6}")
    print(f"  {'-'*80}")
    for name, orig, new in scaled_tensors[:5]:  # Show first 5
        safe = "✓" if new < 65504 else "✗"
        print(f"  {name:<50} {orig:>10.2f} {new:>10.4f} {safe:>6}")
    if len(scaled_tensors) > 5:
        print(f"  ... and {len(scaled_tensors) - 5} more tensors")

    # Check all are FP16 safe
    unsafe = [(n, o, m) for n, o, m in scaled_tensors if m >= 65504]
    if unsafe:
        print(f"\n  ⚠️  WARNING: {len(unsafe)} tensors exceed FP16 max!")
        for name, orig, new in unsafe[:3]:
            print(f"    {name}: {new:.2f}")

    model.eval()
    return model


def run_inference(model, tokenizer, prompt: str, max_tokens: int = 20):
    """Run greedy inference and return tokens + logits."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    generated_tokens = []
    all_logits = []

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # Last position
            all_logits.append(logits.float().cpu())

            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop on EOS
            if next_token.item() in [tokenizer.eos_token_id, 1, 2]:
                break

    return generated_tokens, all_logits


def compare_outputs(tokens_a, logits_a, tokens_b, logits_b, tokenizer):
    """Compare two inference runs."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    # Token match
    min_len = min(len(tokens_a), len(tokens_b))
    matches = sum(1 for i in range(min_len) if tokens_a[i] == tokens_b[i])
    match_rate = matches / min_len if min_len > 0 else 0

    print(f"\nToken Match: {matches}/{min_len} ({match_rate*100:.1f}%)")

    # Show tokens side by side
    print("\nToken comparison:")
    print(f"{'Pos':>4} {'BF16':>8} {'FP16':>8} {'Match':>6}")
    print("-" * 30)
    for i in range(min_len):
        match = "✓" if tokens_a[i] == tokens_b[i] else "✗"
        print(f"{i:>4} {tokens_a[i]:>8} {tokens_b[i]:>8} {match:>6}")

    # Logit statistics
    print("\nLogit statistics per position:")
    print(f"{'Pos':>4} {'Corr':>8} {'MaxDiff':>10} {'KL':>10}")
    print("-" * 40)

    for i in range(min(min_len, len(logits_a), len(logits_b))):
        la = logits_a[i].squeeze()
        lb = logits_b[i].squeeze()

        # Correlation
        corr = torch.corrcoef(torch.stack([la, lb]))[0, 1].item()

        # Max difference
        max_diff = (la - lb).abs().max().item()

        # KL divergence
        pa = torch.softmax(la, dim=-1)
        pb = torch.softmax(lb, dim=-1)
        kl = (pa * (pa.log() - pb.log())).sum().item()

        print(f"{i:>4} {corr:>8.4f} {max_diff:>10.2f} {kl:>10.4f}")

    # Decode text
    print("\n" + "-"*60)
    print("Generated text:")
    print(f"  BF16: {tokenizer.decode(tokens_a)}")
    print(f"  FP16: {tokenizer.decode(tokens_b)}")

    return match_rate


def main():
    parser = argparse.ArgumentParser(description="Test FP16 scaling modes")
    parser.add_argument("--model", type=str,
                        default="google/gemma-3-4b-it-qat-int4-unquantized",
                        help="Model ID")
    parser.add_argument("--alpha", type=float, default=0.1875,
                        help="Scaling factor (default: 0.1875 = 3/16)")
    parser.add_argument("--prompt", type=str,
                        default="The capital of France is",
                        help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Max tokens to generate")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["embed-only", "full", "both", "analyze"],
                        help="Scaling mode: embed-only, full (embed+post-norms), both, or analyze (weights only)")
    args = parser.parse_args()

    # Weight analysis mode
    if args.mode == "analyze":
        analyze_weights(args.model)
        return 0

    print("="*60)
    print("GEMMA3 FP16 SCALING TEST")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Alpha: {args.alpha}")
    print(f"Mode:  {args.mode}")
    print(f"Prompt: {args.prompt}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Test 1: BF16 baseline
    print("\n[1] BF16 Baseline")
    print("-"*40)
    model_bf16 = load_model_bf16(args.model)
    tokens_bf16, logits_bf16 = run_inference(
        model_bf16, tokenizer, args.prompt, args.max_tokens
    )
    del model_bf16
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results = {}

    # Test 2: Embed-only FP16
    if args.mode in ["embed-only", "both"]:
        print("\n[2] Embed-Only FP16 (scaled)")
        print("-"*40)
        model_embed = load_model_embed_fp16(args.model, args.alpha)
        tokens_embed, logits_embed = run_inference(
            model_embed, tokenizer, args.prompt, args.max_tokens
        )
        del model_embed
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        match_rate = compare_outputs(
            tokens_bf16, logits_bf16,
            tokens_embed, logits_embed,
            tokenizer
        )
        results["embed-only"] = match_rate

    # Test 3: Full FP16 (embed + post-norms)
    if args.mode in ["full", "both"]:
        print("\n[3] Full FP16 (embed + 68 post-norms)")
        print("-"*40)
        model_full = load_model_full_fp16(args.model, args.alpha)
        tokens_full, logits_full = run_inference(
            model_full, tokenizer, args.prompt, args.max_tokens
        )
        del model_full

        match_rate = compare_outputs(
            tokens_bf16, logits_bf16,
            tokens_full, logits_full,
            tokenizer
        )
        results["full"] = match_rate

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for mode, rate in results.items():
        status = "✓ PASS" if rate >= 0.95 else "✗ FAIL"
        print(f"  {mode:<15}: {rate*100:.1f}% match  {status}")

    all_pass = all(r >= 0.95 for r in results.values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
