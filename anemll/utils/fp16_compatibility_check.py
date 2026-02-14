#!/usr/bin/env python3
"""
FP16 Compatibility Check for Apple Neural Engine

Universal diagnostic script to check if a HuggingFace model can run on ANE (FP16).
Identifies overflow issues, problematic layers, and provides compatibility report.

Usage:
    python fp16_compatibility_check.py --model <model_id_or_path> [--prompt "test prompt"]

Examples:
    python fp16_compatibility_check.py --model google/gemma-3-4b-it-qat-int4-unquantized
    python fp16_compatibility_check.py --model meta-llama/Llama-3.1-8B-Instruct
    python fp16_compatibility_check.py --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FP16_MAX = 65504.0
FP16_MIN = -65504.0


@dataclass
class LayerStats:
    """Statistics for a single layer."""
    name: str
    weight_min: float = 0.0
    weight_max: float = 0.0
    weight_abs_max: float = 0.0
    activation_min: float = 0.0
    activation_max: float = 0.0
    activation_abs_max: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    overflow_count: int = 0  # Values exceeding FP16 range


@dataclass
class CompatibilityReport:
    """Full compatibility report."""
    model_id: str
    total_params: int = 0

    # Weight analysis
    weight_max_abs: float = 0.0
    weight_max_layer: str = ""
    weights_exceed_fp16: int = 0

    # Activation analysis per precision
    bf16_works: bool = False
    bf16_output: str = ""

    fp16_works: bool = False
    fp16_output: str = ""
    fp16_first_nan_layer: str = ""
    fp16_overflow_layers: List[str] = field(default_factory=list)

    fp16_to_fp32_works: bool = False
    fp16_to_fp32_output: str = ""

    # Layer-by-layer stats
    layer_stats: Dict[str, LayerStats] = field(default_factory=dict)

    # Clamp sweep results (if needed)
    clamp_sweep: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Overall compatibility
    ane_compatible: bool = False
    compatibility_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class FP16CompatibilityChecker:
    """Check FP16/ANE compatibility for HuggingFace models."""

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = torch.device(
            device if device else
            ("mps" if torch.backends.mps.is_available() else
             "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = None
        self.report = CompatibilityReport(model_id=model_id)

    def _load_tokenizer(self):
        """Load tokenizer if not already loaded."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        return self.tokenizer

    def _prepare_input(self, prompt: str) -> torch.Tensor:
        """Prepare input tokens from prompt."""
        tokenizer = self._load_tokenizer()

        # Try chat template first
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return tokenizer(text, return_tensors="pt")["input_ids"]
            except Exception:
                pass

        # Fallback to direct tokenization
        return tokenizer(prompt, return_tensors="pt")["input_ids"]

    def analyze_weights(self, verbose: bool = True) -> None:
        """Analyze model weights for FP16 compatibility."""
        if verbose:
            print("\n" + "=" * 70)
            print("WEIGHT ANALYSIS")
            print("=" * 70)

        # Load in BF16 to see original precision
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

        self.report.total_params = sum(p.numel() for p in model.parameters())

        weight_stats = []
        exceeds_fp16 = 0

        for name, param in model.named_parameters():
            p = param.float()
            abs_max = p.abs().max().item()

            stats = LayerStats(
                name=name,
                weight_min=p.min().item(),
                weight_max=p.max().item(),
                weight_abs_max=abs_max,
            )
            self.report.layer_stats[name] = stats
            weight_stats.append((name, abs_max, param.shape))

            if abs_max > FP16_MAX:
                exceeds_fp16 += 1

        # Find max weight
        weight_stats.sort(key=lambda x: x[1], reverse=True)
        if weight_stats:
            self.report.weight_max_layer = weight_stats[0][0]
            self.report.weight_max_abs = weight_stats[0][1]

        self.report.weights_exceed_fp16 = exceeds_fp16

        if verbose:
            print(f"Total parameters: {self.report.total_params:,}")
            print(f"Max weight magnitude: {self.report.weight_max_abs:.4f}")
            print(f"Max weight layer: {self.report.weight_max_layer[:60]}...")
            print(f"Weights exceeding FP16 range: {exceeds_fp16}")

            if exceeds_fp16 == 0:
                print("\n✓ All weights within FP16 range")
            else:
                print(f"\n✗ {exceeds_fp16} weight tensors exceed FP16 range!")
                self.report.issues.append(f"{exceeds_fp16} weight tensors exceed FP16 max")

            print("\nTop 10 largest weight magnitudes:")
            for name, val, shape in weight_stats[:10]:
                short_name = name if len(name) < 50 else "..." + name[-47:]
                print(f"  {val:12.4f}  {str(shape):25}  {short_name}")

        del model
        self._clear_cache()

    def _clear_cache(self):
        """Clear GPU/MPS cache."""
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def test_precision(
        self,
        dtype: torch.dtype,
        input_ids: torch.Tensor,
        max_tokens: int = 20,
        label: str = "",
        convert_to_fp32: bool = False,
        verbose: bool = True
    ) -> Tuple[bool, str, Dict[str, int]]:
        """Test inference at a specific precision.

        Returns:
            Tuple of (success, output_text, overflow_by_layer)
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, trust_remote_code=True
        )

        if convert_to_fp32:
            model = model.float()

        model = model.to(self.device)
        model.eval()

        # Track overflows and NaN by layer
        overflow_by_layer = defaultdict(int)
        nan_by_layer = defaultdict(int)
        inf_by_layer = defaultdict(int)
        handles = []

        def make_hook(name):
            def hook(module, input, output):
                def check_tensor(t, suffix=""):
                    if isinstance(t, torch.Tensor) and t.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                        key = f"{name}{suffix}"

                        nan_count = torch.isnan(t).sum().item()
                        inf_count = torch.isinf(t).sum().item()
                        overflow = ((t.abs() > FP16_MAX) & ~torch.isinf(t) & ~torch.isnan(t)).sum().item()

                        if nan_count > 0:
                            nan_by_layer[key] += nan_count
                        if inf_count > 0:
                            inf_by_layer[key] += inf_count
                        if overflow > 0:
                            overflow_by_layer[key] += overflow

                if isinstance(output, tuple):
                    for i, o in enumerate(output):
                        check_tensor(o, f"[{i}]")
                else:
                    check_tensor(output)
            return hook

        for name, module in model.named_modules():
            if name:
                handles.append(module.register_forward_hook(make_hook(name)))

        # Generate tokens
        tokenizer = self._load_tokenizer()
        generated = input_ids.to(self.device)
        tokens = []
        success = True

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = model(input_ids=generated)
                logits = outputs.logits[0, -1, :]

            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()

            if nan_count > 0 or inf_count > 0:
                success = False
                break

            next_token = logits.argmax().item()
            tokens.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

            generated = torch.cat([
                generated,
                torch.tensor([[next_token]], device=self.device)
            ], dim=1)

        for h in handles:
            h.remove()

        output_text = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""

        del model
        self._clear_cache()

        # Combine overflow info
        all_issues = {}
        for k, v in overflow_by_layer.items():
            all_issues[k] = {"overflow": v}
        for k, v in nan_by_layer.items():
            if k not in all_issues:
                all_issues[k] = {}
            all_issues[k]["nan"] = v
        for k, v in inf_by_layer.items():
            if k not in all_issues:
                all_issues[k] = {}
            all_issues[k]["inf"] = v

        return success, output_text, all_issues

    def run_precision_tests(
        self,
        prompt: str = "Explain quantum computing in simple terms.",
        max_tokens: int = 30,
        verbose: bool = True
    ) -> None:
        """Run inference tests at different precisions."""
        if verbose:
            print("\n" + "=" * 70)
            print("PRECISION TESTS")
            print("=" * 70)
            print(f"Prompt: {prompt[:50]}...")
            print(f"Max tokens: {max_tokens}")

        input_ids = self._prepare_input(prompt)

        # Test BF16 (baseline)
        if verbose:
            print("\n--- BF16 (baseline) ---")
        success, output, issues = self.test_precision(
            torch.bfloat16, input_ids, max_tokens, "BF16"
        )
        self.report.bf16_works = success
        self.report.bf16_output = output
        if verbose:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"Status: {status}")
            print(f"Output: {output[:100]}..." if len(output) > 100 else f"Output: {output}")

        # Test FP16 (ANE precision)
        if verbose:
            print("\n--- FP16 (ANE precision) ---")
        success, output, issues = self.test_precision(
            torch.float16, input_ids, max_tokens, "FP16"
        )
        self.report.fp16_works = success
        self.report.fp16_output = output

        if issues:
            # Find first NaN layer
            nan_layers = [k for k, v in issues.items() if v.get("nan", 0) > 0]
            inf_layers = [k for k, v in issues.items() if v.get("inf", 0) > 0]
            overflow_layers = [k for k, v in issues.items() if v.get("overflow", 0) > 0]

            if nan_layers:
                self.report.fp16_first_nan_layer = nan_layers[0]
            self.report.fp16_overflow_layers = inf_layers + overflow_layers

        if verbose:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"Status: {status}")
            if success:
                print(f"Output: {output[:100]}..." if len(output) > 100 else f"Output: {output}")
            else:
                print(f"First NaN layer: {self.report.fp16_first_nan_layer}")
                if self.report.fp16_overflow_layers:
                    print(f"Overflow/Inf layers ({len(self.report.fp16_overflow_layers)}):")
                    for layer in self.report.fp16_overflow_layers[:5]:
                        print(f"  - {layer[:60]}...")

        # Test FP16 weights → FP32 compute
        if verbose:
            print("\n--- FP16 weights → FP32 compute ---")
        success, output, issues = self.test_precision(
            torch.float16, input_ids, max_tokens, "FP16→FP32", convert_to_fp32=True
        )
        self.report.fp16_to_fp32_works = success
        self.report.fp16_to_fp32_output = output

        if verbose:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"Status: {status}")
            if success:
                print(f"Output: {output[:100]}..." if len(output) > 100 else f"Output: {output}")

    def run_clamp_sweep(
        self,
        prompt: str = "Explain quantum computing.",
        max_tokens: int = 20,
        clamp_values: Optional[List[int]] = None,
        verbose: bool = True
    ) -> None:
        """Sweep clamp values to find overflow threshold."""
        if clamp_values is None:
            clamp_values = [65000, 60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000, 20000]

        if verbose:
            print("\n" + "=" * 70)
            print("CLAMP VALUE SWEEP")
            print("=" * 70)

        input_ids = self._prepare_input(prompt)
        tokenizer = self._load_tokenizer()

        # Get baseline tokens
        if verbose:
            print("Getting BF16 baseline...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.device)
        model.eval()

        baseline_tokens = []
        generated = input_ids.to(self.device)
        for _ in range(max_tokens):
            with torch.no_grad():
                logits = model(input_ids=generated).logits[0, -1, :]
            tok = logits.argmax().item()
            baseline_tokens.append(tok)
            if tok == tokenizer.eos_token_id:
                break
            generated = torch.cat([generated, torch.tensor([[tok]], device=self.device)], dim=1)

        del model
        self._clear_cache()

        if verbose:
            print(f"\n{'Clamp':>10} {'Status':>8} {'Clamped':>12} {'Match%':>8} Output")
            print("-" * 70)

        for cv in clamp_values:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.float16, trust_remote_code=True
            ).to(self.device)
            model.eval()

            overflow_count = 0
            handles = []

            def make_hook(clamp_val):
                def hook(module, input, output):
                    nonlocal overflow_count
                    if isinstance(output, tuple):
                        for o in output:
                            if isinstance(o, torch.Tensor) and o.dtype == torch.float16:
                                overflow_count += (o.abs() > clamp_val).sum().item()
                                o.clamp_(-clamp_val, clamp_val)
                    elif isinstance(output, torch.Tensor) and output.dtype == torch.float16:
                        overflow_count += (output.abs() > clamp_val).sum().item()
                        output.clamp_(-clamp_val, clamp_val)
                return hook

            for name, module in model.named_modules():
                if name:
                    handles.append(module.register_forward_hook(make_hook(cv)))

            tokens = []
            generated = input_ids.to(self.device)
            success = True

            for _ in range(max_tokens):
                with torch.no_grad():
                    outputs = model(input_ids=generated)
                    logits = outputs.logits[0, -1, :]

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    success = False
                    break

                tok = logits.argmax().item()
                tokens.append(tok)
                if tok == tokenizer.eos_token_id:
                    break
                generated = torch.cat([generated, torch.tensor([[tok]], device=self.device)], dim=1)

            for h in handles:
                h.remove()
            del model
            self._clear_cache()

            # Calculate match rate
            if success and tokens:
                min_len = min(len(tokens), len(baseline_tokens))
                matches = sum(1 for i in range(min_len) if tokens[i] == baseline_tokens[i])
                match_rate = matches / min_len if min_len > 0 else 0
                output_text = tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                match_rate = 0
                output_text = ""

            self.report.clamp_sweep[cv] = {
                "success": success,
                "clamped": overflow_count,
                "match_rate": match_rate,
                "output": output_text[:50]
            }

            if verbose:
                if success:
                    short_text = output_text[:35] + "..." if len(output_text) > 35 else output_text
                    print(f"{cv:>10} {'PASS':>8} {overflow_count:>12,} {match_rate:>7.1%} {short_text}")
                else:
                    print(f"{cv:>10} {'FAIL':>8} {overflow_count:>12,} {'-':>8} (NaN/Inf)")

    def generate_report(self, verbose: bool = True) -> CompatibilityReport:
        """Generate final compatibility report."""
        # Determine overall compatibility
        if self.report.fp16_works:
            self.report.ane_compatible = True
            self.report.compatibility_score = 1.0
            self.report.recommendations.append("Model is fully FP16 compatible - ready for ANE conversion")
        elif self.report.fp16_to_fp32_works and not self.report.fp16_works:
            self.report.ane_compatible = False
            self.report.compatibility_score = 0.5
            self.report.issues.append("FP16 computation overflows - weights are fine but activations exceed FP16 range")

            # Check clamp sweep for mitigation
            if self.report.clamp_sweep:
                passing = [(cv, d) for cv, d in self.report.clamp_sweep.items()
                          if d["success"] and d["match_rate"] > 0.5]
                if passing:
                    best = max(passing, key=lambda x: x[1]["match_rate"])
                    self.report.recommendations.append(
                        f"Apply activation clamping at {best[0]} (achieves {best[1]['match_rate']:.0%} match rate)"
                    )
                    self.report.compatibility_score = 0.7
                else:
                    self.report.recommendations.append(
                        "No clamping value provides acceptable quality - model not suitable for ANE"
                    )
            else:
                self.report.recommendations.append(
                    "Run clamp sweep to find if activation clamping can help"
                )
        else:
            self.report.ane_compatible = False
            self.report.compatibility_score = 0.0
            self.report.issues.append("Model fails in both FP16 and FP16→FP32 modes")
            self.report.recommendations.append("Model has fundamental precision issues - not suitable for ANE")

        if self.report.weights_exceed_fp16 > 0:
            self.report.issues.append(f"{self.report.weights_exceed_fp16} weight tensors exceed FP16 range")
            self.report.recommendations.append("Apply weight clipping before conversion")

        if verbose:
            print("\n" + "=" * 70)
            print("COMPATIBILITY REPORT")
            print("=" * 70)
            print(f"Model: {self.report.model_id}")
            print(f"Parameters: {self.report.total_params:,}")
            print()
            print(f"ANE Compatible: {'✓ YES' if self.report.ane_compatible else '✗ NO'}")
            print(f"Compatibility Score: {self.report.compatibility_score:.0%}")
            print()

            print("Precision Test Results:")
            print(f"  BF16:        {'✓ PASS' if self.report.bf16_works else '✗ FAIL'}")
            print(f"  FP16:        {'✓ PASS' if self.report.fp16_works else '✗ FAIL'}")
            print(f"  FP16→FP32:   {'✓ PASS' if self.report.fp16_to_fp32_works else '✗ FAIL'}")

            if self.report.issues:
                print("\nIssues Found:")
                for issue in self.report.issues:
                    print(f"  ✗ {issue}")

            if self.report.recommendations:
                print("\nRecommendations:")
                for rec in self.report.recommendations:
                    print(f"  → {rec}")

            if self.report.fp16_first_nan_layer:
                print(f"\nFirst NaN occurs at: {self.report.fp16_first_nan_layer}")

            if self.report.fp16_overflow_layers:
                print(f"\nLayers with overflow/inf ({len(self.report.fp16_overflow_layers)}):")
                for layer in self.report.fp16_overflow_layers[:10]:
                    short = layer if len(layer) < 60 else "..." + layer[-57:]
                    print(f"  - {short}")
                if len(self.report.fp16_overflow_layers) > 10:
                    print(f"  ... and {len(self.report.fp16_overflow_layers) - 10} more")

        return self.report

    def analyze_residual_accumulation(
        self,
        prompt: str = "Explain quantum computing.",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Analyze residual accumulation across layers in FP32."""
        if verbose:
            print("\n" + "=" * 70)
            print("RESIDUAL ACCUMULATION ANALYSIS (FP32)")
            print("=" * 70)

        input_ids = self._prepare_input(prompt)

        # Load in FP32 to see all values
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float32, trust_remote_code=True
        ).to(self.device)
        model.eval()

        # Track layer outputs and sub-tensors
        layer_outputs = {}
        sub_tensors = {}
        handles = []

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    t = output[0]
                elif isinstance(output, torch.Tensor):
                    t = output
                else:
                    return

                if t.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    abs_max = t.abs().max().item()
                    overflow = (t.abs() > FP16_MAX).sum().item()

                    # Check if this is a layer output (e.g., model.language_model.layers.5)
                    import re
                    layer_match = re.match(r".*\.layers\.(\d+)$", name)
                    if layer_match:
                        layer_outputs[name] = {
                            "abs_max": abs_max,
                            "overflow": overflow,
                            "exceeds_fp16": abs_max > FP16_MAX,
                            "layer_num": int(layer_match.group(1))
                        }
                    else:
                        sub_tensors[name] = {
                            "abs_max": abs_max,
                            "overflow": overflow,
                            "exceeds_fp16": abs_max > FP16_MAX
                        }

            return hook

        for name, module in model.named_modules():
            if name:
                handles.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            model(input_ids=input_ids.to(self.device))

        for h in handles:
            h.remove()
        del model
        self._clear_cache()

        # Analyze results
        results = {
            "layer_progression": [],
            "first_overflow_layer": None,
            "max_overflow_layer": None,
            "max_value": 0,
            "sub_tensor_overflows": [],
            "is_residual_accumulation": False,
        }

        # Sort layer outputs by layer number
        sorted_layers = sorted(
            layer_outputs.items(),
            key=lambda x: x[1].get("layer_num", 999)
        )

        if verbose:
            print(f"\n{'Layer':<8} {'Max Value':>12} {'Status':>12}")
            print("-" * 35)

        for name, info in sorted_layers:
            lnum = info.get("layer_num", 0)
            results["layer_progression"].append({
                "layer": lnum,
                "max_value": info["abs_max"],
                "exceeds_fp16": info["exceeds_fp16"]
            })

            if info["abs_max"] > results["max_value"]:
                results["max_value"] = info["abs_max"]
                results["max_overflow_layer"] = lnum

            if info["exceeds_fp16"] and results["first_overflow_layer"] is None:
                results["first_overflow_layer"] = lnum

            if verbose:
                status = "✗ OVERFLOW" if info["exceeds_fp16"] else "✓ ok"
                bar = "█" * min(int(info["abs_max"] / 10000), 30)
                print(f"Layer {lnum:<2} {info['abs_max']:>12.1f} {status:>12} {bar}")

        # Check for sub-tensor overflows
        sub_overflows = {k: v for k, v in sub_tensors.items() if v["exceeds_fp16"]}
        results["sub_tensor_overflows"] = list(sub_overflows.keys())

        # Determine if this is residual accumulation pattern
        if results["first_overflow_layer"] is not None:
            if len(sub_overflows) == 0:
                results["is_residual_accumulation"] = True
                self.report.issues.append(
                    f"Residual accumulation overflow starting at layer {results['first_overflow_layer']}"
                )
                self.report.recommendations.append(
                    "Consider layer-wise normalization or residual scaling to reduce accumulation"
                )
            else:
                self.report.issues.append(
                    f"Sub-tensor overflow in: {list(sub_overflows.keys())[:3]}"
                )

        if verbose:
            print()
            if results["is_residual_accumulation"]:
                print("DIAGNOSIS: Residual accumulation overflow")
                print("  - All sub-tensors (attention, MLP, norms) are within FP16 range")
                print("  - Overflow occurs in layer OUTPUT due to cumulative residual")
                print(f"  - First overflow: Layer {results['first_overflow_layer']}")
                print(f"  - Peak value: {results['max_value']:.1f} ({results['max_value']/FP16_MAX:.1f}x FP16 max)")
            elif sub_overflows:
                print("DIAGNOSIS: Sub-tensor overflow")
                print(f"  - {len(sub_overflows)} tensors exceed FP16 range")
                for name in list(sub_overflows.keys())[:5]:
                    print(f"    - {name}")

        # Store results for scaling computation
        self._residual_results = results
        return results

    def compute_recommended_scaling(self, verbose: bool = True) -> Dict[str, Any]:
        """Compute recommended scaling factor based on residual analysis."""
        if not hasattr(self, '_residual_results'):
            return {}

        results = self._residual_results if hasattr(self, '_residual_results') else {}
        max_value = results.get("max_value", 0)

        if max_value == 0:
            return {}

        TARGET_MAX = 50000.0  # With headroom
        recommended_alpha = TARGET_MAX / max_value

        scaling_info = {
            "original_peak": max_value,
            "target_max": TARGET_MAX,
            "recommended_alpha": recommended_alpha,
            "overflow_ratio": max_value / FP16_MAX,
        }

        if verbose and max_value > FP16_MAX:
            print("\n" + "=" * 70)
            print("RECOMMENDED SCALING FIX")
            print("=" * 70)
            print(f"\nOriginal peak: {max_value:.1f} ({max_value/FP16_MAX:.1f}x FP16 max)")
            print(f"Target max: {TARGET_MAX:.1f} (with headroom)")
            print(f"\nRecommended α = {recommended_alpha:.4f}")
            print(f"\nWeight transformations:")
            print(f"  1. embed_tokens.weight *= {recommended_alpha:.4f}")
            print(f"  2. For each layer's post_*_layernorm:")
            print(f"     weight_new = {recommended_alpha:.4f} * (1 + weight_old) - 1")
            print(f"\nThis is a WEIGHT-ONLY transformation - no runtime ops needed.")
            print(f"See: docs/GEMMA3_FP16_SCALING.md for details.")

        self.report.scaling_recommendation = scaling_info
        return scaling_info

    def plot_activation_graph(
        self,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> Optional[str]:
        """Generate activation magnitude graph across layers (similar to Unsloth).

        Args:
            output_path: Path to save the graph image (default: <model_name>_fp16_activations.png)
            show: If True, display the graph interactively

        Returns:
            Path to saved image, or None if no data available
        """
        if not hasattr(self, '_residual_results') or not self._residual_results:
            print("No residual analysis data available. Run analyze_residual_accumulation() first.")
            return None

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return None

        results = self._residual_results
        layer_data = results.get("layer_progression", [])

        if not layer_data:
            print("No layer data available.")
            return None

        # Extract data
        layers = [d["layer"] for d in layer_data]
        max_values = [d["max_value"] for d in layer_data]
        exceeds = [d["exceeds_fp16"] for d in layer_data]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Color bars based on FP16 overflow
        colors = ['#ff4444' if e else '#44aa44' for e in exceeds]

        bars = ax.bar(layers, max_values, color=colors, edgecolor='black', linewidth=0.5)

        # Add FP16 max line
        ax.axhline(y=FP16_MAX, color='red', linestyle='--', linewidth=2, label=f'FP16 Max ({FP16_MAX:,})')

        # Add target line (50k with headroom)
        ax.axhline(y=50000, color='orange', linestyle=':', linewidth=1.5, label='Safe Target (50,000)')

        # Labels and title
        model_name = self.model_id.split('/')[-1] if '/' in self.model_id else self.model_id
        ax.set_xlabel('Layer Number', fontsize=12)
        ax.set_ylabel('Max Activation Magnitude', fontsize=12)
        ax.set_title(f'FP16 Activation Analysis: {model_name}', fontsize=14, fontweight='bold')

        # Format y-axis with comma separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

        # Add legend
        safe_patch = mpatches.Patch(color='#44aa44', label='Within FP16 range')
        overflow_patch = mpatches.Patch(color='#ff4444', label='Exceeds FP16 range')
        ax.legend(handles=[safe_patch, overflow_patch, ax.lines[0], ax.lines[1]], loc='upper left')

        # Add peak annotation
        peak_idx = max_values.index(max(max_values))
        peak_val = max(max_values)
        ax.annotate(
            f'Peak: {peak_val:,.0f}\n({peak_val/FP16_MAX:.1f}x FP16)',
            xy=(layers[peak_idx], peak_val),
            xytext=(layers[peak_idx] + 2, peak_val * 0.9),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )

        # Add first overflow annotation if applicable
        first_overflow = results.get("first_overflow_layer")
        if first_overflow is not None and first_overflow != peak_idx:
            overflow_val = layer_data[first_overflow]["max_value"]
            ax.annotate(
                f'First overflow\nLayer {first_overflow}',
                xy=(first_overflow, overflow_val),
                xytext=(first_overflow - 3, overflow_val * 1.1),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', alpha=0.7)
            )

        # Grid
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

        # Tight layout
        plt.tight_layout()

        # Save or show
        if output_path is None:
            output_path = f"{model_name}_fp16_activations.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nActivation graph saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def save_report(self, path: str) -> None:
        """Save report to JSON file."""
        data = {
            "model_id": self.report.model_id,
            "total_params": self.report.total_params,
            "weight_max_abs": self.report.weight_max_abs,
            "weight_max_layer": self.report.weight_max_layer,
            "weights_exceed_fp16": self.report.weights_exceed_fp16,
            "bf16_works": self.report.bf16_works,
            "fp16_works": self.report.fp16_works,
            "fp16_to_fp32_works": self.report.fp16_to_fp32_works,
            "fp16_first_nan_layer": self.report.fp16_first_nan_layer,
            "fp16_overflow_layers": self.report.fp16_overflow_layers,
            "ane_compatible": self.report.ane_compatible,
            "compatibility_score": self.report.compatibility_score,
            "issues": self.report.issues,
            "recommendations": self.report.recommendations,
            "clamp_sweep": self.report.clamp_sweep,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nReport saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check FP16/ANE compatibility for HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fp16_compatibility_check.py --model google/gemma-3-4b-it-qat-int4-unquantized
  python fp16_compatibility_check.py --model meta-llama/Llama-3.1-8B-Instruct --quick
  python fp16_compatibility_check.py --model Qwen/Qwen3-4B --sweep --save-report report.json
        """
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Explain quantum computing in simple terms.",
        help="Test prompt for inference"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Maximum tokens to generate per test"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run clamp value sweep (slower but more detailed)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode - skip weight analysis"
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Save report to JSON file"
    )
    parser.add_argument(
        "--graph",
        type=str,
        nargs='?',
        const='auto',
        default=None,
        help="Generate activation graph (like Unsloth). Optionally specify output path."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (mps, cuda, cpu)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FP16 COMPATIBILITY CHECK FOR APPLE NEURAL ENGINE")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device or 'auto'}")

    checker = FP16CompatibilityChecker(args.model, args.device)

    # Run checks
    if not args.quick:
        checker.analyze_weights()

    checker.run_precision_tests(args.prompt, args.max_tokens)

    # If FP16 fails but FP16→FP32 works, analyze residual accumulation
    # Also run if --graph is requested (graph needs residual data)
    if (not checker.report.fp16_works and checker.report.fp16_to_fp32_works) or args.graph:
        checker.analyze_residual_accumulation(args.prompt)
        # Compute recommended scaling if residual overflow detected
        checker.compute_recommended_scaling()

    # Generate activation graph if requested
    if args.graph:
        graph_path = args.graph if args.graph != 'auto' else None
        checker.plot_activation_graph(output_path=graph_path)

    # Only run clamp sweep if explicitly requested (we use weight scaling, not clamping)
    if args.sweep:
        checker.run_clamp_sweep(args.prompt, min(args.max_tokens, 20))

    report = checker.generate_report()

    if args.save_report:
        checker.save_report(args.save_report)

    # Exit code based on compatibility
    return 0 if report.ane_compatible else 1


if __name__ == "__main__":
    sys.exit(main())
