#!/usr/bin/env python3
"""
Compare Gemma3 CoreML chunks vs PyTorch layer-by-layer to find divergence.

This script:
1. Loads HF model with optional FP16 scaling (for QAT models)
2. Loads each CoreML chunk
3. Runs same input through both PyTorch layers and CoreML chunk
4. Reports per-chunk metrics: correlation, max diff, KL divergence

Usage:
    # Basic comparison (8 chunks, no scaling)
    python tests/dev/test_gemma3_chunk_compare.py \
        --coreml-dir /Volumes/Models/ANE/gemma3_4b_qat4_scale \
        --hf-dir ~/.cache/huggingface/hub/models--google--gemma-3-4b-it-qat-int4-unquantized/snapshots/... \
        --prompt "Hello world"

    # With FP16 scaling for QAT model
    python tests/dev/test_gemma3_chunk_compare.py \
        --coreml-dir /Volumes/Models/ANE/gemma3_4b_qat4_scale \
        --hf-dir ~/.cache/huggingface/hub/models--google--gemma-3-4b-it-qat-int4-unquantized/snapshots/... \
        --alpha 0.1875 \
        --prompt "Hello world"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

# Add repo root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anemll.models.gemma3_model import Gemma3ForCausalLM, Gemma3Config

try:
    from tests import chat as chat_module
except ImportError:
    chat_module = None
    print("Warning: Could not import chat module, will use direct coremltools loading")

import coremltools as ct


def load_meta(meta_path: Path) -> dict:
    """Load meta.yaml configuration."""
    with meta_path.open("r") as f:
        return yaml.safe_load(f)


def strip_model_ext(value: str) -> str:
    """Remove model file extensions."""
    return value.replace(".mlmodelc", "").replace(".mlpackage", "")


def compute_metrics(name: str, a: np.ndarray, b: np.ndarray, verbose: bool = True) -> dict:
    """Compute comparison metrics between two arrays.

    Returns dict with: max_diff, mean_diff, correlation, kl_divergence
    """
    a_flat = a.astype(np.float32).flatten()
    b_flat = b.astype(np.float32).flatten()

    # Basic diff stats
    diff = np.abs(a_flat - b_flat)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    # Correlation
    if np.std(a_flat) > 1e-6 and np.std(b_flat) > 1e-6:
        corr = float(np.corrcoef(a_flat, b_flat)[0, 1])
    else:
        corr = 0.0

    # Cosine similarity
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a > 1e-6 and norm_b > 1e-6:
        cosine_sim = float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
    else:
        cosine_sim = 0.0

    # L2 distance
    l2_dist = float(np.linalg.norm(a_flat - b_flat))

    # Relative error
    denominator = np.maximum(np.abs(b_flat), 1e-6)
    rel_error = float(np.mean(diff / denominator))

    result = {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "correlation": corr,
        "cosine_sim": cosine_sim,
        "l2_dist": l2_dist,
        "rel_error": rel_error,
    }

    if verbose:
        status = "✓" if corr > 0.99 else ("~" if corr > 0.9 else "✗")
        print(f"{name}: corr={corr:.4f} max_diff={max_diff:.4f} mean_diff={mean_diff:.6f} l2={l2_dist:.4f} {status}")

    return result


def load_hf_model_with_scaling(hf_dir: str, alpha: Optional[float] = None, context_length: int = 1024, device: str = "cpu"):
    """Load HF model using our Gemma3ForCausalLM with optional FP16 scaling.

    Args:
        hf_dir: Path to HF model directory
        alpha: Scaling factor for FP16 conversion (e.g., 0.1875 for 4B QAT)
        context_length: Context length to use
        device: Device to load model on

    Returns:
        Loaded model with weights scaled if alpha provided
    """
    print(f"Loading model from: {hf_dir}")

    # Load config
    config = Gemma3Config.from_json(str(Path(hf_dir) / "config.json"))
    config.context_length = context_length
    config.state_length = context_length

    # Create model (disable KV cache for comparison)
    model = Gemma3ForCausalLM(config, disable_kv_cache=True)

    # Load weights from HF format
    model.load_pretrained_weights(hf_dir)

    if alpha is not None and alpha != 1.0:
        print(f"Applying FP16 scaling with α={alpha}")

        embed = model.model.embed_tokens
        layers = model.model.layers

        scaled_count = 0

        # 1. Scale embedding weights
        with torch.no_grad():
            orig_max = embed.weight.abs().max().item()
            embed.weight.mul_(alpha)
            # Keep as float32 for PyTorch inference (no FP16 cast needed for comparison)
            scaled_count += 1
            print(f"  embed_tokens: {orig_max:.2f} -> {embed.weight.abs().max().item():.4f}")

        # 2. Transform post-norm weights: w_new = α * (1 + w_old) - 1
        for layer_idx, layer in enumerate(layers):
            for norm_name in ['post_attention_layernorm', 'post_feedforward_layernorm']:
                if hasattr(layer, norm_name):
                    norm = getattr(layer, norm_name)
                    if hasattr(norm, 'weight'):
                        with torch.no_grad():
                            # Gemma3 uses (1 + w) gain, so transform is: w_new = α * (1 + w_old) - 1
                            norm.weight.data = alpha * (1 + norm.weight.data) - 1
                            # Keep as float32 for PyTorch inference
                            scaled_count += 1

        print(f"  Scaled {scaled_count} tensors with α={alpha}")

    # Model uses float16 dtype (MODEL_DTYPE in gemma3_model.py)
    model.eval()
    return model


def load_coreml_chunks(coreml_dir: Path, meta: dict, cpu_only: bool = True):
    """Load CoreML FFN chunks from directory.

    Returns:
        List of dicts, each with 'infer' and optionally 'prefill' models
    """
    params = meta["model_info"]["parameters"]
    num_chunks = int(params.get("num_chunks", 1))

    compute_unit = ct.ComputeUnit.CPU_ONLY if cpu_only else ct.ComputeUnit.CPU_AND_NE
    print(f"Loading {num_chunks} CoreML chunks (compute_unit={'CPU_ONLY' if cpu_only else 'CPU_AND_NE'})...")

    ffn_base = params.get("ffn", "gemma3_FFN_PF_chunk_01of01").replace(".mlmodelc", "")
    # Extract prefix pattern for multi-chunk naming
    # e.g., "gemma3_FFN_PF_lut4_chunk_01of08" -> "gemma3_FFN_PF_lut4_chunk_{:02d}of{:02d}"
    if f"_chunk_01of{num_chunks:02d}" in ffn_base:
        ffn_pattern = ffn_base.replace(f"_chunk_01of{num_chunks:02d}", "_chunk_{chunk:02d}of{num_chunks:02d}")
    elif "_chunk_01of" in ffn_base:
        # Handle any num_chunks format
        import re
        match = re.search(r"_chunk_01of(\d+)", ffn_base)
        if match:
            ffn_pattern = re.sub(r"_chunk_01of\d+", "_chunk_{chunk:02d}of{num_chunks:02d}", ffn_base)
        else:
            ffn_pattern = ffn_base
    else:
        ffn_pattern = ffn_base

    chunks = []
    for i in range(1, num_chunks + 1):
        chunk_name = ffn_pattern.format(chunk=i, num_chunks=num_chunks)
        # Try .mlpackage first (more reliable), then .mlmodelc
        chunk_path = coreml_dir / f"{chunk_name}.mlpackage"

        if not chunk_path.exists():
            chunk_path = coreml_dir / f"{chunk_name}.mlmodelc"

        if not chunk_path.exists():
            print(f"  Warning: Chunk {i} not found at {chunk_path}")
            chunks.append(None)
            continue

        print(f"  Loading chunk {i}/{num_chunks}: {chunk_path.name}")
        try:
            # Try to load with 'infer' function first
            model = ct.models.MLModel(str(chunk_path), function_name="infer", compute_units=compute_unit)
            chunks.append({"infer": model, "path": chunk_path})
        except Exception as e:
            # Fall back to loading without function name
            try:
                model = ct.models.MLModel(str(chunk_path), compute_units=compute_unit)
                chunks.append({"infer": model, "path": chunk_path})
            except Exception as e2:
                print(f"  Error loading chunk {i}: {e2}")
                chunks.append(None)

    return chunks


def load_embeddings_model(coreml_dir: Path, meta: dict, cpu_only: bool = True):
    """Load CoreML embeddings model."""
    params = meta["model_info"]["parameters"]
    embed_name = strip_model_ext(params.get("embeddings", "gemma3_embeddings"))

    compute_unit = ct.ComputeUnit.CPU_ONLY if cpu_only else ct.ComputeUnit.CPU_AND_NE

    # Try .mlpackage first (more reliable), then .mlmodelc
    embed_path = coreml_dir / f"{embed_name}.mlpackage"
    if not embed_path.exists():
        embed_path = coreml_dir / f"{embed_name}.mlmodelc"

    print(f"Loading embeddings: {embed_path.name}")
    return ct.models.MLModel(str(embed_path), compute_units=compute_unit)


def make_causal_mask(context_length: int, batch_start: int, batch_size: int = 1) -> np.ndarray:
    """Create causal attention mask for specified positions."""
    # Full causal mask
    mask = np.triu(np.full((context_length, context_length), -np.inf, dtype=np.float16), k=1)
    # Extract the relevant slice
    return mask[np.newaxis, np.newaxis, batch_start:batch_start + batch_size, :]


def compare_chunks(
    pt_model,
    coreml_embed,
    coreml_chunks: list,
    tokenizer,
    prompt: str,
    context_length: int,
    num_layers: int,
    position: int = 0,
    verbose: bool = True,
) -> dict:
    """Compare each CoreML chunk against PyTorch layers.

    Args:
        pt_model: Our Gemma3ForCausalLM model
        coreml_embed: CoreML embeddings model
        coreml_chunks: List of CoreML chunk models
        tokenizer: Tokenizer
        prompt: Test prompt
        context_length: Context length
        num_layers: Total number of transformer layers
        position: Token position to test
        verbose: Print detailed output

    Returns:
        Dict with comparison results for each stage
    """
    results = {}
    num_chunks = len(coreml_chunks)
    layers_per_chunk = num_layers // num_chunks

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    token_id = int(input_ids[0, 0].item())

    if verbose:
        print(f"\nTest setup:")
        print(f"  Prompt: '{prompt}'")
        print(f"  Token ID: {token_id} ('{tokenizer.decode([token_id])}')")
        print(f"  Position: {position}")
        print(f"  Context: {context_length}")
        print(f"  Layers: {num_layers} ({layers_per_chunk} per chunk)")
        print()

    # === 1. Compare Embeddings ===
    if verbose:
        print("=" * 60)
        print("Stage: Embeddings")
        print("=" * 60)

    # CoreML embedding
    input_ids_np = np.array([[token_id]], dtype=np.int32)
    coreml_embed_out = coreml_embed.predict({"input_ids": input_ids_np})["hidden_states"]

    # PyTorch embedding (using our model)
    with torch.no_grad():
        pt_embed_out = pt_model.model.embed_tokens(input_ids[:, :1])
        # Apply Gemma3 embedding scaling (sqrt(hidden_size))
        pt_embed_out = pt_embed_out * pt_model.model.embedding_scale

    pt_embed_np = pt_embed_out.cpu().numpy()
    results["embeddings"] = compute_metrics("embeddings", coreml_embed_out, pt_embed_np, verbose)

    # Use CoreML embedding for subsequent layers (to isolate chunk issues)
    hidden_states_coreml = coreml_embed_out.astype(np.float16)
    # Match model dtype (float16 for gemma3_model)
    hidden_states_pt = torch.from_numpy(coreml_embed_out.astype(np.float32)).half()

    # === 2. Compare each chunk ===
    # Prepare inputs for FFN chunks
    causal_mask = make_causal_mask(context_length, position, 1)
    position_ids_np = np.array([position], dtype=np.int32)
    current_pos_np = np.array([position], dtype=np.int32)

    # Balanced layer distribution (same as converter)
    base = num_layers // num_chunks
    rem = num_layers % num_chunks

    for chunk_idx, chunk in enumerate(coreml_chunks):
        if chunk is None:
            print(f"\nChunk {chunk_idx + 1}: SKIPPED (not loaded)")
            continue

        # Same algorithm as llama/qwen/gemma3 converters
        start_layer = chunk_idx * base + min(chunk_idx, rem)
        end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
        is_last_chunk = (chunk_idx == num_chunks - 1)

        if verbose:
            print()
            print("=" * 60)
            print(f"Stage: Chunk {chunk_idx + 1}/{num_chunks} (layers {start_layer}-{end_layer-1})")
            print("=" * 60)

        # CoreML chunk
        try:
            # Create state if needed
            state = None
            try:
                state = chunk["infer"].make_state()
            except:
                pass  # Model may not have state

            ffn_inputs = {
                "hidden_states": hidden_states_coreml,
                "position_ids": position_ids_np,
                "causal_mask": causal_mask.astype(np.float16),
                "current_pos": current_pos_np,
            }

            if state is not None:
                coreml_out = chunk["infer"].predict(ffn_inputs, state)
            else:
                coreml_out = chunk["infer"].predict(ffn_inputs)

            hidden_states_coreml = coreml_out["output_hidden_states"].astype(np.float16)

        except Exception as e:
            print(f"  CoreML error: {e}")
            results[f"chunk_{chunk_idx + 1}"] = {"error": str(e)}
            continue

        # PyTorch layers (using our model's process_layers)
        with torch.no_grad():
            position_ids_pt = torch.tensor([position], dtype=torch.long)
            causal_mask_pt = torch.from_numpy(causal_mask.astype(np.float32))
            current_pos_pt = torch.tensor([position], dtype=torch.long)

            # Use process_layers to handle the layer range
            hidden_states_pt = pt_model.model.process_layers(
                hidden_states_pt,
                position_ids_pt,
                causal_mask_pt,
                current_pos_pt,
                start_layer=start_layer,
                end_layer=end_layer,
                IN_PREFILL=False
            )

            # Apply final norm only for last chunk
            if is_last_chunk:
                hidden_states_pt = pt_model.model.norm(hidden_states_pt)

        pt_chunk_np = hidden_states_pt.cpu().numpy()
        results[f"chunk_{chunk_idx + 1}"] = compute_metrics(
            f"chunk_{chunk_idx + 1}",
            hidden_states_coreml,
            pt_chunk_np,
            verbose
        )

        # Update PyTorch hidden states to match CoreML for next chunk
        # This isolates each chunk's comparison
        hidden_states_pt = torch.from_numpy(hidden_states_coreml.astype(np.float32)).half()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare Gemma3 CoreML chunks vs PyTorch layer-by-layer"
    )
    parser.add_argument("--coreml-dir", required=True, help="CoreML model directory")
    parser.add_argument("--hf-dir", required=True, help="HF model directory")
    parser.add_argument("--meta", default=None, help="Path to meta.yaml")
    parser.add_argument("--prompt", default="Hello", help="Test prompt")
    parser.add_argument("--alpha", type=float, default=None,
                        help="FP16 scaling factor (e.g., 0.1875 for 4B QAT)")
    parser.add_argument("--position", type=int, default=0, help="Token position")
    parser.add_argument("--context-length", type=int, default=None, help="Override context length")
    parser.add_argument("--cpu", action="store_true", default=True, help="Use CPU only (default)")
    parser.add_argument("--ane", action="store_true", help="Use ANE (CPU_AND_NE)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    # Paths
    coreml_dir = Path(args.coreml_dir).resolve()
    hf_dir = Path(args.hf_dir).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else (coreml_dir / "meta.yaml")

    if not meta_path.exists():
        print(f"Error: meta.yaml not found at {meta_path}")
        return 1

    # Load meta
    meta = load_meta(meta_path)
    params = meta["model_info"]["parameters"]
    context_length = args.context_length or int(params.get("context_length", 1024))

    print("=" * 70)
    print("GEMMA3 CHUNK-BY-CHUNK COMPARISON")
    print("=" * 70)
    print(f"CoreML dir: {coreml_dir}")
    print(f"HF dir:     {hf_dir}")
    print(f"Context:    {context_length}")
    if args.alpha:
        print(f"Alpha:      {args.alpha}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir), use_fast=False, local_files_only=True)

    # Load model with optional scaling
    pt_model = load_hf_model_with_scaling(str(hf_dir), alpha=args.alpha, context_length=context_length)

    # Get number of layers from config
    num_layers = pt_model.model.config.num_hidden_layers

    # Load CoreML models
    cpu_only = not args.ane
    coreml_embed = load_embeddings_model(coreml_dir, meta, cpu_only)
    coreml_chunks = load_coreml_chunks(coreml_dir, meta, cpu_only)

    print()

    # Run comparison
    results = compare_chunks(
        pt_model=pt_model,
        coreml_embed=coreml_embed,
        coreml_chunks=coreml_chunks,
        tokenizer=tokenizer,
        prompt=args.prompt,
        context_length=context_length,
        num_layers=num_layers,
        position=args.position,
        verbose=not args.quiet,
    )

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Stage':<20} {'Correlation':>12} {'Max Diff':>12} {'Status':>8}")
    print("-" * 55)

    all_good = True
    for stage, metrics in results.items():
        if "error" in metrics:
            print(f"{stage:<20} {'ERROR':>12} {'-':>12} {'✗':>8}")
            all_good = False
        else:
            corr = metrics["correlation"]
            max_diff = metrics["max_diff"]
            if corr > 0.99:
                status = "✓ GOOD"
            elif corr > 0.9:
                status = "~ OK"
                all_good = False
            else:
                status = "✗ BAD"
                all_good = False
            print(f"{stage:<20} {corr:>12.4f} {max_diff:>12.4f} {status:>8}")

    print()
    if all_good:
        print("All stages have good correlation (>0.99)")
    else:
        print("Some stages show divergence - investigate further")
        # Find first divergent stage
        for stage, metrics in results.items():
            if "error" in metrics or metrics.get("correlation", 1.0) < 0.99:
                print(f"First issue: {stage}")
                break

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
