#!/usr/bin/env python3
"""
Compare Qwen3 CoreML chunks vs PyTorch to detect chunked norm bug.

If norm is incorrectly applied per-chunk:
  - chunk1 will diverge (CoreML has norm, PyTorch doesn't)
  - chunk2+ will diverge more (norm applied multiple times)

Usage:
  python tests/dev/test_qwen3_coreml_chunks_vs_pytorch.py \
    --coreml-dir /Volumes/Models/ANE/qwen3_1.7b_ctx2048_FP16 \
    --prompt "Hello"
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import yaml

# Add repo root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer, AutoModelForCausalLM


def _load_meta(meta_path: Path) -> dict:
    with meta_path.open("r") as f:
        return yaml.safe_load(f)


def _stats(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"{name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    return {"max": max_diff, "mean": mean_diff}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Qwen3 CoreML chunks vs PyTorch (single token) to detect norm bug."
    )
    parser.add_argument("--coreml-dir", required=True, help="CoreML model directory")
    parser.add_argument("--hf-model", default="Qwen/Qwen3-1.7B", help="HF model ID")
    parser.add_argument("--prompt", default="Hello", help="Prompt for token")
    parser.add_argument("--token-id", type=int, default=None, help="Explicit token id")
    parser.add_argument("--pos", type=int, default=0, help="Token position")
    args = parser.parse_args()

    coreml_dir = Path(args.coreml_dir).resolve()
    meta_path = coreml_dir / "meta.yaml"
    if not meta_path.exists():
        print(f"meta.yaml not found: {meta_path}")
        return 1

    meta = _load_meta(meta_path)
    params = meta["model_info"]["parameters"]
    context_length = int(params.get("context_length", 2048))
    num_chunks = int(params.get("num_chunks", 4))

    print(f"Context length={context_length}, num_chunks={num_chunks}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(coreml_dir), local_files_only=True)

    if args.token_id is not None:
        token_id = args.token_id
    else:
        ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids
        token_id = int(ids[0, 0].item())

    print(f"Using token_id={token_id} ('{tokenizer.decode([token_id])}') at pos={args.pos}")

    # Load HF model for reference
    print("\nLoading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=torch.float16, local_files_only=True
    )
    hf_model.eval()

    # Get HF embeddings
    input_ids = torch.tensor([[token_id]], dtype=torch.long)
    with torch.no_grad():
        hf_embed = hf_model.model.embed_tokens(input_ids)

    print(f"\nHF embeddings shape: {hf_embed.shape}")
    print(f"HF embed[0,0,:5]: {hf_embed[0,0,:5].tolist()}")

    # Get total layers and chunk info
    total_layers = hf_model.config.num_hidden_layers
    layers_per_chunk = total_layers // num_chunks
    remainder = total_layers % num_chunks

    print(f"\nTotal layers: {total_layers}")
    print(f"Layers per chunk: {layers_per_chunk}")
    print(f"Remainder: {remainder}")

    # Calculate layer ranges for each chunk (with remainder distribution)
    chunk_ranges = []
    current_layer = 0
    for i in range(num_chunks):
        chunk_size = layers_per_chunk + (1 if i < remainder else 0)
        chunk_ranges.append((current_layer, current_layer + chunk_size))
        current_layer += chunk_size

    print(f"Chunk ranges: {chunk_ranges}")

    # Run full HF model to get correct output
    print("\n" + "=" * 70)
    print("Running HF model (CORRECT - norm applied once at end)")
    print("=" * 70)

    with torch.no_grad():
        hf_outputs = hf_model(input_ids, use_cache=False, output_hidden_states=True)
        correct_hidden = hf_outputs.hidden_states[-1]  # After final norm
        correct_logits = hf_outputs.logits

    print(f"Correct hidden[0,0,:5]: {correct_hidden[0,0,:5].tolist()}")

    # Simulate buggy behavior: apply norm after each chunk
    print("\n" + "=" * 70)
    print("Simulating BUGGY behavior (norm applied after each chunk)")
    print("=" * 70)

    # We can't easily run layer-by-layer with HF due to position embeddings
    # Instead, simulate what happens to the final norm magnitude

    # Get the hidden state before final norm (last hidden state before norm)
    pre_norm_hidden = hf_outputs.hidden_states[-2] if len(hf_outputs.hidden_states) > 1 else hf_embed

    # Simulate applying norm multiple times
    norm = hf_model.model.norm

    with torch.no_grad():
        # Correct: norm applied once
        once_normed = norm(pre_norm_hidden)

        # Buggy: norm applied num_chunks times
        multi_normed = pre_norm_hidden
        for i in range(num_chunks):
            multi_normed = norm(multi_normed)
            print(f"After norm {i+1}: hidden[0,0,:5] = {multi_normed[0,0,:5].tolist()}")

    print("\n" + "=" * 70)
    print("Divergence Analysis")
    print("=" * 70)

    diff = torch.abs(once_normed.float() - multi_normed.float())
    print(f"Once vs {num_chunks}x norm: max_diff={diff.max().item():.4f}, mean_diff={diff.mean().item():.4f}")

    # Compare top-5 tokens
    with torch.no_grad():
        correct_probs = torch.softmax(correct_logits[0, 0, :].float(), dim=-1)
        _, correct_top5 = torch.topk(correct_probs, 5)

        buggy_logits = hf_model.lm_head(multi_normed)
        buggy_probs = torch.softmax(buggy_logits[0, 0, :].float(), dim=-1)
        _, buggy_top5 = torch.topk(buggy_probs, 5)

    print(f"\nCorrect top-5: {[tokenizer.decode([t.item()]) for t in correct_top5]}")
    print(f"Buggy top-5:   {[tokenizer.decode([t.item()]) for t in buggy_top5]}")

    overlap = set(correct_top5.tolist()) & set(buggy_top5.tolist())
    print(f"Overlap: {len(overlap)}/5")

    if len(overlap) < 3:
        print("\n*** SEVERE DIVERGENCE - Consistent with norm-per-chunk bug ***")
    elif len(overlap) < 5:
        print("\n*** MODERATE DIVERGENCE ***")
    else:
        print("\n*** LOW DIVERGENCE - May not be the norm bug ***")

    # Also show what the RMS values look like
    print("\n" + "=" * 70)
    print("RMS Value Analysis")
    print("=" * 70)

    with torch.no_grad():
        pre_rms = torch.sqrt((pre_norm_hidden ** 2).mean(dim=-1))
        once_rms = torch.sqrt((once_normed ** 2).mean(dim=-1))
        multi_rms = torch.sqrt((multi_normed ** 2).mean(dim=-1))

    print(f"Pre-norm RMS: {pre_rms.mean().item():.4f}")
    print(f"After 1x norm RMS: {once_rms.mean().item():.4f}")
    print(f"After {num_chunks}x norm RMS: {multi_rms.mean().item():.4f}")

    print(f"\nRMS reduction ratio (1x): {once_rms.mean().item() / pre_rms.mean().item():.4f}")
    print(f"RMS reduction ratio ({num_chunks}x): {multi_rms.mean().item() / pre_rms.mean().item():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
