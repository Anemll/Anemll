#!/usr/bin/env python3
"""Compare Gemma3n CoreML embeddings and LM head against PyTorch weights.

This test avoids full model loading by pulling only the needed weights from
the Gemma3n safetensors shards.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoTokenizer

from anemll.models.gemma3n_model import Gemma3nConfig, Gemma3nRMSNorm


SAFETENSOR_FILES = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]


def load_weight(model_path: Path, key: str) -> Optional[torch.Tensor]:
    for filename in SAFETENSOR_FILES:
        path = model_path / filename
        if not path.exists():
            continue
        with safe_open(path, framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    return None


def load_required_weights(model_path: Path) -> Dict[str, torch.Tensor]:
    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    lm_head_key = "model.language_model.lm_head.weight"

    embed_weight = load_weight(model_path, embed_key)
    if embed_weight is None:
        raise ValueError(f"Missing {embed_key} in safetensors")

    norm_weight = load_weight(model_path, norm_key)
    if norm_weight is None:
        raise ValueError(f"Missing {norm_key} in safetensors")

    lm_head_weight = load_weight(model_path, lm_head_key)
    if lm_head_weight is None:
        # Gemma3n can tie LM head to embeddings.
        lm_head_weight = embed_weight

    return {
        "embed_weight": embed_weight,
        "norm_weight": norm_weight,
        "lm_head_weight": lm_head_weight,
    }


def compare_arrays(name: str, pt: np.ndarray, cm: np.ndarray, tol: float) -> bool:
    if pt.shape != cm.shape:
        print(f"{name}: shape mismatch PT={pt.shape} CM={cm.shape}")
        return False
    diff = np.abs(pt - cm)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"{name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} tol={tol}")
    return max_diff <= tol


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma3n embeddings + LM head consistency test")
    parser.add_argument("--bundle", required=True, help="Directory with .mlpackage models and tokenizer files")
    parser.add_argument("--model", required=True, help="Path to Gemma3n model folder (with safetensors)")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt text")
    parser.add_argument("--context-length", type=int, default=512, help="Context length used for embedding export")
    parser.add_argument("--embed-tol", type=float, default=1e-3, help="Tolerance for embedding diff")
    parser.add_argument("--lmhead-tol", type=float, default=2e-2, help="Tolerance for LM head diff")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                        help="PyTorch dtype for LM head comparison (match CoreML with float16)")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    model_path = Path(args.model)
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Load config
    config = Gemma3nConfig.from_json(str(model_path / "config.json"))
    config.context_length = args.context_length

    # Load CoreML models
    embeddings_cm = ct.models.MLModel(str(bundle / "gemma3n_embeddings.mlpackage"))
    lm_head_cm = ct.models.MLModel(str(bundle / "gemma3n_lm_head.mlpackage"))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(bundle), use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # Load weights (minimal)
    weights = load_required_weights(model_path)

    pt_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Build PyTorch embedding
    embed = nn.Embedding(config.vocab_size, config.hidden_size)
    embed.weight.data = weights["embed_weight"].to(torch.float32)
    embed.eval()

    # Build PyTorch LM head (split conv2d)
    norm = Gemma3nRMSNorm(config.hidden_size, config.rms_norm_eps, with_scale=True)
    norm.weight.data = weights["norm_weight"].to(pt_dtype)
    norm.eval()

    lm_weight = weights["lm_head_weight"].to(pt_dtype)
    vocab_split = config.vocab_size // 16
    vocab_remainder = config.vocab_size % 16
    split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(16)]
    splits = torch.split(lm_weight, split_sizes, dim=0)

    lm_heads = nn.ModuleList()
    for split_weight in splits:
        head = nn.Conv2d(config.hidden_size, split_weight.shape[0], 1, bias=False)
        head.weight.data = split_weight.unsqueeze(-1).unsqueeze(-1).to(pt_dtype)
        lm_heads.append(head)
    lm_heads.eval()

    # Prepare input ids (pad to context length)
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].long()
    prompt_len = input_ids.shape[1]
    if prompt_len > args.context_length:
        raise ValueError(f"Prompt length {prompt_len} exceeds context length {args.context_length}")

    padded = torch.full((1, args.context_length), tokenizer.pad_token_id, dtype=torch.long)
    padded[0, :prompt_len] = input_ids[0]

    # CoreML embeddings
    cm_embed_out = embeddings_cm.predict({"input_ids": padded.numpy().astype(np.int32)})
    cm_embed = cm_embed_out.get("embeddings", cm_embed_out.get("hidden_states"))
    if cm_embed is None:
        raise KeyError(f"Embeddings output missing. Keys: {list(cm_embed_out.keys())}")

    # PyTorch embeddings (no scaling to match converter)
    with torch.no_grad():
        pt_embed = embed(padded).detach().numpy()  # [1, T, H]
    pt_embed = np.transpose(pt_embed, (0, 2, 1))[:, :, :, None]  # [1, H, T, 1]

    # Compare embeddings on prompt range only
    pt_slice = pt_embed[:, :, :prompt_len, :]
    cm_slice = cm_embed[:, :, :prompt_len, :]
    print("\nEmbedding consistency:")
    embed_ok = compare_arrays("embeddings", pt_slice.astype(np.float32), cm_slice.astype(np.float32), args.embed_tol)

    # LM head consistency on last prompt token
    last_hidden = pt_embed[:, :, prompt_len - 1, 0]  # [1, H]
    lm_input = torch.from_numpy(last_hidden).to(pt_dtype).unsqueeze(1)  # [1, 1, H]
    with torch.no_grad():
        normed = norm(lm_input).permute(0, 2, 1).unsqueeze(-1)  # [1, H, 1, 1]
        pt_logits = [head(normed).squeeze(-1).transpose(1, 2) for head in lm_heads]

    cm_lm_out = lm_head_cm.predict({"hidden_states": lm_input.numpy().astype(np.float16)})
    cm_logits = []
    for i in range(1, 17):
        key = f"logits_split_{i}"
        if key not in cm_lm_out:
            raise KeyError(f"Missing {key} in LM head outputs: {list(cm_lm_out.keys())}")
        cm_logits.append(cm_lm_out[key])

    print("\nLM head split consistency:")
    lm_ok = True
    for i, (pt_part, cm_part) in enumerate(zip(pt_logits, cm_logits), start=1):
        ok = compare_arrays(f"logits_split_{i}", pt_part.detach().numpy().astype(np.float32), cm_part.astype(np.float32), args.lmhead_tol)
        lm_ok = lm_ok and ok

    # Top-1 check across concatenated logits
    pt_full = torch.cat(pt_logits, dim=-1).detach().cpu().numpy()
    cm_full = np.concatenate(cm_logits, axis=-1)
    pt_top = int(np.argmax(pt_full))
    cm_top = int(np.argmax(cm_full))
    print("\nTop-1 argmax check:")
    print(f"  PyTorch top-1: {pt_top}")
    print(f"  CoreML top-1:  {cm_top}")

    # Summary
    print("\nSummary:")
    print(f"  Embeddings match: {'PASS' if embed_ok else 'FAIL'}")
    print(f"  LM head match:   {'PASS' if lm_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
