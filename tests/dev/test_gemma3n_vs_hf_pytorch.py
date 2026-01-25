#!/usr/bin/env python3
"""Compare ANEMLL Gemma3n PyTorch implementation against HF reference (text-only)."""

import argparse
import os
import sys
import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nForCausalLM

import anemll.models.gemma3n_model as gemma3n_mod
from anemll.models.gemma3n_model import Gemma3nModel, Gemma3nConfig


DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/0330734afc91972a1aa1ba1bc4495e2723666854"
)


def _concat_logits_if_split(logits):
    if isinstance(logits, (tuple, list)):
        return torch.cat(list(logits), dim=-1)
    return logits


def _build_masks(seq_len, sliding_window, device, dtype):
    idx = torch.arange(seq_len, device=device)
    i = idx[:, None]
    j = idx[None, :]
    causal = (j > i)
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    zeros = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    causal_mask = torch.where(causal, neg_inf, zeros)
    if sliding_window and sliding_window > 0:
        window = j < (i - sliding_window + 1)
        mask = causal | window
        sliding_mask = torch.where(mask, neg_inf, zeros)
    else:
        sliding_mask = causal_mask.clone()
    return causal_mask.unsqueeze(0).unsqueeze(0), sliding_mask.unsqueeze(0).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Compare ANEMLL Gemma3n vs HF logits")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to local Gemma3n model")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt for comparison")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Dtype")
    parser.add_argument("--topk", type=int, default=20, help="Top-k for overlap comparison")
    parser.add_argument("--external-mask", action="store_true",
                        help="Pass precomputed global/local masks into ANEMLL model")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"❌ Model path not found: {args.model}")
        sys.exit(1)

    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # Force PyTorch-friendly logits
    gemma3n_mod.ENABLE_COREML = False
    gemma3n_mod.ENABLE_CONV2D = False
    gemma3n_mod.ENABLE_VACAB_SPLIT16 = False
    gemma3n_mod.ENABLE_LOGITS2 = False

    print("🔧 Loading HF config/tokenizer...")
    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Run HF first to reduce peak memory (especially on MPS), then free it.
    print("🔧 Loading HF text-only reference model...")
    text_config = hf_config.text_config
    hf_model = Gemma3nForCausalLM.from_pretrained(
        args.model,
        config=text_config,
        dtype=torch_dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    hf_model.eval()

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits.detach().cpu()

    # Free HF model before loading ANEMLL to reduce peak memory usage.
    del hf_model
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    print("🔧 Building ANEMLL Gemma3n model...")
    ane_config = Gemma3nConfig.from_pretrained_config(hf_config)
    ane_model = Gemma3nModel(ane_config)
    ane_model.load_weights(args.model, config=ane_config)
    ane_model = ane_model.to(device=device, dtype=torch_dtype)
    ane_model.eval()

    with torch.no_grad():
        if args.external_mask:
            seq_len = input_ids.shape[1]
            full_mask, _ = _build_masks(seq_len, ane_config.sliding_window, device, torch_dtype)
            ane_logits = ane_model(
                input_ids,
                attention_mask=full_mask,
            ).detach().cpu()
        else:
            ane_logits = ane_model(input_ids).detach().cpu()

    hf_logits = _concat_logits_if_split(hf_logits).float()
    ane_logits = _concat_logits_if_split(ane_logits).float()

    if hf_logits.shape != ane_logits.shape:
        print(f"❌ Shape mismatch: HF {hf_logits.shape} vs ANE {ane_logits.shape}")
        sys.exit(1)

    # Compare last token
    hf_last = hf_logits[:, -1, :]
    ane_last = ane_logits[:, -1, :]

    cos = F.cosine_similarity(hf_last, ane_last, dim=-1).mean().item()
    mse = F.mse_loss(ane_last, hf_last).item()

    k = min(args.topk, hf_last.shape[-1])
    hf_topk = torch.topk(hf_last, k=k, dim=-1).indices
    ane_topk = torch.topk(ane_last, k=k, dim=-1).indices
    overlap = torch.isin(ane_topk, hf_topk).float().mean().item()

    print("\n✅ Gemma3n HF vs ANEMLL (last-token logits)")
    print(f"  cosine: {cos:.6f}")
    print(f"  mse:    {mse:.6f}")
    print(f"  top-{k} overlap: {overlap:.4f}")

    # Optional: decode top-1 tokens
    hf_top1 = hf_topk[0, 0].item()
    ane_top1 = ane_topk[0, 0].item()
    print(f"  HF top-1:  {hf_top1} -> {tokenizer.decode([hf_top1])!r}")
    print(f"  ANE top-1: {ane_top1} -> {tokenizer.decode([ane_top1])!r}")


if __name__ == "__main__":
    main()
