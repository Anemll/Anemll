#!/usr/bin/env python3
"""Verify Gemma3n outputs match when using external vs internal masks."""

import argparse
import math

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from anemll.models.gemma3n_model import Gemma3nModel, Gemma3nConfig


def build_masks(seq_len: int, sliding_window: int, device, dtype):
    idx = torch.arange(seq_len, device=device)
    i = idx[:, None]
    j = idx[None, :]

    causal_bool = (j > i)
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    zeros = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    causal = torch.where(causal_bool, neg_inf, zeros)

    if sliding_window and sliding_window > 0:
        window = (j < (i - sliding_window + 1))
        local = torch.where(causal_bool | window, neg_inf, zeros)
    else:
        local = causal.clone()

    causal = causal.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
    local = local.unsqueeze(0).unsqueeze(0)
    return causal, local


def main():
    parser = argparse.ArgumentParser(description="Gemma3n external mask consistency test")
    parser.add_argument("--model", required=True, help="Path to Gemma3n model directory")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt text")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="Device")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Dtype")
    parser.add_argument("--max-layers", type=int, default=4, help="Limit layers for faster test")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    config = Gemma3nConfig.from_json(f"{args.model}/config.json")
    model = Gemma3nModel(config)
    model.load_weights(args.model)
    model.to(device=device, dtype=dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        # Baseline (internal masks)
        logits_internal = model(input_ids, debug_layer_limit=args.max_layers)

        # External mask (single causal mask)
        seq_len = input_ids.shape[1]
        full_mask, _ = build_masks(seq_len, config.sliding_window, device, dtype)
        logits_external = model(
            input_ids,
            debug_layer_limit=args.max_layers,
            attention_mask=full_mask,
        )

    # Compare logits
    if isinstance(logits_internal, tuple):
        logits_internal = torch.cat(list(logits_internal), dim=-1)
    if isinstance(logits_external, tuple):
        logits_external = torch.cat(list(logits_external), dim=-1)

    internal = logits_internal.float().flatten()
    external = logits_external.float().flatten()

    cos = F.cosine_similarity(internal, external, dim=0).item()
    mse = torch.mean((internal - external) ** 2).item()

    top_internal = int(torch.argmax(logits_internal[0, -1]).item())
    top_external = int(torch.argmax(logits_external[0, -1]).item())

    print(f"cosine: {cos:.6f}")
    print(f"mse: {mse:.6f}")
    print(f"top-1 internal: {top_internal}")
    print(f"top-1 external: {top_external}")
    print("match:", top_internal == top_external)


if __name__ == "__main__":
    main()
