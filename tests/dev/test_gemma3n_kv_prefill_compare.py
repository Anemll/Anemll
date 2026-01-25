#!/usr/bin/env python3
"""Compare CoreML vs PyTorch KV cache after prefill."""

import argparse
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
from transformers import AutoConfig, AutoTokenizer

from anemll.models.gemma3n_model import (
    Gemma3nConfig,
    Gemma3nModel,
    MODEL_DTYPE,
    TEST_DEVICE,
    create_causal_mask_4d,
)
import anemll.models.gemma3n_model as gemma3n_mod

from gemma3n_coreml_inputs import (
    create_position_mask,
    create_position_one_hot,
    create_rotary_embeddings,
)


def load_mlpackage(path: Path) -> ct.models.MLModel:
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-8
    return float(np.dot(a_flat, b_flat) / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--context-length", type=int, default=512)
    args = parser.parse_args()

    bundle = Path(args.bundle)
    tokenizer = AutoTokenizer.from_pretrained(str(bundle), use_fast=False, trust_remote_code=True)

    tokens = tokenizer(args.prompt, return_tensors="np")["input_ids"][0].tolist()
    print(f"Prompt tokens ({len(tokens)}): {tokens}")

    # ===== CoreML prefill =====
    infer_init = load_mlpackage(bundle / "gemma3n_infer_init.mlpackage")
    chunk_paths = sorted(bundle.glob("gemma3n_infer_chunk_*of*.mlpackage"))
    if not chunk_paths:
        infer_single = bundle / "gemma3n_infer.mlpackage"
        if infer_single.exists():
            chunk_paths = [infer_single]
        else:
            raise FileNotFoundError("No infer chunk(s) or gemma3n_infer.mlpackage found")

    chunks = [load_mlpackage(p) for p in chunk_paths]
    state = chunks[0].make_state()
    kv_name = "model_kv_cache_0"
    kv_cache = state.read_state(kv_name)
    ctx_len = kv_cache.shape[2]
    head_dim = kv_cache.shape[3]

    for pos, tok in enumerate(tokens):
        init_out = infer_init.predict({"input_ids": np.array([[int(tok)]], dtype=np.int32)})
        hidden = init_out["hidden_states"]
        pli = init_out["per_layer_inputs"]
        pos_mask = create_position_mask(pos, ctx_len)
        pos_one_hot = create_position_one_hot(pos, ctx_len)
        cos_local, sin_local, cos_global, sin_global = create_rotary_embeddings(pos, head_dim)

        for chunk in chunks:
            out = chunk.predict(
                {
                    "hidden_states": hidden,
                    "per_layer_inputs": pli,
                    "causal_mask": pos_mask,
                    "current_pos": np.array([pos], dtype=np.int32),
                    "position_one_hot": pos_one_hot,
                    "rotary_cos_local": cos_local,
                    "rotary_sin_local": sin_local,
                    "rotary_cos_global": cos_global,
                    "rotary_sin_global": sin_global,
                },
                state,
            )
            hidden = out["output_hidden_states"]

    coreml_kv = state.read_state(kv_name)
    print(f"CoreML KV cache: shape={coreml_kv.shape} dtype={coreml_kv.dtype}")

    # ===== PyTorch prefill =====
    gemma3n_mod.ENABLE_COREML = False
    hf_config = AutoConfig.from_pretrained(args.model)
    config = Gemma3nConfig.from_pretrained_config(hf_config)
    config.context_length = args.context_length
    config.state_length = max(config.state_length, args.context_length)
    model = Gemma3nModel(config)
    model.load_weights(args.model, config=config)
    model.to(device=TEST_DEVICE, dtype=MODEL_DTYPE)
    model.eval()
    model.reset_kv_cache()

    with torch.no_grad():
        causal_mask = create_causal_mask_4d(config.state_length, 1, TEST_DEVICE, MODEL_DTYPE)
        for pos, tok in enumerate(tokens):
            input_ids = torch.tensor([[tok]], dtype=torch.int32, device=TEST_DEVICE)
            embeds, per_layer = model._compute_inputs_and_per_layer(input_ids)
            hidden_states = model._init_hidden_states(embeds)
            for layer_idx in range(config.num_hidden_layers):
                per_layer_input = per_layer[:, :, layer_idx, :]
                hidden_states = model._process_layer_regular(
                    layer_idx,
                    hidden_states,
                    per_layer_input,
                    causal_mask,
                    torch.tensor([pos], dtype=torch.int32, device=TEST_DEVICE),
                )

    pt_kv = model.kv_cache_0.detach().cpu().numpy()
    print(f"PyTorch KV cache: shape={pt_kv.shape} dtype={pt_kv.dtype}")

    # ===== Compare =====
    diff = np.abs(pt_kv - coreml_kv)
    print("\nKV cache comparison after prefill:")
    print(f"  cosine similarity: {cos_sim(pt_kv, coreml_kv):.6f}")
    print(f"  max abs diff: {diff.max():.6f}")
    print(f"  mean abs diff: {diff.mean():.6f}")


if __name__ == "__main__":
    main()
