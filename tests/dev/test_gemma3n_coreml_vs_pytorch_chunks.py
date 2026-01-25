#!/usr/bin/env python3
"""Compare CoreML infer chunks vs PyTorch Gemma3n outputs per chunk."""

import argparse
import os
from pathlib import Path
import numpy as np
import torch

import coremltools as ct

import anemll.models.gemma3n_model as gemma3n_mod
from anemll.models.gemma3n_model import Gemma3nModel, Gemma3nConfig
from transformers import AutoConfig

from gemma3n_coreml_inputs import (
    create_position_mask,
    create_position_one_hot,
    create_rotary_embeddings,
)


def load_mlpackage(path: Path) -> ct.models.MLModel:
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return ct.models.MLModel(str(path))


def build_causal_mask(context_length: int, dtype=np.float16) -> np.ndarray:
    causal = np.zeros((1, 1, context_length, context_length), dtype=dtype)
    i_idx, j_idx = np.triu_indices(context_length, k=1)
    causal[:, :, i_idx, j_idx] = float("-inf")
    return causal


def build_causal_mask_torch(context_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    causal_np = build_causal_mask(context_length, dtype=np.float32)
    return torch.tensor(causal_np, dtype=dtype, device=device)

def summarize(arr: np.ndarray, name: str, max_vals: int = 5) -> None:
    flat = arr.reshape(-1)
    sample = flat[:max_vals]
    print(
        f"{name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min():.6f} max={arr.max():.6f} "
        f"mean={arr.mean():.6f} std={arr.std():.6f} "
        f"sample={sample}"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CoreML infer chunks vs PyTorch")
    parser.add_argument("--bundle", default="/tmp/gemma3n-infer/bundle", help="CoreML bundle directory")
    parser.add_argument("--model", required=True, help="Path to local Gemma3n model")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--token-id", type=int, default=2, help="Token id to test")
    parser.add_argument("--pos", type=int, default=0, help="Current position for KV cache")
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    args = parser.parse_args()

    bundle = Path(args.bundle)
    infer_init = load_mlpackage(bundle / "gemma3n_infer_init.mlpackage")
    combine = load_mlpackage(bundle / "gemma3n_combine_streams.mlpackage")
    infer_chunks = sorted(bundle.glob("gemma3n_infer_chunk_*of*.mlpackage"))
    if not infer_chunks:
        raise FileNotFoundError(f"No infer chunks found in {bundle}")
    infer_chunk_models = [load_mlpackage(p) for p in infer_chunks]

    # CoreML run
    init_out = infer_init.predict({"input_ids": np.array([[args.token_id]], dtype=np.int32)})
    cm_hidden = init_out["hidden_states"]
    cm_per_layer = init_out["per_layer_inputs"]
    state = infer_chunk_models[0].make_state()
    kv_cache = state.read_state("model_kv_cache_0")
    ctx_len = kv_cache.shape[2]
    head_dim = kv_cache.shape[3]
    pos_mask = create_position_mask(args.pos, ctx_len)
    pos_one_hot = create_position_one_hot(args.pos, ctx_len)
    cos_local, sin_local, cos_global, sin_global = create_rotary_embeddings(args.pos, head_dim)

    cm_chunk_outputs = []
    for model in infer_chunk_models:
        out = model.predict(
            {
                "hidden_states": cm_hidden,
                "per_layer_inputs": cm_per_layer,
                "causal_mask": pos_mask,
                "current_pos": np.array([args.pos], dtype=np.int32),
                "position_one_hot": pos_one_hot,
                "rotary_cos_local": cos_local,
                "rotary_sin_local": sin_local,
                "rotary_cos_global": cos_global,
                "rotary_sin_global": sin_global,
            },
            state,
        )
        cm_hidden = out["output_hidden_states"]
        cm_chunk_outputs.append(cm_hidden)

    # PyTorch run
    if not os.path.isdir(args.model):
        raise FileNotFoundError(f"Model path not found: {args.model}")

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    ane_config = Gemma3nConfig.from_pretrained_config(hf_config)
    ane_config.context_length = args.context_length
    ane_config.state_length = args.context_length

    model = Gemma3nModel(ane_config)
    model.load_weights(args.model, config=ane_config)
    model.to(device=device, dtype=torch_dtype)
    model.eval()
    model.reset_kv_cache()

    input_ids = torch.tensor([[args.token_id]], dtype=torch.int32, device=device)
    inputs_embeds, per_layer_inputs = model._compute_inputs_and_per_layer(input_ids)
    hidden_states = model._init_hidden_states(inputs_embeds)
    causal_mask = build_causal_mask_torch(args.context_length, device, hidden_states.dtype)
    current_pos = torch.tensor(args.pos, device=device)

    total_layers = ane_config.num_hidden_layers
    layers_per_chunk = total_layers // args.chunk

    print("\nChunk comparison (CoreML vs PyTorch):")
    for idx, cm_out in enumerate(cm_chunk_outputs):
        start = idx * layers_per_chunk
        end = total_layers if idx == args.chunk - 1 else (idx + 1) * layers_per_chunk
        with torch.no_grad():
            hidden_states = model.process_layers(
                hidden_states,
                per_layer_inputs,
                causal_mask,
                current_pos,
                start_layer=start,
                end_layer=end,
                prefill=False,
            )
        pt_out = hidden_states.detach().cpu().numpy().astype(np.float32)
        cm_out_f = cm_out.astype(np.float32)
        diff = np.abs(pt_out - cm_out_f)
        print(
            f"  chunk {idx} layers {start}-{end-1}: "
            f"max={diff.max():.6f} mean={diff.mean():.6f}"
        )

    # Optional: compare combined stream
    combined_pt = model._combine_streams(hidden_states).detach().cpu().numpy().astype(np.float32)
    combined_cm = combine.predict({"hidden_states": cm_hidden})["output_hidden_states"].astype(np.float32)
    diff = np.abs(combined_pt - combined_cm)
    print(
        f"\nCombined streams diff: max={diff.max():.6f} mean={diff.mean():.6f}"
    )
    summarize(combined_pt, "PyTorch combine.hidden_states")
    summarize(combined_cm, "CoreML  combine.hidden_states")
    summarize(diff, "ABS diff combine.hidden_states")


if __name__ == "__main__":
    main()
