#!/usr/bin/env python3
"""
Compare Gemma3 CoreML vs PyTorch at:
  1) Embedding output
  2) Chunk 1 output (layers [0, L/2))
  3) Chunk 2 output (layers [L/2, L), with final norm)

This is a single-token comparison (no prefill) to isolate conversion issues.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

# Add repo root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests import chat as chat
from anemll.models.gemma3_model import Gemma3Config, Gemma3ForCausalLM


def _load_meta(meta_path: Path) -> dict:
    with meta_path.open("r") as f:
        return yaml.safe_load(f)


def _strip_model_ext(value: str) -> str:
    return value.replace(".mlmodelc", "").replace(".mlpackage", "")


def _stats(name: str, a: np.ndarray, b: np.ndarray) -> None:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    print(f"{name}: max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Gemma3 CoreML chunks vs PyTorch (single token)."
    )
    parser.add_argument("--coreml-dir", required=True, help="CoreML model directory")
    parser.add_argument("--hf-dir", required=True, help="HF snapshot dir with safetensors")
    parser.add_argument("--meta", default=None, help="Path to meta.yaml (defaults to <coreml-dir>/meta.yaml)")
    parser.add_argument("--prompt", default=None, help="Prompt to derive a token id")
    parser.add_argument("--token-id", type=int, default=None, help="Explicit token id (overrides --prompt)")
    parser.add_argument("--pos", type=int, default=0, help="Token position (default: 0)")
    parser.add_argument("--context-length", type=int, default=None, help="Override context length")
    parser.add_argument("--num-chunks", type=int, default=None, help="Override num chunks")
    args = parser.parse_args()

    coreml_dir = Path(args.coreml_dir).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else (coreml_dir / "meta.yaml")
    if not meta_path.exists():
        print(f"meta.yaml not found: {meta_path}")
        return 1

    meta = _load_meta(meta_path)
    params = meta["model_info"]["parameters"]
    context_length = args.context_length or int(params.get("context_length", 4096))
    num_chunks = args.num_chunks or int(params.get("num_chunks", 2))

    # Load tokenizer for prompt -> token id
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, use_fast=False, local_files_only=True)
    if args.token_id is not None:
        token_id = args.token_id
    elif args.prompt:
        ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True).input_ids
        token_id = int(ids[0, 0].item())
    else:
        token_id = int(tokenizer.bos_token_id or 2)

    print(f"Using token_id={token_id} ('{tokenizer.decode([token_id])}') at pos={args.pos}")
    print(f"Context length={context_length}, num_chunks={num_chunks}")

    # Prepare CoreML models via chat.py loader
    class _Args:
        pass

    load_args = _Args()
    load_args.eval = True
    load_args.cpu = True
    load_args.mem_report = False
    load_args.context_length = context_length
    load_args.batch_size = int(params.get("batch_size", 64))
    load_args.num_chunks = num_chunks
    load_args.d = str(coreml_dir)
    load_args.embed = str(coreml_dir / _strip_model_ext(params.get("embeddings", "gemma3_embeddings")))
    load_args.lmhead = str(coreml_dir / _strip_model_ext(params.get("lm_head", "gemma3_lm_head")))
    load_args.ffn = str(coreml_dir / _strip_model_ext(params.get("ffn", "gemma3_FFN_PF_chunk_01of02")))

    embed_model, ffn_models, _, _ = chat.load_models(load_args, {})
    state = ffn_models[0]["prefill"].make_state()

    # CoreML embedding output
    input_ids_np = np.array([[token_id]], dtype=np.int32)
    coreml_embed = embed_model.predict({"input_ids": input_ids_np})["hidden_states"]

    # CoreML chunk 1 -> chunk 2
    causal_mask = chat.make_causal_mask(context_length, 0)[:, :, args.pos : args.pos + 1, :]
    position_ids_np = np.array([args.pos], dtype=np.int32)
    current_pos_np = np.array([args.pos], dtype=np.int32)

    ffn_inputs = {
        "hidden_states": coreml_embed.astype(np.float16),
        "position_ids": position_ids_np,
        "causal_mask": causal_mask.astype(np.float16),
        "current_pos": current_pos_np,
    }
    coreml_chunk1 = ffn_models[0]["infer"].predict(ffn_inputs, state)["output_hidden_states"]

    ffn_inputs["hidden_states"] = coreml_chunk1.astype(np.float16)
    coreml_chunk2 = ffn_models[1]["infer"].predict(ffn_inputs, state)["output_hidden_states"]

    # PyTorch model
    config = Gemma3Config.from_json(str(Path(args.hf_dir) / "config.json"))
    config.context_length = context_length
    config.state_length = context_length
    model = Gemma3ForCausalLM(config, disable_kv_cache=True)
    model.load_pretrained_weights(args.hf_dir)
    model.eval()

    input_ids = torch.tensor([[token_id]], dtype=torch.int64)
    position_ids = torch.tensor([args.pos], dtype=torch.long)
    current_pos = torch.tensor([args.pos], dtype=torch.long)
    causal_mask_t = torch.tensor(causal_mask, dtype=torch.float16)

    with torch.no_grad():
        pt_embed = model.model.embed_tokens(input_ids) * model.model.embedding_scale

        total_layers = config.num_hidden_layers
        layers_per_chunk = total_layers // num_chunks
        start2 = layers_per_chunk

        pt_chunk1 = model.model.process_layers(
            pt_embed, position_ids, causal_mask_t, current_pos,
            start_layer=0, end_layer=start2, IN_PREFILL=False
        )
        pt_chunk2 = model.model.process_layers(
            pt_chunk1, position_ids, causal_mask_t, current_pos,
            start_layer=start2, end_layer=None, IN_PREFILL=False
        )
        # Final norm for last chunk
        pt_chunk2 = model.model.norm(pt_chunk2)

    # Compare
    _stats("embeddings", coreml_embed, pt_embed.cpu().numpy())
    _stats("chunk1_out", coreml_chunk1, pt_chunk1.cpu().numpy())
    _stats("chunk2_out", coreml_chunk2, pt_chunk2.cpu().numpy())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
