#!/usr/bin/env python3
"""Smoke checks for state-transition chunk functions.

Modes:
1) infer-prefill:
   Prefill token-by-token using infer_ctx{context} only, then decode 1 token.
2) prefill-project:
   Prefill at max context using infer(4096), project KV state into infer_ctx{context}
   state object, then decode 1 token.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer


def _load_chunks(base_dir: Path, num_chunks: int, contexts: list[int]):
    chunks = []
    for idx in range(1, num_chunks + 1):
        path = base_dir / f"qwen25_FFN_PF_statex_chunk_{idx:02d}of{num_chunks:02d}.mlmodelc"
        entry = {
            "prefill": ct.models.CompiledMLModel(
                str(path), ct.ComputeUnit.CPU_AND_NE, function_name="prefill"
            ),
            "infer_4096": ct.models.CompiledMLModel(
                str(path), ct.ComputeUnit.CPU_AND_NE, function_name="infer"
            ),
        }
        for ctx in contexts:
            fn = f"infer_ctx{ctx}"
            entry[fn] = ct.models.CompiledMLModel(
                str(path), ct.ComputeUnit.CPU_AND_NE, function_name=fn
            )
        chunks.append(entry)
    return chunks


def _concat_logits(lm_out: dict[str, np.ndarray]) -> np.ndarray:
    if "output_logits" in lm_out:
        return lm_out["output_logits"]
    parts: list[tuple[int, np.ndarray]] = []
    for key, value in lm_out.items():
        m = re.fullmatch(r"logits(\d+)", key)
        if m:
            parts.append((int(m.group(1)), value))
    if not parts:
        raise RuntimeError(f"LM head output does not contain logits keys: {sorted(lm_out.keys())}")
    parts.sort(key=lambda x: x[0])
    return np.concatenate([v for _, v in parts], axis=-1)


def _copy_state_prefix(src_state, dst_state, context: int):
    src = src_state.read_state("model_model_kv_cache_0")
    dst = dst_state.read_state("model_model_kv_cache_0")
    n = min(context, int(src.shape[2]), int(dst.shape[2]))
    projected = np.zeros_like(dst)
    projected[:, :, :n, :] = src[:, :, :n, :]
    dst_state.write_state("model_model_kv_cache_0", projected)


def _prefill_tokenwise(
    *,
    chunks,
    embed_model,
    input_ids: torch.Tensor,
    state,
    context_len: int,
    infer_key: str,
):
    mask = np.zeros((1, 1, 1, context_len), dtype=np.float16)
    context_pos = int(input_ids.size(1))
    for pos in range(context_pos):
        token = input_ids[:, pos : pos + 1].numpy().astype(np.int32)
        hidden = embed_model.predict({"input_ids": token})["hidden_states"]
        position_ids = np.array([pos], dtype=np.int32)
        current_pos = np.array([pos], dtype=np.int32)
        for chunk in chunks:
            out = chunk[infer_key].predict(
                {
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": position_ids,
                    "causal_mask": mask,
                    "current_pos": current_pos,
                },
                state,
            )
            hidden = out["output_hidden_states"]


def _decode_one(
    *,
    chunks,
    embed_model,
    lm_head_model,
    input_ids: torch.Tensor,
    state,
    context_len: int,
    infer_key: str,
) -> int:
    pos = int(input_ids.size(1))
    token = input_ids[:, pos - 1 : pos].numpy().astype(np.int32)
    hidden = embed_model.predict({"input_ids": token})["hidden_states"]
    position_ids = np.array([pos - 1], dtype=np.int32)
    current_pos = np.array([pos - 1], dtype=np.int32)
    mask = np.zeros((1, 1, 1, context_len), dtype=np.float16)
    for chunk in chunks:
        out = chunk[infer_key].predict(
            {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": position_ids,
                "causal_mask": mask,
                "current_pos": current_pos,
            },
            state,
        )
        hidden = out["output_hidden_states"]
    lm_out = lm_head_model.predict({"hidden_states": hidden.astype(np.float16)})
    logits = _concat_logits(lm_out)
    return int(np.argmax(logits[0, -1, :]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="Combined state-transition output directory")
    ap.add_argument("--tokenizer", required=True, help="Tokenizer path")
    ap.add_argument("--contexts", default="512,1024,2048,3072,4096")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--num-chunks", type=int, default=3)
    ap.add_argument("--prompt", default="What is the capital of France?")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    contexts = [int(x) for x in args.contexts.replace(",", " ").split() if x.strip()]
    if args.max_context not in contexts:
        contexts.append(args.max_context)
        contexts.sort()

    embed_model = ct.models.CompiledMLModel(
        str(model_dir / "qwen25_embeddings.mlmodelc"), ct.ComputeUnit.CPU_AND_NE
    )
    lm_head_model = ct.models.CompiledMLModel(
        str(model_dir / "qwen25_lm_head.mlmodelc"), ct.ComputeUnit.CPU_AND_NE
    )
    chunks = _load_chunks(model_dir, args.num_chunks, contexts)

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.tokenizer).expanduser().resolve()),
        use_fast=False,
        trust_remote_code=True,
    )
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(torch.int32)

    print(f"Prompt tokens: {input_ids.size(1)}")

    print("\n[Step1] infer-prefill")
    for ctx in contexts:
        infer_key = f"infer_ctx{ctx}"
        state = chunks[0][infer_key].make_state()
        _prefill_tokenwise(
            chunks=chunks,
            embed_model=embed_model,
            input_ids=input_ids,
            state=state,
            context_len=ctx,
            infer_key=infer_key,
        )
        token = _decode_one(
            chunks=chunks,
            embed_model=embed_model,
            lm_head_model=lm_head_model,
            input_ids=input_ids,
            state=state,
            context_len=ctx,
            infer_key=infer_key,
        )
        print(f"  ctx{ctx}: ok (next_token={token})")

    print("\n[Step2] prefill@max then project")
    state_max = chunks[0]["prefill"].make_state()
    _prefill_tokenwise(
        chunks=chunks,
        embed_model=embed_model,
        input_ids=input_ids,
        state=state_max,
        context_len=args.max_context,
        infer_key="infer_4096",
    )
    for ctx in contexts:
        infer_key = f"infer_ctx{ctx}"
        state_ctx = chunks[0][infer_key].make_state()
        _copy_state_prefix(state_max, state_ctx, ctx)
        token = _decode_one(
            chunks=chunks,
            embed_model=embed_model,
            lm_head_model=lm_head_model,
            input_ids=input_ids,
            state=state_ctx,
            context_len=ctx,
            infer_key=infer_key,
        )
        print(f"  ctx{ctx}: ok (next_token={token})")

    print("\nState-transition smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

