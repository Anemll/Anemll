#!/usr/bin/env python3
"""State-transition decode runner with visible token output and timing stats."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer


def _concat_logits(lm_out: dict[str, np.ndarray]) -> np.ndarray:
    if "output_logits" in lm_out:
        return lm_out["output_logits"]
    parts: list[tuple[int, np.ndarray]] = []
    for key, value in lm_out.items():
        m = re.fullmatch(r"logits(\d+)", key)
        if m:
            parts.append((int(m.group(1)), value))
    if not parts:
        raise RuntimeError(f"LM head output missing logits: {sorted(lm_out.keys())}")
    parts.sort(key=lambda x: x[0])
    return np.concatenate([v for _, v in parts], axis=-1)


def _load_chunks(model_dir: Path, num_chunks: int, context: int):
    chunks = []
    for i in range(1, num_chunks + 1):
        p = model_dir / f"qwen25_FFN_PF_statex_chunk_{i:02d}of{num_chunks:02d}.mlmodelc"
        chunks.append(
            {
                "prefill": ct.models.CompiledMLModel(
                    str(p), ct.ComputeUnit.CPU_AND_NE, function_name="prefill"
                ),
                "infer_4096": ct.models.CompiledMLModel(
                    str(p), ct.ComputeUnit.CPU_AND_NE, function_name="infer"
                ),
                "infer_ctx": ct.models.CompiledMLModel(
                    str(p), ct.ComputeUnit.CPU_AND_NE, function_name=f"infer_ctx{context}"
                ),
            }
        )
    return chunks


def _prefill_tokenwise(
    *,
    chunks,
    embed_model,
    input_ids: torch.Tensor,
    state,
    mask_len: int,
    infer_key: str,
) -> None:
    mask = np.zeros((1, 1, 1, mask_len), dtype=np.float16)
    context_pos = int(input_ids.size(1))
    for pos in range(context_pos):
        token = input_ids[:, pos : pos + 1].numpy().astype(np.int32)
        hidden = embed_model.predict({"input_ids": token})["hidden_states"]
        pid = np.array([pos], dtype=np.int32)
        cur = np.array([pos], dtype=np.int32)
        for ch in chunks:
            out = ch[infer_key].predict(
                {
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": pid,
                    "causal_mask": mask,
                    "current_pos": cur,
                },
                state,
            )
            hidden = out["output_hidden_states"]


def _project_state_4096_to_ctx(src_state, dst_state, context: int) -> None:
    src = src_state.read_state("model_model_kv_cache_0")
    dst = dst_state.read_state("model_model_kv_cache_0")
    n = min(context, int(src.shape[2]), int(dst.shape[2]))
    projected = np.zeros_like(dst)
    projected[:, :, :n, :] = src[:, :, :n, :]
    dst_state.write_state("model_model_kv_cache_0", projected)


def _decode_step(
    *,
    chunks,
    embed_model,
    lm_head_model,
    token_id: int,
    pos: int,
    context: int,
    state,
) -> int:
    token = np.array([[token_id]], dtype=np.int32)
    hidden = embed_model.predict({"input_ids": token})["hidden_states"]
    pid = np.array([pos], dtype=np.int32)
    cur = np.array([pos], dtype=np.int32)
    mask = np.zeros((1, 1, 1, context), dtype=np.float16)
    for ch in chunks:
        out = ch["infer_ctx"].predict(
            {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pid,
                "causal_mask": mask,
                "current_pos": cur,
            },
            state,
        )
        hidden = out["output_hidden_states"]
    lm_out = lm_head_model.predict({"hidden_states": hidden.astype(np.float16)})
    logits = _concat_logits(lm_out)
    return int(np.argmax(logits[0, -1, :]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--num-chunks", type=int, default=3)
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument(
        "--mode",
        choices=["infer-prefill", "prefill-project"],
        default="infer-prefill",
        help="infer-prefill: prefill via infer_ctx; prefill-project: prefill@4096 then project",
    )
    ap.add_argument("--no-think", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    tok_dir = Path(args.tokenizer).expanduser().resolve()

    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_dir), use_fast=False, trust_remote_code=True
    )
    tpl_kwargs = {"enable_thinking": False} if args.no_think else {}
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        **tpl_kwargs,
    ).to(torch.int32)

    embed = ct.models.CompiledMLModel(
        str(model_dir / "qwen25_embeddings.mlmodelc"), ct.ComputeUnit.CPU_AND_NE
    )
    lm_head = ct.models.CompiledMLModel(
        str(model_dir / "qwen25_lm_head.mlmodelc"), ct.ComputeUnit.CPU_AND_NE
    )
    chunks = _load_chunks(model_dir, args.num_chunks, args.context)

    prefill_start = time.time()
    if args.mode == "infer-prefill":
        state = chunks[0]["infer_ctx"].make_state()
        _prefill_tokenwise(
            chunks=chunks,
            embed_model=embed,
            input_ids=input_ids,
            state=state,
            mask_len=args.context,
            infer_key="infer_ctx",
        )
    else:
        state_4096 = chunks[0]["prefill"].make_state()
        _prefill_tokenwise(
            chunks=chunks,
            embed_model=embed,
            input_ids=input_ids,
            state=state_4096,
            mask_len=args.max_context,
            infer_key="infer_4096",
        )
        state = chunks[0]["infer_ctx"].make_state()
        _project_state_4096_to_ctx(state_4096, state, args.context)
    prefill_time = time.time() - prefill_start

    context_pos = int(input_ids.size(1))
    generated: list[int] = []
    last_token = int(input_ids[0, context_pos - 1].item())
    decode_start = time.time()
    for i in range(args.max_tokens):
        pos = context_pos + i - 1
        if pos >= args.context - 1:
            break
        token = _decode_step(
            chunks=chunks,
            embed_model=embed,
            lm_head_model=lm_head,
            token_id=last_token,
            pos=pos,
            context=args.context,
            state=state,
        )
        generated.append(token)
        piece = tokenizer.decode([token], skip_special_tokens=False)
        print(piece, end="", flush=True)
        last_token = token
        if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            break
    print()
    decode_time = time.time() - decode_start

    prefill_tps = context_pos / prefill_time if prefill_time > 0 else 0.0
    decode_tps = len(generated) / decode_time if decode_time > 0 else 0.0
    print(
        f"mode={args.mode} context={args.context} prefill={prefill_time*1000:.1f}ms "
        f"({prefill_tps:.1f} t/s) decode={decode_tps:.1f} t/s tokens={len(generated)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

