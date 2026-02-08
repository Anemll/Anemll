#!/usr/bin/env python3
"""Prototype pipeline for Qwen2.5-style models:

Stage 0 (CPU/FP32):
  input_ids -> embeddings + layer0 attention (no layer0 MLP)

Chunk 1 (FP16):
  layer0 post-attention MLP + layers1..N/2

Chunk 2 (FP16):
  remaining layers + final norm

This is a prototype builder/runner only. It does not modify main converters.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anemll.ane_converter.qwen2_5_converter import Qwen25Converter
from anemll.models.qwen2_5_model import Qwen25Config, Qwen25ForCausalLM


def _resolve_model_path(model_path: str) -> Path:
    p = Path(model_path).expanduser().resolve()
    if (p / "config.json").exists():
        return p
    snaps = [s for s in p.glob("snapshots/*") if (s / "config.json").exists()]
    if snaps:
        return snaps[0]
    raise FileNotFoundError(f"config.json not found under: {p}")


def _reset_kv_buffers(module: torch.nn.Module | torch.jit.ScriptModule) -> None:
    with torch.no_grad():
        for n, b in module.named_buffers():
            if "kv_cache_" in n:
                b.zero_()


def _make_mask(context_length: int, pos: int) -> np.ndarray:
    m = np.zeros((1, 1, 1, context_length), dtype=np.float16)
    if pos + 1 < context_length:
        m[0, 0, 0, pos + 1 :] = np.float16(-65504.0)
    return m


def _find_output_key(pred: dict, preferred: str) -> str:
    if preferred in pred:
        return preferred
    if len(pred) == 1:
        return next(iter(pred.keys()))
    for key in ("output_hidden_states", "hidden_states", "output_logits"):
        if key in pred:
            return key
    raise KeyError(f"Unable to resolve output key from: {list(pred.keys())}")


class EmbedLayer0AttentionWrapper(torch.nn.Module):
    """Embeddings + layer0 attention residual only (no layer0 MLP)."""

    def __init__(self, model: Qwen25ForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, causal_mask, current_pos):
        del position_ids  # kept for interface compatibility
        hidden_states = self.model.model.embed_tokens(input_ids)

        layer = self.model.model.layers[0]
        normalized_states = layer.input_layernorm(hidden_states)
        rotary = self.model.model.get_rotary_embeddings_s(current_pos)
        query_states, key_states, value_states = layer.self_attn.get_new_kv_cache(
            normalized_states,
            current_pos,
            rotary,
        )

        kv_cache = self.model.model.kv_cache_0
        layers_per_group = self.model.config.num_hidden_layers
        key_idx = 0
        value_idx = layers_per_group
        pos = current_pos

        kv_cache[key_idx : key_idx + 1, :, pos : pos + 1, :] = key_states
        kv_cache[value_idx : value_idx + 1, :, pos : pos + 1, :] = value_states

        key_cache = kv_cache[key_idx : key_idx + 1].squeeze(0)
        value_cache = kv_cache[value_idx : value_idx + 1].squeeze(0)
        attn_out = layer.self_attn.forward_regular(
            hidden_states=normalized_states,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
            current_pos=current_pos,
        )
        return hidden_states + attn_out


class Chunk1AfterAttentionWrapper(torch.nn.Module):
    """layer0 post-attn MLP + a range of full layers."""

    def __init__(self, model: Qwen25ForCausalLM, end_layer: int):
        super().__init__()
        self.model = model
        self.end_layer = end_layer

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        layer0 = self.model.model.layers[0]
        post = layer0.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer0.mlp(post)

        rotary = self.model.model.get_rotary_embeddings_s(current_pos)
        return self.model.model.process_layers(
            hidden_states,
            position_ids,
            causal_mask,
            current_pos,
            rotary,
            start_layer=1,
            end_layer=self.end_layer,
            IN_PREFILL=False,
        )


class TailRangeWrapper(torch.nn.Module):
    def __init__(self, model: Qwen25ForCausalLM, start_layer: int):
        super().__init__()
        self.model = model
        self.start_layer = start_layer

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        rotary = self.model.model.get_rotary_embeddings_s(current_pos)
        out = self.model.model.process_layers(
            hidden_states,
            position_ids,
            causal_mask,
            current_pos,
            rotary,
            start_layer=self.start_layer,
            end_layer=None,
            IN_PREFILL=False,
        )
        return self.model.model.norm(out)


def _convert_stage0_wrapper(
    wrapper: torch.nn.Module,
    model: Qwen25ForCausalLM,
    context_length: int,
    out_path: Path,
) -> None:
    input_ids = torch.zeros((1, 1), dtype=torch.int32)
    position_ids = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, 1, context_length), dtype=torch.float16)
    current_pos = torch.zeros((1,), dtype=torch.int32)

    _reset_kv_buffers(wrapper)
    traced = torch.jit.trace(wrapper, (input_ids, position_ids, causal_mask, current_pos))
    _reset_kv_buffers(wrapper)
    _reset_kv_buffers(traced)

    states = Qwen25Converter.GetTransformerStates(model, part="2", prefix="model.model.")
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="hidden_states", dtype=np.float16)],
        states=states,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    if out_path.exists():
        shutil.rmtree(out_path)
    ml.save(str(out_path))


def _convert_hidden_wrapper(
    wrapper: torch.nn.Module,
    model: Qwen25ForCausalLM,
    hidden_size: int,
    context_length: int,
    out_path: Path,
) -> None:
    hidden_states = torch.zeros((1, 1, hidden_size), dtype=torch.float16)
    position_ids = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, 1, context_length), dtype=torch.float16)
    current_pos = torch.zeros((1,), dtype=torch.int32)

    _reset_kv_buffers(wrapper)
    traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos))
    _reset_kv_buffers(wrapper)
    _reset_kv_buffers(traced)

    states = Qwen25Converter.GetTransformerStates(model, part="2", prefix="model.model.")
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    if out_path.exists():
        shutil.rmtree(out_path)
    ml.save(str(out_path))


def _resolve_lmhead_path(explicit_path: str | None, source_dir: Path) -> Path:
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    candidates = sorted(source_dir.glob("*lm_head*.mlmodelc"))
    if not candidates:
        candidates = sorted(source_dir.glob("*lm_head*.mlpackage"))
    if not candidates:
        raise FileNotFoundError(
            f"No LM head model found in {source_dir}. "
            "Use --lmhead-path to set it explicitly."
        )
    return candidates[0]


def _pick_next_token(lm_out: dict, vocab_size: int) -> int:
    if "argmax_idx" in lm_out and "argmax_val" in lm_out:
        idx = lm_out["argmax_idx"].astype(np.int64).reshape(-1)
        val = lm_out["argmax_val"].astype(np.float32).reshape(-1)
        best = int(np.argmax(val))
        num_chunks = max(len(idx), 1)
        base, rem = divmod(vocab_size, num_chunks)
        offsets: list[int] = []
        cur = 0
        for i in range(num_chunks):
            offsets.append(cur)
            cur += base + (1 if i < rem else 0)
        return int(idx[best] + offsets[best])

    if "output_logits" in lm_out:
        logits = lm_out["output_logits"]
        return int(np.argmax(logits[0, -1, :]))

    logit_keys = [k for k in lm_out.keys() if k.startswith("logits")]
    if not logit_keys:
        raise KeyError(f"LM head outputs do not contain logits/argmax fields: {list(lm_out.keys())}")

    def _order(k: str) -> int:
        suffix = k.replace("logits", "")
        return int(suffix) if suffix.isdigit() else 10**9

    pieces = [lm_out[k] for k in sorted(logit_keys, key=_order)]
    logits = np.concatenate(pieces, axis=-1)
    return int(np.argmax(logits[0, -1, :]))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prototype: embeddings+layer0-attn(FP32) + adjusted FFN chunks."
    )
    ap.add_argument(
        "--model-path",
        default="~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B",
        help="Local HF snapshot dir or cache root",
    )
    ap.add_argument("--prefix", default="qwen25")
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument(
        "--out-dir",
        default="/Volumes/Models/ANE/vibethinker_1.5b_proto_embed_attn_fp32",
    )
    ap.add_argument(
        "--lmhead-path",
        default=None,
        help="Path to existing LM head model (.mlmodelc/.mlpackage). If omitted, search in --source-model-dir.",
    )
    ap.add_argument(
        "--source-model-dir",
        default="/Volumes/Models/ANE/vibethinker_1.5b_ctx2048_lut6",
        help="Directory to search for LM head when --lmhead-path is not provided.",
    )
    ap.add_argument("--reuse-models", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--prompt", default="2+2=")
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    model_path = _resolve_model_path(args.model_path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Qwen25Config.from_json(str(model_path / "config.json"))
    cfg.context_length = args.context_length
    cfg.state_length = args.context_length
    model = Qwen25ForCausalLM(cfg, enable_coreml=True)
    model.load_pretrained_weights(str(model_path))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    num_layers = int(cfg.num_hidden_layers)
    split_layer = num_layers // 2

    emb_attn_wrapper = EmbedLayer0AttentionWrapper(model).eval()
    chunk1_wrapper = Chunk1AfterAttentionWrapper(model, end_layer=split_layer).eval()
    chunk2_wrapper = TailRangeWrapper(model, start_layer=split_layer).eval()

    emb_name = f"{args.prefix}_embeddings"
    chunk1_name = f"{args.prefix}_FFN_PF_chunk_01of02"
    chunk2_name = f"{args.prefix}_FFN_PF_chunk_02of02"

    emb_path = out_dir / f"{emb_name}.mlpackage"
    chunk1_path = out_dir / f"{chunk1_name}.mlpackage"
    chunk2_path = out_dir / f"{chunk2_name}.mlpackage"

    if not args.reuse_models or not emb_path.exists():
        print("Converting stage0 embeddings+layer0-attention FP32 (CPU_ONLY)...")
        _convert_stage0_wrapper(
            emb_attn_wrapper,
            model,
            args.context_length,
            emb_path,
        )
    if not args.reuse_models or not chunk1_path.exists():
        print("Converting chunk1 (layer0 MLP + layers1..split) FP16...")
        _convert_hidden_wrapper(
            chunk1_wrapper,
            model,
            cfg.hidden_size,
            args.context_length,
            chunk1_path,
        )
    if not args.reuse_models or not chunk2_path.exists():
        print("Converting chunk2 (split..end + norm) FP16...")
        _convert_hidden_wrapper(
            chunk2_wrapper,
            model,
            cfg.hidden_size,
            args.context_length,
            chunk2_path,
        )

    print("\nPrototype artifacts")
    print(f"  Stage0: {emb_path}")
    print(f"  Chunk1: {chunk1_path}")
    print(f"  Chunk2: {chunk2_path}")

    if args.build_only:
        return 0

    source_model_dir = Path(args.source_model_dir).expanduser().resolve()
    lmhead_path = _resolve_lmhead_path(args.lmhead_path, source_model_dir)
    print(f"  LM head: {lmhead_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if args.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids

    emb_model = ct.models.MLModel(str(emb_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    chunk1_model = ct.models.MLModel(str(chunk1_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    chunk2_model = ct.models.MLModel(str(chunk2_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    lm_model = ct.models.MLModel(str(lmhead_path), compute_units=ct.ComputeUnit.CPU_AND_NE)

    s0 = emb_model.make_state()
    s1 = chunk1_model.make_state()
    s2 = chunk2_model.make_state()

    emb_out_key = None

    def step(token_id: int, pos: int) -> np.ndarray:
        nonlocal emb_out_key
        token_arr = np.array([[token_id]], dtype=np.int32)
        pos_arr = np.array([pos], dtype=np.int32)
        mask = _make_mask(args.context_length, pos)

        emb_out = emb_model.predict(
            {
                "input_ids": token_arr,
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
            },
            state=s0,
        )
        if emb_out_key is None:
            emb_out_key = _find_output_key(emb_out, "hidden_states")
        h0 = emb_out[emb_out_key].astype(np.float16)

        h1 = chunk1_model.predict(
            {
                "hidden_states": h0,
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
            },
            state=s1,
        )["output_hidden_states"].astype(np.float16)

        h2 = chunk2_model.predict(
            {
                "hidden_states": h1,
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
            },
            state=s2,
        )["output_hidden_states"].astype(np.float16)

        return h2

    pos = 0
    last_hidden = None
    for tok in input_ids[0].tolist():
        last_hidden = step(int(tok), pos)
        pos += 1

    generated: list[int] = []
    for _ in range(args.max_new_tokens):
        lm_out = lm_model.predict({"hidden_states": last_hidden.astype(np.float16)})
        next_id = _pick_next_token(lm_out, int(cfg.vocab_size))
        generated.append(next_id)
        last_hidden = step(next_id, pos)
        pos += 1

    text = tokenizer.decode(generated, skip_special_tokens=False)
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Generated token ids ({len(generated)}): {generated}")
    print(f"Generated text: {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
