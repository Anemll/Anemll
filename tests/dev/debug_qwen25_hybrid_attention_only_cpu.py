#!/usr/bin/env python3
"""Hybrid experiment for Qwen2.5-style models:

Baseline:
  embeddings(FP16) -> layer0_attn(FP16) -> layer0_ffn(FP16) -> rest(FP16)

Hybrid:
  embeddings(FP16) -> layer0_attn(FP32/CPU) -> layer0_ffn(FP16) -> rest(FP16)

This isolates whether first-layer attention precision is sufficient to recover
parity without moving the whole first layer to FP32.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def _make_mask(context_length: int, pos: int) -> np.ndarray:
    m = np.zeros((1, 1, 1, context_length), dtype=np.float16)
    if pos + 1 < context_length:
        m[0, 0, 0, pos + 1 :] = np.float16(-65504.0)
    return m


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12))


def _reset_kv_buffers(module: torch.nn.Module | torch.jit.ScriptModule) -> None:
    with torch.no_grad():
        for n, b in module.named_buffers():
            if "kv_cache_" in n:
                b.zero_()


class Layer0AttentionWrapper(torch.nn.Module):
    def __init__(self, model: Qwen25ForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        # Exactly layer0 attention+residual path, no MLP.
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


class Layer0FFNWrapper(torch.nn.Module):
    def __init__(self, model: Qwen25ForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, hidden_states):
        # Exactly layer0 post-attn norm + MLP + residual path.
        layer = self.model.model.layers[0]
        post = layer.post_attention_layernorm(hidden_states)
        return hidden_states + layer.mlp(post)


class RangeWrapper(torch.nn.Module):
    def __init__(self, model: Qwen25ForCausalLM, start_layer: int, end_layer: int | None):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        rotary = self.model.model.get_rotary_embeddings_s(current_pos)
        out = self.model.model.process_layers(
            hidden_states,
            position_ids,
            causal_mask,
            current_pos,
            rotary,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            IN_PREFILL=False,
        )
        if self.end_layer is None or self.end_layer == len(self.model.model.layers):
            out = self.model.model.norm(out)
        return out


def _convert_stateful_wrapper(
    wrapper: torch.nn.Module,
    model: Qwen25ForCausalLM,
    hidden_size: int,
    context_length: int,
    out_path: Path,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
) -> None:
    hs = torch.zeros((1, 1, hidden_size), dtype=torch.float16)
    pid = torch.zeros((1,), dtype=torch.int32)
    cmask = torch.zeros((1, 1, 1, context_length), dtype=torch.float16)
    cp = torch.zeros((1,), dtype=torch.int32)

    _reset_kv_buffers(wrapper)
    traced = torch.jit.trace(wrapper, (hs, pid, cmask, cp))
    _reset_kv_buffers(wrapper)
    _reset_kv_buffers(traced)

    states = Qwen25Converter.GetTransformerStates(model, part="2", prefix="model.model.")
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hs.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=pid.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=cmask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=cp.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=states,
        compute_precision=compute_precision,
        compute_units=compute_units,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    if out_path.exists():
        shutil.rmtree(out_path)
    ml.save(str(out_path))


def _convert_ffn_only_wrapper(
    wrapper: torch.nn.Module,
    hidden_size: int,
    out_path: Path,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
) -> None:
    hs = torch.zeros((1, 1, hidden_size), dtype=torch.float16)
    traced = torch.jit.trace(wrapper, (hs,))
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=hs.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=compute_units,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    if out_path.exists():
        shutil.rmtree(out_path)
    ml.save(str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid attention-only FP32 first-stage diagnostic.")
    ap.add_argument(
        "--model-path",
        default="~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B",
        help="Local HF snapshot dir or cache root",
    )
    ap.add_argument("--hf-model", default="WeiboAI/VibeThinker-1.5B")
    ap.add_argument("--prompt", default="2+2=")
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument(
        "--out-dir",
        default="/Volumes/Models/ANE/debug_qwen25_hybrid_attn_only_cpu",
    )
    ap.add_argument("--reuse-models", action="store_true")
    args = ap.parse_args()

    model_path = _resolve_model_path(args.model_path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    layer0_attn_fp16_path = out_dir / "layer0_attn_fp16.mlpackage"
    layer0_attn_fp32_path = out_dir / "layer0_attn_fp32.mlpackage"
    layer0_ffn_fp16_path = out_dir / "layer0_ffn_fp16.mlpackage"
    mid_fp16_path = out_dir / "layers01_13_fp16.mlpackage"
    tail_fp16_path = out_dir / "layers14_end_fp16.mlpackage"

    print(f"Model path: {model_path}")
    print(f"HF model: {args.hf_model}")

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
    seq_len = int(input_ids.shape[1])
    print(f"Prompt token count: {seq_len}")

    hf = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    hf.eval()
    with torch.no_grad():
        hf_out = hf(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    hf_final = hf_out.hidden_states[-1].detach().cpu().to(torch.float16).numpy().astype(np.float16)
    hf_next = int(hf_out.logits[0, -1].argmax().item())

    cfg = Qwen25Config.from_json(str(model_path / "config.json"))
    cfg.context_length = args.context_length
    cfg.state_length = args.context_length
    model = Qwen25ForCausalLM(cfg, enable_coreml=True)
    model.load_pretrained_weights(str(model_path))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Use separate wrapper instances to avoid state-name mutation cross-talk.
    attn_fp16_wrapper = Layer0AttentionWrapper(model).eval()
    attn_fp32_wrapper = Layer0AttentionWrapper(model).eval()
    ffn_wrapper = Layer0FFNWrapper(model).eval()
    mid_wrapper = RangeWrapper(model, start_layer=1, end_layer=14).eval()
    tail_wrapper = RangeWrapper(model, start_layer=14, end_layer=None).eval()

    if not args.reuse_models or not layer0_attn_fp16_path.exists():
        print("Converting layer0 attention FP16...")
        _convert_stateful_wrapper(
            attn_fp16_wrapper,
            model,
            cfg.hidden_size,
            args.context_length,
            layer0_attn_fp16_path,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
    if not args.reuse_models or not layer0_attn_fp32_path.exists():
        print("Converting layer0 attention FP32...")
        _convert_stateful_wrapper(
            attn_fp32_wrapper,
            model,
            cfg.hidden_size,
            args.context_length,
            layer0_attn_fp32_path,
            compute_precision=ct.precision.FLOAT32,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
    if not args.reuse_models or not layer0_ffn_fp16_path.exists():
        print("Converting layer0 FFN FP16...")
        _convert_ffn_only_wrapper(ffn_wrapper, cfg.hidden_size, layer0_ffn_fp16_path)
    if not args.reuse_models or not mid_fp16_path.exists():
        print("Converting layers1-13 FP16...")
        _convert_stateful_wrapper(
            mid_wrapper,
            model,
            cfg.hidden_size,
            args.context_length,
            mid_fp16_path,
            compute_precision=ct.precision.FLOAT16,
        )
    if not args.reuse_models or not tail_fp16_path.exists():
        print("Converting layers14-end FP16...")
        _convert_stateful_wrapper(
            tail_wrapper,
            model,
            cfg.hidden_size,
            args.context_length,
            tail_fp16_path,
            compute_precision=ct.precision.FLOAT16,
        )

    attn_fp16 = ct.models.MLModel(str(layer0_attn_fp16_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    attn_fp32 = ct.models.MLModel(str(layer0_attn_fp32_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    ffn_fp16 = ct.models.MLModel(str(layer0_ffn_fp16_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    mid_fp16 = ct.models.MLModel(str(mid_fp16_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    tail_fp16 = ct.models.MLModel(str(tail_fp16_path), compute_units=ct.ComputeUnit.CPU_AND_NE)

    def run_chain(attn_model: ct.models.MLModel):
        s_attn = attn_model.make_state()
        s_mid = mid_fp16.make_state()
        s_tail = tail_fp16.make_state()
        means: list[float] = []
        coss: list[float] = []
        last_out = None
        for pos in range(seq_len):
            tok_id = int(input_ids[0, pos].item())
            with torch.no_grad():
                emb = model.model.embed_tokens(torch.tensor([[tok_id]], dtype=torch.long)).cpu().numpy().astype(np.float16)
            p = np.array([pos], dtype=np.int32)
            m = _make_mask(args.context_length, pos)
            h_attn = attn_model.predict(
                {"hidden_states": emb, "position_ids": p, "causal_mask": m, "current_pos": p},
                state=s_attn,
            )["output_hidden_states"].astype(np.float16)
            h0 = ffn_fp16.predict({"hidden_states": h_attn})["output_hidden_states"].astype(np.float16)
            h1 = mid_fp16.predict(
                {"hidden_states": h0, "position_ids": p, "causal_mask": m, "current_pos": p},
                state=s_mid,
            )["output_hidden_states"].astype(np.float16)
            h2 = tail_fp16.predict(
                {"hidden_states": h1, "position_ids": p, "causal_mask": m, "current_pos": p},
                state=s_tail,
            )["output_hidden_states"].astype(np.float16)

            ref = hf_final[:, pos : pos + 1, :]
            means.append(float(np.abs(h2.astype(np.float32) - ref.astype(np.float32)).mean()))
            coss.append(_cos(h2, ref))
            last_out = h2
        return means, coss, last_out

    def run_generate(attn_model: ct.models.MLModel, max_new_tokens: int):
        s_attn = attn_model.make_state()
        s_mid = mid_fp16.make_state()
        s_tail = tail_fp16.make_state()

        def step(token_id: int, pos: int) -> np.ndarray:
            with torch.no_grad():
                emb = model.model.embed_tokens(
                    torch.tensor([[token_id]], dtype=torch.long)
                ).cpu().numpy().astype(np.float16)
            p = np.array([pos], dtype=np.int32)
            m = _make_mask(args.context_length, pos)
            h_attn = attn_model.predict(
                {"hidden_states": emb, "position_ids": p, "causal_mask": m, "current_pos": p},
                state=s_attn,
            )["output_hidden_states"].astype(np.float16)
            h0 = ffn_fp16.predict({"hidden_states": h_attn})["output_hidden_states"].astype(
                np.float16
            )
            h1 = mid_fp16.predict(
                {"hidden_states": h0, "position_ids": p, "causal_mask": m, "current_pos": p},
                state=s_mid,
            )["output_hidden_states"].astype(np.float16)
            h2 = tail_fp16.predict(
                {"hidden_states": h1, "position_ids": p, "causal_mask": m, "current_pos": p},
                state=s_tail,
            )["output_hidden_states"].astype(np.float16)
            return h2

        last_hidden = None
        pos = 0
        for tok in input_ids[0].tolist():
            last_hidden = step(int(tok), pos)
            pos += 1

        generated_ids: list[int] = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = hf.lm_head(torch.from_numpy(last_hidden).to(torch.float16))[0, 0]
            next_id = int(logits.argmax().item())
            generated_ids.append(next_id)
            last_hidden = step(next_id, pos)
            pos += 1

        return generated_ids

    print("Running token-by-token attention-only split evaluation...")
    b_means, b_coss, b_last = run_chain(attn_fp16)
    h_means, h_coss, h_last = run_chain(attn_fp32)

    with torch.no_grad():
        b_logits = hf.lm_head(torch.from_numpy(b_last).to(torch.float16))[0, 0]
        h_logits = hf.lm_head(torch.from_numpy(h_last).to(torch.float16))[0, 0]
    b_next = int(b_logits.argmax().item())
    h_next = int(h_logits.argmax().item())

    print("\nResults")
    print(
        f"Baseline (attn FP16): last_mean={b_means[-1]:.6f}, "
        f"worst_mean={max(b_means):.6f}, last_cos={b_coss[-1]:.6f}"
    )
    print(
        f"Hybrid (attn FP32): last_mean={h_means[-1]:.6f}, "
        f"worst_mean={max(h_means):.6f}, last_cos={h_coss[-1]:.6f}"
    )
    print(f"HF next token: {hf_next}")
    print(f"Baseline next token: {b_next}")
    print(f"Hybrid next token: {h_next}")

    print("\nRunning end-to-end greedy generation (HF lm_head on chain hidden)...")
    base_ids = run_generate(attn_fp16, args.max_new_tokens)
    hyb_ids = run_generate(attn_fp32, args.max_new_tokens)
    with torch.no_grad():
        hf_gen = hf.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    hf_ids = hf_gen[0, input_ids.shape[1] :].tolist()

    print(f"Generated tokens: {args.max_new_tokens}")
    print(f"HF ids      : {hf_ids}")
    print(f"Baseline ids: {base_ids}")
    print(f"Hybrid ids  : {hyb_ids}")
    print(f"HF text      : {tokenizer.decode(hf_ids, skip_special_tokens=False)!r}")
    print(f"Baseline text: {tokenizer.decode(base_ids, skip_special_tokens=False)!r}")
    print(f"Hybrid text  : {tokenizer.decode(hyb_ids, skip_special_tokens=False)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
