#!/usr/bin/env python3
"""Probe FP16 overflow in Qwen2.5-style attention at layer/token granularity.

This script compares fp16 vs fp32 attention-logit behavior for selected layers,
using HF hidden_states as layer inputs and ANEMLL projection/cache code.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anemll.models.qwen2_5_model import Qwen25Config, Qwen25ForCausalLM


def _resolve_model_path(model_path: str) -> Path:
    p = Path(model_path).expanduser().resolve()
    if (p / "config.json").exists():
        return p
    snapshots = [s for s in p.glob("snapshots/*") if (s / "config.json").exists()]
    if snapshots:
        return snapshots[0]
    raise FileNotFoundError(f"Could not find config.json in {p}")


def _build_input_ids(tokenizer, prompt: str, use_chat_template: bool) -> torch.Tensor:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids


def _parse_layers(arg: str, total_layers: int) -> list[int]:
    if arg.strip().lower() == "auto":
        cands = [0, 1, 2, total_layers // 4, total_layers // 2, total_layers - 1]
        out = sorted(set([x for x in cands if 0 <= x < total_layers]))
        return out
    layers: list[int] = []
    for part in arg.split(","):
        v = int(part.strip())
        if v < 0 or v >= total_layers:
            raise ValueError(f"Layer {v} out of range [0, {total_layers - 1}]")
        layers.append(v)
    return sorted(set(layers))


def _make_single_token_mask(context_length: int, pos: int) -> torch.Tensor:
    m = torch.zeros((1, 1, 1, context_length), dtype=torch.float16)
    if pos + 1 < context_length:
        m[:, :, :, pos + 1 :] = float("-inf")
    return m


def _fmt_f(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x:.4f}"


def probe_model(
    model_path: Path,
    hf_model_id: str,
    prompt: str,
    use_chat_template: bool,
    context_length: int,
    layers_arg: str,
    sweep_layer0_bias: bool,
    sweep_scales_arg: str,
) -> int:
    print("=" * 90)
    print(f"Model path: {model_path}")
    print(f"HF reference: {hf_model_id}")
    print(f"Prompt: {prompt!r}")
    print(f"Use chat template: {use_chat_template}")
    print(f"Context length: {context_length}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    input_ids = _build_input_ids(tokenizer, prompt, use_chat_template)
    seq_len = int(input_ids.shape[1])
    print(f"Token count: {seq_len}")
    print(f"First tokens: {input_ids[0, : min(16, seq_len)].tolist()}")

    print("\nLoading HF model for hidden states...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    hf_model.eval()
    with torch.no_grad():
        hf_out = hf_model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    hidden_states = [h.detach().cpu().to(torch.float16) for h in hf_out.hidden_states]

    print("Loading ANEMLL Qwen2.5 model...")
    cfg = Qwen25Config.from_json(str(model_path / "config.json"))
    cfg.context_length = context_length
    cfg.state_length = context_length
    model = Qwen25ForCausalLM(cfg, enable_coreml=True)
    model.load_pretrained_weights(str(model_path))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    total_layers = int(model.config.num_hidden_layers)
    layers = _parse_layers(layers_arg, total_layers)
    print(f"Probing layers: {layers}")

    if seq_len > context_length:
        raise ValueError(f"seq_len={seq_len} exceeds context_length={context_length}")

    kv_cache = model.model.kv_cache_0
    layers_per_group = total_layers
    n_rep = model.config.num_attention_heads // model.config.num_key_value_heads

    print("\nPer-layer overflow summary:")
    print(
        f"{'layer':>5} {'ovf_pos':>7} {'nan_pre':>7} {'max_+inf':>9} "
        f"{'q|max|':>8} {'k|max|':>8} {'logit16|max|':>12} {'logit32|max|':>12} "
        f"{'pmax16':>8} {'pmax32':>8} {'logit32_p99.9':>14}"
    )

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        key_idx = layer_idx
        value_idx = layer_idx + layers_per_group

        with torch.no_grad():
            kv_cache[key_idx : key_idx + 1].zero_()
            kv_cache[value_idx : value_idx + 1].zero_()

        pos_with_overflow = 0
        pos_with_nan_pre = 0
        max_pos_inf_count = 0
        q_absmax = 0.0
        k_absmax = 0.0
        max_logit16 = -np.inf
        max_logit32 = -np.inf
        pmax16 = 0.0
        pmax32 = 0.0
        finite32_vals: list[np.ndarray] = []

        for pos in range(seq_len):
            hs_in = hidden_states[layer_idx][:, pos : pos + 1, :]
            pos_t = torch.tensor([pos], dtype=torch.int32)
            mask = _make_single_token_mask(context_length, pos)

            with torch.no_grad():
                normed = layer.input_layernorm(hs_in)
                rotary = model.model.get_rotary_embeddings_s(pos_t)
                q, k, v = layer.self_attn.get_new_kv_cache(normed, pos_t, rotary)

                kv_cache[key_idx : key_idx + 1, :, pos : pos + 1, :] = k
                kv_cache[value_idx : value_idx + 1, :, pos : pos + 1, :] = v

                key_cache = kv_cache[key_idx : key_idx + 1].squeeze(0)
                key_states = layer.self_attn.repeat_kv(key_cache, n_rep)

                q_absmax = max(q_absmax, float(q.abs().max()))
                k_absmax = max(k_absmax, float(key_states.abs().max()))

                # FP16 path (problematic): check raw QK overflow before mask.
                logits16_pre = (
                    torch.matmul(
                        q.to(torch.float16),
                        key_states.transpose(-1, -2).to(torch.float16),
                    )
                    * layer.self_attn.scale
                )
                pos_inf_count = int(torch.isposinf(logits16_pre).sum().item())
                nan_pre_count = int(torch.isnan(logits16_pre).sum().item())
                max_pos_inf_count = max(max_pos_inf_count, pos_inf_count)
                if pos_inf_count > 0:
                    pos_with_overflow += 1
                if nan_pre_count > 0:
                    pos_with_nan_pre += 1

                logits16 = logits16_pre + mask[:, :, :, : logits16_pre.shape[-1]]
                finite16 = logits16_pre[torch.isfinite(logits16_pre)]
                if finite16.numel() > 0:
                    max_logit16 = max(max_logit16, float(finite16.abs().max()))
                    sm16 = torch.softmax(logits16, dim=-1)
                    if torch.isfinite(sm16).all():
                        pmax16 = max(pmax16, float(sm16.max()))

                # FP32 reference
                logits32 = (
                    torch.matmul(
                        q.to(torch.float32),
                        key_states.transpose(-1, -2).to(torch.float32),
                    )
                    * float(layer.self_attn.scale)
                )
                logits32 = logits32 + mask[:, :, :, : logits32.shape[-1]].to(torch.float32)

                finite32 = logits32[torch.isfinite(logits32)]
                if finite32.numel() > 0:
                    finite32_vals.append(finite32.detach().cpu().numpy().astype(np.float32))
                    max_logit32 = max(max_logit32, float(finite32.abs().max()))
                    sm32 = torch.softmax(logits32, dim=-1)
                    if torch.isfinite(sm32).all():
                        pmax32 = max(pmax32, float(sm32.max()))

        if finite32_vals:
            all32 = np.concatenate([x.reshape(-1) for x in finite32_vals], axis=0)
            p999 = float(np.percentile(np.abs(all32), 99.9))
        else:
            p999 = np.nan

        print(
            f"{layer_idx:5d} {pos_with_overflow:7d} {pos_with_nan_pre:7d} {max_pos_inf_count:9d} "
            f"{q_absmax:8.1f} {k_absmax:8.1f} {_fmt_f(max_logit16):>12} {_fmt_f(max_logit32):>12} "
            f"{_fmt_f(pmax16):>8} {_fmt_f(pmax32):>8} {_fmt_f(p999):>14}"
        )

    if sweep_layer0_bias:
        if 0 not in layers:
            print("\n[Bias Sweep] Layer 0 not in --layers, skipping sweep.")
            return 0

        try:
            scales = [float(x.strip()) for x in sweep_scales_arg.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError(f"Invalid --sweep-scales value: {sweep_scales_arg}") from exc
        if not scales:
            raise ValueError("--sweep-scales produced no values")

        print("\nLayer-0 Bias Sweep (optional workflow)")
        print(
            f"{'scale':>8} {'ovf_pos':>8} {'max_+inf':>9} "
            f"{'pt_vs_hf_last':>14} {'pt_vs_hf_worst':>15}"
        )

        layer0 = model.model.layers[0]
        k_bias_orig = layer0.self_attn.k_proj.bias.detach().clone()
        q_bias_orig = layer0.self_attn.q_proj.bias.detach().clone()
        key_idx0 = 0
        value_idx0 = total_layers

        for scale in scales:
            with torch.no_grad():
                layer0.self_attn.k_proj.bias.copy_(k_bias_orig * scale)
                layer0.self_attn.q_proj.bias.copy_(q_bias_orig * scale)
                kv_cache[key_idx0 : key_idx0 + 1].zero_()
                kv_cache[value_idx0 : value_idx0 + 1].zero_()

            ovf_pos = 0
            max_pos_inf = 0
            pt_hf_means: list[float] = []

            for pos in range(seq_len):
                hs_in = hidden_states[0][:, pos : pos + 1, :]
                pos_t = torch.tensor([pos], dtype=torch.int32)
                mask = _make_single_token_mask(context_length, pos)

                with torch.no_grad():
                    normed = layer0.input_layernorm(hs_in)
                    rotary = model.model.get_rotary_embeddings_s(pos_t)
                    q, k, v = layer0.self_attn.get_new_kv_cache(normed, pos_t, rotary)

                    kv_cache[key_idx0 : key_idx0 + 1, :, pos : pos + 1, :] = k
                    kv_cache[value_idx0 : value_idx0 + 1, :, pos : pos + 1, :] = v
                    key_cache = kv_cache[key_idx0 : key_idx0 + 1].squeeze(0)
                    key_states = layer0.self_attn.repeat_kv(key_cache, n_rep)

                    logits16_pre = (
                        torch.matmul(
                            q.to(torch.float16),
                            key_states.transpose(-1, -2).to(torch.float16),
                        )
                        * layer0.self_attn.scale
                    )
                    pos_inf_count = int(torch.isposinf(logits16_pre).sum().item())
                    if pos_inf_count > 0:
                        ovf_pos += 1
                    max_pos_inf = max(max_pos_inf, pos_inf_count)

                    out_h = model.model.process_layer(
                        0,
                        hs_in,
                        pos_t,
                        mask,
                        pos_t,
                        model.model.get_rotary_embeddings_s(pos_t),
                        layer_offset=0,
                        IN_PREFILL=False,
                    )

                ref_h = hidden_states[1][:, pos : pos + 1, :]
                pt_hf_means.append(float((out_h - ref_h).abs().float().mean()))

            print(
                f"{scale:8.3f} {ovf_pos:8d} {max_pos_inf:9d} "
                f"{pt_hf_means[-1]:14.6f} {max(pt_hf_means):15.6f}"
            )

        with torch.no_grad():
            layer0.self_attn.k_proj.bias.copy_(k_bias_orig)
            layer0.self_attn.q_proj.bias.copy_(q_bias_orig)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deep FP16 overflow probe for Qwen2.5-style single-layer attention."
    )
    parser.add_argument("--model-path", required=True, help="HF local model path or cache root")
    parser.add_argument("--hf-model", required=True, help="HF model id for hidden state reference")
    parser.add_argument("--prompt", default="2+2=", help="Prompt")
    parser.add_argument("--use-chat-template", action="store_true", help="Use chat template")
    parser.add_argument("--context-length", type=int, default=2048, help="Context length/state length")
    parser.add_argument(
        "--layers",
        default="auto",
        help="Comma-separated layer ids, or 'auto' for key layers",
    )
    parser.add_argument(
        "--sweep-layer0-bias",
        action="store_true",
        help="Optional: sweep layer-0 q/k bias scaling and report overflow vs PT-HF drift.",
    )
    parser.add_argument(
        "--sweep-scales",
        default="1.0,0.75,0.5,0.25,0.0",
        help="Comma-separated scales for --sweep-layer0-bias.",
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model_path)
    return probe_model(
        model_path=model_path,
        hf_model_id=args.hf_model,
        prompt=args.prompt,
        use_chat_template=args.use_chat_template,
        context_length=args.context_length,
        layers_arg=args.layers,
        sweep_layer0_bias=args.sweep_layer0_bias,
        sweep_scales_arg=args.sweep_scales,
    )


if __name__ == "__main__":
    raise SystemExit(main())
