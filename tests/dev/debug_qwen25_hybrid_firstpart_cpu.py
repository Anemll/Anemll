#!/usr/bin/env python3
"""Prototype split execution: first part FP32 + rest FP16 CoreML for Qwen2.5-style models.

Pipeline Baseline:
  embeddings (PyTorch FP16) -> layer0 (CoreML FP16) ->
  layers1-13 (CoreML FP16) -> layers14-end+norm (CoreML FP16)

Pipeline Hybrid:
  embeddings (PyTorch FP16) -> layer0 (CoreML FP32/CPU) ->
  layers1-13 (CoreML FP16) -> layers14-end+norm (CoreML FP16)

This tests whether moving only the first execution stage to higher precision
improves parity while keeping the rest on the standard FP16 path.
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


def _convert_wrapper(
    wrapper: RangeWrapper,
    hidden_size: int,
    context_length: int,
    out_path: Path,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
):
    hs = torch.zeros((1, 1, hidden_size), dtype=torch.float16)
    pid = torch.zeros((1,), dtype=torch.int32)
    cmask = torch.zeros((1, 1, 1, context_length), dtype=torch.float16)
    cp = torch.zeros((1,), dtype=torch.int32)

    _reset_kv_buffers(wrapper)
    traced = torch.jit.trace(wrapper, (hs, pid, cmask, cp))
    _reset_kv_buffers(wrapper)
    _reset_kv_buffers(traced)

    # Build fresh state descriptors per conversion call.
    # coremltools may normalize/mutate names during conversion.
    states = Qwen25Converter.GetTransformerStates(wrapper.model, part="2", prefix="model.model.")

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


def main() -> int:
    ap = argparse.ArgumentParser(description="Split first-part CPU + rest CoreML diagnostics.")
    ap.add_argument(
        "--model-path",
        default="~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B",
        help="Local HF snapshot dir or cache root",
    )
    ap.add_argument("--hf-model", default="WeiboAI/VibeThinker-1.5B")
    ap.add_argument("--prompt", default="2+2=")
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument(
        "--out-dir",
        default="/Volumes/Models/ANE/debug_qwen25_hybrid_firstpart_cpu",
    )
    ap.add_argument("--reuse-models", action="store_true")
    args = ap.parse_args()

    model_path = _resolve_model_path(args.model_path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    layer0_fp16_path = out_dir / "layer0_fp16.mlpackage"
    layer0_fp32_path = out_dir / "layer0_fp32.mlpackage"
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
        args.hf_model, torch_dtype=torch.float16, local_files_only=True
    )
    hf.eval()
    with torch.no_grad():
        hf_out = hf(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    hf_final = hf_out.hidden_states[-1].detach().cpu().to(torch.float16).numpy().astype(np.float16)
    hf_next_token = int(hf_out.logits[0, -1].argmax().item())

    cfg = Qwen25Config.from_json(str(model_path / "config.json"))
    cfg.context_length = args.context_length
    cfg.state_length = args.context_length
    model = Qwen25ForCausalLM(cfg, enable_coreml=True)
    model.load_pretrained_weights(str(model_path))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # 28-layer models: split as [0], [1..13], [14..end]
    layer0_wrapper_fp16 = RangeWrapper(model, start_layer=0, end_layer=1).eval()
    layer0_wrapper_fp32 = RangeWrapper(model, start_layer=0, end_layer=1).eval()
    mid_wrapper = RangeWrapper(model, start_layer=1, end_layer=14).eval()
    tail_wrapper = RangeWrapper(model, start_layer=14, end_layer=None).eval()

    if not args.reuse_models or not layer0_fp16_path.exists():
        print("Converting layer0 FP16...")
        _convert_wrapper(
            layer0_wrapper_fp16,
            cfg.hidden_size,
            args.context_length,
            layer0_fp16_path,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
    if not args.reuse_models or not layer0_fp32_path.exists():
        print("Converting layer0 FP32...")
        _convert_wrapper(
            layer0_wrapper_fp32,
            cfg.hidden_size,
            args.context_length,
            layer0_fp32_path,
            compute_precision=ct.precision.FLOAT32,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
    if not args.reuse_models or not mid_fp16_path.exists():
        print("Converting layers1-13 FP16...")
        _convert_wrapper(mid_wrapper, cfg.hidden_size, args.context_length, mid_fp16_path)
    if not args.reuse_models or not tail_fp16_path.exists():
        print("Converting layers14-end FP16...")
        _convert_wrapper(tail_wrapper, cfg.hidden_size, args.context_length, tail_fp16_path)

    layer0_fp16 = ct.models.MLModel(str(layer0_fp16_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    layer0_fp32 = ct.models.MLModel(str(layer0_fp32_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    mid_fp16 = ct.models.MLModel(str(mid_fp16_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    tail_fp16 = ct.models.MLModel(str(tail_fp16_path), compute_units=ct.ComputeUnit.CPU_AND_NE)

    _reset_kv_buffers(model)
    baseline_means = []
    hybrid_means = []
    baseline_cos = []
    hybrid_cos = []
    baseline_last = None
    hybrid_last = None

    def run_chain(layer0_model: ct.models.MLModel):
        s0 = layer0_model.make_state()
        s1 = mid_fp16.make_state()
        s2 = tail_fp16.make_state()
        means = []
        coss = []
        last_out = None
        for pos in range(seq_len):
            token_id = int(input_ids[0, pos].item())
            with torch.no_grad():
                emb = model.model.embed_tokens(torch.tensor([[token_id]], dtype=torch.long)).cpu().numpy().astype(np.float16)
            p = np.array([pos], dtype=np.int32)
            m = _make_mask(args.context_length, pos)
            h0 = layer0_model.predict(
                {"hidden_states": emb, "position_ids": p, "causal_mask": m, "current_pos": p},
                state=s0,
            )["output_hidden_states"].astype(np.float16)
            h1 = mid_fp16.predict(
                {"hidden_states": h0, "position_ids": p, "causal_mask": m, "current_pos": p},
                state=s1,
            )["output_hidden_states"].astype(np.float16)
            h2 = tail_fp16.predict(
                {"hidden_states": h1, "position_ids": p, "causal_mask": m, "current_pos": p},
                state=s2,
            )["output_hidden_states"].astype(np.float16)
            ref = hf_final[:, pos : pos + 1, :]
            means.append(float(np.abs(h2.astype(np.float32) - ref.astype(np.float32)).mean()))
            coss.append(_cos(h2, ref))
            last_out = h2
        return means, coss, last_out

    print("Running token-by-token split evaluation...")
    baseline_means, baseline_cos, baseline_last = run_chain(layer0_fp16)
    hybrid_means, hybrid_cos, hybrid_last = run_chain(layer0_fp32)

    # Use HF lm_head for diagnostic next-token compare from last prompt position
    with torch.no_grad():
        b_logits = hf.lm_head(torch.from_numpy(baseline_last).to(torch.float16))[0, 0]
        h_logits = hf.lm_head(torch.from_numpy(hybrid_last).to(torch.float16))[0, 0]
    b_next = int(b_logits.argmax().item())
    h_next = int(h_logits.argmax().item())

    print("\nResults")
    print(
        f"Baseline CoreML FP16: last_mean={baseline_means[-1]:.6f}, "
        f"worst_mean={max(baseline_means):.6f}, last_cos={baseline_cos[-1]:.6f}"
    )
    print(
        f"Hybrid CPU(layer0)+CoreML(rest): last_mean={hybrid_means[-1]:.6f}, "
        f"worst_mean={max(hybrid_means):.6f}, last_cos={hybrid_cos[-1]:.6f}"
    )
    print(f"HF next token: {hf_next_token}")
    print(f"Baseline next token (HF lm_head on baseline hidden): {b_next}")
    print(f"Hybrid next token (HF lm_head on hybrid hidden): {h_next}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
