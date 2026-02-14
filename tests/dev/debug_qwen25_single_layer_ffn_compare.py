#!/usr/bin/env python3
"""Convert one Qwen2.5 FFN layer to CoreML and compare against PyTorch/HF.

This isolates layer-level divergence by feeding the exact hidden-state inputs
for a chosen layer (from HF hidden_states) token-by-token while sharing KV
cache state across steps.
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


def _make_single_token_mask(context_length: int, pos: int) -> np.ndarray:
    mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
    if pos + 1 < context_length:
        mask[0, 0, 0, pos + 1 :] = -np.inf
    return mask


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.astype(np.float64).reshape(-1)
    bv = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def _diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return float(d.max()), float(d.mean()), _cosine(a, b)


def _reset_kv_cache(module: torch.nn.Module | torch.jit.ScriptModule) -> None:
    with torch.no_grad():
        for name, buf in module.named_buffers():
            if "kv_cache_" in name:
                buf.zero_()


class SingleLayerInferWrapper(torch.nn.Module):
    def __init__(self, model: Qwen25ForCausalLM, layer_idx: int) -> None:
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.states = Qwen25Converter.GetTransformerStates(
            model, part="2", prefix="model.model."
        )

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        rotary = self.model.model.get_rotary_embeddings_s(current_pos)
        return self.model.model.process_layer(
            self.layer_idx,
            hidden_states,
            position_ids,
            causal_mask,
            current_pos,
            rotary,
            layer_offset=0,
            IN_PREFILL=False,
        )


def _resolve_model_path(model_path: str) -> Path:
    p = Path(model_path).expanduser().resolve()
    if (p / "config.json").exists():
        return p

    snapshots = list(p.glob("snapshots/*"))
    snapshots = [s for s in snapshots if (s / "config.json").exists()]
    if snapshots:
        return snapshots[0]
    raise FileNotFoundError(f"Could not find config.json under: {p}")


def _build_input_ids(tokenizer, prompt: str, use_chat_template: bool) -> torch.Tensor:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        msg = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            msg,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids


def _convert_single_layer(
    model: Qwen25ForCausalLM,
    layer_idx: int,
    context_length: int,
    out_path: Path,
) -> ct.models.MLModel:
    wrapper = SingleLayerInferWrapper(model, layer_idx)
    wrapper.eval()

    hidden_states = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float16)
    position_ids = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, 1, context_length), dtype=torch.float16)
    current_pos = torch.zeros((1,), dtype=torch.int32)

    _reset_kv_cache(wrapper)
    traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos))
    _reset_kv_cache(wrapper)
    _reset_kv_cache(traced)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    return mlmodel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert one Qwen2.5 layer and compare token-wise outputs (HF/PT/CoreML)."
    )
    parser.add_argument(
        "--model-path",
        default="~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B",
        help="HF model snapshot dir or hub cache root.",
    )
    parser.add_argument(
        "--hf-model",
        default="WeiboAI/VibeThinker-1.5B",
        help="HF model id for reference hidden states.",
    )
    parser.add_argument("--layer-idx", type=int, default=0, help="Layer index to isolate.")
    parser.add_argument("--prompt", default="2+2=", help="Prompt text.")
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Apply tokenizer chat template + generation prompt.",
    )
    parser.add_argument(
        "--context-length", type=int, default=2048, help="State/context length for conversion."
    )
    parser.add_argument(
        "--target-pos",
        type=int,
        default=-1,
        help="Token position to report as final. -1 means last prompt token.",
    )
    parser.add_argument(
        "--out-dir",
        default="/Volumes/Models/ANE/debug_single_layer_qwen25",
        help="Where to save converted single-layer mlpackage.",
    )
    parser.add_argument(
        "--reuse-converted",
        action="store_true",
        help="Skip conversion if output mlpackage already exists.",
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model_path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / f"qwen25_layer_{args.layer_idx:02d}.mlpackage"

    print(f"Model path: {model_path}")
    print(f"HF reference: {args.hf_model}")
    print(f"Layer index: {args.layer_idx}")
    print(f"Context length: {args.context_length}")
    print(f"Output model: {out_model}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    input_ids = _build_input_ids(tokenizer, args.prompt, args.use_chat_template)
    seq_len = int(input_ids.shape[1])
    target_pos = seq_len - 1 if args.target_pos < 0 else args.target_pos
    if target_pos < 0 or target_pos >= seq_len:
        raise ValueError(f"target_pos {target_pos} out of range for seq_len={seq_len}")
    if seq_len > args.context_length:
        raise ValueError(
            f"Prompt produced {seq_len} tokens, exceeds context_length={args.context_length}"
        )

    print(f"Prompt token count: {seq_len}")
    print(f"Target token position: {target_pos}")
    print(f"First 16 tokens: {input_ids[0, : min(16, seq_len)].tolist()}")

    print("\nLoading HF reference model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    hf_model.eval()

    with torch.no_grad():
        hf_out = hf_model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    hf_hs = hf_out.hidden_states
    if args.layer_idx < 0 or args.layer_idx + 1 >= len(hf_hs):
        raise ValueError(
            f"layer_idx {args.layer_idx} invalid for hidden_states length {len(hf_hs)}"
        )
    hf_pre_layer = hf_hs[args.layer_idx].detach().cpu().to(torch.float16)
    hf_post_layer = hf_hs[args.layer_idx + 1].detach().cpu().to(torch.float16)
    print(
        f"HF layer tensors: pre={tuple(hf_pre_layer.shape)}, post={tuple(hf_post_layer.shape)}"
    )

    print("\nLoading ANEMLL Qwen2.5 model...")
    cfg = Qwen25Config.from_json(str(model_path / "config.json"))
    cfg.context_length = args.context_length
    cfg.state_length = args.context_length
    custom_model = Qwen25ForCausalLM(cfg, enable_coreml=True)
    custom_model.load_pretrained_weights(str(model_path))
    custom_model.eval()
    for p in custom_model.parameters():
        p.requires_grad = False

    if not args.reuse_converted or not out_model.exists():
        print("\nConverting single layer to CoreML...")
        _convert_single_layer(custom_model, args.layer_idx, args.context_length, out_model)
    else:
        print("\nReusing existing converted single-layer model.")

    print("Loading CoreML single-layer model...")
    cm_model = ct.models.MLModel(str(out_model), compute_units=ct.ComputeUnit.CPU_AND_NE)
    cm_state = cm_model.make_state()

    layer_wrapper = SingleLayerInferWrapper(custom_model, args.layer_idx)
    layer_wrapper.eval()
    _reset_kv_cache(layer_wrapper)

    print("\nRunning sequential token-by-token layer parity...")
    worst_cm_hf_mean = -1.0
    worst_pos = -1

    for pos in range(target_pos + 1):
        hs_in = hf_pre_layer[:, pos : pos + 1, :].numpy().astype(np.float16)
        pos_ids_np = np.array([pos], dtype=np.int32)
        cur_pos_np = np.array([pos], dtype=np.int32)
        causal_mask_np = _make_single_token_mask(args.context_length, pos)

        with torch.no_grad():
            pt_out = layer_wrapper(
                torch.from_numpy(hs_in),
                torch.from_numpy(pos_ids_np),
                torch.from_numpy(causal_mask_np),
                torch.from_numpy(cur_pos_np),
            ).detach().cpu().numpy().astype(np.float16)

        cm_out = cm_model.predict(
            {
                "hidden_states": hs_in,
                "position_ids": pos_ids_np,
                "causal_mask": causal_mask_np,
                "current_pos": cur_pos_np,
            },
            state=cm_state,
        )["output_hidden_states"].astype(np.float16)

        hf_expected = hf_post_layer[:, pos : pos + 1, :].numpy().astype(np.float16)

        pt_hf_max, pt_hf_mean, pt_hf_cos = _diff_stats(pt_out, hf_expected)
        cm_hf_max, cm_hf_mean, cm_hf_cos = _diff_stats(cm_out, hf_expected)
        cm_pt_max, cm_pt_mean, cm_pt_cos = _diff_stats(cm_out, pt_out)

        if cm_hf_mean > worst_cm_hf_mean:
            worst_cm_hf_mean = cm_hf_mean
            worst_pos = pos

        if pos < 3 or pos == target_pos:
            print(
                f"pos={pos:4d} | "
                f"PTvsHF mean={pt_hf_mean:.6f} cos={pt_hf_cos:.6f} | "
                f"CMvsHF mean={cm_hf_mean:.6f} cos={cm_hf_cos:.6f} | "
                f"CMvsPT mean={cm_pt_mean:.6f} cos={cm_pt_cos:.6f}"
            )
            if pos == target_pos:
                print(
                    f"  max diff @target: PTvsHF={pt_hf_max:.6f} "
                    f"CMvsHF={cm_hf_max:.6f} CMvsPT={cm_pt_max:.6f}"
                )

    print("\nSummary")
    print(f"  Worst CMvsHF mean diff: {worst_cm_hf_mean:.6f} at pos {worst_pos}")
    print(f"  Converted single-layer model: {out_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
