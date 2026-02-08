#!/usr/bin/env python3
"""Prototype: standard-compatible 3-chunk layout with FP32 attention-only first chunk.

Target runtime layout (chat.py compatible):
  embeddings -> FFN_PF_chunk_01of03 -> FFN_PF_chunk_02of03 -> FFN_PF_chunk_03of03 -> lm_head

Where:
  - chunk_01of03: layer0 attention residual only (FP32, CPU_ONLY)
  - chunk_02of03: layer0 post-attention MLP + layers1..split (FP16; optional LUT)
  - chunk_03of03: tail layers + norm (rebuilt; optional LUT)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
import yaml

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


def _chunk_bounds(total_layers: int, total_chunks: int, chunk_idx_zero_based: int) -> tuple[int, int | None]:
    if total_chunks <= 1:
        return 0, None
    base, rem = divmod(total_layers, total_chunks)
    start = chunk_idx_zero_based * base + min(chunk_idx_zero_based, rem)
    end = start + base + (1 if chunk_idx_zero_based < rem else 0)
    return start, end


def _compile_mlpackage(package_path: Path, output_dir: Path) -> None:
    target = output_dir / f"{package_path.stem}.mlmodelc"
    if target.exists():
        shutil.rmtree(target)
    cmd = [
        "xcrun",
        "coremlcompiler",
        "compile",
        str(package_path),
        str(output_dir),
        "--add-mlprogram-if-eligible",
        "force",
    ]
    subprocess.run(cmd, check=True)


def _combine_infer_prefill(infer_path: Path, prefill_path: Path, output_path: Path) -> None:
    temp_path = output_path.parent / f"temp_{output_path.name}"
    if temp_path.exists():
        shutil.rmtree(temp_path)

    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(str(infer_path), "main", "infer")
    desc.add_function(str(prefill_path), "main", "prefill")
    desc.default_function_name = "infer"
    ct.utils.save_multifunction(desc, str(temp_path))

    if output_path.exists():
        shutil.rmtree(output_path)
    temp_path.rename(output_path)


def _save_model(model: ct.models.MLModel, out_path: Path) -> None:
    if out_path.exists():
        shutil.rmtree(out_path)
    model.save(str(out_path))


def _meta_lut_to_tuple(bits_raw: Any, per_channel_raw: Any) -> tuple[int | None, int | None]:
    bits = bits_raw
    if isinstance(bits, str):
        bits_s = bits.strip().lower()
        if bits_s in ("none", "no", "false", ""):
            return None, None
        bits = int(bits_s)
    elif bits is None:
        return None, None
    else:
        bits = int(bits)

    per = per_channel_raw
    if per is None:
        per = 8
    elif isinstance(per, str):
        per_s = per.strip().lower()
        if per_s in ("none", "no", "false", "", "tensor", "t", "0"):
            per = 0
        else:
            per = int(per_s)
    else:
        per = int(per)

    return bits, per


def _parse_lut_arg(raw: str | None, default_bits: int | None, default_per: int | None) -> tuple[int | None, int | None]:
    if raw is None:
        return default_bits, default_per
    s = str(raw).strip().lower()
    if s in ("none", "no", "false", ""):
        return None, None
    if "," in s:
        lhs, rhs = s.split(",", 1)
        bits = int(lhs)
        rhs = rhs.strip().lower()
        if rhs in ("tensor", "t", "0"):
            per = 0
        else:
            per = int(rhs)
        return bits, per
    bits = int(s)
    per = 8 if default_per is None else int(default_per)
    return bits, per


def _lut_suffix(bits: int | None) -> str:
    return f"_lut{bits}" if bits is not None else ""


def _quantize_with_lut(
    mlmodel: ct.models.MLModel,
    model: Qwen25ForCausalLM,
    context_length: int,
    batch_size: int,
    lut_bits: int,
    per_channel: int,
) -> ct.models.MLModel:
    converter = Qwen25Converter(
        model=model,
        context_length=context_length,
        batch_size=batch_size,
        lut_bits=lut_bits,
        per_channel=per_channel,
        num_chunks=1,
    )
    converter.converted_model = mlmodel
    # Single-worker avoids known multiprocessing instability for chunked paths.
    converter.postprocess(num_workers=None)
    return converter.converted_model


class Chunk1AttnInferWrapper(torch.nn.Module):
    """layer0 attention-only residual path for single-token infer."""

    def __init__(self, model: Qwen25ForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        del position_ids
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


class Chunk1AttnPrefillWrapper(torch.nn.Module):
    """layer0 attention-only residual path for prefill batch."""

    def __init__(self, model: Qwen25ForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        layer = self.model.model.layers[0]
        normalized_states = layer.input_layernorm(hidden_states)
        rotary = self.model.model.get_rotary_embedding_prefill(position_ids)
        query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
            normalized_states,
            current_pos,
            rotary,
        )

        kv_cache = self.model.model.kv_cache_0
        layers_per_group = self.model.config.num_hidden_layers
        key_idx = 0
        value_idx = layers_per_group

        seq_length = key_states.shape[2]
        kv_cache[key_idx : key_idx + 1, :, current_pos : current_pos + seq_length, :] = key_states
        kv_cache[value_idx : value_idx + 1, :, current_pos : current_pos + seq_length, :] = value_states

        key_cache = kv_cache[key_idx : key_idx + 1].squeeze(0)
        value_cache = kv_cache[value_idx : value_idx + 1].squeeze(0)
        attn_out = layer.self_attn.forward_prefill(
            hidden_states=normalized_states,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
        )
        return hidden_states + attn_out


class Chunk2AfterAttnInferWrapper(torch.nn.Module):
    """layer0 MLP + middle layers for infer."""

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


class Chunk2AfterAttnPrefillWrapper(torch.nn.Module):
    """layer0 MLP + middle layers for prefill."""

    def __init__(self, model: Qwen25ForCausalLM, end_layer: int):
        super().__init__()
        self.model = model
        self.end_layer = end_layer

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        layer0 = self.model.model.layers[0]
        post = layer0.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer0.mlp(post)

        rotary = self.model.model.get_rotary_embedding_prefill(position_ids)
        return self.model.model.process_layers(
            hidden_states,
            position_ids,
            causal_mask,
            current_pos,
            rotary,
            start_layer=1,
            end_layer=self.end_layer,
            IN_PREFILL=True,
        )


class Chunk3TailInferWrapper(torch.nn.Module):
    """tail layers + final norm for infer."""

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


class Chunk3TailPrefillWrapper(torch.nn.Module):
    """tail layers for prefill; return first token only (matches converter behavior)."""

    def __init__(self, model: Qwen25ForCausalLM, start_layer: int):
        super().__init__()
        self.model = model
        self.start_layer = start_layer

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        rotary = self.model.model.get_rotary_embedding_prefill(position_ids)
        out = self.model.model.process_layers(
            hidden_states,
            position_ids,
            causal_mask,
            current_pos,
            rotary,
            start_layer=self.start_layer,
            end_layer=None,
            IN_PREFILL=True,
        )
        return out[:, 0:1, :]


def _convert_infer_wrapper(
    wrapper: torch.nn.Module,
    model: Qwen25ForCausalLM,
    context_length: int,
    hidden_size: int,
    precision: ct.precision,
    compute_units: ct.ComputeUnit,
) -> ct.models.MLModel:
    hidden_states = torch.zeros((1, 1, hidden_size), dtype=torch.float16)
    position_ids = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, 1, context_length), dtype=torch.float16)
    current_pos = torch.zeros((1,), dtype=torch.int32)

    _reset_kv_buffers(wrapper)
    traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos))
    _reset_kv_buffers(wrapper)
    _reset_kv_buffers(traced)

    return ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=Qwen25Converter.GetTransformerStates(model, part=None, prefix="model.model."),
        compute_precision=precision,
        compute_units=compute_units,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )


def _convert_prefill_wrapper(
    wrapper: torch.nn.Module,
    model: Qwen25ForCausalLM,
    context_length: int,
    hidden_size: int,
    batch_size: int,
    precision: ct.precision,
    compute_units: ct.ComputeUnit,
) -> ct.models.MLModel:
    hidden_states = torch.zeros((1, batch_size, hidden_size), dtype=torch.float16)
    position_ids = torch.zeros((batch_size,), dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, batch_size, context_length), dtype=torch.float16)
    current_pos = torch.zeros((1,), dtype=torch.int32)

    _reset_kv_buffers(wrapper)
    traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos))
    _reset_kv_buffers(wrapper)
    _reset_kv_buffers(traced)

    return ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=Qwen25Converter.GetTransformerStates(model, part="2_prefill", prefix="model.model."),
        compute_precision=precision,
        compute_units=compute_units,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )


def _build_lm_head(
    model: Qwen25ForCausalLM,
    context_length: int,
    batch_size: int,
    lut3_bits: int | None,
    lut3_per: int | None,
    argmax_in_model: bool,
) -> ct.models.MLModel:
    converter = Qwen25Converter(
        model=model,
        context_length=context_length,
        batch_size=batch_size,
        lut_bits=lut3_bits,
        per_channel=(8 if lut3_per is None else int(lut3_per)),
        num_chunks=1,
        argmax_in_model=argmax_in_model,
    )
    return converter.convert_part_3(model, argmax_in_model=argmax_in_model)


def _load_meta_params(source_dir: Path) -> dict[str, Any]:
    meta_path = source_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = yaml.safe_load(meta_path.read_text())
    params = meta.get("model_info", {}).get("parameters", {})
    if not params:
        raise ValueError("meta.yaml missing model_info.parameters")
    return params


def _clean_existing_ffn_pf_chunks(out_dir: Path, model_prefix: str) -> None:
    for p in sorted(out_dir.glob(f"{model_prefix}_FFN_PF*_chunk_*of*.mlpackage")):
        shutil.rmtree(p, ignore_errors=True)
    for p in sorted(out_dir.glob(f"{model_prefix}_FFN_PF*_chunk_*of*.mlmodelc")):
        shutil.rmtree(p, ignore_errors=True)


def _copy_tail_chunk_from_source(source_dir: Path, out_dir: Path, source_base: str, target_base: str) -> Path:
    source_tail_pkg = source_dir / f"{source_base}_chunk_02of02.mlpackage"
    if not source_tail_pkg.exists():
        raise FileNotFoundError(source_tail_pkg)
    out_tail_pkg = out_dir / f"{target_base}_chunk_03of03.mlpackage"
    shutil.copytree(source_tail_pkg, out_tail_pkg)
    return out_tail_pkg


def _lut_to_meta_value(bits: int | None) -> str | int:
    return "none" if bits is None else int(bits)


def _update_meta_for_3chunks(
    out_dir: Path,
    new_ffn_mlmodelc: str,
    new_lm_head_mlmodelc: str,
    lut1_bits: int | None,
    lut2_bits: int | None,
    lut2_per: int | None,
    lut3_bits: int | None,
    lut3_per: int | None,
    argmax_in_model: bool,
    recommended_sampling: dict[str, Any] | None,
) -> None:
    meta_path = out_dir / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    params = meta.setdefault("model_info", {}).setdefault("parameters", {})
    params["num_chunks"] = 3
    params["ffn"] = new_ffn_mlmodelc
    params["lm_head"] = new_lm_head_mlmodelc
    params["lut_embeddings"] = _lut_to_meta_value(lut1_bits)
    params["lut_ffn"] = _lut_to_meta_value(lut2_bits)
    params["lut_lmhead"] = _lut_to_meta_value(lut3_bits)
    params["lut_ffn_per_channel"] = 0 if lut2_per is None else int(lut2_per)
    params["lut_lmhead_per_channel"] = 0 if lut3_per is None else int(lut3_per)
    params["argmax_in_model"] = bool(argmax_in_model)
    if recommended_sampling is not None:
        params["recommended_sampling"] = recommended_sampling
    desc = meta["model_info"].get("description", "")
    if isinstance(desc, str):
        meta["model_info"]["description"] = desc.replace("Chunks: 2", "Chunks: 3")
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))


def _assert_required_artifacts(out_dir: Path, target_base: str, lm_head_stem: str) -> None:
    required = [
        out_dir / "meta.yaml",
        out_dir / f"{lm_head_stem}.mlmodelc",
        out_dir / f"{target_base}_chunk_01of03.mlpackage",
        out_dir / f"{target_base}_chunk_01of03.mlmodelc",
        out_dir / f"{target_base}_chunk_02of03.mlpackage",
        out_dir / f"{target_base}_chunk_02of03.mlmodelc",
        out_dir / f"{target_base}_chunk_03of03.mlpackage",
        out_dir / f"{target_base}_chunk_03of03.mlmodelc",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        lines = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Prototype build incomplete. Missing required artifacts:\n"
            f"  - {lines}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Prototype: chunk01 attention FP32, chunk02 after-attention, chunk03 tail.")
    ap.add_argument(
        "--model-path",
        default="~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B",
    )
    ap.add_argument(
        "--source-dir",
        default="/Volumes/Models/ANE/vibethinker_1.5b_ctx2048_lut6",
        help="Existing converted model directory to clone from.",
    )
    ap.add_argument(
        "--out-dir",
        default="/Volumes/Models/ANE/vibethinker_1.5b_ctx2048_lut6_attn3",
        help="Output prototype directory.",
    )
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--reuse-out-dir", action="store_true", help="Do not recopy source dir if out dir exists.")
    ap.add_argument("--no-quantize-chunk2", action="store_true", help="Skip LUT quantization for chunk2 (legacy flag).")
    ap.add_argument("--no-quantize-ffn", action="store_true", help="Skip LUT quantization for FFN chunks 2 and 3.")
    ap.add_argument("--copy-tail-from-source", action="store_true", help="Reuse source tail chunk (may stay quantized).")
    ap.add_argument("--lut1", type=str, default=None, help="LUT for embeddings, e.g. 'none' or '6,4'")
    ap.add_argument("--lut2", type=str, default=None, help="LUT for FFN chunks, e.g. 'none' or '6,4'")
    ap.add_argument("--lut3", type=str, default=None, help="LUT for lm_head, e.g. 'none' or '6,4'")
    ap.add_argument(
        "--argmax-in-model",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override argmax mode for rebuilt lm_head/meta. auto = keep source meta setting.",
    )
    ap.add_argument(
        "--recommended-do-sample",
        choices=["auto", "true", "false"],
        default="auto",
        help="Set recommended_sampling.do_sample in meta (auto keeps source value).",
    )
    ap.add_argument("--recommended-temperature", type=float, default=None, help="Set recommended_sampling.temperature in meta.")
    ap.add_argument("--recommended-top-p", type=float, default=None, help="Set recommended_sampling.top_p in meta.")
    ap.add_argument("--recommended-top-k", type=int, default=None, help="Set recommended_sampling.top_k in meta.")
    args = ap.parse_args()

    model_path = _resolve_model_path(args.model_path)
    source_dir = Path(args.source_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    if out_dir.exists() and not args.reuse_out_dir:
        shutil.rmtree(out_dir)
    if not out_dir.exists():
        print(f"Cloning source directory to: {out_dir}")
        shutil.copytree(source_dir, out_dir)

    meta_params = _load_meta_params(source_dir)
    model_prefix = str(meta_params.get("model_prefix", "qwen25"))
    src_lut1_bits, src_lut1_per = _meta_lut_to_tuple(
        meta_params.get("lut_embeddings"),
        meta_params.get("lut_embeddings_per_channel"),
    )
    src_lut2_bits, src_lut2_per = _meta_lut_to_tuple(
        meta_params.get("lut_ffn"),
        meta_params.get("lut_ffn_per_channel"),
    )
    src_lut3_bits, src_lut3_per = _meta_lut_to_tuple(
        meta_params.get("lut_lmhead"),
        meta_params.get("lut_lmhead_per_channel"),
    )
    lut1_bits, lut1_per = _parse_lut_arg(args.lut1, src_lut1_bits, src_lut1_per)
    lut2_bits, lut2_per = _parse_lut_arg(args.lut2, src_lut2_bits, src_lut2_per)
    lut3_bits, lut3_per = _parse_lut_arg(args.lut3, src_lut3_bits, src_lut3_per)

    if lut1_bits is not None:
        print("Warning: this prototype keeps chunk_01of03 FP32 unquantized; --lut1 is metadata-only here.")

    source_base = f"{model_prefix}_FFN_PF{_lut_suffix(src_lut2_bits)}"
    target_base = f"{model_prefix}_FFN_PF{_lut_suffix(lut2_bits)}"

    print(f"Source chunk base: {source_base}")
    print(f"Target chunk base: {target_base}")
    print(f"LUT config: lut1={args.lut1 or src_lut1_bits}, lut2={args.lut2 or src_lut2_bits}, lut3={args.lut3 or src_lut3_bits}")
    quantize_ffn = not (args.no_quantize_chunk2 or args.no_quantize_ffn)

    _clean_existing_ffn_pf_chunks(out_dir, model_prefix)

    cfg = Qwen25Config.from_json(str(model_path / "config.json"))
    cfg.context_length = args.context_length
    cfg.state_length = args.context_length
    model = Qwen25ForCausalLM(cfg, enable_coreml=True)
    if not model.load_pretrained_weights(str(model_path)):
        raise RuntimeError("Failed loading pretrained weights")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Use split matching original 2-ch tail (layers 14..end for 28-layer models).
    split_layer, _ = _chunk_bounds(cfg.num_hidden_layers, 2, 1)
    print(f"Middle/tail split layer: {split_layer}")

    chunk1_infer = Chunk1AttnInferWrapper(model).eval()
    chunk1_prefill = Chunk1AttnPrefillWrapper(model).eval()
    chunk2_infer = Chunk2AfterAttnInferWrapper(model, end_layer=split_layer).eval()
    chunk2_prefill = Chunk2AfterAttnPrefillWrapper(model, end_layer=split_layer).eval()
    chunk3_infer = Chunk3TailInferWrapper(model, start_layer=split_layer).eval()
    chunk3_prefill = Chunk3TailPrefillWrapper(model, start_layer=split_layer).eval()

    # Build lm_head according to --lut3 so metadata can point to a matching artifact.
    source_argmax_in_model = bool(meta_params.get("argmax_in_model", False))
    if args.argmax_in_model == "auto":
        argmax_in_model = source_argmax_in_model
    else:
        argmax_in_model = args.argmax_in_model == "true"

    source_sampling = meta_params.get("recommended_sampling")
    if not isinstance(source_sampling, dict):
        source_sampling = {}

    if args.recommended_do_sample == "auto":
        rec_do_sample = source_sampling.get("do_sample")
    else:
        rec_do_sample = args.recommended_do_sample == "true"
    rec_temperature = args.recommended_temperature
    if rec_temperature is None:
        rec_temperature = source_sampling.get("temperature")
    rec_top_p = args.recommended_top_p
    if rec_top_p is None:
        rec_top_p = source_sampling.get("top_p")
    rec_top_k = args.recommended_top_k
    if rec_top_k is None:
        rec_top_k = source_sampling.get("top_k")

    recommended_sampling = None
    if any(v is not None for v in (rec_do_sample, rec_temperature, rec_top_p, rec_top_k)):
        if rec_do_sample is None:
            rec_do_sample = True
        recommended_sampling = {
            "do_sample": bool(rec_do_sample),
            "temperature": float(rec_temperature if rec_temperature is not None else 0.6),
            "top_p": float(rec_top_p if rec_top_p is not None else 0.95),
            "top_k": int(rec_top_k if rec_top_k is not None else 0),
        }

    lm_head_stem = f"{model_prefix}_lm_head{_lut_suffix(lut3_bits)}"
    lm_head_pkg = out_dir / f"{lm_head_stem}.mlpackage"
    lm_head_mlmodelc = out_dir / f"{lm_head_stem}.mlmodelc"
    print(f"Building LM head: {lm_head_pkg.name} (argmax={argmax_in_model})")
    lm_head_ml = _build_lm_head(
        model=model,
        context_length=args.context_length,
        batch_size=args.batch_size,
        lut3_bits=lut3_bits,
        lut3_per=lut3_per,
        argmax_in_model=argmax_in_model,
    )
    _save_model(lm_head_ml, lm_head_pkg)
    _compile_mlpackage(lm_head_pkg, out_dir)

    # Build chunk 01of03 (FP32 attention-only)
    print("Converting chunk_01of03 infer (FP32, CPU_ONLY)...")
    c1_infer_ml = _convert_infer_wrapper(
        chunk1_infer,
        model,
        args.context_length,
        cfg.hidden_size,
        precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    print("Converting chunk_01of03 prefill (FP32, CPU_ONLY)...")
    c1_prefill_ml = _convert_prefill_wrapper(
        chunk1_prefill,
        model,
        args.context_length,
        cfg.hidden_size,
        args.batch_size,
        precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    c1_infer_tmp = out_dir / f"{target_base}_infer_fp32_chunk_01of03.mlpackage"
    c1_prefill_tmp = out_dir / f"{target_base}_prefill_fp32_chunk_01of03.mlpackage"
    _save_model(c1_infer_ml, c1_infer_tmp)
    _save_model(c1_prefill_ml, c1_prefill_tmp)
    c1_pkg = out_dir / f"{target_base}_chunk_01of03.mlpackage"
    _combine_infer_prefill(c1_infer_tmp, c1_prefill_tmp, c1_pkg)
    shutil.rmtree(c1_infer_tmp, ignore_errors=True)
    shutil.rmtree(c1_prefill_tmp, ignore_errors=True)
    _compile_mlpackage(c1_pkg, out_dir)

    # Build chunk 02of03 (post-attn + middle layers)
    print("Converting chunk_02of03 infer (FP16)...")
    c2_infer_ml = _convert_infer_wrapper(
        chunk2_infer,
        model,
        args.context_length,
        cfg.hidden_size,
        precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print("Converting chunk_02of03 prefill (FP16)...")
    c2_prefill_ml = _convert_prefill_wrapper(
        chunk2_prefill,
        model,
        args.context_length,
        cfg.hidden_size,
        args.batch_size,
        precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    if quantize_ffn and lut2_bits is not None:
        print(f"Quantizing chunk_02of03 with LUT {lut2_bits},{lut2_per} ...")
        c2_infer_ml = _quantize_with_lut(
            c2_infer_ml,
            model,
            args.context_length,
            args.batch_size,
            lut2_bits,
            0 if lut2_per is None else lut2_per,
        )
        c2_prefill_ml = _quantize_with_lut(
            c2_prefill_ml,
            model,
            args.context_length,
            args.batch_size,
            lut2_bits,
            0 if lut2_per is None else lut2_per,
        )

    c2_infer_tmp = out_dir / f"{target_base}_infer_tmp_chunk_02of03.mlpackage"
    c2_prefill_tmp = out_dir / f"{target_base}_prefill_tmp_chunk_02of03.mlpackage"
    _save_model(c2_infer_ml, c2_infer_tmp)
    _save_model(c2_prefill_ml, c2_prefill_tmp)
    c2_pkg = out_dir / f"{target_base}_chunk_02of03.mlpackage"
    _combine_infer_prefill(c2_infer_tmp, c2_prefill_tmp, c2_pkg)
    shutil.rmtree(c2_infer_tmp, ignore_errors=True)
    shutil.rmtree(c2_prefill_tmp, ignore_errors=True)
    _compile_mlpackage(c2_pkg, out_dir)

    # Build chunk 03of03 (tail).
    if args.copy_tail_from_source:
        print("Copying chunk_03of03 from source tail chunk (02of02)...")
        c3_pkg = _copy_tail_chunk_from_source(source_dir, out_dir, source_base, target_base)
        _compile_mlpackage(c3_pkg, out_dir)
    else:
        print("Converting chunk_03of03 infer (FP16)...")
        c3_infer_ml = _convert_infer_wrapper(
            chunk3_infer,
            model,
            args.context_length,
            cfg.hidden_size,
            precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        print("Converting chunk_03of03 prefill (FP16)...")
        c3_prefill_ml = _convert_prefill_wrapper(
            chunk3_prefill,
            model,
            args.context_length,
            cfg.hidden_size,
            args.batch_size,
            precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

        if quantize_ffn and lut2_bits is not None:
            print(f"Quantizing chunk_03of03 with LUT {lut2_bits},{lut2_per} ...")
            c3_infer_ml = _quantize_with_lut(
                c3_infer_ml,
                model,
                args.context_length,
                args.batch_size,
                lut2_bits,
                0 if lut2_per is None else lut2_per,
            )
            c3_prefill_ml = _quantize_with_lut(
                c3_prefill_ml,
                model,
                args.context_length,
                args.batch_size,
                lut2_bits,
                0 if lut2_per is None else lut2_per,
            )

        c3_infer_tmp = out_dir / f"{target_base}_infer_tmp_chunk_03of03.mlpackage"
        c3_prefill_tmp = out_dir / f"{target_base}_prefill_tmp_chunk_03of03.mlpackage"
        _save_model(c3_infer_ml, c3_infer_tmp)
        _save_model(c3_prefill_ml, c3_prefill_tmp)
        c3_pkg = out_dir / f"{target_base}_chunk_03of03.mlpackage"
        _combine_infer_prefill(c3_infer_tmp, c3_prefill_tmp, c3_pkg)
        shutil.rmtree(c3_infer_tmp, ignore_errors=True)
        shutil.rmtree(c3_prefill_tmp, ignore_errors=True)
        _compile_mlpackage(c3_pkg, out_dir)

    # Patch metadata for 3-chunk loading.
    _update_meta_for_3chunks(
        out_dir,
        new_ffn_mlmodelc=f"{target_base}_chunk_01of03.mlmodelc",
        new_lm_head_mlmodelc=f"{lm_head_stem}.mlmodelc",
        lut1_bits=lut1_bits,
        lut2_bits=lut2_bits,
        lut2_per=lut2_per,
        lut3_bits=lut3_bits,
        lut3_per=lut3_per,
        argmax_in_model=argmax_in_model,
        recommended_sampling=recommended_sampling,
    )
    _assert_required_artifacts(out_dir, target_base=target_base, lm_head_stem=lm_head_stem)

    print("\nDone.")
    print(f"Prototype directory: {out_dir}")
    print(f"Chunk 1: {c1_pkg.name} (FP32 attention-only)")
    print(f"Chunk 2: {c2_pkg.name} (post-attention + middle layers)")
    print(f"Chunk 3: {c3_pkg.name} (tail)")
    print(f"Run test:")
    print(f"  python tests/chat.py --meta {out_dir / 'meta.yaml'} --prompt \"2+2=\" --max-tokens 32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
