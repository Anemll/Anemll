#!/usr/bin/env python3
"""Prototype hybrid chunk export with FP32 attention-only first stage.

Runtime layout (chat.py compatible):
  embeddings -> FFN_PF_chunk_01ofNN -> ... -> FFN_PF_chunk_NNofNN -> lm_head

Where:
  - chunk_01ofNN: layer0 attention residual only (FP32, CPU_ONLY)
  - chunk_02ofNN..chunk_NNofNN: remaining model split across configurable chunks
    (chunk_02 applies layer0 post-attention MLP, final chunk applies model norm).
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


def _remaining_layer_segments(total_layers: int, remaining_chunks: int) -> list[tuple[int, int]]:
    """Partition layers [1, total_layers) into `remaining_chunks` contiguous ranges."""
    if remaining_chunks <= 0:
        raise ValueError("remaining_chunks must be > 0")
    remaining_layers = max(total_layers - 1, 0)
    if remaining_layers == 0:
        return [(1, 1)] * remaining_chunks
    base, rem = divmod(remaining_layers, remaining_chunks)
    segments: list[tuple[int, int]] = []
    start = 1
    for idx in range(remaining_chunks):
        size = base + (1 if idx < rem else 0)
        end = start + size
        segments.append((start, end))
        start = end
    return segments


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


class Chunk1Fp32FirstAttnInferWrapper(torch.nn.Module):
    """Full chunk-1 contract with FP32 first-layer attention path.

    This wrapper is intended for infer-only standalone export used by
    state-transition combine. It preserves default converter chunk semantics:
    - chunk_01 must output the same hidden state contract as regular FFN chunk_01
      for the given num_chunks partition.
    """

    def __init__(self, model: Qwen25ForCausalLM, end_layer: int | None):
        super().__init__()
        self.model = model
        self.end_layer = end_layer

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        layer0 = self.model.model.layers[0]
        normalized_states = layer0.input_layernorm(hidden_states)
        rotary = self.model.model.get_rotary_embeddings_s(current_pos)
        query_states, key_states, value_states = layer0.self_attn.get_new_kv_cache(
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
        attn_out = layer0.self_attn.forward_regular(
            hidden_states=normalized_states,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
            current_pos=current_pos,
        )
        hidden_states = hidden_states + attn_out

        # Complete layer0 MLP path so chunk output matches default converter chunk_01.
        post = layer0.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer0.mlp(post)

        if self.end_layer is None or self.end_layer > 1:
            hidden_states = self.model.model.process_layers(
                hidden_states,
                position_ids,
                causal_mask,
                current_pos,
                rotary,
                start_layer=1,
                end_layer=self.end_layer,
                IN_PREFILL=False,
            )

        # Mirror converter behavior: only last chunk applies final norm.
        if self.end_layer is None or self.end_layer == len(self.model.model.layers):
            hidden_states = self.model.model.norm(hidden_states)

        return hidden_states


class RemainingSegmentInferWrapper(torch.nn.Module):
    """Infer wrapper for one post-attention segment."""

    def __init__(
        self,
        model: Qwen25ForCausalLM,
        start_layer: int,
        end_layer: int,
        *,
        include_layer0_mlp: bool = False,
        apply_final_norm: bool = False,
    ):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.include_layer0_mlp = include_layer0_mlp
        self.apply_final_norm = apply_final_norm

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        if self.include_layer0_mlp:
            layer0 = self.model.model.layers[0]
            post = layer0.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states + layer0.mlp(post)

        rotary = self.model.model.get_rotary_embeddings_s(current_pos)
        if self.start_layer < self.end_layer:
            hidden_states = self.model.model.process_layers(
                hidden_states,
                position_ids,
                causal_mask,
                current_pos,
                rotary,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                IN_PREFILL=False,
            )
        if self.apply_final_norm:
            hidden_states = self.model.model.norm(hidden_states)
        return hidden_states


class RemainingSegmentPrefillWrapper(torch.nn.Module):
    """Prefill wrapper for one post-attention segment."""

    def __init__(
        self,
        model: Qwen25ForCausalLM,
        start_layer: int,
        end_layer: int,
        *,
        include_layer0_mlp: bool = False,
        return_first_token: bool = False,
    ):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.include_layer0_mlp = include_layer0_mlp
        self.return_first_token = return_first_token

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        if self.include_layer0_mlp:
            layer0 = self.model.model.layers[0]
            post = layer0.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states + layer0.mlp(post)

        rotary = self.model.model.get_rotary_embedding_prefill(position_ids)
        if self.start_layer < self.end_layer:
            hidden_states = self.model.model.process_layers(
                hidden_states,
                position_ids,
                causal_mask,
                current_pos,
                rotary,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                IN_PREFILL=True,
            )
        if self.return_first_token:
            return hidden_states[:, 0:1, :]
        return hidden_states


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


def _load_meta_params(source_dir: Path, allow_missing: bool = False) -> dict[str, Any]:
    meta_path = source_dir / "meta.yaml"
    if not meta_path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(meta_path)
    meta = yaml.safe_load(meta_path.read_text())
    params = meta.get("model_info", {}).get("parameters", {})
    if not params:
        if allow_missing:
            return {}
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


def _has_chunk_artifact(out_dir: Path, base: str, idx: int, num_chunks: int) -> bool:
    stem = f"{base}_chunk_{idx:02d}of{num_chunks:02d}"
    return (out_dir / f"{stem}.mlpackage").exists() or (out_dir / f"{stem}.mlmodelc").exists()


def _update_meta_for_chunks(
    out_dir: Path,
    new_ffn_mlmodelc: str,
    new_lm_head_mlmodelc: str,
    num_chunks: int,
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
    params["num_chunks"] = int(num_chunks)
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
    desc = meta.setdefault("model_info", {}).get("description")
    if not isinstance(desc, str):
        desc = ""
    if f"Chunks: {num_chunks}" not in desc:
        suffix = f" Chunks: {num_chunks}."
        meta["model_info"]["description"] = (desc.strip() + suffix).strip()
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))


def _assert_required_artifacts(
    out_dir: Path,
    target_base: str,
    lm_head_stem: str,
    num_chunks: int,
) -> None:
    required = [
        out_dir / "meta.yaml",
        out_dir / f"{lm_head_stem}.mlmodelc",
    ]
    for idx in range(1, num_chunks + 1):
        required.append(out_dir / f"{target_base}_chunk_{idx:02d}of{num_chunks:02d}.mlpackage")
        required.append(out_dir / f"{target_base}_chunk_{idx:02d}of{num_chunks:02d}.mlmodelc")

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        lines = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Prototype build incomplete. Missing required artifacts:\n"
            f"  - {lines}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Prototype hybrid export: chunk01 attention-only FP32 + configurable "
            "post-attention segment chunks."
        )
    )
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
    ap.add_argument(
        "--num-chunks",
        type=int,
        default=3,
        help="Chunk count for infer-only patch naming (default: 3).",
    )
    ap.add_argument(
        "--remaining-chunks",
        type=int,
        default=3,
        help=(
            "Number of post-attention segments for full hybrid mode "
            "(total output chunks = 1 + remaining_chunks). Default: 3."
        ),
    )
    ap.add_argument("--reuse-out-dir", action="store_true", help="Do not recopy source dir if out dir exists.")
    ap.add_argument(
        "--infer-only",
        action="store_true",
        help="Only rebuild FP32 attention chunk_01 infer (no prefill).",
    )
    ap.add_argument(
        "--prefill-only",
        action="store_true",
        help="Only rebuild FP32 attention chunk_01 prefill (no infer).",
    )
    ap.add_argument(
        "--infer-only-out-base",
        default=None,
        help=(
            "Optional output base stem for FP32 chunk export "
            "(e.g. qwen25_FFN_attn_fp32). If omitted, overwrites the regular FFN chunk_01 base."
        ),
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="Do not compile newly produced .mlpackage outputs to .mlmodelc.",
    )
    ap.add_argument("--no-quantize-chunk2", action="store_true", help="Skip LUT quantization for chunk2 (legacy flag).")
    ap.add_argument("--no-quantize-ffn", action="store_true", help="Skip LUT quantization for FFN chunks 2 and 3.")
    ap.add_argument("--copy-tail-from-source", action="store_true", help="Reuse source tail chunk (may stay quantized).")
    ap.add_argument("--lut1", type=str, default=None, help="LUT for embeddings, e.g. 'none' or '6,4'")
    ap.add_argument("--lut2", type=str, default=None, help="LUT for FFN chunks, e.g. 'none' or '6,4'")
    ap.add_argument("--lut3", type=str, default=None, help="LUT for lm_head, e.g. 'none' or '6,4'")
    ap.add_argument(
        "--reuse-lm-head",
        action="store_true",
        help="Reuse existing lm_head artifact in out dir instead of rebuilding it.",
    )
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
    ap.add_argument(
        "--rebuild-ffn-chunk",
        type=int,
        default=None,
        help=(
            "Rebuild a single FFN+prefill chunk (1-indexed) with layer0 MLP + layers[start:end]. "
            "Uses standard converter layer partition (not hybrid). "
            "Applies LUT if --lut2 is set."
        ),
    )
    ap.add_argument(
        "--rebuild-prefill-chunk",
        type=int,
        default=None,
        help=(
            "Rebuild only the prefill for a single chunk (1-indexed) with layer0 MLP + layers[start:end]. "
            "Uses standard converter layer partition. Applies LUT if --lut2 is set."
        ),
    )
    ap.add_argument(
        "--rebuild-hybrid-chunk1",
        action="store_true",
        help=(
            "Rebuild all 4 hybrid chunk1 artifacts in one shot: "
            "FP32 attention infer+prefill and FFN+prefill chunk_01 (layer0 MLP + layers). "
            "Reads config from meta.yaml in --out-dir. No --source-dir needed."
        ),
    )
    args = ap.parse_args()

    model_path = _resolve_model_path(args.model_path)
    out_dir = Path(args.out_dir).expanduser().resolve()

    # --rebuild-hybrid-chunk1 doesn't need source-dir, reads meta from out-dir
    if args.rebuild_hybrid_chunk1:
        source_dir = out_dir
    else:
        source_dir = Path(args.source_dir).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    if out_dir.exists() and not (args.reuse_out_dir or args.rebuild_hybrid_chunk1):
        shutil.rmtree(out_dir)
    if not out_dir.exists() and not args.rebuild_hybrid_chunk1:
        print(f"Cloning source directory to: {out_dir}")
        shutil.copytree(source_dir, out_dir)
    if args.rebuild_hybrid_chunk1 and not out_dir.exists():
        raise FileNotFoundError(f"--rebuild-hybrid-chunk1 requires existing out-dir: {out_dir}")

    meta_params = _load_meta_params(source_dir, allow_missing=args.infer_only)
    model_prefix = str(meta_params.get("model_prefix", "qwen25"))

    # For rebuild modes, override CLI defaults with meta.yaml values when not
    # explicitly provided on the command line.  This prevents the common mistake
    # of building all contexts with the default --context-length 2048.
    _rebuild_mode = (
        args.rebuild_hybrid_chunk1
        or args.rebuild_ffn_chunk is not None
        or args.rebuild_prefill_chunk is not None
    )
    if _rebuild_mode and meta_params:
        _cli_set = {a.split("=")[0].lstrip("-") for a in sys.argv[1:] if a.startswith("-")}
        _cli_set_normalized = set()
        for k in _cli_set:
            _cli_set_normalized.add(k.replace("-", "_"))

        if "context_length" not in _cli_set_normalized:
            meta_ctx = meta_params.get("context_length")
            if meta_ctx is not None:
                args.context_length = int(meta_ctx)
                print(f"[meta] context_length={args.context_length} (from meta.yaml)")
        if "batch_size" not in _cli_set_normalized:
            meta_bs = meta_params.get("batch_size")
            if meta_bs is not None:
                args.batch_size = int(meta_bs)
                print(f"[meta] batch_size={args.batch_size} (from meta.yaml)")
        if "num_chunks" not in _cli_set_normalized:
            meta_nc = meta_params.get("num_chunks")
            if meta_nc is not None:
                args.num_chunks = int(meta_nc)
                print(f"[meta] num_chunks={args.num_chunks} (from meta.yaml)")
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
        print(
            "Warning: this prototype keeps chunk_01 FP32 unquantized; "
            "--lut1 is metadata-only here."
        )

    if args.infer_only:
        source_base = f"{model_prefix}_FFN{_lut_suffix(src_lut2_bits)}"
        target_base = f"{model_prefix}_FFN{_lut_suffix(lut2_bits)}"
    else:
        source_base = f"{model_prefix}_FFN_PF{_lut_suffix(src_lut2_bits)}"
        target_base = f"{model_prefix}_FFN_PF{_lut_suffix(lut2_bits)}"

    print(f"Source chunk base: {source_base}")
    print(f"Target chunk base: {target_base}")
    print(f"LUT config: lut1={args.lut1 or src_lut1_bits}, lut2={args.lut2 or src_lut2_bits}, lut3={args.lut3 or src_lut3_bits}")
    quantize_ffn = not (args.no_quantize_chunk2 or args.no_quantize_ffn)

    if not (args.rebuild_hybrid_chunk1 or args.rebuild_ffn_chunk is not None
            or args.rebuild_prefill_chunk is not None or args.infer_only or args.prefill_only):
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

    # --rebuild-ffn-chunk / --rebuild-prefill-chunk: export a single chunk with layer0-MLP + layers[start:end]
    rebuild_chunk = args.rebuild_ffn_chunk if args.rebuild_ffn_chunk is not None else args.rebuild_prefill_chunk
    build_ffn = args.rebuild_ffn_chunk is not None
    build_prefill = args.rebuild_prefill_chunk is not None or args.rebuild_ffn_chunk is not None
    if args.rebuild_prefill_chunk is not None and args.rebuild_ffn_chunk is None:
        build_ffn = False
    if rebuild_chunk is not None:
        chunk_idx_1 = rebuild_chunk  # 1-indexed
        total_chunks = args.num_chunks
        if chunk_idx_1 < 1 or chunk_idx_1 > total_chunks:
            raise ValueError(f"--rebuild-ffn-chunk {chunk_idx_1} out of range [1, {total_chunks}]")
        chunk_idx_0 = chunk_idx_1 - 1

        # Use standard converter layer partition
        total_layers = cfg.num_hidden_layers
        base_c, rem_c = divmod(total_layers, total_chunks)
        start_layer = chunk_idx_0 * base_c + min(chunk_idx_0, rem_c)
        end_layer = start_layer + base_c + (1 if chunk_idx_0 < rem_c else 0)

        # For chunk 1 in hybrid: skip layer0 attention, include layer0 MLP
        include_layer0_mlp = (start_layer == 0)
        if include_layer0_mlp:
            start_layer = 1  # attention handled by FP32 chunk

        is_last = (chunk_idx_1 == total_chunks)
        ffn_base = f"{model_prefix}_FFN{_lut_suffix(lut2_bits)}"
        chunk_stem = f"{ffn_base}_chunk_{chunk_idx_1:02d}of{total_chunks:02d}"

        seg_infer_ml = None
        seg_prefill_ml = None

        if build_ffn:
            print(
                f"Rebuilding FFN {chunk_stem}: "
                f"layers[{start_layer}:{end_layer}) "
                f"include_l0_mlp={include_layer0_mlp} final_norm={is_last}"
            )
            seg_infer = RemainingSegmentInferWrapper(
                model,
                start_layer=start_layer,
                end_layer=end_layer,
                include_layer0_mlp=include_layer0_mlp,
                apply_final_norm=is_last,
            ).eval()
            seg_infer_ml = _convert_infer_wrapper(
                seg_infer, model, args.context_length, cfg.hidden_size,
                precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
            )

        prefill_base = f"{model_prefix}_prefill{_lut_suffix(lut2_bits)}"
        prefill_stem = f"{prefill_base}_chunk_{chunk_idx_1:02d}of{total_chunks:02d}"

        if build_prefill:
            print(
                f"Rebuilding prefill {prefill_stem}: "
                f"layers[{start_layer}:{end_layer}) "
                f"include_l0_mlp={include_layer0_mlp} return_first_token={is_last}"
            )
            seg_prefill = RemainingSegmentPrefillWrapper(
                model,
                start_layer=start_layer,
                end_layer=end_layer,
                include_layer0_mlp=include_layer0_mlp,
                return_first_token=is_last,
            ).eval()
            seg_prefill_ml = _convert_prefill_wrapper(
                seg_prefill, model, args.context_length, cfg.hidden_size,
                args.batch_size,
                precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
            )

        if quantize_ffn and lut2_bits is not None:
            print(f"Quantizing with LUT {lut2_bits},{lut2_per} ...")
            if seg_infer_ml is not None:
                seg_infer_ml = _quantize_with_lut(
                    seg_infer_ml, model, args.context_length, args.batch_size,
                    lut2_bits, 0 if lut2_per is None else lut2_per,
                )
            if seg_prefill_ml is not None:
                seg_prefill_ml = _quantize_with_lut(
                    seg_prefill_ml, model, args.context_length, args.batch_size,
                    lut2_bits, 0 if lut2_per is None else lut2_per,
                )

        print(f"\nDone. Rebuilt:")
        if seg_infer_ml is not None:
            ffn_pkg = out_dir / f"{chunk_stem}.mlpackage"
            _save_model(seg_infer_ml, ffn_pkg)
            if not args.no_compile:
                _compile_mlpackage(ffn_pkg, out_dir)
            print(f"  FFN:     {ffn_pkg.name}")

        if seg_prefill_ml is not None:
            prefill_pkg = out_dir / f"{prefill_stem}.mlpackage"
            _save_model(seg_prefill_ml, prefill_pkg)
            if not args.no_compile:
                _compile_mlpackage(prefill_pkg, out_dir)
            print(f"  Prefill: {prefill_pkg.name}")
        return 0

    # --rebuild-hybrid-chunk1: all 4 chunk1 artifacts in one shot
    if args.rebuild_hybrid_chunk1:
        total_chunks = args.num_chunks
        total_layers = cfg.num_hidden_layers
        base_c, rem_c = divmod(total_layers, total_chunks)
        end_layer = base_c + (1 if 0 < rem_c else 0)

        ffn_base = f"{model_prefix}_FFN{_lut_suffix(lut2_bits)}"
        ffn_stem = f"{ffn_base}_chunk_01of{total_chunks:02d}"
        prefill_base = f"{model_prefix}_prefill{_lut_suffix(lut2_bits)}"
        prefill_stem = f"{prefill_base}_chunk_01of{total_chunks:02d}"
        fp32_infer_base = f"{model_prefix}_FFN_attn_fp32"
        fp32_prefill_base = f"{model_prefix}_prefill_attn_fp32"
        fp32_infer_stem = f"{fp32_infer_base}_chunk_01of{total_chunks:02d}"
        fp32_prefill_stem = f"{fp32_prefill_base}_chunk_01of{total_chunks:02d}"

        print(f"=== Hybrid chunk1 rebuild (total_chunks={total_chunks}, layers=28) ===")
        print(f"  FP32 attention: layer0 attention only")
        print(f"  ANE chunk1: layer0 MLP + layers[1:{end_layer})")

        # 1. FP32 attention infer
        print(f"\n[1/4] FP32 attention infer: {fp32_infer_stem}")
        fp32_infer_wrapper = Chunk1AttnInferWrapper(model).eval()
        fp32_infer_ml = _convert_infer_wrapper(
            fp32_infer_wrapper, model, args.context_length, cfg.hidden_size,
            precision=ct.precision.FLOAT32,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        fp32_infer_pkg = out_dir / f"{fp32_infer_stem}.mlpackage"
        _save_model(fp32_infer_ml, fp32_infer_pkg)

        # 2. FP32 attention prefill
        print(f"\n[2/4] FP32 attention prefill: {fp32_prefill_stem}")
        fp32_prefill_wrapper = Chunk1AttnPrefillWrapper(model).eval()
        fp32_prefill_ml = _convert_prefill_wrapper(
            fp32_prefill_wrapper, model, args.context_length, cfg.hidden_size,
            args.batch_size,
            precision=ct.precision.FLOAT32,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        fp32_prefill_pkg = out_dir / f"{fp32_prefill_stem}.mlpackage"
        _save_model(fp32_prefill_ml, fp32_prefill_pkg)

        # 3. ANE FFN chunk1 (layer0 MLP + layers 1..end_layer)
        print(f"\n[3/4] ANE FFN infer: {ffn_stem} (layer0 MLP + layers[1:{end_layer}))")
        ffn_infer_wrapper = RemainingSegmentInferWrapper(
            model, start_layer=1, end_layer=end_layer,
            include_layer0_mlp=True, apply_final_norm=False,
        ).eval()
        ffn_infer_ml = _convert_infer_wrapper(
            ffn_infer_wrapper, model, args.context_length, cfg.hidden_size,
            precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

        # 4. ANE prefill chunk1 (layer0 MLP + layers 1..end_layer)
        print(f"\n[4/4] ANE prefill: {prefill_stem} (layer0 MLP + layers[1:{end_layer}))")
        prefill_wrapper = RemainingSegmentPrefillWrapper(
            model, start_layer=1, end_layer=end_layer,
            include_layer0_mlp=True, return_first_token=False,
        ).eval()
        prefill_ml = _convert_prefill_wrapper(
            prefill_wrapper, model, args.context_length, cfg.hidden_size,
            args.batch_size,
            precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

        # Quantize ANE chunks
        if quantize_ffn and lut2_bits is not None:
            print(f"\nQuantizing ANE chunks with LUT {lut2_bits},{lut2_per} ...")
            ffn_infer_ml = _quantize_with_lut(
                ffn_infer_ml, model, args.context_length, args.batch_size,
                lut2_bits, 0 if lut2_per is None else lut2_per,
            )
            prefill_ml = _quantize_with_lut(
                prefill_ml, model, args.context_length, args.batch_size,
                lut2_bits, 0 if lut2_per is None else lut2_per,
            )

        # Save ANE chunks
        ffn_pkg = out_dir / f"{ffn_stem}.mlpackage"
        _save_model(ffn_infer_ml, ffn_pkg)
        prefill_pkg = out_dir / f"{prefill_stem}.mlpackage"
        _save_model(prefill_ml, prefill_pkg)

        # Compile all 4
        if not args.no_compile:
            print("\nCompiling...")
            _compile_mlpackage(fp32_infer_pkg, out_dir)
            _compile_mlpackage(fp32_prefill_pkg, out_dir)
            _compile_mlpackage(ffn_pkg, out_dir)
            _compile_mlpackage(prefill_pkg, out_dir)

        print(f"\nDone. Rebuilt 4 hybrid chunk1 artifacts:")
        print(f"  FP32 infer:   {fp32_infer_pkg.name}")
        print(f"  FP32 prefill: {fp32_prefill_pkg.name}")
        print(f"  ANE FFN:      {ffn_pkg.name}")
        print(f"  ANE prefill:  {prefill_pkg.name}")
        return 0

    if args.remaining_chunks <= 0:
        raise ValueError("--remaining-chunks must be > 0")
    total_output_chunks = 1 + int(args.remaining_chunks)
    segment_ranges = _remaining_layer_segments(cfg.num_hidden_layers, args.remaining_chunks)
    print(
        "Hybrid segmentation: "
        f"chunk_01 attention-only + {args.remaining_chunks} remaining chunks "
        f"(total={total_output_chunks})"
    )
    print(
        "Remaining ranges: "
        + ", ".join(
            f"chunk_{idx + 2:02d}=[{start}:{end})"
            for idx, (start, end) in enumerate(segment_ranges)
        )
    )

    chunk1_infer = Chunk1AttnInferWrapper(model).eval()
    chunk1_prefill = Chunk1AttnPrefillWrapper(model).eval()

    if args.infer_only or args.prefill_only:
        if args.num_chunks <= 0:
            raise ValueError("--num-chunks must be > 0")

        infer_only_base = args.infer_only_out_base.strip() if args.infer_only_out_base else target_base
        overwrite_mode = infer_only_base == target_base
        prefill_attn_base = infer_only_base.replace("_FFN_", "_prefill_")
        build_infer = not args.prefill_only
        build_prefill = not args.infer_only

        # If both flags given, build both
        if args.infer_only and args.prefill_only:
            build_infer = True
            build_prefill = True

        c1_infer_pkg = None
        c1_prefill_pkg = None

        if build_infer:
            print(
                f"FP32 attention infer: exporting "
                f"{infer_only_base}_chunk_01of{args.num_chunks:02d} "
                f"(layer0 attention residual only)"
            )
            chunk1_attn_infer = Chunk1AttnInferWrapper(model).eval()
            c1_infer_ml = _convert_infer_wrapper(
                chunk1_attn_infer,
                model,
                args.context_length,
                cfg.hidden_size,
                precision=ct.precision.FLOAT32,
                compute_units=ct.ComputeUnit.CPU_ONLY,
            )
            c1_infer_pkg = out_dir / f"{infer_only_base}_chunk_01of{args.num_chunks:02d}.mlpackage"
            _save_model(c1_infer_ml, c1_infer_pkg)
            if not args.no_compile:
                _compile_mlpackage(c1_infer_pkg, out_dir)

        if build_prefill:
            print(
                f"FP32 attention prefill: exporting "
                f"{prefill_attn_base}_chunk_01of{args.num_chunks:02d} "
                f"(layer0 attention residual only)"
            )
            chunk1_attn_prefill = Chunk1AttnPrefillWrapper(model).eval()
            c1_prefill_ml = _convert_prefill_wrapper(
                chunk1_attn_prefill,
                model,
                args.context_length,
                cfg.hidden_size,
                args.batch_size,
                precision=ct.precision.FLOAT32,
                compute_units=ct.ComputeUnit.CPU_ONLY,
            )
            c1_prefill_pkg = out_dir / f"{prefill_attn_base}_chunk_01of{args.num_chunks:02d}.mlpackage"
            _save_model(c1_prefill_ml, c1_prefill_pkg)
            if not args.no_compile:
                _compile_mlpackage(c1_prefill_pkg, out_dir)

        print("\nDone (FP32 attention).")
        if c1_infer_pkg:
            print(f"  Infer:   {c1_infer_pkg.name}")
        if c1_prefill_pkg:
            print(f"  Prefill: {c1_prefill_pkg.name}")
        return 0

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
    if args.reuse_lm_head:
        existing_lm_head = str(meta_params.get("lm_head", "")).strip()
        if existing_lm_head:
            lm_head_stem = Path(existing_lm_head).stem
        lm_head_pkg = out_dir / f"{lm_head_stem}.mlpackage"
        lm_head_mlmodelc = out_dir / f"{lm_head_stem}.mlmodelc"
        print(f"Reusing LM head: {lm_head_stem} (argmax meta override={argmax_in_model})")
        if not lm_head_mlmodelc.exists():
            if lm_head_pkg.exists():
                _compile_mlpackage(lm_head_pkg, out_dir)
            else:
                raise FileNotFoundError(
                    f"--reuse-lm-head requested but neither {lm_head_pkg} nor {lm_head_mlmodelc} exists"
                )
    else:
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

    # Build chunk 01ofNN (FP32 attention-only).
    print(f"Converting chunk_01of{total_output_chunks:02d} infer (FP32, CPU_ONLY)...")
    c1_infer_ml = _convert_infer_wrapper(
        chunk1_infer,
        model,
        args.context_length,
        cfg.hidden_size,
        precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    print(f"Converting chunk_01of{total_output_chunks:02d} prefill (FP32, CPU_ONLY)...")
    c1_prefill_ml = _convert_prefill_wrapper(
        chunk1_prefill,
        model,
        args.context_length,
        cfg.hidden_size,
        args.batch_size,
        precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    c1_infer_tmp = out_dir / f"{target_base}_infer_fp32_chunk_01of{total_output_chunks:02d}.mlpackage"
    c1_prefill_tmp = out_dir / f"{target_base}_prefill_fp32_chunk_01of{total_output_chunks:02d}.mlpackage"
    _save_model(c1_infer_ml, c1_infer_tmp)
    _save_model(c1_prefill_ml, c1_prefill_tmp)
    c1_pkg = out_dir / f"{target_base}_chunk_01of{total_output_chunks:02d}.mlpackage"
    _combine_infer_prefill(c1_infer_tmp, c1_prefill_tmp, c1_pkg)
    shutil.rmtree(c1_infer_tmp, ignore_errors=True)
    shutil.rmtree(c1_prefill_tmp, ignore_errors=True)
    _compile_mlpackage(c1_pkg, out_dir)

    if args.copy_tail_from_source and total_output_chunks != 3:
        raise ValueError(
            "--copy-tail-from-source is only supported for total output chunks == 3 "
            "(set --remaining-chunks 2)."
        )

    built_chunk_pkgs: list[Path] = [c1_pkg]

    # Build post-attention chunks 02..NN.
    for seg_idx, (start_layer, end_layer) in enumerate(segment_ranges):
        out_idx = seg_idx + 2
        include_layer0_mlp = seg_idx == 0
        is_last = seg_idx == len(segment_ranges) - 1

        if args.copy_tail_from_source and is_last:
            print("Copying final tail chunk from source chunk_02of02...")
            copied = _copy_tail_chunk_from_source(source_dir, out_dir, source_base, target_base)
            _compile_mlpackage(copied, out_dir)
            built_chunk_pkgs.append(copied)
            continue

        print(
            f"Converting chunk_{out_idx:02d}of{total_output_chunks:02d} infer "
            f"(FP16) start={start_layer} end={end_layer} "
            f"include_l0_mlp={include_layer0_mlp} final_norm={is_last}"
        )
        seg_infer = RemainingSegmentInferWrapper(
            model,
            start_layer=start_layer,
            end_layer=end_layer,
            include_layer0_mlp=include_layer0_mlp,
            apply_final_norm=is_last,
        ).eval()
        seg_infer_ml = _convert_infer_wrapper(
            seg_infer,
            model,
            args.context_length,
            cfg.hidden_size,
            precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

        print(
            f"Converting chunk_{out_idx:02d}of{total_output_chunks:02d} prefill (FP16)"
        )
        seg_prefill = RemainingSegmentPrefillWrapper(
            model,
            start_layer=start_layer,
            end_layer=end_layer,
            include_layer0_mlp=include_layer0_mlp,
            return_first_token=is_last,
        ).eval()
        seg_prefill_ml = _convert_prefill_wrapper(
            seg_prefill,
            model,
            args.context_length,
            cfg.hidden_size,
            args.batch_size,
            precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

        if quantize_ffn and lut2_bits is not None:
            print(
                f"Quantizing chunk_{out_idx:02d}of{total_output_chunks:02d} "
                f"with LUT {lut2_bits},{lut2_per} ..."
            )
            seg_infer_ml = _quantize_with_lut(
                seg_infer_ml,
                model,
                args.context_length,
                args.batch_size,
                lut2_bits,
                0 if lut2_per is None else lut2_per,
            )
            seg_prefill_ml = _quantize_with_lut(
                seg_prefill_ml,
                model,
                args.context_length,
                args.batch_size,
                lut2_bits,
                0 if lut2_per is None else lut2_per,
            )

        infer_tmp = out_dir / f"{target_base}_infer_tmp_chunk_{out_idx:02d}of{total_output_chunks:02d}.mlpackage"
        prefill_tmp = out_dir / f"{target_base}_prefill_tmp_chunk_{out_idx:02d}of{total_output_chunks:02d}.mlpackage"
        _save_model(seg_infer_ml, infer_tmp)
        _save_model(seg_prefill_ml, prefill_tmp)
        out_pkg = out_dir / f"{target_base}_chunk_{out_idx:02d}of{total_output_chunks:02d}.mlpackage"
        _combine_infer_prefill(infer_tmp, prefill_tmp, out_pkg)
        shutil.rmtree(infer_tmp, ignore_errors=True)
        shutil.rmtree(prefill_tmp, ignore_errors=True)
        _compile_mlpackage(out_pkg, out_dir)
        built_chunk_pkgs.append(out_pkg)

    # Patch metadata for N-chunk loading.
    _update_meta_for_chunks(
        out_dir,
        new_ffn_mlmodelc=f"{target_base}_chunk_01of{total_output_chunks:02d}.mlmodelc",
        new_lm_head_mlmodelc=f"{lm_head_stem}.mlmodelc",
        num_chunks=total_output_chunks,
        lut1_bits=lut1_bits,
        lut2_bits=lut2_bits,
        lut2_per=lut2_per,
        lut3_bits=lut3_bits,
        lut3_per=lut3_per,
        argmax_in_model=argmax_in_model,
        recommended_sampling=recommended_sampling,
    )
    _assert_required_artifacts(
        out_dir,
        target_base=target_base,
        lm_head_stem=lm_head_stem,
        num_chunks=total_output_chunks,
    )

    print("\nDone.")
    print(f"Prototype directory: {out_dir}")
    for idx, pkg in enumerate(built_chunk_pkgs, start=1):
        role = "FP32 attention-only" if idx == 1 else "post-attention segment"
        print(f"Chunk {idx}: {pkg.name} ({role})")
    print(f"Run test:")
    print(f"  python tests/chat.py --meta {out_dir / 'meta.yaml'} --prompt \"2+2=\" --max-tokens 32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
