#!/usr/bin/env python3
"""Decode across state-transition context exports by growing KV cache state.

This runner uses per-context chunked exports (e.g. ctx512/1024/2048/3072/4096)
and transitions KV state upward with anemll.utils.state_transition.transition_kv_state
as token count grows.

It also supports combined multi-context exports via --meta (single folder containing
infer_ctx*/prefill_ctx* functions, including split/no-alias layouts).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import coremltools as ct
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anemll.utils.state_transition import transition_kv_state

ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"
ANSI_YELLOW = "\033[93m"


def _colorize(text: str, color: str, stream) -> str:
    if stream is None or (not hasattr(stream, "isatty")) or (not stream.isatty()):
        return text
    return f"{color}{text}{ANSI_RESET}"


def _pick_log_stream(name: str):
    key = str(name).strip().lower()
    if key == "stderr":
        return sys.stderr
    if key == "stdout":
        return sys.stdout
    if key == "none":
        return None
    raise ValueError(f"Invalid --progress-stream: {name}")


def _log_line(stream, text: str = "", *, flush: bool = False, color: Optional[str] = None) -> None:
    if stream is None:
        return
    if color:
        text = _colorize(text, color, stream)
    print(text, file=stream, flush=flush)


def _parse_context_dir_entries(entries: Iterable[str]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"Invalid --context-dirs entry '{raw}'. Expected N=/path")
        lhs, rhs = raw.split("=", 1)
        ctx = int(lhs)
        p = Path(rhs).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        out[ctx] = p
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _resolve_context_dirs(
    *,
    contexts: List[int],
    root: Path,
    name_template: str,
    explicit_entries: Optional[List[str]],
) -> Dict[int, Path]:
    if explicit_entries:
        mapping = _parse_context_dir_entries(explicit_entries)
        missing = [c for c in contexts if c not in mapping]
        if missing:
            raise ValueError(f"Missing contexts in --context-dirs: {missing}")
        return {c: mapping[c] for c in contexts}

    out: Dict[int, Path] = {}
    for ctx in contexts:
        name = name_template.format(context=ctx)
        p = (root / name).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Missing context dir for {ctx}: {p}")
        out[ctx] = p
    return out


def _split_chunk_stem(ffn_value: str) -> Tuple[str, int]:
    stem = re.sub(r"\.(mlmodelc|mlpackage)$", "", str(ffn_value))
    m = re.match(r"^(.+)_chunk_(\d+)of(\d+)$", stem)
    if not m:
        raise ValueError(f"Could not parse chunk stem from ffn='{ffn_value}'")
    return m.group(1), int(m.group(3))


def _format_function_name(template: str, context: int) -> str:
    try:
        return str(template).format(context=int(context))
    except Exception:
        return str(template)


def _concat_logits(lm_out: Dict[str, np.ndarray]) -> np.ndarray:
    if "output_logits" in lm_out:
        return lm_out["output_logits"]
    parts: List[Tuple[int, np.ndarray]] = []
    for key, value in lm_out.items():
        m = re.fullmatch(r"logits(\d+)", key)
        if m:
            parts.append((int(m.group(1)), value))
    if not parts:
        raise RuntimeError(f"LM head output missing logits keys: {sorted(lm_out.keys())}")
    parts.sort(key=lambda x: x[0])
    return np.concatenate([v for _, v in parts], axis=-1)


def _compute_unit(name: str) -> ct.ComputeUnit:
    table = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    key = name.strip().upper()
    if key not in table:
        raise ValueError(f"Invalid compute unit '{name}'. Choose from: {sorted(table)}")
    return table[key]


def _detect_fp32_chunk1_bases(
    *,
    model_dir: Path,
    num_chunks: int,
    model_prefix: Optional[str],
) -> List[str]:
    bases: List[str] = []
    seen = set()
    # Match both *_FFN_attn_fp32_chunk_01of03 and *_FFN_attn_fp32_*_chunk_01of03
    # (the xstates combine inserts a suffix like '_statex' before '_chunk_').
    patterns = [
        f"*_FFN_attn_fp32_chunk_01of{num_chunks:02d}",
        f"*_FFN_attn_fp32_*_chunk_01of{num_chunks:02d}",
    ]

    for ext in ("mlmodelc", "mlpackage"):
        for pattern in patterns:
            for p in sorted(model_dir.glob(f"{pattern}.{ext}")):
                stem = p.name[: -len(f".{ext}")]
                m = re.match(r"^(.+)_chunk_(\d+)of(\d+)$", stem)
                if not m:
                    continue
                base = m.group(1)
                if base in seen:
                    continue
                seen.add(base)
                bases.append(base)

    if model_prefix:
        # Prefer exact prefix match, then prefix with any suffix
        preferred_exact = f"{model_prefix}_FFN_attn_fp32"
        preferred_any = [b for b in bases if b.startswith(f"{model_prefix}_FFN_attn_fp32")]
        if preferred_exact in bases:
            bases = [preferred_exact] + [b for b in bases if b != preferred_exact]
        elif preferred_any:
            rest = [b for b in bases if b not in preferred_any]
            bases = preferred_any + rest
    return bases


def _detect_prefill_fp32_chunk1_bases(
    *,
    model_dir: Path,
    num_chunks: int,
    model_prefix: Optional[str],
) -> List[str]:
    bases: List[str] = []
    seen = set()
    patterns = [
        f"*_prefill_attn_fp32_chunk_01of{num_chunks:02d}",
        f"*_prefill_attn_fp32_*_chunk_01of{num_chunks:02d}",
    ]

    for ext in ("mlmodelc", "mlpackage"):
        for pattern in patterns:
            for p in sorted(model_dir.glob(f"{pattern}.{ext}")):
                stem = p.name[: -len(f".{ext}")]
                m = re.match(r"^(.+)_chunk_(\d+)of(\d+)$", stem)
                if not m:
                    continue
                base = m.group(1)
                if base in seen:
                    continue
                seen.add(base)
                bases.append(base)

    if model_prefix:
        preferred_exact = f"{model_prefix}_prefill_attn_fp32"
        preferred_any = [b for b in bases if b.startswith(f"{model_prefix}_prefill_attn_fp32")]
        if preferred_exact in bases:
            bases = [preferred_exact] + [b for b in bases if b != preferred_exact]
        elif preferred_any:
            rest = [b for b in bases if b not in preferred_any]
            bases = preferred_any + rest
    return bases


def _make_causal_mask(context_length: int) -> np.ndarray:
    """Create a full causal mask [1,1,context,context] in fp16."""
    mask = np.full((1, 1, context_length, context_length), -np.inf, dtype=np.float16)
    row = np.arange(context_length).reshape(context_length, 1)
    col = np.arange(context_length).reshape(1, context_length)
    mask[:, :, col <= row] = 0.0
    return mask


def _load_compiled_model(path: Path, compute_unit: ct.ComputeUnit, preferred_fns: List[str]):
    errors: List[str] = []
    is_package = str(path).endswith(".mlpackage")
    for fn in preferred_fns:
        try:
            if is_package:
                cfg = ct.ComputeUnit
                m = ct.models.MLModel(str(path), compute_units=compute_unit, function_name=fn)
            else:
                m = ct.models.CompiledMLModel(str(path), compute_unit, function_name=fn)
            return m, fn
        except Exception as exc:
            errors.append(f"{fn}: {exc}")
    try:
        if is_package:
            m = ct.models.MLModel(str(path), compute_units=compute_unit)
        else:
            m = ct.models.CompiledMLModel(str(path), compute_unit)
        return m, "default"
    except Exception as exc:
        errors.append(f"default: {exc}")
    raise RuntimeError(f"Failed to load {path} with functions {preferred_fns}: {errors}")


@dataclass
class ContextRuntime:
    context: int
    model_dir: Path
    batch_size: int
    num_chunks: int
    infer_chunks: List[ct.models.CompiledMLModel]
    prefill_chunks: List[ct.models.CompiledMLModel]
    infer_pipeline: List[str]
    prefill_pipeline: List[str]
    infer_chunk1_fp32_enabled: bool = False
    prefill_chunk1_fp32_enabled: bool = False


@dataclass
class SamplingConfig:
    do_sample: bool
    temperature: float


def _resolve_sampling_config(
    *,
    sampling_mode: str,
    max_params: Dict,
    cli_temperature: Optional[float],
) -> SamplingConfig:
    rec = max_params.get("recommended_sampling", {})
    if not isinstance(rec, dict):
        rec = {}

    rec_do_sample = bool(rec.get("do_sample", False))
    rec_temperature = float(rec.get("temperature", 1.0))
    mode = sampling_mode.strip().lower()
    if mode not in {"auto", "greedy"}:
        raise ValueError("sampling_mode must be one of: auto, greedy")

    if mode == "auto":
        do_sample = rec_do_sample
    else:
        do_sample = False

    temperature = float(cli_temperature if cli_temperature is not None else rec_temperature)

    if temperature <= 0:
        temperature = 1.0

    return SamplingConfig(
        do_sample=do_sample,
        temperature=temperature,
    )


def _pick_next_token(logits_1d: np.ndarray, cfg: SamplingConfig, rng: np.random.Generator) -> int:
    if not cfg.do_sample:
        return int(np.argmax(logits_1d))

    logits = logits_1d.astype(np.float64)
    logits = logits / max(cfg.temperature, 1e-6)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0:
        return int(np.argmax(logits_1d))
    probs /= s

    s2 = float(np.sum(probs))
    if (not np.isfinite(s2)) or s2 <= 0:
        return int(np.argmax(logits_1d))
    probs /= s2
    return int(rng.choice(probs.size, p=probs))


def _build_context_runtime(
    *,
    context: int,
    model_dir: Path,
    compute_unit: ct.ComputeUnit,
    allow_mlpackage_fallback: bool,
    infer_chunk1_mode: str,
    log_stream,
) -> ContextRuntime:
    meta_path = model_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = yaml.safe_load(meta_path.read_text())
    params = meta.get("model_info", {}).get("parameters", {})
    if not isinstance(params, dict) or not params:
        raise ValueError(f"Invalid meta parameters in {meta_path}")

    ctx_from_meta = int(params.get("context_length", context))
    if ctx_from_meta != context:
        raise ValueError(
            f"Context mismatch in {model_dir}: requested={context}, meta={ctx_from_meta}"
        )

    batch_size = int(params.get("batch_size", 32))
    num_chunks = int(params.get("num_chunks", 1))
    ffn = str(params.get("ffn", "")).strip()
    if not ffn:
        raise ValueError(f"meta.yaml missing ffn in {model_dir}")

    chunk_base, total_chunks = _split_chunk_stem(ffn)
    if total_chunks != num_chunks:
        raise ValueError(
            f"num_chunks mismatch in {model_dir}: meta={num_chunks}, ffn={total_chunks}"
        )

    infer_chunks: List[ct.models.CompiledMLModel] = []
    prefill_chunks: List[ct.models.CompiledMLModel] = []
    infer_pipeline: List[str] = []
    prefill_pipeline: List[str] = []
    infer_chunk_bases = [chunk_base]
    prefill_chunk_bases = [chunk_base]
    model_prefix = str(params.get("model_prefix", "")).strip() or None

    # Backward/forward compatibility:
    # Some context exports write meta ffn as FFN_PF_* while artifacts are split into
    # FFN_* (infer) and prefill_* files.  Prefer split single-function models
    # (FFN for infer, prefill for prefill) since compiled multi-function
    # .mlmodelc files can't load with function_name.
    if "_FFN_PF" in chunk_base:
        split_infer_base = chunk_base.replace("_FFN_PF", "_FFN")
        split_prefill_base = chunk_base.replace("_FFN_PF", "_prefill")
        if split_infer_base not in infer_chunk_bases:
            infer_chunk_bases.insert(0, split_infer_base)
        if split_prefill_base not in prefill_chunk_bases:
            prefill_chunk_bases.insert(0, split_prefill_base)

    ffn_prefill = str(params.get("ffn_prefill", "")).strip()
    if ffn_prefill:
        try:
            prefill_base_from_meta, prefill_total_chunks = _split_chunk_stem(ffn_prefill)
            if prefill_total_chunks == num_chunks and prefill_base_from_meta not in prefill_chunk_bases:
                prefill_chunk_bases.insert(0, prefill_base_from_meta)
        except Exception:
            # Keep runtime robust even when optional prefill metadata is malformed.
            pass

    infer_chunk1_fp32_bases: List[str] = []
    if infer_chunk1_mode in ("auto", "on"):
        infer_chunk1_fp32_bases = _detect_fp32_chunk1_bases(
            model_dir=model_dir,
            num_chunks=num_chunks,
            model_prefix=model_prefix,
        )
        if infer_chunk1_mode == "on" and not infer_chunk1_fp32_bases:
            raise FileNotFoundError(
                f"Requested FP32 chunk1 infer path, but no *_FFN_attn_fp32_chunk_01of{num_chunks:02d} "
                f"artifact exists in {model_dir}"
            )
    prefill_chunk1_fp32_bases = _detect_prefill_fp32_chunk1_bases(
        model_dir=model_dir,
        num_chunks=num_chunks,
        model_prefix=model_prefix,
    )

    def _resolve_chunk_from_bases(
        *,
        bases: List[str],
        idx: int,
        num_chunks: int,
    ) -> tuple[Path, str]:
        package_only: List[str] = []
        for base in bases:
            stem = f"{base}_chunk_{idx:02d}of{num_chunks:02d}"
            p = model_dir / f"{stem}.mlmodelc"
            if p.exists():
                return p, stem
            pkg = model_dir / f"{stem}.mlpackage"
            if allow_mlpackage_fallback:
                if pkg.exists():
                    return pkg, stem
            elif pkg.exists():
                package_only.append(stem)
        stems = ", ".join(f"{base}_chunk_{idx:02d}of{num_chunks:02d}" for base in bases)
        if package_only:
            names = ", ".join(f"{s}.mlpackage" for s in package_only)
            raise FileNotFoundError(
                f"Missing compiled chunk artifact in {model_dir}. "
                f"Found package-only candidates: {names}. "
                "Re-run with --allow-mlpackage-fallback or compile to .mlmodelc."
            )
        raise FileNotFoundError(
            f"Missing chunk artifact candidates ({stems}) in {model_dir}"
        )

    for idx in range(1, num_chunks + 1):
        infer_attn_path: Optional[Path] = None
        infer_attn_fn: Optional[str] = None
        if idx == 1 and infer_chunk1_fp32_bases:
            infer_attn_path, _ = _resolve_chunk_from_bases(
                bases=infer_chunk1_fp32_bases, idx=idx, num_chunks=num_chunks
            )
            infer_attn_model, infer_attn_fn = _load_compiled_model(
                infer_attn_path, compute_unit, ["infer", "main"]
            )
            infer_chunks.append(infer_attn_model)
            infer_pipeline.append(f"{infer_attn_path.name}:{infer_attn_fn}")

        infer_bases_for_idx = infer_chunk_bases
        if idx == 1 and infer_chunk1_fp32_bases:
            infer_bases_for_idx = [b for b in infer_chunk_bases if b not in infer_chunk1_fp32_bases]
        infer_path, infer_stem = _resolve_chunk_from_bases(
            bases=infer_bases_for_idx, idx=idx, num_chunks=num_chunks
        )
        infer_model, infer_fn = _load_compiled_model(infer_path, compute_unit, ["infer", "main"])
        infer_chunks.append(infer_model)
        infer_pipeline.append(f"{infer_path.name}:{infer_fn}")

        prefill_attn_path: Optional[Path] = None
        prefill_attn_fn: Optional[str] = None
        if idx == 1 and prefill_chunk1_fp32_bases:
            prefill_attn_path, _ = _resolve_chunk_from_bases(
                bases=prefill_chunk1_fp32_bases, idx=idx, num_chunks=num_chunks
            )
            prefill_attn_model, prefill_attn_fn = _load_compiled_model(
                prefill_attn_path, compute_unit, ["prefill", "main", "infer"]
            )
            prefill_chunks.append(prefill_attn_model)
            prefill_pipeline.append(f"{prefill_attn_path.name}:{prefill_attn_fn}")

        prefill_path, prefill_stem = _resolve_chunk_from_bases(
            bases=prefill_chunk_bases, idx=idx, num_chunks=num_chunks
        )
        prefill_model, prefill_fn = _load_compiled_model(
            prefill_path, compute_unit, ["prefill", "main", "infer"]
        )
        prefill_chunks.append(prefill_model)
        prefill_pipeline.append(f"{prefill_path.name}:{prefill_fn}")

        if idx == 1:
            infer_attn_stage = ""
            if infer_attn_path is not None:
                infer_attn_stage = (
                    f"infer_attn_chunk={infer_attn_path.name} "
                    f"infer_attn_fn={infer_attn_fn} "
                )
            prefill_attn_stage = ""
            if prefill_attn_path is not None:
                prefill_attn_stage = (
                    f"prefill_attn_chunk={prefill_attn_path.name} "
                    f"prefill_attn_fn={prefill_attn_fn} "
                )
            _log_line(
                log_stream,
                (
                    f"[load] ctx{context}: {infer_attn_stage}infer_chunk={infer_path.name} infer_fn={infer_fn} "
                    f"{prefill_attn_stage}prefill_chunk={prefill_path.name} prefill_fn={prefill_fn}"
                ),
            )

    return ContextRuntime(
        context=context,
        model_dir=model_dir,
        batch_size=batch_size,
        num_chunks=num_chunks,
        infer_chunks=infer_chunks,
        prefill_chunks=prefill_chunks,
        infer_pipeline=infer_pipeline,
        prefill_pipeline=prefill_pipeline,
        infer_chunk1_fp32_enabled=bool(infer_chunk1_fp32_bases),
        prefill_chunk1_fp32_enabled=bool(prefill_chunk1_fp32_bases),
    )


def _resolve_chunk_artifact(
    *,
    model_dir: Path,
    stem: str,
    allow_mlpackage_fallback: bool,
) -> Path:
    p = model_dir / f"{stem}.mlmodelc"
    if p.exists():
        return p
    pkg = model_dir / f"{stem}.mlpackage"
    if pkg.exists() and not allow_mlpackage_fallback:
        raise FileNotFoundError(
            f"Missing compiled chunk artifact: {stem}.mlmodelc in {model_dir}; "
            f"found {stem}.mlpackage. Re-run with --allow-mlpackage-fallback "
            "or compile the package to .mlmodelc."
        )
    if allow_mlpackage_fallback and pkg.exists():
        return pkg
    raise FileNotFoundError(f"Missing chunk artifact: {stem} in {model_dir}")


def _build_context_runtimes_from_combined_meta(
    *,
    contexts: List[int],
    meta_path: Path,
    compute_unit: ct.ComputeUnit,
    allow_mlpackage_fallback: bool,
    log_stream,
) -> Tuple[List[ContextRuntime], Dict, Path]:
    meta = yaml.safe_load(meta_path.read_text())
    params = meta.get("model_info", {}).get("parameters", {})
    if not isinstance(params, dict) or not params:
        raise ValueError(f"Invalid meta parameters in {meta_path}")

    model_dir = meta_path.parent.resolve()
    batch_size = int(params.get("batch_size", 32))
    num_chunks = int(params.get("num_chunks", 1))

    infer_ffn = str(params.get("ffn", "")).strip()
    if not infer_ffn:
        raise ValueError(f"meta.yaml missing ffn in {model_dir}")
    infer_chunk_base, infer_total_chunks = _split_chunk_stem(infer_ffn)
    if infer_total_chunks != num_chunks:
        raise ValueError(
            f"num_chunks mismatch in {model_dir}: meta={num_chunks}, infer_ffn={infer_total_chunks}"
        )

    prefill_ffn = str(params.get("ffn_prefill", infer_ffn)).strip() or infer_ffn
    prefill_chunk_base, prefill_total_chunks = _split_chunk_stem(prefill_ffn)
    if prefill_total_chunks != num_chunks:
        raise ValueError(
            f"num_chunks mismatch in {model_dir}: meta={num_chunks}, prefill_ffn={prefill_total_chunks}"
        )

    infer_template = str(
        params.get("state_transition_infer_function_template", "infer_ctx{context}")
    ).strip()
    prefill_template = str(
        params.get("state_transition_prefill_function_template", "prefill_ctx{context}")
    ).strip()

    no_alias = bool(params.get("state_transition_no_alias_functions", False))
    has_all_prefill_ctx = bool(params.get("state_transition_all_context_prefill", False))
    prefill_ctx_from_meta = [int(c) for c in params.get("state_transition_prefill_contexts", [])]
    prefill_ctx_set = set(prefill_ctx_from_meta)
    prefill_default_context = int(params.get("state_transition_prefill_context", max(contexts)))

    infer_ctx_from_meta = params.get("state_transition_infer_contexts", [])
    if isinstance(infer_ctx_from_meta, list) and infer_ctx_from_meta:
        infer_ctx_set = {int(c) for c in infer_ctx_from_meta}
        missing = [c for c in contexts if c not in infer_ctx_set]
        if missing:
            raise ValueError(
                f"Requested contexts missing from state_transition_infer_contexts in {meta_path}: {missing}"
            )

    # Detect FP32 chunk1 artifacts (same logic as per-folder path).
    model_prefix = str(params.get("model_prefix", "")).strip() or None
    infer_chunk1_fp32_enabled_meta = bool(
        params.get("state_transition_chunk1_fp32_enabled", False)
    )
    prefill_chunk1_fp32_enabled_meta = bool(
        params.get("state_transition_prefill_chunk1_fp32_enabled", False)
    )
    infer_chunk1_fp32_bases: List[str] = []
    prefill_chunk1_fp32_bases: List[str] = []
    if infer_chunk1_fp32_enabled_meta:
        infer_chunk1_fp32_bases = _detect_fp32_chunk1_bases(
            model_dir=model_dir,
            num_chunks=num_chunks,
            model_prefix=model_prefix,
        )
    if prefill_chunk1_fp32_enabled_meta:
        prefill_chunk1_fp32_bases = _detect_prefill_fp32_chunk1_bases(
            model_dir=model_dir,
            num_chunks=num_chunks,
            model_prefix=model_prefix,
        )

    def _resolve_fp32_chunk_from_bases(
        bases: List[str], num_chunks: int
    ) -> Path:
        """Resolve FP32 chunk1 artifact from detected bases."""
        for base in bases:
            stem = f"{base}_chunk_01of{num_chunks:02d}"
            p = model_dir / f"{stem}.mlmodelc"
            if p.exists():
                return p
            pkg = model_dir / f"{stem}.mlpackage"
            if allow_mlpackage_fallback and pkg.exists():
                return pkg
        stems = ", ".join(f"{b}_chunk_01of{num_chunks:02d}" for b in bases)
        raise FileNotFoundError(
            f"Missing FP32 chunk1 artifact candidates ({stems}) in {model_dir}"
        )

    runtimes: List[ContextRuntime] = []
    for context in contexts:
        infer_chunks: List[ct.models.CompiledMLModel] = []
        prefill_chunks: List[ct.models.CompiledMLModel] = []
        infer_pipeline: List[str] = []
        prefill_pipeline: List[str] = []

        infer_fn = _format_function_name(infer_template, context)

        if has_all_prefill_ctx and (not prefill_ctx_set or context in prefill_ctx_set):
            prefill_fn = _format_function_name(prefill_template, context)
        else:
            prefill_fn = str(
                params.get(
                    "state_transition_prefill_default_function",
                    _format_function_name(prefill_template, prefill_default_context),
                )
            )

        infer_fn_preferences = [infer_fn]
        prefill_fn_preferences = [prefill_fn]
        if not no_alias:
            infer_fn_preferences.append("infer")
            prefill_fn_preferences.append("prefill")
        infer_fn_preferences.append("main")
        prefill_fn_preferences.extend(["main", "infer"])

        # Preserve order while deduping
        seen_infer = set()
        infer_fn_preferences = [
            fn for fn in infer_fn_preferences if not (fn in seen_infer or seen_infer.add(fn))
        ]
        seen_prefill = set()
        prefill_fn_preferences = [
            fn
            for fn in prefill_fn_preferences
            if not (fn in seen_prefill or seen_prefill.add(fn))
        ]

        for idx in range(1, num_chunks + 1):
            # Prepend FP32 attn chunk before regular chunk 1 (mirrors per-folder path).
            if idx == 1 and infer_chunk1_fp32_bases:
                fp32_path = _resolve_fp32_chunk_from_bases(
                    infer_chunk1_fp32_bases, num_chunks
                )
                fp32_infer_model, fp32_infer_fn = _load_compiled_model(
                    fp32_path, compute_unit, infer_fn_preferences
                )
                infer_chunks.append(fp32_infer_model)
                infer_pipeline.append(f"{fp32_path.name}:{fp32_infer_fn}")

                # Same FP32 attn chunk also has prefill functions — prepend to prefill too.
                try:
                    fp32_prefill_model, fp32_prefill_fn = _load_compiled_model(
                        fp32_path, compute_unit, prefill_fn_preferences
                    )
                    prefill_chunks.append(fp32_prefill_model)
                    prefill_pipeline.append(f"{fp32_path.name}:{fp32_prefill_fn}")
                except RuntimeError:
                    pass  # FP32 chunk has no prefill functions — prefill stays 3 stages

            elif idx == 1 and prefill_chunk1_fp32_bases:
                fp32_prefill_path = _resolve_fp32_chunk_from_bases(
                    prefill_chunk1_fp32_bases, num_chunks
                )
                fp32_prefill_model, fp32_prefill_fn = _load_compiled_model(
                    fp32_prefill_path, compute_unit, prefill_fn_preferences
                )
                prefill_chunks.append(fp32_prefill_model)
                prefill_pipeline.append(f"{fp32_prefill_path.name}:{fp32_prefill_fn}")

            infer_stem = f"{infer_chunk_base}_chunk_{idx:02d}of{num_chunks:02d}"
            prefill_stem = f"{prefill_chunk_base}_chunk_{idx:02d}of{num_chunks:02d}"
            infer_path = _resolve_chunk_artifact(
                model_dir=model_dir,
                stem=infer_stem,
                allow_mlpackage_fallback=allow_mlpackage_fallback,
            )
            prefill_path = _resolve_chunk_artifact(
                model_dir=model_dir,
                stem=prefill_stem,
                allow_mlpackage_fallback=allow_mlpackage_fallback,
            )

            infer_model, infer_fn_loaded = _load_compiled_model(
                infer_path, compute_unit, infer_fn_preferences
            )
            prefill_model, prefill_fn_loaded = _load_compiled_model(
                prefill_path, compute_unit, prefill_fn_preferences
            )
            infer_chunks.append(infer_model)
            prefill_chunks.append(prefill_model)
            infer_pipeline.append(f"{infer_path.name}:{infer_fn_loaded}")
            prefill_pipeline.append(f"{prefill_path.name}:{prefill_fn_loaded}")

            if idx == 1:
                fp32_infer_stage = ""
                if infer_chunk1_fp32_bases:
                    fp32_infer_stage = (
                        f"fp32_attn={infer_pipeline[0]} "
                    )
                fp32_prefill_stage = ""
                if prefill_chunk1_fp32_bases:
                    fp32_prefill_stage = (
                        f"fp32_prefill_attn={prefill_pipeline[0]} "
                    )
                _log_line(
                    log_stream,
                    (
                        f"[load] ctx{context}: {fp32_infer_stage}infer_chunk={infer_path.name} "
                        f"infer_fn={infer_fn_loaded} {fp32_prefill_stage}prefill_chunk={prefill_path.name} "
                        f"prefill_fn={prefill_fn_loaded}"
                    ),
                )

        runtimes.append(
            ContextRuntime(
                context=context,
                model_dir=model_dir,
                batch_size=batch_size,
                num_chunks=num_chunks,
                infer_chunks=infer_chunks,
                prefill_chunks=prefill_chunks,
                infer_pipeline=infer_pipeline,
                prefill_pipeline=prefill_pipeline,
                infer_chunk1_fp32_enabled=bool(infer_chunk1_fp32_bases),
                prefill_chunk1_fp32_enabled=bool(prefill_chunk1_fp32_bases),
            )
        )

    return runtimes, params, model_dir


def _resolve_shared_model_path(model_dir: Path, value: str) -> Path:
    stem = re.sub(r"\.(mlmodelc|mlpackage)$", "", value)
    p = model_dir / f"{stem}.mlmodelc"
    if p.exists():
        return p
    p = model_dir / f"{stem}.mlpackage"
    if p.exists():
        return p
    raise FileNotFoundError(f"Shared model not found for '{value}' in {model_dir}")


def _detect_state_name(state_obj, explicit: Optional[str]) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(["model_model_kv_cache_0", "kv_cache", "model.model.kv_cache_0"])

    tried = []
    for name in candidates:
        if name in tried:
            continue
        tried.append(name)
        try:
            _ = state_obj.read_state(name)
            return name
        except Exception:
            continue
    raise RuntimeError(f"Could not detect state name. Tried: {tried}")


def _prefill_with_prefill_fn(
    *,
    runtime: ContextRuntime,
    embed_model,
    state,
    input_ids: np.ndarray,
    full_causal_mask: np.ndarray,
    log_stream=None,
) -> None:
    seq_len = int(input_ids.shape[1])
    batch = max(1, int(runtime.batch_size))
    start = 0
    batch_prefill_supported = True

    # Run prefill only on full-size batches (avoids enumerated-shape failures).
    while start + batch <= seq_len and batch_prefill_supported:
        end = start + batch
        token_batch = input_ids[:, start:end].astype(np.int32)
        try:
            hidden = embed_model.predict({"input_ids": token_batch})["hidden_states"]
        except RuntimeError as exc:
            msg = str(exc)
            if "MultiArray Shape" in msg and "enumerated set of allowed shapes" in msg:
                batch_prefill_supported = False
                _log_line(
                    log_stream,
                    (
                        f"[prefill] batch mode unsupported at shape {token_batch.shape}; "
                        "falling back to token-infer prefill."
                    ),
                )
                break
            raise

        position_ids = np.arange(start, end, dtype=np.int32)
        current_pos = np.array([start], dtype=np.int32)
        causal_mask = full_causal_mask[:, :, start:end, :]

        for chunk in runtime.prefill_chunks:
            out = chunk.predict(
                {
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": position_ids,
                    "causal_mask": causal_mask,
                    "current_pos": current_pos,
                },
                state,
            )
            hidden = out["output_hidden_states"]
        start = end

    # Process remainder tokens one-by-one via infer path.
    if start < seq_len:
        for pos in range(start, seq_len):
            token = input_ids[:, pos : pos + 1].astype(np.int32)
            hidden = embed_model.predict({"input_ids": token})["hidden_states"]
            position_ids = np.array([pos], dtype=np.int32)
            current_pos = np.array([pos], dtype=np.int32)
            causal_mask = full_causal_mask[:, :, pos : pos + 1, :]
            for chunk in runtime.infer_chunks:
                out = chunk.predict(
                    {
                        "hidden_states": hidden.astype(np.float16),
                        "position_ids": position_ids,
                        "causal_mask": causal_mask,
                        "current_pos": current_pos,
                    },
                    state,
                )
                hidden = out["output_hidden_states"]


def _prefill_with_infer_fn(
    *,
    runtime: ContextRuntime,
    embed_model,
    state,
    input_ids: np.ndarray,
    full_causal_mask: np.ndarray,
) -> None:
    seq_len = int(input_ids.shape[1])
    for pos in range(seq_len):
        token = input_ids[:, pos : pos + 1].astype(np.int32)
        hidden = embed_model.predict({"input_ids": token})["hidden_states"]
        position_ids = np.array([pos], dtype=np.int32)
        current_pos = np.array([pos], dtype=np.int32)
        causal_mask = full_causal_mask[:, :, pos : pos + 1, :]

        for chunk in runtime.infer_chunks:
            out = chunk.predict(
                {
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": position_ids,
                    "causal_mask": causal_mask,
                    "current_pos": current_pos,
                },
                state,
            )
            hidden = out["output_hidden_states"]


def _decode_step(
    *,
    runtime: ContextRuntime,
    embed_model,
    lm_head_model,
    state,
    token_id: int,
    pos: int,
    full_causal_mask: np.ndarray,
    sampling_cfg: SamplingConfig,
    rng: np.random.Generator,
) -> int:
    token = np.array([[token_id]], dtype=np.int32)
    hidden = embed_model.predict({"input_ids": token})["hidden_states"]
    position_ids = np.array([pos], dtype=np.int32)
    current_pos = np.array([pos], dtype=np.int32)
    causal_mask = full_causal_mask[:, :, pos : pos + 1, :]

    for chunk in runtime.infer_chunks:
        out = chunk.predict(
            {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": position_ids,
                "causal_mask": causal_mask,
                "current_pos": current_pos,
            },
            state,
        )
        hidden = out["output_hidden_states"]

    lm_out = lm_head_model.predict({"hidden_states": hidden.astype(np.float16)})
    logits = _concat_logits(lm_out)
    return _pick_next_token(logits[0, -1, :], sampling_cfg, rng)


def _transition_state(
    *,
    source_state,
    source_state_name: str,
    target_runtime: ContextRuntime,
    valid_tokens: int,
) -> Tuple[object, str]:
    src = source_state.read_state(source_state_name)

    target_state = target_runtime.infer_chunks[0].make_state()
    target_state_name = _detect_state_name(target_state, source_state_name)
    target_probe = target_state.read_state(target_state_name)

    transitioned = transition_kv_state(
        source_state=src,
        target_seq_length=int(target_probe.shape[2]),
        current_position=int(valid_tokens),
        pad_value=0.0,
    )
    target_state.write_state(target_state_name, transitioned)
    return target_state, target_state_name


def _compute_shift_keep_tokens(
    *,
    context_length: int,
    batch_size: int,
    valid_tokens: int,
    reserve_batches: int,
) -> int:
    """Compute number of tail tokens to keep during shift-refill overflow handling."""
    if valid_tokens <= 1:
        return valid_tokens

    batch = max(1, int(batch_size))
    max_batches = max(1, int(context_length) // batch)
    desired_batches = max(1, max_batches - max(0, int(reserve_batches)))
    keep_tokens = desired_batches * batch

    # Keep room for at least one new token after refill.
    max_keep = int(context_length) - min(batch, max(1, int(context_length) - 1))
    max_keep = max(1, max_keep)

    keep_tokens = min(int(valid_tokens), int(context_length), keep_tokens, max_keep)
    keep_tokens = max(1, keep_tokens)
    return keep_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--contexts",
        default="512,1024,2048,3072,4096",
        help=(
            "Comma/space-separated contexts. "
            "If --meta is set and --contexts is not explicitly provided, "
            "state_transition_infer_contexts from meta.yaml is used."
        ),
    )
    ap.add_argument(
        "--meta",
        default=None,
        help=(
            "Optional combined model meta.yaml path. "
            "When set, contexts are loaded from one shared model directory."
        ),
    )
    ap.add_argument("--contexts-root", default="/Volumes/Models/ANE", help="Root directory for per-context model folders.")
    ap.add_argument(
        "--name-template",
        default="vibethinker_1.5b_ctx{context}_L6_4_hybrid",
        help="Folder template under --contexts-root when --context-dirs is not set.",
    )
    ap.add_argument(
        "--context-dirs",
        nargs="+",
        default=None,
        help="Optional explicit context dirs, entries like: 512=/path 1024=/path ...",
    )
    ap.add_argument("--tokenizer", default=None, help="Tokenizer path override")
    ap.add_argument("--prompt", default="What is the capital of France?", help="Input prompt text.")
    ap.add_argument("--max-tokens", type=int, default=256, help="Maximum decode tokens to generate.")
    ap.add_argument(
        "--max-time",
        type=float,
        default=None,
        help=(
            "Maximum wall time in seconds from prefill start "
            "(includes prefill, decode, transitions, and compactions)."
        ),
    )
    ap.add_argument(
        "--max-context-size",
        "--max-active-context",
        dest="max_context_size",
        type=int,
        default=None,
        help=(
            "Optional cap for active context growth. "
            "Only contexts <= this value are used (e.g., 2048 or 3072)."
        ),
    )
    ap.add_argument(
        "--prefill-mode",
        choices=["batch-prefill", "token-infer"],
        default="batch-prefill",
        help=(
            "Prefill strategy: 'batch-prefill' uses prefill chunks in batch_size batches, "
            "'token-infer' prefills token-by-token via infer chunks. "
            "Auto-switches to token-infer when FP32 infer chunk1 is present but no FP32 prefill chunk."
        ),
    )
    ap.add_argument(
        "--compute-unit",
        default="CPU_AND_NE",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
        help="CoreML compute unit for model loading.",
    )
    ap.add_argument(
        "--allow-mlpackage-fallback",
        action="store_true",
        help="Allow loading .mlpackage when .mlmodelc is missing (default: false, compiled-only).",
    )
    ap.add_argument(
        "--per-context-infer-chunk1",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Per-folder mode only: chunk1 infer source selection. "
            "'auto' prefers *_FFN_attn_fp32* when present, "
            "'on' requires FP32 chunk1, 'off' uses regular infer chunks."
        ),
    )
    ap.add_argument("--state-name", default=None, help="Override KV cache state name (auto-detected by default).")
    ap.add_argument(
        "--sampling-mode",
        choices=["auto", "greedy"],
        default="auto",
        help=(
            "Decode policy: auto=use meta.yaml recommended_sampling.do_sample, "
            "greedy=argmax."
        ),
    )
    ap.add_argument("--temperature", type=float, default=None, help="Sampling temperature override for auto mode.")
    ap.add_argument("--seed", type=int, default=None, help="Sampling RNG seed.")
    ap.add_argument(
        "--overflow-policy",
        choices=["stop", "shift-refill"],
        default="stop",
        help=(
            "Behavior when decode exceeds largest context. "
            "'stop' ends generation, 'shift-refill' keeps a tail window and refills state."
        ),
    )
    ap.add_argument(
        "--overflow-reserve-batches",
        type=int,
        default=2,
        help=(
            "For --overflow-policy shift-refill: keep (max_batches - reserve) batches "
            "of recent tokens before refilling state."
        ),
    )
    ap.add_argument(
        "--overflow-preserve-prompt",
        action="store_true",
        help=(
            "With --overflow-policy shift-refill, preserve original prompt tokens "
            "as a fixed prefix on each compact/refill."
        ),
    )
    ap.add_argument("--no-think", action="store_true", help="Pass enable_thinking=False to chat template.")
    ap.add_argument("--no-eos-stop", action="store_true", help="Don't stop on EOS token (continue generating).")
    ap.add_argument(
        "--progress-stream",
        choices=["stderr", "stdout", "none"],
        default="stderr",
        help=(
            "Where to print system/progress logs. "
            "Decode tokens are always streamed to stdout."
        ),
    )
    ap.add_argument(
        "--live-events",
        action="store_true",
        help="Print transition/compact/stop events live during decode (default: summary-only).",
    )
    args = ap.parse_args()
    if args.max_time is not None and float(args.max_time) <= 0:
        raise ValueError(f"Invalid --max-time: {args.max_time}")
    if args.max_tokens is not None and int(args.max_tokens) <= 0:
        raise ValueError(f"Invalid --max-tokens: {args.max_tokens}")

    user_set_max_tokens = any(
        a == "--max-tokens" or a.startswith("--max-tokens=") for a in sys.argv[1:]
    )
    user_set_contexts = any(
        a == "--contexts" or a.startswith("--contexts=") for a in sys.argv[1:]
    )
    # If max-time is explicitly set and max-tokens is not, let time budget govern.
    token_cap_disabled = bool(args.max_time is not None and not user_set_max_tokens)
    effective_max_tokens = (2**31 - 1) if token_cap_disabled else int(args.max_tokens)
    log_stream = _pick_log_stream(args.progress_stream)

    contexts = [int(x) for x in args.contexts.replace(",", " ").split() if x.strip()]
    if args.meta and (not user_set_contexts):
        combined_meta_path_probe = Path(args.meta).expanduser().resolve()
        if not combined_meta_path_probe.exists():
            raise FileNotFoundError(combined_meta_path_probe)
        meta_probe = yaml.safe_load(combined_meta_path_probe.read_text())
        params_probe = meta_probe.get("model_info", {}).get("parameters", {})
        if isinstance(params_probe, dict):
            meta_contexts = params_probe.get("state_transition_infer_contexts", [])
            if isinstance(meta_contexts, list) and meta_contexts:
                contexts = [int(c) for c in meta_contexts]
                _log_line(log_stream, f"[plan] contexts from meta: {contexts}")
    contexts = sorted(dict.fromkeys(contexts))
    if not contexts:
        raise ValueError("No contexts parsed")
    if args.max_context_size is not None:
        max_ctx_cap = int(args.max_context_size)
        if max_ctx_cap <= 0:
            raise ValueError(f"Invalid --max-context-size: {args.max_context_size}")
        capped_contexts = [c for c in contexts if c <= max_ctx_cap]
        if not capped_contexts:
            raise ValueError(
                f"--max-context-size={max_ctx_cap} excludes all contexts from {contexts}"
            )
        contexts = capped_contexts

    cu = _compute_unit(args.compute_unit)
    combined_meta_path: Optional[Path] = None
    max_params: Dict = {}
    if args.meta:
        combined_meta_path = Path(args.meta).expanduser().resolve()
        if not combined_meta_path.exists():
            raise FileNotFoundError(combined_meta_path)
        runtimes, max_params, combined_model_dir = _build_context_runtimes_from_combined_meta(
            contexts=contexts,
            meta_path=combined_meta_path,
            compute_unit=cu,
            allow_mlpackage_fallback=args.allow_mlpackage_fallback,
            log_stream=log_stream,
        )
    else:
        context_dirs = _resolve_context_dirs(
            contexts=contexts,
            root=Path(args.contexts_root).expanduser().resolve(),
            name_template=args.name_template,
            explicit_entries=args.context_dirs,
        )

        runtimes = []
        for ctx in contexts:
            runtimes.append(
                _build_context_runtime(
                    context=ctx,
                    model_dir=context_dirs[ctx],
                    compute_unit=cu,
                    allow_mlpackage_fallback=args.allow_mlpackage_fallback,
                    infer_chunk1_mode=args.per_context_infer_chunk1,
                    log_stream=log_stream,
                )
            )

        max_runtime = runtimes[-1]
        max_meta = yaml.safe_load((max_runtime.model_dir / "meta.yaml").read_text())
        max_params = max_meta.get("model_info", {}).get("parameters", {})

    expected_chunks = runtimes[0].num_chunks
    for rt in runtimes:
        if rt.num_chunks != expected_chunks:
            raise ValueError(
                f"num_chunks mismatch across contexts: {runtimes[0].context}->{expected_chunks}, "
                f"ctx{rt.context}->{rt.num_chunks}"
            )

    max_runtime = runtimes[-1]
    sampling_cfg = _resolve_sampling_config(
        sampling_mode=args.sampling_mode,
        max_params=max_params,
        cli_temperature=args.temperature,
    )
    rng = np.random.default_rng(args.seed)

    embed_path = _resolve_shared_model_path(max_runtime.model_dir, str(max_params["embeddings"]))
    lm_head_path = _resolve_shared_model_path(max_runtime.model_dir, str(max_params["lm_head"]))

    embed_model, _ = _load_compiled_model(embed_path, cu, ["main"])
    lm_head_model, lm_fn = _load_compiled_model(lm_head_path, cu, ["main", "infer"])
    _log_line(log_stream, f"[load] lm_head function={lm_fn}")

    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    elif isinstance(max_params.get("tokenizer_path"), str) and str(max_params.get("tokenizer_path")).strip():
        tokenizer_path = Path(str(max_params.get("tokenizer_path"))).expanduser().resolve()
    else:
        tokenizer_path = max_runtime.model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        use_fast=False,
        trust_remote_code=True,
    )
    causal_masks_by_context = {
        rt.context: _make_causal_mask(rt.context) for rt in runtimes
    }

    tpl_kwargs = {"enable_thinking": False} if args.no_think else {}
    input_ids_t = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        **tpl_kwargs,
    ).to(torch.int32)
    input_ids = input_ids_t.cpu().numpy().astype(np.int32)

    prompt_tokens = int(input_ids.shape[1])
    if prompt_tokens < 1:
        raise ValueError("Prompt tokenization produced empty input")

    start_idx = None
    for i, rt in enumerate(runtimes):
        if prompt_tokens <= rt.context:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(
            f"Prompt length ({prompt_tokens}) exceeds largest context ({runtimes[-1].context})"
        )

    active_idx = int(start_idx)
    active = runtimes[active_idx]

    # Check infer vs prefill chunk count parity (FP32 attn prepend).
    # If both pipelines have the same stage count, FP32 attn is shared — no action needed.
    if runtimes:
        rt0 = runtimes[0]
        if len(rt0.infer_chunks) != len(rt0.prefill_chunks):
            prefill_mode_explicit = any(
                a.startswith("--prefill-mode") or a.startswith("--prefill_mode")
                for a in sys.argv[1:]
            )
            if args.prefill_mode == "batch-prefill":
                if prefill_mode_explicit:
                    _log_line(
                        log_stream,
                        (
                            f"[warn] infer has {len(rt0.infer_chunks)} stages but "
                            f"prefill has {len(rt0.prefill_chunks)} stages "
                            "(FP32 attn chunk not in prefill pipeline)."
                        ),
                        color=ANSI_YELLOW,
                    )
                else:
                    args.prefill_mode = "token-infer"
                    _log_line(
                        log_stream,
                        (
                            f"[auto] prefill_mode -> token-infer "
                            f"(infer has {len(rt0.infer_chunks)} stages, "
                            f"prefill has {len(rt0.prefill_chunks)})"
                        ),
                        color=ANSI_YELLOW,
                    )

    _log_line(
        log_stream,
        f"[plan] contexts={contexts} prompt_tokens={prompt_tokens} "
        f"start_ctx={active.context} prefill_mode={args.prefill_mode}",
    )
    if args.max_context_size is not None:
        _log_line(log_stream, f"[plan] max_context_cap={int(args.max_context_size)}")
    if token_cap_disabled:
        _log_line(log_stream, "[plan] max_tokens=unbounded (time-limited)")
    else:
        _log_line(log_stream, f"[plan] max_tokens={effective_max_tokens}")
    if args.max_time is not None:
        _log_line(log_stream, f"[plan] max_time={float(args.max_time):.1f}s")
    for i, rt in enumerate(runtimes):
        infer_chain = " -> ".join(rt.infer_pipeline) if rt.infer_pipeline else "(empty)"
        prefill_chain = " -> ".join(rt.prefill_pipeline) if rt.prefill_pipeline else "(empty)"
        if i == 0:
            # Full pipeline readout for first context
            _log_line(log_stream, f"[pipeline] ctx{rt.context} infer ({len(rt.infer_chunks)} stages): {infer_chain}")
            _log_line(log_stream, f"[pipeline] ctx{rt.context} prefill ({len(rt.prefill_chunks)} stages): {prefill_chain}")
        else:
            # Compact: just context + function names
            infer_fns = [entry.split(":")[-1] for entry in rt.infer_pipeline]
            prefill_fns = [entry.split(":")[-1] for entry in rt.prefill_pipeline]
            _log_line(log_stream, f"[pipeline] ctx{rt.context} infer: {','.join(infer_fns)}  prefill: {','.join(prefill_fns)}")
    if sampling_cfg.do_sample:
        _log_line(
            log_stream,
            f"[sampling] mode=sample temperature={sampling_cfg.temperature} "
            f"seed={'random' if args.seed is None else args.seed}",
        )
    else:
        _log_line(log_stream, "[sampling] mode=greedy")
    state = active.prefill_chunks[0].make_state()
    state_name = _detect_state_name(state, args.state_name)
    _log_line(log_stream, f"[state] using state_name='{state_name}'")

    prefill_start = time.perf_counter()
    if args.prefill_mode == "batch-prefill":
        _prefill_with_prefill_fn(
            runtime=active,
            embed_model=embed_model,
            state=state,
            input_ids=input_ids,
            full_causal_mask=causal_masks_by_context[active.context],
            log_stream=log_stream,
        )
    else:
        _prefill_with_infer_fn(
            runtime=active,
            embed_model=embed_model,
            state=state,
            input_ids=input_ids,
            full_causal_mask=causal_masks_by_context[active.context],
        )
    prefill_s = time.perf_counter() - prefill_start
    max_time_limit_s = float(args.max_time) if args.max_time is not None else None
    max_time_hit = False
    if max_time_limit_s is not None and prefill_s >= max_time_limit_s:
        max_time_hit = True
        if args.live_events:
            _log_line(
                log_stream,
                f"\n[stop] max-time reached during prefill "
                f"({prefill_s:.1f}s >= {max_time_limit_s:.1f}s)",
                flush=True,
            )

    valid_tokens = prompt_tokens
    last_token = int(input_ids[0, prompt_tokens - 1])
    prompt_anchor_tokens: List[int] = [int(x) for x in input_ids[0, :prompt_tokens].tolist()]
    token_history: List[int] = list(prompt_anchor_tokens)
    preserve_prompt_truncated_warned = False

    generated: List[int] = []
    decode_stats = {rt.context: {"tokens": 0, "time": 0.0} for rt in runtimes}
    transitions: List[Tuple[int, int, int, float, float]] = []
    compactions: List[Tuple[int, int, int, float, float]] = []

    if args.live_events:
        _log_line(log_stream, "\n[decode]", flush=True)
    stop_reason = "running"
    for _ in range(effective_max_tokens):
        if max_time_limit_s is not None:
            elapsed_s = time.perf_counter() - prefill_start
            if elapsed_s >= max_time_limit_s:
                max_time_hit = True
                stop_reason = "max-time"
                if args.live_events:
                    _log_line(
                        log_stream,
                        f"\n[stop] max-time reached ({elapsed_s:.1f}s >= {max_time_limit_s:.1f}s)",
                        flush=True,
                    )
                break

        while active_idx + 1 < len(runtimes) and valid_tokens >= active.context:
            next_runtime = runtimes[active_idx + 1]
            decoded_so_far = len(generated)
            decode_s_so_far = sum(v["time"] for v in decode_stats.values())
            avg_decode_tps_so_far = (
                decoded_so_far / decode_s_so_far if decode_s_so_far > 0 else 0.0
            )
            t0 = time.perf_counter()
            state, state_name = _transition_state(
                source_state=state,
                source_state_name=state_name,
                target_runtime=next_runtime,
                valid_tokens=valid_tokens,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            transitions.append(
                (
                    active.context,
                    next_runtime.context,
                    valid_tokens,
                    dt,
                    avg_decode_tps_so_far,
                )
            )
            transition_msg = (
                f"\n[transition] ctx{active.context} -> ctx{next_runtime.context} "
                f"at tokens={valid_tokens} ({dt:.1f} ms, avg decode {avg_decode_tps_so_far:.1f} t/s)"
            )
            if args.live_events:
                _log_line(log_stream, transition_msg, flush=True, color=ANSI_CYAN)
            active_idx += 1
            active = next_runtime

        if valid_tokens > runtimes[-1].context:
            if args.overflow_policy == "shift-refill" and active_idx == (len(runtimes) - 1):
                keep_tokens = _compute_shift_keep_tokens(
                    context_length=active.context,
                    batch_size=active.batch_size,
                    valid_tokens=valid_tokens,
                    reserve_batches=args.overflow_reserve_batches,
                )

                if args.overflow_preserve_prompt:
                    max_anchor = max(1, int(active.context) - 1)
                    anchor_tokens = prompt_anchor_tokens[:max_anchor]
                    if (
                        (len(prompt_anchor_tokens) > max_anchor)
                        and (not preserve_prompt_truncated_warned)
                    ):
                        if args.live_events:
                            _log_line(
                                log_stream,
                                f"\n[compact] prompt truncated for anchor preserve: "
                                f"{len(prompt_anchor_tokens)} -> {max_anchor}",
                                flush=True,
                                color=ANSI_YELLOW,
                            )
                        preserve_prompt_truncated_warned = True

                    keep_tokens = max(keep_tokens, len(anchor_tokens))
                    tail_budget = max(0, keep_tokens - len(anchor_tokens))
                    if token_history[: len(anchor_tokens)] == anchor_tokens:
                        tail_pool = token_history[len(anchor_tokens) :]
                    else:
                        tail_pool = token_history
                    tail_tokens = tail_pool[-tail_budget:] if tail_budget > 0 else []
                    refill_tokens = anchor_tokens + tail_tokens
                else:
                    refill_tokens = token_history[-keep_tokens:]

                drop_tokens = valid_tokens - len(refill_tokens)
                if drop_tokens <= 0:
                    stop_reason = "overflow-no-drop"
                    if args.live_events:
                        _log_line(
                            log_stream,
                            "\n[stop] overflow shift computed no drop; stopping",
                            flush=True,
                        )
                    break

                decoded_so_far = len(generated)
                decode_s_so_far = sum(v["time"] for v in decode_stats.values())
                avg_decode_tps_so_far = (
                    decoded_so_far / decode_s_so_far if decode_s_so_far > 0 else 0.0
                )

                token_history = refill_tokens
                refill_ids = np.array([token_history], dtype=np.int32)
                t0 = time.perf_counter()
                state = active.prefill_chunks[0].make_state()
                state_name = _detect_state_name(state, state_name)
                if args.prefill_mode == "batch-prefill":
                    _prefill_with_prefill_fn(
                        runtime=active,
                        embed_model=embed_model,
                        state=state,
                        input_ids=refill_ids,
                        full_causal_mask=causal_masks_by_context[active.context],
                        log_stream=log_stream,
                    )
                else:
                    _prefill_with_infer_fn(
                        runtime=active,
                        embed_model=embed_model,
                        state=state,
                        input_ids=refill_ids,
                        full_causal_mask=causal_masks_by_context[active.context],
                    )
                compact_ms = (time.perf_counter() - t0) * 1000.0
                transition_ms_so_far = sum(ms for _, _, _, ms, _ in transitions)
                prior_compact_ms_so_far = sum(ms for _, _, _, ms, _ in compactions)
                infer_elapsed_s = (
                    decode_s_so_far
                    + (transition_ms_so_far / 1000.0)
                    + ((prior_compact_ms_so_far + compact_ms) / 1000.0)
                )
                total_elapsed_s = time.perf_counter() - prefill_start

                valid_tokens = keep_tokens
                last_token = int(token_history[-1])
                compactions.append(
                    (
                        active.context,
                        drop_tokens,
                        keep_tokens,
                        compact_ms,
                        avg_decode_tps_so_far,
                    )
                )
                compact_msg = (
                    f"\n[compact] ctx{active.context} drop={drop_tokens} keep={keep_tokens} "
                    f"({compact_ms:.1f} ms, avg decode {avg_decode_tps_so_far:.1f} t/s, "
                    f"infer_elapsed {infer_elapsed_s:.1f}s, total_elapsed {total_elapsed_s:.1f}s)"
                )
                if args.live_events:
                    _log_line(log_stream, compact_msg, flush=True, color=ANSI_YELLOW)
                continue

            stop_reason = "exceeded-largest-capacity"
            if args.live_events:
                _log_line(log_stream, "\n[stop] exceeded largest state capacity", flush=True)
            break

        pos = valid_tokens - 1
        if pos >= active.context:
            stop_reason = "current-pos-out-of-range"
            if args.live_events:
                _log_line(log_stream, "\n[stop] current_pos out of range for active context", flush=True)
            break

        t0 = time.perf_counter()
        next_token = _decode_step(
            runtime=active,
            embed_model=embed_model,
            lm_head_model=lm_head_model,
            state=state,
            token_id=last_token,
            pos=pos,
            full_causal_mask=causal_masks_by_context[active.context],
            sampling_cfg=sampling_cfg,
            rng=rng,
        )
        dt = time.perf_counter() - t0
        decode_stats[active.context]["tokens"] += 1
        decode_stats[active.context]["time"] += dt

        generated.append(next_token)
        token_history.append(next_token)
        piece = tokenizer.decode([next_token], skip_special_tokens=False)
        print(piece, end="", flush=True)

        last_token = next_token
        valid_tokens += 1

        if not args.no_eos_stop and tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            stop_reason = "eos"
            if args.live_events:
                _log_line(log_stream, "\n[stop] eos", flush=True)
            break
    else:
        if not token_cap_disabled:
            stop_reason = "max-tokens"
            if args.live_events:
                _log_line(log_stream, f"\n[stop] max-tokens reached ({effective_max_tokens})", flush=True)

    print("\n")

    run_end = time.perf_counter()
    prefill_tps = prompt_tokens / prefill_s if prefill_s > 0 else 0.0
    total_decode_tokens = len(generated)
    total_decode_s = sum(v["time"] for v in decode_stats.values())
    total_decode_tps = total_decode_tokens / total_decode_s if total_decode_s > 0 else 0.0
    total_transition_ms = sum(ms for _, _, _, ms, _ in transitions)
    total_compact_ms = sum(ms for _, _, _, ms, _ in compactions)
    decode_with_overheads_s = (
        total_decode_s + (total_transition_ms / 1000.0) + (total_compact_ms / 1000.0)
    )
    decode_with_overheads_tps = (
        total_decode_tokens / decode_with_overheads_s
        if decode_with_overheads_s > 0
        else 0.0
    )
    total_run_s = run_end - prefill_start
    total_run_tps = total_decode_tokens / total_run_s if total_run_s > 0 else 0.0

    if stop_reason == "running":
        if max_time_hit:
            stop_reason = "max-time"
        elif (not token_cap_disabled) and (total_decode_tokens >= effective_max_tokens):
            stop_reason = "max-tokens"
        else:
            stop_reason = "completed"

    _log_line(log_stream, "=== Summary ===")
    _log_line(log_stream, f"prompt_tokens={prompt_tokens}")
    _log_line(log_stream, f"stop_reason={stop_reason}")
    if max_time_limit_s is not None:
        _log_line(log_stream, f"time_limit={max_time_limit_s:.1f}s hit={'yes' if max_time_hit else 'no'}")
    _log_line(
        log_stream,
        f"prefill={prefill_s*1000.0:.1f}ms ({prefill_tps:.1f} t/s) context={runtimes[start_idx].context}",
    )
    _log_line(log_stream, f"decode_tokens={total_decode_tokens} decode_tps={total_decode_tps:.1f} final_context={active.context}")
    _log_line(log_stream, f"decode_time={total_decode_s:.2f}s (token-step only)")
    _log_line(
        log_stream,
        f"decode_time_with_overheads={decode_with_overheads_s:.2f}s "
        f"({decode_with_overheads_tps:.1f} t/s incl transitions+compactions)",
    )
    _log_line(log_stream, f"total_time={total_run_s:.2f}s ({total_run_tps:.1f} t/s from prefill start)")

    if transitions:
        _log_line(log_stream, "transitions:")
        for src, dst, tok, ms, avg_tps in transitions:
            _log_line(
                log_stream,
                f"  ctx{src}->ctx{dst} at token_count={tok} "
                f"({ms:.1f} ms, avg decode {avg_tps:.1f} t/s)",
            )
    else:
        _log_line(log_stream, "transitions: none")

    if compactions:
        _log_line(log_stream, "compactions:")
        for ctx, drop, keep, ms, avg_tps in compactions:
            _log_line(
                log_stream,
                f"  ctx{ctx} drop={drop} keep={keep} "
                f"({ms:.1f} ms, avg decode {avg_tps:.1f} t/s)",
            )
        total_compact_ms = sum(ms for _, _, _, ms, _ in compactions)
        _log_line(log_stream, f"compact_total={total_compact_ms:.1f}ms events={len(compactions)}")
    else:
        _log_line(log_stream, "compactions: none")

    _log_line(log_stream, "per-context decode:")
    for rt in runtimes:
        tok = int(decode_stats[rt.context]["tokens"])
        sec = float(decode_stats[rt.context]["time"])
        tps = tok / sec if sec > 0 else 0.0
        _log_line(log_stream, f"  ctx{rt.context}: tokens={tok} tps={tps:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
