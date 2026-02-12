#!/usr/bin/env python3
"""Combine context exports into multi-function chunk packages.

Expected input layout (from scripts/export_vibethinker_infer_contexts.sh):
- Max context dir: embeddings + lm_head + FFN chunks + prefill chunks (all .mlpackage)
- Other context dirs: FFN chunks only
- Optional per-context standalone FP32 attention artifact for chunk 01:
  <prefix>_FFN_attn_fp32_chunk_01ofNN(.mlpackage/.mlmodelc)

Output per chunk (default):
- infer_ctx{N} for each context
- infer alias (from max context)
- prefill alias (from max context prefill chunk)

Optional modes:
- add prefill_ctx{N} via --prefill-all-contexts
- split infer/prefill outputs via --split-infer-prefill
- remove infer/prefill aliases via --no-alias-functions
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable

_DEDUP_DIAG_LABEL = "unlabeled"
_ANEMLL_DEDUP_AVAILABLE = False

# Ensure repo root is on sys.path so anemll.utils.dedup_weights is importable
# even when running as a standalone script via the shell wrapper.
import sys as _sys
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

try:
    from anemll.utils.dedup_weights import prepare_dedup_sources as _prepare_dedup_sources
    _ANEMLL_DEDUP_AVAILABLE = True
except ImportError:
    _prepare_dedup_sources = None


def _save_multifunction_with_dedup(
    sources: list,
    output_path: str,
    default_function_name: str,
    anemll_dedup: bool = True,
    verbose: bool = True,
    label: str = "",
    source_groups: list | None = None,
):
    """Save multifunction package with optional anemll-dedup pre-processing.

    sources: list of (mlpackage_path_str, src_function_name, target_function_name)
             Used when all sources share the same weight structure.
    source_groups: list of lists of (path, src_fn, tgt_fn) tuples.
             Each group is deduped independently (own anchor), then all results
             are combined into one save_multifunction call.  Use this when
             infer and prefill have different weight structures (e.g. FP32 vs LUT6).
             When provided, ``sources`` is ignored.
    """
    ct = _require_coremltools()

    # Normalise into groups
    if source_groups is not None:
        groups = source_groups
    else:
        groups = [sources]

    total_sources = sum(len(g) for g in groups)

    if anemll_dedup and _ANEMLL_DEDUP_AVAILABLE and total_sources > 1:
        _log(f"[anemll-dedup]{' '+label if label else ''} dedup {total_sources} functions "
             f"in {len(groups)} group(s) ...")

        # Dedup each group independently, collecting all results
        all_deduped: list[tuple[str, str, str]] = []
        # We need to nest context managers; use ExitStack
        from contextlib import ExitStack
        with ExitStack() as stack:
            for gi, group in enumerate(groups):
                if len(group) < 2:
                    # Single-source group: nothing to dedup
                    all_deduped.extend(group)
                    continue
                grp_label = f"group{gi+1}/{len(groups)}"
                _log(f"[anemll-dedup]   {grp_label}: {len(group)} sources")
                deduped = stack.enter_context(
                    _prepare_dedup_sources(group, verbose=verbose)
                )
                all_deduped.extend(deduped)

            # Build and save the multifunction descriptor with all deduped sources
            desc = ct.utils.MultiFunctionDescriptor()
            for path, src_fn, tgt_fn in all_deduped:
                desc.add_function(path, src_fn, tgt_fn)
            desc.default_function_name = default_function_name
            ct.utils.save_multifunction(desc, output_path)

        _log(f"[anemll-dedup]{' '+label if label else ''} done")
    else:
        if anemll_dedup and not _ANEMLL_DEDUP_AVAILABLE:
            _log("[anemll-dedup] WARNING: dedup_weights module not available, using standard combine")
        desc = ct.utils.MultiFunctionDescriptor()
        flat = [s for g in groups for s in g] if source_groups is not None else sources
        for path, src_fn, tgt_fn in flat:
            desc.add_function(path, src_fn, tgt_fn)
        desc.default_function_name = default_function_name
        ct.utils.save_multifunction(desc, output_path)


def _require_coremltools():
    try:
        import coremltools as ct  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "coremltools is required to combine infer context exports."
        ) from exc
    return ct


def _require_yaml():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required.") from exc
    return yaml


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _set_dedup_diag_label(label: str) -> None:
    global _DEDUP_DIAG_LABEL
    _DEDUP_DIAG_LABEL = label


def _enable_coreml_dedup_diagnostics() -> None:
    """Monkey-patch CoreMLTools const dedup pass to print per-save diagnostics."""
    try:
        from coremltools.converters.mil.mil.passes.defs.cleanup.const_deduplication import (  # type: ignore
            const_deduplication,
        )
    except Exception as exc:
        _log(f"[dedup] diagnostics unavailable: failed to import dedup pass ({exc})")
        return

    if getattr(const_deduplication, "_anemll_dedup_diag_wrapped", False):
        _log("[dedup] diagnostics already enabled")
        return

    original = const_deduplication._deduplicate_const_across_functions

    def _var_nbytes(var) -> int:
        val = getattr(var, "val", None)
        if val is None:
            return 0
        try:
            return int(val.nbytes)
        except Exception:
            try:
                import numpy as np

                return int(np.asarray(val).nbytes)
            except Exception:
                return 0

    def wrapped(self, prog) -> None:
        blocks = list(prog.functions.values())

        # Compute potential dedup matches with the same aggressive threshold used
        # by CoreMLTools across-functions dedup.
        old_threshold = self.const_threshold
        try:
            self.const_threshold = 1
            unique2duplicates = self.find_constants(blocks)
        finally:
            self.const_threshold = old_threshold

        dup_groups = 0
        dup_consts = 0
        dup_bytes = 0
        for duplicates in unique2duplicates.values():
            if not duplicates:
                continue
            dup_groups += 1
            dup_consts += len(duplicates)
            dup_bytes += sum(_var_nbytes(v) for v in duplicates)

        pre_const_total = 0
        for block in blocks:
            for op in block.operations:
                if op.op_type == "const":
                    pre_const_total += 1

        fn_names = list(prog.functions.keys())
        _log(
            f"[dedup][{_DEDUP_DIAG_LABEL}] pre: functions={len(fn_names)} "
            f"const_total={pre_const_total} dup_groups={dup_groups} "
            f"dup_consts={dup_consts} dup_bytes={dup_bytes/1024/1024:.2f} MiB"
        )

        original(self, prog)

        post_const_total = 0
        with_weight_id = 0
        weight_id_hist: dict[str, int] = {}
        for block in blocks:
            for op in block.operations:
                if op.op_type != "const":
                    continue
                post_const_total += 1
                wid = getattr(op, "weight_id", None)
                if wid is not None:
                    with_weight_id += 1
                    key = str(wid)
                    weight_id_hist[key] = weight_id_hist.get(key, 0) + 1

        shared_groups = sum(1 for n in weight_id_hist.values() if n > 1)
        shared_consts = sum(n for n in weight_id_hist.values() if n > 1)
        _log(
            f"[dedup][{_DEDUP_DIAG_LABEL}] post: const_total={post_const_total} "
            f"with_weight_id={with_weight_id} shared_groups={shared_groups} "
            f"shared_consts={shared_consts}"
        )

    const_deduplication._deduplicate_const_across_functions = wrapped
    const_deduplication._anemll_dedup_diag_wrapped = True
    _log("[dedup] diagnostics enabled (common::const_deduplication)")


def _parse_context_entries(entries: Iterable[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"Invalid --contexts entry '{raw}'. Expected N=/path")
        lhs, rhs = raw.split("=", 1)
        ctx = int(lhs)
        model_dir = Path(rhs).expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(model_dir)
        out[ctx] = model_dir
    if not out:
        raise ValueError("No contexts provided")
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _parse_chunk_indices(raw: str | None, num_chunks: int) -> list[int]:
    if raw is None or str(raw).strip() == "":
        return list(range(1, num_chunks + 1))

    out: list[int] = []
    seen: set[int] = set()
    for part in str(raw).replace(",", " ").split():
        idx = int(part)
        if idx < 1 or idx > num_chunks:
            raise ValueError(
                f"--chunk-indices contains out-of-range chunk {idx}; "
                f"valid range is 1..{num_chunks}"
            )
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    if not out:
        raise ValueError("--chunk-indices resolved to empty list")
    return sorted(out)


def _find_chunk_model(
    model_dir: Path,
    prefix: str,
    kind: str,
    chunk_idx: int,
    num_chunks: int,
) -> Path:
    suffix = f"_chunk_{chunk_idx:02d}of{num_chunks:02d}"
    exact = [
        model_dir / f"{prefix}_{kind}{suffix}.mlpackage",
        model_dir / f"{prefix}_{kind}{suffix}.mlmodelc",
    ]
    for p in exact:
        if p.exists():
            return p

    hits = sorted(model_dir.glob(f"{prefix}_{kind}_lut*{suffix}.mlpackage"))
    if not hits:
        hits = sorted(model_dir.glob(f"{prefix}_{kind}_lut*{suffix}.mlmodelc"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"Missing {kind} chunk {chunk_idx:02d}of{num_chunks:02d} in {model_dir}"
    )


def _find_first_available_kind(
    model_dir: Path,
    prefix: str,
    num_chunks: int,
    candidates: list[str],
) -> str | None:
    """Return first chunk-kind candidate present in model_dir, checking chunk_01 first."""
    for kind in candidates:
        try:
            _find_chunk_model(
                model_dir=model_dir,
                prefix=prefix,
                kind=kind,
                chunk_idx=1,
                num_chunks=num_chunks,
            )
            return kind
        except FileNotFoundError:
            continue
    return None


def _resolve_infer_kind_for_context(
    requested_kind: str,
    model_dir: Path,
    prefix: str,
    num_chunks: int,
    context: int,
) -> str:
    requested = (requested_kind or "").strip()
    if requested == "auto":
        candidates = ["FFN", "FFN_PF"]
    elif requested == "FFN_PF":
        candidates = ["FFN_PF", "FFN"]
    elif requested == "FFN":
        # Keep explicit FFN strict; prefill handling is resolved separately
        # from max context only.
        candidates = ["FFN"]
    else:
        candidates = [requested]

    resolved = _find_first_available_kind(
        model_dir=model_dir,
        prefix=prefix,
        num_chunks=num_chunks,
        candidates=candidates,
    )
    if resolved is None:
        raise FileNotFoundError(
            f"Missing infer chunks in context {context} ({model_dir}). "
            f"Tried kinds: {candidates}"
        )
    if requested not in ("", "auto") and resolved != requested:
        print(
            f"[combine] Context {context}: infer kind '{requested}' not found; "
            f"falling back to '{resolved}'."
        )
    return resolved


def _resolve_prefill_kind_for_context(
    requested_kind: str,
    model_dir: Path,
    prefix: str,
    num_chunks: int,
    context: int,
) -> str:
    requested = (requested_kind or "").strip()
    if requested == "auto":
        candidates = ["FFN_PF", "prefill"]
    elif requested == "FFN_PF":
        candidates = ["FFN_PF", "prefill"]
    elif requested == "prefill":
        candidates = ["prefill", "FFN_PF"]
    else:
        candidates = [requested]

    resolved = _find_first_available_kind(
        model_dir=model_dir,
        prefix=prefix,
        num_chunks=num_chunks,
        candidates=candidates,
    )
    if resolved is None:
        raise FileNotFoundError(
            f"Missing prefill chunks in context {context} ({model_dir}). "
            f"Tried kinds: {candidates}"
        )
    if requested not in ("", "auto") and resolved != requested:
        print(
            f"[combine] Context {context}: prefill kind '{requested}' not found; "
            f"falling back to '{resolved}'."
        )
    return resolved


def _dedup_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_infer_chunk1_kind_for_context(
    requested_kind: str,
    default_infer_kind: str,
    model_dir: Path,
    prefix: str,
    num_chunks: int,
    context: int,
) -> str:
    requested = (requested_kind or "").strip()
    if requested in ("", "auto"):
        # Default to regular per-context infer chunks. FP32 chunk1 should be
        # explicit via --infer-chunk1-kind FFN_attn_fp32 to avoid accidental
        # selection when standalone debug artifacts are present.
        candidates = _dedup_preserve_order(
            [default_infer_kind, "FFN", "FFN_PF"]
        )
    else:
        candidates = _dedup_preserve_order([requested, default_infer_kind])

    for kind in candidates:
        try:
            _find_chunk_model(
                model_dir=model_dir,
                prefix=prefix,
                kind=kind,
                chunk_idx=1,
                num_chunks=num_chunks,
            )
            if requested not in ("", "auto") and kind != requested:
                print(
                    f"[combine] Context {context}: chunk1 kind '{requested}' not found; "
                    f"falling back to '{kind}'."
                )
            return kind
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"Missing infer chunk1 in context {context} ({model_dir}). "
        f"Tried kinds: {candidates}"
    )


def _available_functions(model_path: Path, ct_module) -> list[str]:
    model = ct_module.models.MLModel(str(model_path))
    spec = model.get_spec()
    names: list[str] = []

    desc = getattr(spec, "description", None)
    if desc is not None and hasattr(desc, "functions"):
        try:
            names.extend([f.name for f in desc.functions if getattr(f, "name", None)])
        except Exception:
            pass

    mlprog = getattr(spec, "mlProgram", None)
    if mlprog is not None and hasattr(mlprog, "functions"):
        try:
            names.extend(list(mlprog.functions.keys()))
        except Exception:
            pass

    if not names:
        names = ["main"]

    dedup: list[str] = []
    seen = set()
    for n in names:
        if n not in seen:
            seen.add(n)
            dedup.append(n)
    return dedup


def _resolve_source_function(
    model_path: Path,
    preferred: str,
    ct_module,
    alternates: list[str] | None = None,
) -> str:
    # Most source exports used in state-transition combine are single-function
    # "main" models. Fast-path this to avoid loading each model spec.
    if preferred in ("", "main"):
        names = _available_functions(model_path, ct_module)
        if "main" in names:
            return "main"
        # Legacy/split exports can be multifunction without "main".
        for fn in (alternates or []):
            if fn in names:
                return fn
        if len(names) == 1:
            return names[0]
        raise RuntimeError(
            f"Could not resolve source function '{preferred}' from {model_path}; available={names}"
        )

    names = _available_functions(model_path, ct_module)
    if preferred in names:
        return preferred
    for fn in (alternates or []):
        if fn in names:
            return fn
    if "main" in names:
        return "main"
    if len(names) == 1:
        return names[0]
    raise RuntimeError(
        f"Could not resolve source function '{preferred}' from {model_path}; available={names}"
    )


def _copy_path(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _compile_mlpackage(package_path: Path, output_dir: Path) -> Path:
    if not package_path.exists():
        raise FileNotFoundError(f"Cannot compile missing package: {package_path}")
    target = output_dir / f"{package_path.stem}.mlmodelc"
    if target.exists():
        shutil.rmtree(target)
    cmd = [
        "xcrun",
        "coremlcompiler",
        "compile",
        str(package_path),
        str(output_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "coremlcompiler compile failed for "
            f"{package_path} (exit={proc.returncode})\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if not target.exists():
        raise RuntimeError(f"Expected compiled artifact not found: {target}")
    return target


def _copy_shared_assets(max_ctx_dir: Path, out_dir: Path, prefix: str) -> dict[str, str | None]:
    optional_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "config.json",
    ]
    for fname in optional_files:
        src = max_ctx_dir / fname
        if src.exists():
            _copy_path(src, out_dir / fname)

    copied = {"embeddings": None, "lm_head": None}
    for key in ("embeddings", "lm_head"):
        compiled_matches = sorted(max_ctx_dir.glob(f"{prefix}_{key}*.mlmodelc"))
        package_matches = sorted(max_ctx_dir.glob(f"{prefix}_{key}*.mlpackage"))
        src: Path | None = None
        if compiled_matches:
            src = compiled_matches[0]
        elif package_matches:
            src = package_matches[0]
        if src is not None:
            _copy_path(src, out_dir / src.name)
            copied[key] = src.name

            # Keep package sidecar when it exists (helpful for tooling fallback).
            if src.suffix == ".mlmodelc":
                sidecar_pkg = max_ctx_dir / f"{src.stem}.mlpackage"
                if sidecar_pkg.exists():
                    _copy_path(sidecar_pkg, out_dir / sidecar_pkg.name)
    return copied


def _copy_model_artifact_with_sidecars(src: Path, out_dir: Path) -> list[str]:
    copied: list[str] = []

    def _copy_one(path: Path) -> None:
        dst = out_dir / path.name
        _copy_path(path, dst)
        copied.append(dst.name)

    _copy_one(src)
    sidecar: Path | None = None
    if src.suffix == ".mlpackage":
        sidecar = src.with_suffix(".mlmodelc")
    elif src.suffix == ".mlmodelc":
        sidecar = src.with_suffix(".mlpackage")

    if sidecar is not None and sidecar.exists():
        _copy_one(sidecar)
    return copied


def _copy_source_chunk_artifacts(
    *,
    context_dir: Path,
    out_dir: Path,
    prefix: str,
    num_chunks: int,
    infer_kind: str,
    chunk1_infer_kind: str,
    prefill_kind: str,
) -> dict[str, object]:
    copied_names: list[str] = []
    infer_sources: dict[str, str] = {}
    prefill_sources: dict[str, str] = {}
    chunk1_regular_infer_source: str | None = None

    for chunk_idx in range(1, num_chunks + 1):
        infer_kind_this_chunk = chunk1_infer_kind if chunk_idx == 1 else infer_kind
        infer_src = _find_chunk_model(
            model_dir=context_dir,
            prefix=prefix,
            kind=infer_kind_this_chunk,
            chunk_idx=chunk_idx,
            num_chunks=num_chunks,
        )
        infer_sources[str(chunk_idx)] = infer_src.name
        copied_names.extend(_copy_model_artifact_with_sidecars(infer_src, out_dir))

        if chunk_idx == 1 and chunk1_infer_kind != infer_kind:
            # Keep the regular chunk_01 infer artifact too so output layout mimics
            # per-context export directories (both FP32 and non-FP32 chunk1).
            regular_src = _find_chunk_model(
                model_dir=context_dir,
                prefix=prefix,
                kind=infer_kind,
                chunk_idx=chunk_idx,
                num_chunks=num_chunks,
            )
            chunk1_regular_infer_source = regular_src.name
            copied_names.extend(_copy_model_artifact_with_sidecars(regular_src, out_dir))

        prefill_src = _find_chunk_model(
            model_dir=context_dir,
            prefix=prefix,
            kind=prefill_kind,
            chunk_idx=chunk_idx,
            num_chunks=num_chunks,
        )
        prefill_sources[str(chunk_idx)] = prefill_src.name
        copied_names.extend(_copy_model_artifact_with_sidecars(prefill_src, out_dir))

    return {
        "context_dir": str(context_dir),
        "infer_kind": infer_kind,
        "chunk1_infer_kind": chunk1_infer_kind,
        "prefill_kind": prefill_kind,
        "infer_sources": infer_sources,
        "chunk1_regular_infer_source": chunk1_regular_infer_source,
        "prefill_sources": prefill_sources,
        "artifacts": _dedup_preserve_order(copied_names),
    }


def _ensure_shared_assets_compiled(
    out_dir: Path,
    copied_shared: dict[str, str | None],
) -> dict[str, str | None]:
    updated = dict(copied_shared)
    for key, name in copied_shared.items():
        if not name:
            continue
        src = out_dir / name
        if src.suffix != ".mlpackage":
            continue
        compiled = _compile_mlpackage(src, out_dir)
        updated[key] = compiled.name
        print(f"[combine] Compiled shared {key}: {compiled.name}")
    return updated


def _copy_tokenizer_assets_if_missing(out_dir: Path, tokenizer_dir: Path) -> None:
    """Backfill tokenizer assets into out_dir when combine inputs don't contain them."""
    if not tokenizer_dir.exists():
        return

    optional_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "tokenizer.model",
    ]
    for fname in optional_files:
        dst = out_dir / fname
        if dst.exists():
            continue
        src = tokenizer_dir / fname
        if src.exists():
            _copy_path(src, dst)


def _sampling_from_generation_config(config_path: Path) -> dict | None:
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    out: dict = {}
    for key in (
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
    ):
        if key in data:
            out[key] = data[key]
    return out or None


def _load_int_param_from_meta(model_dir: Path, key: str) -> int | None:
    meta_path = model_dir / "meta.yaml"
    if not meta_path.exists():
        return None
    try:
        yaml = _require_yaml()
        meta = yaml.safe_load(meta_path.read_text())
        params = meta.get("model_info", {}).get("parameters", {})
        if not isinstance(params, dict):
            return None
        value = params.get(key, None)
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _load_recommended_sampling(max_ctx_dir: Path, tokenizer_dir: Path | None = None) -> dict | None:
    meta_path = max_ctx_dir / "meta.yaml"
    if not meta_path.exists():
        rec = None
    else:
        try:
            yaml = _require_yaml()
            meta = yaml.safe_load(meta_path.read_text())
            params = meta.get("model_info", {}).get("parameters", {})
            rec = params.get("recommended_sampling")
            if isinstance(rec, dict):
                return rec
        except Exception:
            rec = None

    # Fallback: derive from generation_config.json
    rec = _sampling_from_generation_config(max_ctx_dir / "generation_config.json")
    if rec:
        return rec
    if tokenizer_dir is not None:
        rec = _sampling_from_generation_config(tokenizer_dir / "generation_config.json")
        if rec:
            return rec
    return None


def _lut_from_name(name: str | None) -> str | int:
    if not name:
        return "none"
    stem = re.sub(r"\.(mlmodelc|mlpackage)$", "", name)
    m = re.search(r"_lut(\d+)$", stem)
    return int(m.group(1)) if m else "none"


def _write_meta_yaml(
    out_dir: Path,
    model_name: str,
    architecture: str,
    prefix: str,
    context_length: int,
    batch_size: int,
    num_chunks: int,
    split_lm_head: int,
    contexts: list[int],
    max_context: int,
    output_base: str,
    output_base_infer: str | None,
    output_base_prefill: str | None,
    split_infer_prefill: bool,
    no_alias_functions: bool,
    embeddings_name: str | None,
    lm_head_name: str | None,
    chunk1_infer_kinds: dict[int, str],
    prefill_chunk1_kinds: dict[int, str],
    include_prefill_ctx_functions: bool,
    prefill_contexts: list[int] | None,
    recommended_sampling: dict | None,
    tokenizer_path: str | None,
    copied_source_chunks: dict[str, object] | None,
) -> None:
    yaml = _require_yaml()

    lut_embeddings = _lut_from_name(embeddings_name)
    lut_lmhead = _lut_from_name(lm_head_name)
    lut_ffn = _lut_from_name(output_base_infer if split_infer_prefill else output_base)
    chunk1_kind_values = sorted(set(chunk1_infer_kinds.values()))
    chunk1_kind_summary = chunk1_kind_values[0] if len(chunk1_kind_values) == 1 else "mixed"
    fp32_chunk1_contexts = [
        int(ctx) for ctx, kind in chunk1_infer_kinds.items() if kind == "FFN_attn_fp32"
    ]
    prefill_chunk1_kind_values = sorted(set(prefill_chunk1_kinds.values()))
    prefill_chunk1_kind_summary = (
        prefill_chunk1_kind_values[0] if len(prefill_chunk1_kind_values) == 1 else "mixed"
    )
    prefill_fp32_chunk1_contexts = [
        int(ctx) for ctx, kind in prefill_chunk1_kinds.items() if kind == "FFN_attn_fp32"
    ]
    prefill_ctxs = [int(c) for c in (prefill_contexts or [max_context])]
    multi_prefill = len(prefill_ctxs) > 1
    if split_infer_prefill:
        if not output_base_infer or not output_base_prefill:
            raise ValueError("split_infer_prefill requires output_base_infer/output_base_prefill")
        ffn_name = f"{output_base_infer}_chunk_01of{num_chunks:02d}.mlpackage"
        ffn_prefill_name = f"{output_base_prefill}_chunk_01of{num_chunks:02d}.mlpackage"
    else:
        ffn_name = f"{output_base}_chunk_01of{num_chunks:02d}.mlpackage"
        ffn_prefill_name = None

    if split_infer_prefill:
        if include_prefill_ctx_functions:
            layout = (
                "split:infer_ctx+prefill_ctx+aliases"
                if not no_alias_functions
                else "split:infer_ctx+prefill_ctx"
            )
        else:
            layout = (
                "split:infer_ctx+prefill_alias"
                if not no_alias_functions
                else "split:infer_ctx_only"
            )
    else:
        if include_prefill_ctx_functions:
            layout = (
                "infer_ctx+prefill_ctx+aliases"
                if not no_alias_functions
                else "infer_ctx+prefill_ctx"
            )
        else:
            layout = "infer_ctx+prefill_alias"

    infer_default_fn = f"infer_ctx{max_context}" if no_alias_functions else "infer"
    if include_prefill_ctx_functions:
        prefill_default_fn = f"prefill_ctx{max_context}" if no_alias_functions else "prefill"
    else:
        prefill_default_fn = "prefill"

    params = {
        "context_length": int(context_length),
        "batch_size": int(batch_size),
        "num_chunks": int(num_chunks),
        "model_prefix": prefix,
        "lut_embeddings": lut_embeddings,
        "lut_ffn": lut_ffn,
        "lut_lmhead": lut_lmhead,
        "embeddings": embeddings_name or f"{prefix}_embeddings.mlpackage",
        "lm_head": lm_head_name or f"{prefix}_lm_head.mlpackage",
        "ffn": ffn_name,
        "split_lm_head": int(split_lm_head),
        "argmax_in_model": False,
        "state_transition_infer_contexts": [int(c) for c in contexts],
        "state_transition_infer_function_template": "infer_ctx{context}",
        "state_transition_infer_default_function": infer_default_fn,
        "state_transition_prefill_context": int(max_context),
        "state_transition_prefill_default_function": prefill_default_fn,
        "state_transition_all_context_prefill": bool(multi_prefill),
        "state_transition_combined_functions_layout": layout,
        "state_transition_no_alias_functions": bool(no_alias_functions),
        "state_transition_split_infer_prefill": bool(split_infer_prefill),
        "state_transition_chunk1_infer_kind": chunk1_kind_summary,
        "state_transition_chunk1_infer_kinds": {
            str(ctx): kind for ctx, kind in chunk1_infer_kinds.items()
        },
        "state_transition_chunk1_fp32_kind": "FFN_attn_fp32",
        "state_transition_chunk1_fp32_contexts": fp32_chunk1_contexts,
        "state_transition_chunk1_fp32_enabled": bool(fp32_chunk1_contexts),
        "state_transition_prefill_chunk1_kind": prefill_chunk1_kind_summary,
        "state_transition_prefill_chunk1_kinds": {
            str(ctx): kind for ctx, kind in prefill_chunk1_kinds.items()
        },
        "state_transition_prefill_chunk1_fp32_contexts": prefill_fp32_chunk1_contexts,
        "state_transition_prefill_chunk1_fp32_enabled": bool(prefill_fp32_chunk1_contexts),
        "state_transition_chunk1_fp32_prefill_mismatch": bool(
            fp32_chunk1_contexts and not prefill_fp32_chunk1_contexts
        ),
    }
    if include_prefill_ctx_functions:
        params["state_transition_prefill_contexts"] = prefill_ctxs
        params["state_transition_prefill_function_template"] = "prefill_ctx{context}"
    if split_infer_prefill:
        params["ffn_prefill"] = ffn_prefill_name
        params["state_transition_infer_output_base"] = output_base_infer
        params["state_transition_prefill_output_base"] = output_base_prefill
    if isinstance(recommended_sampling, dict):
        params["recommended_sampling"] = recommended_sampling
    if tokenizer_path:
        params["tokenizer_path"] = str(tokenizer_path)
    if copied_source_chunks:
        params["state_transition_copied_source_chunks"] = True
        copied_ctx = copied_source_chunks.get("context", None)
        if copied_ctx is not None:
            params["state_transition_copied_source_chunks_context"] = int(copied_ctx)
        copied_artifacts = copied_source_chunks.get("artifacts", None)
        if isinstance(copied_artifacts, list) and copied_artifacts:
            params["state_transition_copied_source_chunk_artifacts"] = copied_artifacts

    description_bits = [f"State-transition chunks with infer_ctx for contexts {contexts}"]
    if include_prefill_ctx_functions:
        description_bits.append(f"prefill_ctx for contexts {prefill_ctxs}")
    else:
        description_bits.append(f"prefill from {max_context}")
    if split_infer_prefill:
        description_bits.append("infer/prefill in separate packages")
    if no_alias_functions:
        description_bits.append("no alias functions")

    meta = {
        "model_info": {
            "name": model_name,
            "version": "0.3.5",
            "description": "; ".join(description_bits),
            "license": "MIT",
            "author": "Anemll",
            "framework": "Core ML",
            "language": "Python",
            "architecture": architecture,
            "parameters": params,
        }
    }
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Combine infer-only context exports into multi-function chunk packages "
            "(infer_ctx* + optional prefill_ctx* + optional aliases)."
        )
    )
    ap.add_argument(
        "--contexts",
        nargs="+",
        required=True,
        help="List of context dirs in form N=/path/to/context_export",
    )
    ap.add_argument("--output-dir", required=True, help="Combined output directory")
    ap.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Context to use for prefill source (default: highest provided context)",
    )
    ap.add_argument("--prefix", default="qwen25", help="Model prefix (default: qwen25)")
    ap.add_argument(
        "--num-chunks",
        type=int,
        default=3,
        help="Number of chunks to combine (default: 3)",
    )
    ap.add_argument(
        "--chunk-indices",
        default=None,
        help=(
            "Optional subset of chunk indices to build, e.g. '2' or '1,3'. "
            "Default: all chunks 1..num-chunks."
        ),
    )
    ap.add_argument(
        "--output-base",
        default=None,
        help="Output chunk base (default: <prefix>_FFN_PF_statex)",
    )
    ap.add_argument(
        "--infer-fn",
        default="infer",
        help="Preferred infer source function name (fallback: main)",
    )
    ap.add_argument(
        "--prefill-fn",
        default="prefill",
        help="Preferred prefill source function name (fallback: main)",
    )
    ap.add_argument(
        "--infer-kind",
        default="auto",
        help="Input infer chunk kind token (default: auto; FFN preferred, fallback FFN_PF)",
    )
    ap.add_argument(
        "--infer-chunk1-kind",
        default="auto",
        help=(
            "Input infer chunk kind token for chunk_01 only "
            "(default: auto; uses resolved infer kind. "
            "Set FFN_attn_fp32 explicitly to force FP32 chunk1.)"
        ),
    )
    ap.add_argument(
        "--prefill-kind",
        default="auto",
        help="Input prefill chunk kind token in max context (default: auto; FFN_PF preferred, fallback prefill)",
    )
    ap.add_argument(
        "--prefill-all-contexts",
        action="store_true",
        help=(
            "Add one prefill function per context (prefill_ctx{N}) in each output chunk. "
            "Alias behavior is controlled by --no-alias-functions."
        ),
    )
    ap.add_argument(
        "--split-infer-prefill",
        action="store_true",
        help=(
            "Emit separate multifunction packages per chunk: one infer-only and one prefill-only. "
            "Useful for dedup/size experiments."
        ),
    )
    ap.add_argument(
        "--no-alias-functions",
        action="store_true",
        help=(
            "Do not emit compatibility aliases 'infer'/'prefill'. "
            "Only context-routed names (infer_ctx*/prefill_ctx*) are written."
        ),
    )
    ap.add_argument(
        "--output-base-infer",
        default=None,
        help=(
            "Infer output base when --split-infer-prefill is enabled "
            "(default: <output-base>_infer)."
        ),
    )
    ap.add_argument(
        "--output-base-prefill",
        default=None,
        help=(
            "Prefill output base when --split-infer-prefill is enabled "
            "(default: <output-base>_prefill)."
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size to write to meta.yaml (default: read from max-context meta.yaml, fallback 32)",
    )
    ap.add_argument(
        "--split-lm-head",
        type=int,
        default=None,
        help="split_lm_head to write to meta.yaml (default: read from max-context meta.yaml, fallback 16)",
    )
    ap.add_argument(
        "--architecture",
        default="qwen2",
        help="Architecture label for meta.yaml (default: qwen2)",
    )
    ap.add_argument(
        "--model-name",
        default="anemll-vibethinker-1.5b-state-transition",
        help="Model name for meta.yaml",
    )
    ap.add_argument(
        "--no-copy-shared",
        action="store_true",
        help="Do not copy tokenizer/embeddings/lm_head assets from max context dir",
    )
    ap.add_argument(
        "--copy-source-chunks",
        action="store_true",
        help=(
            "Copy source chunk artifacts into output dir so layout mirrors a context folder "
            "(infer/prefill source chunks plus chunk1 FP32 and regular infer when different)."
        ),
    )
    ap.add_argument(
        "--copy-source-chunks-context",
        type=int,
        default=None,
        help=(
            "Context to copy source chunk artifacts from when --copy-source-chunks is set "
            "(default: max-context)."
        ),
    )
    ap.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Optional tokenizer directory fallback. "
            "When provided, missing tokenizer files are copied from this directory "
            "and tokenizer_path is written into meta.yaml."
        ),
    )
    ap.add_argument(
        "--no-meta",
        action="store_true",
        help="Do not write meta.yaml",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Allow overwrite of existing output chunk packages",
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip compiling output chunk packages to .mlmodelc",
    )
    ap.add_argument(
        "--dedup-diagnostics",
        action="store_true",
        help=(
            "Enable per-save CoreMLTools dedup diagnostics "
            "(duplicate const candidates and weight_id sharing stats)."
        ),
    )
    ap.add_argument(
        "--anemll-dedup",
        action="store_true",
        default=True,
        dest="anemll_dedup",
        help=(
            "Enable anemll-dedup surgical weight dedup before combining "
            "(default: enabled). Replaces semantically-identical palettized "
            "weights across contexts with byte-exact copies for maximum "
            "CoreML dedup sharing."
        ),
    )
    ap.add_argument(
        "--skip-anemll-dedup",
        dest="anemll_dedup",
        action="store_false",
        help="Skip anemll-dedup weight dedup (use standard combine)",
    )
    args = ap.parse_args()

    if args.num_chunks <= 0:
        raise ValueError("--num-chunks must be > 0")

    ct = _require_coremltools()
    yaml = _require_yaml()
    if args.dedup_diagnostics:
        _enable_coreml_dedup_diagnostics()

    if args.anemll_dedup:
        if _ANEMLL_DEDUP_AVAILABLE:
            _log("[anemll-dedup] weight dedup: enabled")
        else:
            _log("[anemll-dedup] weight dedup: UNAVAILABLE (dedup_weights module not found)")
    else:
        _log("[anemll-dedup] weight dedup: disabled (--skip-anemll-dedup)")

    context_dirs = _parse_context_entries(args.contexts)
    selected_chunks = _parse_chunk_indices(args.chunk_indices, args.num_chunks)
    context_list = list(context_dirs.keys())
    max_context = args.max_context if args.max_context is not None else max(context_list)
    if max_context not in context_dirs:
        raise ValueError(f"max_context={max_context} not present in --contexts")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    output_base = args.output_base or f"{args.prefix}_FFN_PF_statex"
    if not args.split_infer_prefill and (
        args.output_base_infer is not None or args.output_base_prefill is not None
    ):
        raise ValueError(
            "--output-base-infer/--output-base-prefill require --split-infer-prefill"
        )
    output_base_infer = args.output_base_infer or f"{output_base}_infer"
    output_base_prefill = args.output_base_prefill or f"{output_base}_prefill"
    max_ctx_dir = context_dirs[max_context]
    resolved_batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else (_load_int_param_from_meta(max_ctx_dir, "batch_size") or 32)
    )
    resolved_split_lm_head = (
        int(args.split_lm_head)
        if args.split_lm_head is not None
        else (_load_int_param_from_meta(max_ctx_dir, "split_lm_head") or 16)
    )
    if args.batch_size is None:
        _log(f"[combine] batch_size auto-resolved to {resolved_batch_size}")
    if args.split_lm_head is None:
        _log(f"[combine] split_lm_head auto-resolved to {resolved_split_lm_head}")

    copied_shared = {"embeddings": None, "lm_head": None}
    if not args.no_copy_shared:
        copied_shared = _copy_shared_assets(max_ctx_dir, out_dir, args.prefix)
        if not args.no_compile:
            copied_shared = _ensure_shared_assets_compiled(out_dir, copied_shared)
    tokenizer_path: str | None = None
    if args.tokenizer_path:
        tok_dir = Path(args.tokenizer_path).expanduser().resolve()
        if not tok_dir.exists():
            raise FileNotFoundError(f"--tokenizer-path does not exist: {tok_dir}")
        _copy_tokenizer_assets_if_missing(out_dir, tok_dir)
        tokenizer_path = str(tok_dir)
    tokenizer_dir_path = Path(tokenizer_path) if tokenizer_path else None
    recommended_sampling = _load_recommended_sampling(
        max_ctx_dir=max_ctx_dir,
        tokenizer_dir=tokenizer_dir_path,
    )

    manifest: dict = {
        "contexts": context_list,
        "max_context": max_context,
        "num_chunks": args.num_chunks,
        "selected_chunk_indices": selected_chunks,
        "output_base": output_base,
        "output_base_infer": output_base_infer if args.split_infer_prefill else None,
        "output_base_prefill": output_base_prefill if args.split_infer_prefill else None,
        "infer_kind": args.infer_kind,
        "infer_chunk1_kind": args.infer_chunk1_kind,
        "prefill_kind": args.prefill_kind,
        "prefill_all_contexts": bool(args.prefill_all_contexts),
        "split_infer_prefill": bool(args.split_infer_prefill),
        "no_alias_functions": bool(args.no_alias_functions),
        "chunks": [],
    }

    resolved_infer_kinds: dict[int, str] = {}
    for ctx, model_dir in context_dirs.items():
        resolved_infer_kinds[ctx] = _resolve_infer_kind_for_context(
            requested_kind=args.infer_kind,
            model_dir=model_dir,
            prefix=args.prefix,
            num_chunks=args.num_chunks,
            context=ctx,
        )
    resolved_prefill_kinds: dict[int, str] = {}
    if args.prefill_all_contexts:
        for ctx, model_dir in context_dirs.items():
            resolved_prefill_kinds[ctx] = _resolve_prefill_kind_for_context(
                requested_kind=args.prefill_kind,
                model_dir=model_dir,
                prefix=args.prefix,
                num_chunks=args.num_chunks,
                context=ctx,
            )
    else:
        resolved_prefill_kinds[max_context] = _resolve_prefill_kind_for_context(
            requested_kind=args.prefill_kind,
            model_dir=max_ctx_dir,
            prefix=args.prefix,
            num_chunks=args.num_chunks,
            context=max_context,
        )
    _log(
        "[combine] Resolved infer kinds: "
        + ", ".join(f"{ctx}={resolved_infer_kinds[ctx]}" for ctx in context_list)
    )
    if args.prefill_all_contexts:
        _log(
            "[combine] Resolved prefill kinds: "
            + ", ".join(f"{ctx}={resolved_prefill_kinds[ctx]}" for ctx in context_list)
        )
    else:
        _log(
            f"[combine] Resolved prefill kind (max context {max_context}): "
            f"{resolved_prefill_kinds[max_context]}"
        )
    manifest["resolved_infer_kinds"] = {str(k): v for k, v in resolved_infer_kinds.items()}
    manifest["resolved_prefill_kinds"] = {
        str(k): v for k, v in sorted(resolved_prefill_kinds.items(), key=lambda kv: kv[0])
    }

    resolved_chunk1_infer_kinds: dict[int, str] = {}
    for ctx, model_dir in context_dirs.items():
        resolved_chunk1_infer_kinds[ctx] = _resolve_infer_chunk1_kind_for_context(
            requested_kind=args.infer_chunk1_kind,
            default_infer_kind=resolved_infer_kinds[ctx],
            model_dir=model_dir,
            prefix=args.prefix,
            num_chunks=args.num_chunks,
            context=ctx,
        )
    _log(
        "[combine] Resolved chunk1 infer kinds: "
        + ", ".join(f"{ctx}={resolved_chunk1_infer_kinds[ctx]}" for ctx in context_list)
    )
    manifest["resolved_chunk1_infer_kinds"] = {
        str(k): v for k, v in resolved_chunk1_infer_kinds.items()
    }

    copied_source_chunks_summary: dict[str, object] | None = None
    if args.copy_source_chunks:
        copy_ctx = (
            int(args.copy_source_chunks_context)
            if args.copy_source_chunks_context is not None
            else int(max_context)
        )
        if copy_ctx not in context_dirs:
            raise ValueError(
                f"--copy-source-chunks-context={copy_ctx} not present in --contexts"
            )
        copy_ctx_dir = context_dirs[copy_ctx]
        _log(
            f"[combine] Copy source chunk artifacts from context {copy_ctx}: {copy_ctx_dir}"
        )
        copied_source_chunks = _copy_source_chunk_artifacts(
            context_dir=copy_ctx_dir,
            out_dir=out_dir,
            prefix=args.prefix,
            num_chunks=args.num_chunks,
            infer_kind=resolved_infer_kinds[copy_ctx],
            chunk1_infer_kind=resolved_chunk1_infer_kinds[copy_ctx],
            prefill_kind=resolved_prefill_kinds[copy_ctx],
        )
        copied_source_chunks_summary = {
            "enabled": True,
            "context": copy_ctx,
            "artifacts": copied_source_chunks.get("artifacts", []),
        }
        manifest["copied_source_chunks"] = copied_source_chunks

    prefill_contexts = context_list if args.prefill_all_contexts else [max_context]
    include_prefill_ctx_functions = bool(
        args.prefill_all_contexts or args.split_infer_prefill or args.no_alias_functions
    )
    prefill_chunk1_kinds = {
        int(ctx): resolved_prefill_kinds[int(ctx)] for ctx in prefill_contexts
    }
    infer_chunk1_fp32_enabled = any(
        kind == "FFN_attn_fp32" for kind in resolved_chunk1_infer_kinds.values()
    )
    prefill_chunk1_fp32_enabled = any(
        kind == "FFN_attn_fp32" for kind in prefill_chunk1_kinds.values()
    )
    manifest["resolved_prefill_chunk1_kinds"] = {
        str(k): v for k, v in prefill_chunk1_kinds.items()
    }
    manifest["prefill_chunk1_fp32_enabled"] = bool(prefill_chunk1_fp32_enabled)
    if infer_chunk1_fp32_enabled and not prefill_chunk1_fp32_enabled:
        _log(
            "[warn] infer chunk1 uses FFN_attn_fp32, but prefill chunk1 does not. "
            "Batch-prefill path will not use FP32 layer0. "
            "Use token-infer prefill for parity."
        )

    for chunk_idx in selected_chunks:
        _log(f"[combine] Chunk {chunk_idx}/{args.num_chunks}: prepare inputs")
        out_pkg = out_dir / f"{output_base}_chunk_{chunk_idx:02d}of{args.num_chunks:02d}.mlpackage"
        infer_out_pkg = (
            out_dir / f"{output_base_infer}_chunk_{chunk_idx:02d}of{args.num_chunks:02d}.mlpackage"
        )
        prefill_out_pkg = (
            out_dir / f"{output_base_prefill}_chunk_{chunk_idx:02d}of{args.num_chunks:02d}.mlpackage"
        )

        # For the LUT6 combined package, always use resolved_infer_kinds
        # (e.g. "FFN" = MLP-only).  The FP32 attn models (layer 0 attention)
        # go into a *separate* package built later.
        infer_sources: list[tuple[int, Path, str]] = []
        fp32_infer_sources: list[tuple[int, Path, str]] = []
        for ctx, model_dir in context_dirs.items():
            infer_kind = resolved_infer_kinds[ctx]
            infer_path = _find_chunk_model(
                model_dir=model_dir,
                prefix=args.prefix,
                kind=infer_kind,
                chunk_idx=chunk_idx,
                num_chunks=args.num_chunks,
            )
            infer_sources.append((ctx, infer_path, infer_kind))
            # Collect FP32 attn infer sources separately for chunk 1
            if chunk_idx == 1:
                fp32_kind = resolved_chunk1_infer_kinds[ctx]
                if fp32_kind != infer_kind:
                    fp32_path = _find_chunk_model(
                        model_dir=model_dir,
                        prefix=args.prefix,
                        kind=fp32_kind,
                        chunk_idx=chunk_idx,
                        num_chunks=args.num_chunks,
                    )
                    fp32_infer_sources.append((ctx, fp32_path, fp32_kind))

        prefill_sources: list[tuple[int, Path, str]] = []
        for ctx in prefill_contexts:
            model_dir = context_dirs[ctx]
            prefill_kind = resolved_prefill_kinds[ctx]
            prefill_path = _find_chunk_model(
                model_dir=model_dir,
                prefix=args.prefix,
                kind=prefill_kind,
                chunk_idx=chunk_idx,
                num_chunks=args.num_chunks,
            )
            prefill_sources.append((ctx, prefill_path, prefill_kind))

        max_infer_path = next(path for (ctx, path, _) in infer_sources if ctx == max_context)
        max_prefill_path = next(path for (ctx, path, _) in prefill_sources if ctx == max_context)
        infer_fn_names = [f"infer_ctx{c}" for c in context_list]
        prefill_fn_names = (
            [f"prefill_ctx{c}" for c in prefill_contexts]
            if include_prefill_ctx_functions
            else []
        )
        has_aliases = not args.no_alias_functions
        infer_default_fn = "infer" if has_aliases else f"infer_ctx{max_context}"
        prefill_default_fn = "prefill" if has_aliases else f"prefill_ctx{max_context}"

        def _prepare_output_path(path: Path) -> None:
            if path.exists() and not args.force:
                raise FileExistsError(
                    f"Output chunk already exists: {path}. Use --force to overwrite."
                )
            if path.exists():
                shutil.rmtree(path)

        def _collect_infer_sources() -> list[tuple[str, str, str]]:
            result: list[tuple[str, str, str]] = []
            for ctx, infer_path, _ in infer_sources:
                src_fn = _resolve_source_function(
                    infer_path, args.infer_fn, ct, alternates=["infer"]
                )
                result.append((str(infer_path), src_fn, f"infer_ctx{ctx}"))
            if has_aliases:
                infer_src_fn = _resolve_source_function(
                    max_infer_path, args.infer_fn, ct, alternates=["infer"]
                )
                result.append((str(max_infer_path), infer_src_fn, "infer"))
            return result

        def _collect_prefill_sources() -> list[tuple[str, str, str]]:
            result: list[tuple[str, str, str]] = []
            if include_prefill_ctx_functions:
                for ctx, prefill_path, _ in prefill_sources:
                    src_fn = _resolve_source_function(
                        prefill_path, args.prefill_fn, ct, alternates=["prefill", "infer"]
                    )
                    result.append((str(prefill_path), src_fn, f"prefill_ctx{ctx}"))
            if has_aliases:
                prefill_src_fn = _resolve_source_function(
                    max_prefill_path, args.prefill_fn, ct, alternates=["prefill", "infer"]
                )
                result.append((str(max_prefill_path), prefill_src_fn, "prefill"))
            return result

        infer_compiled = None
        prefill_compiled = None
        combined_compiled = None

        if args.split_infer_prefill:
            _prepare_output_path(infer_out_pkg)
            _prepare_output_path(prefill_out_pkg)

            infer_src_list = _collect_infer_sources()
            infer_count = len(infer_src_list)
            _log(
                f"[combine] Chunk {chunk_idx}/{args.num_chunks}: "
                f"save infer package ({infer_count} functions)"
            )
            if args.dedup_diagnostics:
                _set_dedup_diag_label(f"chunk{chunk_idx:02d}:infer")
            _save_multifunction_with_dedup(
                sources=infer_src_list,
                output_path=str(infer_out_pkg),
                default_function_name=infer_default_fn,
                anemll_dedup=args.anemll_dedup,
                label=f"chunk{chunk_idx:02d}:infer",
            )
            _log(f"[combine] Chunk {chunk_idx}/{args.num_chunks}: saved {infer_out_pkg.name}")

            prefill_src_list = _collect_prefill_sources()
            prefill_count = len(prefill_src_list)
            _log(
                f"[combine] Chunk {chunk_idx}/{args.num_chunks}: "
                f"save prefill package ({prefill_count} functions)"
            )
            if args.dedup_diagnostics:
                _set_dedup_diag_label(f"chunk{chunk_idx:02d}:prefill")
            _save_multifunction_with_dedup(
                sources=prefill_src_list,
                output_path=str(prefill_out_pkg),
                default_function_name=prefill_default_fn,
                anemll_dedup=args.anemll_dedup,
                label=f"chunk{chunk_idx:02d}:prefill",
            )
            _log(f"[combine] Chunk {chunk_idx}/{args.num_chunks}: saved {prefill_out_pkg.name}")

            if not args.no_compile:
                _log(f"[combine] Chunk {chunk_idx}/{args.num_chunks}: compile infer package")
                infer_compiled = _compile_mlpackage(infer_out_pkg, out_dir)
                _log(
                    f"[combine] Chunk {chunk_idx}/{args.num_chunks}: "
                    f"compiled {infer_compiled.name}"
                )
                _log(f"[combine] Chunk {chunk_idx}/{args.num_chunks}: compile prefill package")
                prefill_compiled = _compile_mlpackage(prefill_out_pkg, out_dir)
                _log(
                    f"[combine] Chunk {chunk_idx}/{args.num_chunks}: "
                    f"compiled {prefill_compiled.name}"
                )
        else:
            _prepare_output_path(out_pkg)
            infer_src_list = _collect_infer_sources()
            prefill_src_list = _collect_prefill_sources()
            total_functions = len(infer_src_list) + len(prefill_src_list)
            _log(
                f"[combine] Chunk {chunk_idx}/{args.num_chunks}: "
                f"save multifunction package ({total_functions} functions: "
                f"{len(infer_src_list)} infer + {len(prefill_src_list)} prefill)"
            )
            if args.dedup_diagnostics:
                _set_dedup_diag_label(f"chunk{chunk_idx:02d}:combined")
            # Check if infer and prefill have different weight structures.
            # When chunk 1 uses FFN_attn_fp32 (FP32 attention-only, ~49 weights)
            # and prefill uses full LUT6 (~662 weights), a single dedup anchor
            # can't cover both.  Split into groups so each gets its own anchor.
            # For chunks where both are LUT6 (even with different kind names
            # like "FFN" vs "prefill"), use a single flat list for cross-dedup.
            has_fp32_attn = any(
                kind == "FFN_attn_fp32" for _, _, kind in infer_sources
            )
            if has_fp32_attn and prefill_src_list:
                groups = [g for g in [infer_src_list, prefill_src_list] if g]
                _log(
                    f"[combine] Chunk {chunk_idx}: FP32 attn infer + LUT prefill, "
                    f"using {len(groups)} dedup groups"
                )
                _save_multifunction_with_dedup(
                    sources=[],  # ignored when source_groups is set
                    source_groups=groups,
                    output_path=str(out_pkg),
                    default_function_name=infer_default_fn,
                    anemll_dedup=args.anemll_dedup,
                    label=f"chunk{chunk_idx:02d}:combined",
                )
            else:
                all_sources = infer_src_list + prefill_src_list
                _save_multifunction_with_dedup(
                    sources=all_sources,
                    output_path=str(out_pkg),
                    default_function_name=infer_default_fn,
                    anemll_dedup=args.anemll_dedup,
                    label=f"chunk{chunk_idx:02d}:combined",
                )
            _log(f"[combine] Chunk {chunk_idx}/{args.num_chunks}: saved {out_pkg.name}")
            if not args.no_compile:
                _log(f"[combine] Chunk {chunk_idx}/{args.num_chunks}: compile to .mlmodelc")
                combined_compiled = _compile_mlpackage(out_pkg, out_dir)
                _log(
                    f"[combine] Chunk {chunk_idx}/{args.num_chunks}: "
                    f"compiled {combined_compiled.name}"
                )

        # --- FP32 attn: separate combined package for chunk 1 ---------------
        fp32_attn_compiled = None
        fp32_attn_out_pkg = None
        if chunk_idx == 1 and infer_chunk1_fp32_enabled:
            fp32_base = f"{args.prefix}_FFN_attn_fp32_statex"
            fp32_attn_out_pkg = (
                out_dir / f"{fp32_base}_chunk_{chunk_idx:02d}of{args.num_chunks:02d}.mlpackage"
            )
            _prepare_output_path(fp32_attn_out_pkg)

            # Collect FP32 attn infer sources from fp32_infer_sources
            # (separated from infer_sources so the LUT6 combined package
            # uses MLP-only models, not FP32 attn models)
            fp32_src_list: list[tuple[str, str, str]] = []
            for ctx, fp32_path, kind in fp32_infer_sources:
                src_fn = _resolve_source_function(
                    fp32_path, args.infer_fn, ct, alternates=["infer"]
                )
                fp32_src_list.append((str(fp32_path), src_fn, f"infer_ctx{ctx}"))
            if has_aliases:
                max_fp32_path = next(
                    (path for ctx, path, kind in fp32_infer_sources
                     if ctx == max_context),
                    None,
                )
                if max_fp32_path is not None:
                    fp32_alias_fn = _resolve_source_function(
                        max_fp32_path, args.infer_fn, ct, alternates=["infer"]
                    )
                    fp32_src_list.append((str(max_fp32_path), fp32_alias_fn, "infer"))

            # Collect FP32 attn prefill sources (prefill_attn_fp32 variant)
            fp32_prefill_src_list: list[tuple[str, str, str]] = []
            for ctx in prefill_contexts:
                model_dir = context_dirs[ctx]
                try:
                    fp32_prefill_path = _find_chunk_model(
                        model_dir=model_dir,
                        prefix=args.prefix,
                        kind="prefill_attn_fp32",
                        chunk_idx=chunk_idx,
                        num_chunks=args.num_chunks,
                    )
                except FileNotFoundError:
                    fp32_prefill_path = None
                if fp32_prefill_path is not None:
                    src_fn = _resolve_source_function(
                        fp32_prefill_path, args.prefill_fn, ct,
                        alternates=["prefill", "main"],
                    )
                    fp32_prefill_src_list.append(
                        (str(fp32_prefill_path), src_fn, f"prefill_ctx{ctx}")
                    )
            if fp32_prefill_src_list:
                if has_aliases:
                    max_fp32_pf = next(
                        (p for p, _, fn in fp32_prefill_src_list
                         if fn == f"prefill_ctx{max_context}"),
                        None,
                    )
                    if max_fp32_pf is not None:
                        pf_alias_fn = _resolve_source_function(
                            Path(max_fp32_pf), args.prefill_fn, ct,
                            alternates=["prefill", "main"],
                        )
                        fp32_prefill_src_list.append(
                            (max_fp32_pf, pf_alias_fn, "prefill")
                        )
                fp32_src_list.extend(fp32_prefill_src_list)
                _log(
                    f"[combine] Chunk {chunk_idx}: added {len(fp32_prefill_src_list)} "
                    f"FP32 prefill functions to FP32 attn package"
                )

            if fp32_src_list:
                fp32_default = "infer" if has_aliases else f"infer_ctx{max_context}"
                _log(
                    f"[combine] Chunk {chunk_idx}/{args.num_chunks}: "
                    f"save FP32 attn package ({len(fp32_src_list)} functions)"
                )
                _save_multifunction_with_dedup(
                    sources=fp32_src_list,
                    output_path=str(fp32_attn_out_pkg),
                    default_function_name=fp32_default,
                    anemll_dedup=args.anemll_dedup,
                    label=f"chunk{chunk_idx:02d}:fp32_attn",
                )
                _log(
                    f"[combine] Chunk {chunk_idx}/{args.num_chunks}: "
                    f"saved {fp32_attn_out_pkg.name}"
                )
                if not args.no_compile:
                    _log(
                        f"[combine] Chunk {chunk_idx}/{args.num_chunks}: "
                        f"compile FP32 attn package"
                    )
                    fp32_attn_compiled = _compile_mlpackage(fp32_attn_out_pkg, out_dir)
                    _log(
                        f"[combine] Chunk {chunk_idx}/{args.num_chunks}: "
                        f"compiled {fp32_attn_compiled.name}"
                    )
            else:
                fp32_attn_out_pkg = None

        chunk_entry: dict[str, object] = {
            "chunk": chunk_idx,
            "split_infer_prefill": bool(args.split_infer_prefill),
            "no_alias_functions": bool(args.no_alias_functions),
            "infer_sources": {str(ctx): str(path) for ctx, path, _ in infer_sources},
            "infer_source_kinds": {str(ctx): kind for ctx, _, kind in infer_sources},
            "prefill_sources": {str(ctx): str(path) for ctx, path, _ in prefill_sources},
            "prefill_source_kinds": {str(ctx): kind for ctx, _, kind in prefill_sources},
            "infer_functions": infer_fn_names + (["infer"] if has_aliases else []),
            "prefill_functions": prefill_fn_names + (["prefill"] if has_aliases else []),
        }
        if fp32_attn_out_pkg is not None:
            chunk_entry["fp32_attn_output"] = fp32_attn_out_pkg.name
            chunk_entry["fp32_attn_compiled_output"] = (
                fp32_attn_compiled.name if fp32_attn_compiled is not None else None
            )
        if args.split_infer_prefill:
            chunk_entry["infer_output"] = infer_out_pkg.name
            chunk_entry["infer_compiled_output"] = (
                infer_compiled.name if infer_compiled is not None else None
            )
            chunk_entry["prefill_output"] = prefill_out_pkg.name
            chunk_entry["prefill_compiled_output"] = (
                prefill_compiled.name if prefill_compiled is not None else None
            )
        else:
            chunk_entry["output"] = out_pkg.name
            chunk_entry["compiled_output"] = (
                combined_compiled.name if combined_compiled is not None else None
            )
            chunk_entry["functions"] = (
                infer_fn_names
                + prefill_fn_names
                + (["infer", "prefill"] if has_aliases else [])
            )
        manifest["chunks"].append(chunk_entry)

        if args.split_infer_prefill:
            built = [infer_out_pkg.name, prefill_out_pkg.name]
            if infer_compiled is not None:
                built.append(infer_compiled.name)
            if prefill_compiled is not None:
                built.append(prefill_compiled.name)
            _log(f"Built chunk {chunk_idx}/{args.num_chunks}: " + " + ".join(built))
        elif combined_compiled is not None:
            _log(
                f"Built chunk {chunk_idx}/{args.num_chunks}: "
                f"{out_pkg.name} + {combined_compiled.name}"
            )
        else:
            _log(f"Built chunk {chunk_idx}/{args.num_chunks}: {out_pkg.name}")
        if fp32_attn_out_pkg is not None:
            extra = fp32_attn_out_pkg.name
            if fp32_attn_compiled is not None:
                extra += f" + {fp32_attn_compiled.name}"
            _log(f"Built chunk {chunk_idx}/{args.num_chunks} FP32 attn: {extra}")

    if not args.no_meta:
        _write_meta_yaml(
            out_dir=out_dir,
            model_name=args.model_name,
            architecture=args.architecture,
            prefix=args.prefix,
            context_length=max_context,
            batch_size=resolved_batch_size,
            num_chunks=args.num_chunks,
            split_lm_head=resolved_split_lm_head,
            contexts=context_list,
            max_context=max_context,
            output_base=output_base,
            output_base_infer=(output_base_infer if args.split_infer_prefill else None),
            output_base_prefill=(output_base_prefill if args.split_infer_prefill else None),
            split_infer_prefill=bool(args.split_infer_prefill),
            no_alias_functions=bool(args.no_alias_functions),
            embeddings_name=copied_shared.get("embeddings"),
            lm_head_name=copied_shared.get("lm_head"),
            chunk1_infer_kinds=resolved_chunk1_infer_kinds,
            prefill_chunk1_kinds=prefill_chunk1_kinds,
            include_prefill_ctx_functions=include_prefill_ctx_functions,
            prefill_contexts=prefill_contexts,
            recommended_sampling=recommended_sampling,
            tokenizer_path=tokenizer_path,
            copied_source_chunks=copied_source_chunks_summary,
        )

    (out_dir / "state_transition_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    _log("Done")
    _log(f"Output dir: {out_dir}")
    _log(f"Contexts: {context_list}")
    if args.prefill_all_contexts:
        _log(f"Prefill contexts: {context_list}")
    else:
        _log(f"Prefill context: {max_context}")
    _log(f"Output base: {output_base}")
    if args.split_infer_prefill:
        _log(f"Infer output base: {output_base_infer}")
        _log(f"Prefill output base: {output_base_prefill}")
    _log(f"Alias functions enabled: {not args.no_alias_functions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
