#!/usr/bin/env python3
"""Combine infer-only context exports into multi-function chunk packages.

Expected input layout (from scripts/export_vibethinker_infer_contexts.sh):
- Max context dir: embeddings + lm_head + FFN chunks + prefill chunks (all .mlpackage)
- Other context dirs: FFN chunks only
- Optional per-context standalone FP32 attention artifact for chunk 01:
  <prefix>_FFN_attn_fp32_chunk_01ofNN(.mlpackage/.mlmodelc)

Output per chunk:
- infer_ctx{N} for each context
- infer alias (from max context)
- prefill (from max context prefill chunk)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


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


def _resolve_prefill_kind_for_max_context(
    requested_kind: str,
    model_dir: Path,
    prefix: str,
    num_chunks: int,
    max_context: int,
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
            f"Missing prefill chunks in max context {max_context} ({model_dir}). "
            f"Tried kinds: {candidates}"
        )
    if requested not in ("", "auto") and resolved != requested:
        print(
            f"[combine] Max context {max_context}: prefill kind '{requested}' not found; "
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


def _resolve_source_function(model_path: Path, preferred: str, ct_module) -> str:
    # Most source exports used in state-transition combine are single-function
    # "main" models. Fast-path this to avoid loading each model spec.
    if preferred in ("", "main"):
        return "main"

    names = _available_functions(model_path, ct_module)
    if preferred in names:
        return preferred
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
    embeddings_name: str | None,
    lm_head_name: str | None,
    chunk1_infer_kinds: dict[int, str],
    recommended_sampling: dict | None,
    tokenizer_path: str | None,
) -> None:
    yaml = _require_yaml()

    lut_embeddings = _lut_from_name(embeddings_name)
    lut_lmhead = _lut_from_name(lm_head_name)
    lut_ffn = _lut_from_name(output_base)
    chunk1_kind_values = sorted(set(chunk1_infer_kinds.values()))
    chunk1_kind_summary = chunk1_kind_values[0] if len(chunk1_kind_values) == 1 else "mixed"
    fp32_chunk1_contexts = [
        int(ctx) for ctx, kind in chunk1_infer_kinds.items() if kind == "FFN_attn_fp32"
    ]

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
        "ffn": f"{output_base}_chunk_01of{num_chunks:02d}.mlpackage",
        "split_lm_head": int(split_lm_head),
        "argmax_in_model": False,
        "state_transition_infer_contexts": [int(c) for c in contexts],
        "state_transition_infer_function_template": "infer_ctx{context}",
        "state_transition_prefill_context": int(max_context),
        "state_transition_chunk1_infer_kind": chunk1_kind_summary,
        "state_transition_chunk1_infer_kinds": {
            str(ctx): kind for ctx, kind in chunk1_infer_kinds.items()
        },
        "state_transition_chunk1_fp32_kind": "FFN_attn_fp32",
        "state_transition_chunk1_fp32_contexts": fp32_chunk1_contexts,
        "state_transition_chunk1_fp32_enabled": bool(fp32_chunk1_contexts),
    }
    if isinstance(recommended_sampling, dict):
        params["recommended_sampling"] = recommended_sampling
    if tokenizer_path:
        params["tokenizer_path"] = str(tokenizer_path)

    meta = {
        "model_info": {
            "name": model_name,
            "version": "0.3.5",
            "description": (
                f"Inference state-transition chunks with infer_ctx functions "
                f"for contexts {contexts}; prefill from {max_context}"
            ),
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
            "(infer_ctx* + infer alias + prefill from max context)."
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to write to meta.yaml (default: 32)",
    )
    ap.add_argument(
        "--split-lm-head",
        type=int,
        default=16,
        help="split_lm_head to write to meta.yaml (default: 16)",
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
    args = ap.parse_args()

    if args.num_chunks <= 0:
        raise ValueError("--num-chunks must be > 0")

    ct = _require_coremltools()
    yaml = _require_yaml()

    context_dirs = _parse_context_entries(args.contexts)
    context_list = list(context_dirs.keys())
    max_context = args.max_context if args.max_context is not None else max(context_list)
    if max_context not in context_dirs:
        raise ValueError(f"max_context={max_context} not present in --contexts")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    output_base = args.output_base or f"{args.prefix}_FFN_PF_statex"
    max_ctx_dir = context_dirs[max_context]

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
        "output_base": output_base,
        "infer_kind": args.infer_kind,
        "infer_chunk1_kind": args.infer_chunk1_kind,
        "prefill_kind": args.prefill_kind,
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
    resolved_prefill_kind = _resolve_prefill_kind_for_max_context(
        requested_kind=args.prefill_kind,
        model_dir=max_ctx_dir,
        prefix=args.prefix,
        num_chunks=args.num_chunks,
        max_context=max_context,
    )
    print(
        "[combine] Resolved infer kinds: "
        + ", ".join(f"{ctx}={resolved_infer_kinds[ctx]}" for ctx in context_list)
    )
    print(
        f"[combine] Resolved prefill kind (max context {max_context}): "
        f"{resolved_prefill_kind}"
    )
    manifest["resolved_infer_kinds"] = {str(k): v for k, v in resolved_infer_kinds.items()}
    manifest["resolved_prefill_kind"] = resolved_prefill_kind

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
    print(
        "[combine] Resolved chunk1 infer kinds: "
        + ", ".join(f"{ctx}={resolved_chunk1_infer_kinds[ctx]}" for ctx in context_list)
    )
    manifest["resolved_chunk1_infer_kinds"] = {
        str(k): v for k, v in resolved_chunk1_infer_kinds.items()
    }

    for chunk_idx in range(1, args.num_chunks + 1):
        out_pkg = out_dir / f"{output_base}_chunk_{chunk_idx:02d}of{args.num_chunks:02d}.mlpackage"
        if out_pkg.exists() and not args.force:
            raise FileExistsError(
                f"Output chunk already exists: {out_pkg}. Use --force to overwrite."
            )
        if out_pkg.exists():
            shutil.rmtree(out_pkg)

        infer_sources: list[tuple[int, Path, str]] = []
        for ctx, model_dir in context_dirs.items():
            if chunk_idx == 1:
                infer_kind = resolved_chunk1_infer_kinds[ctx]
            else:
                infer_kind = resolved_infer_kinds[ctx]
            infer_path = _find_chunk_model(
                model_dir=model_dir,
                prefix=args.prefix,
                kind=infer_kind,
                chunk_idx=chunk_idx,
                num_chunks=args.num_chunks,
            )
            infer_sources.append((ctx, infer_path, infer_kind))

        max_infer_path = next(path for (ctx, path, _) in infer_sources if ctx == max_context)
        max_prefill_path = _find_chunk_model(
            model_dir=max_ctx_dir,
            prefix=args.prefix,
            kind=resolved_prefill_kind,
            chunk_idx=chunk_idx,
            num_chunks=args.num_chunks,
        )

        desc = ct.utils.MultiFunctionDescriptor()

        for ctx, infer_path, _ in infer_sources:
            src_fn = _resolve_source_function(infer_path, args.infer_fn, ct)
            desc.add_function(str(infer_path), src_fn, f"infer_ctx{ctx}")

        infer_src_fn = _resolve_source_function(max_infer_path, args.infer_fn, ct)
        prefill_src_fn = _resolve_source_function(max_prefill_path, args.prefill_fn, ct)
        desc.add_function(str(max_infer_path), infer_src_fn, "infer")
        desc.add_function(str(max_prefill_path), prefill_src_fn, "prefill")
        desc.default_function_name = "infer"

        ct.utils.save_multifunction(desc, str(out_pkg))
        compiled_chunk = None
        if not args.no_compile:
            compiled_chunk = _compile_mlpackage(out_pkg, out_dir)

        manifest["chunks"].append(
            {
                "chunk": chunk_idx,
                "output": out_pkg.name,
                "compiled_output": compiled_chunk.name if compiled_chunk is not None else None,
                "infer_sources": {str(ctx): str(path) for ctx, path, _ in infer_sources},
                "infer_source_kinds": {str(ctx): kind for ctx, _, kind in infer_sources},
                "prefill_source": str(max_prefill_path),
                "functions": [f"infer_ctx{c}" for c in context_list] + ["infer", "prefill"],
            }
        )
        if compiled_chunk is not None:
            print(
                f"Built chunk {chunk_idx}/{args.num_chunks}: "
                f"{out_pkg.name} + {compiled_chunk.name}"
            )
        else:
            print(f"Built chunk {chunk_idx}/{args.num_chunks}: {out_pkg.name}")

    if not args.no_meta:
        _write_meta_yaml(
            out_dir=out_dir,
            model_name=args.model_name,
            architecture=args.architecture,
            prefix=args.prefix,
            context_length=max_context,
            batch_size=args.batch_size,
            num_chunks=args.num_chunks,
            split_lm_head=args.split_lm_head,
            contexts=context_list,
            max_context=max_context,
            output_base=output_base,
            embeddings_name=copied_shared.get("embeddings"),
            lm_head_name=copied_shared.get("lm_head"),
            chunk1_infer_kinds=resolved_chunk1_infer_kinds,
            recommended_sampling=recommended_sampling,
            tokenizer_path=tokenizer_path,
        )

    (out_dir / "state_transition_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    print("\nDone")
    print(f"Output dir: {out_dir}")
    print(f"Contexts: {context_list}")
    print(f"Prefill context: {max_context}")
    print(f"Output base: {output_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
