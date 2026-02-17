#!/usr/bin/env python3
# Copyright (c) 2026, Anemll. All rights reserved.
#
# Use of this source code is governed by an MIT license that can be
# found in the LICENSE.txt file or at https://opensource.org/license/mit

"""Combine per-context CoreML exports into state-transition multifunction models.

This utility merges chunked per-context exports (e.g. ctx512/1024/2048/4096)
into a single output folder whose chunk models expose:
  - infer_ctx{context}
  - prefill_ctx{context}

Optional modes:
  - split infer/prefill into separate chunk files
  - omit alias functions (infer/prefill)
  - add chunk1 specialized stage(s), e.g. FFN_attn_fp32
  - compile generated mlpackage outputs to mlmodelc

It also writes state-transition metadata into output meta.yaml so
tests/dev/state_transition_growing_inference.py can load it directly.
"""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


if __name__ == "__main__":
    # Allow `python anemll/utils/combine_state_transition_contexts.py ...`
    package_root = str(Path(__file__).resolve().parents[2])
    if package_root not in sys.path:
        sys.path.insert(0, package_root)

try:
    import coremltools as ct
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "coremltools is required. Activate env-anemll or install dependencies first."
    ) from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Activate env-anemll or install dependencies first.") from exc


KNOWN_KINDS: Tuple[str, ...] = (
    "FFN_attn_fp32",
    "prefill_attn_fp32",
    "FFN_PF",
    "FFN",
    "prefill",
)

COMMON_FILES_TO_COPY: Tuple[str, ...] = (
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "tokenizer_config_search.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
)

KIND_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<kind>FFN_attn_fp32|prefill_attn_fp32|FFN_PF|FFN|prefill)(?P<suffix>(?:_.+)?)$"
)
CHUNK_RE = re.compile(r"^(.+)_chunk_(\d+)of(\d+)$")


@dataclass
class ContextExport:
    context: int
    model_dir: Path
    meta: dict
    params: dict
    ffn_base: str
    ffn_prefill_base: Optional[str]
    num_chunks: int
    batch_size: int
    model_prefix: Optional[str]


@dataclass
class ChunkBuildInfo:
    chunk: int
    infer_sources: Dict[str, str]
    prefill_sources: Dict[str, str]
    infer_functions: List[str]
    prefill_functions: List[str]
    output: Optional[str]
    output_infer: Optional[str]
    output_prefill: Optional[str]


def _parse_context_list(raw: str) -> List[int]:
    vals: List[int] = []
    for part in str(raw).replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return sorted(dict.fromkeys(vals))


def _parse_context_dir_entries(entries: Sequence[str]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"Invalid --context-dir entry '{raw}'. Expected N=/path")
        lhs, rhs = raw.split("=", 1)
        ctx = int(lhs.strip())
        p = Path(rhs).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        out[ctx] = p
    return out


def _resolve_context_dirs(
    *,
    contexts: Sequence[int],
    context_root: Path,
    name_template: str,
    explicit_context_dirs: Dict[int, Path],
) -> Dict[int, Path]:
    if explicit_context_dirs:
        missing = [c for c in contexts if c not in explicit_context_dirs]
        if missing:
            raise ValueError(f"Missing contexts in --context-dir entries: {missing}")
        return {c: explicit_context_dirs[c] for c in contexts}

    out: Dict[int, Path] = {}
    for ctx in contexts:
        path = (context_root / name_template.format(context=ctx)).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Missing context directory for {ctx}: {path} "
                "(set --context-dir for explicit mapping if needed)"
            )
        out[ctx] = path
    return out


def _strip_model_ext(value: str) -> str:
    return re.sub(r"\.(mlmodelc|mlpackage)$", "", value)


def _split_chunk_stem(value: str) -> Tuple[str, int]:
    stem = _strip_model_ext(str(value))
    m = CHUNK_RE.match(stem)
    if not m:
        raise ValueError(f"Could not parse chunk stem from '{value}'")
    return m.group(1), int(m.group(3))


def _classify_kind(base: str) -> Optional[str]:
    if re.search(r"_FFN_attn_fp32(?:_|$)", base):
        return "FFN_attn_fp32"
    if re.search(r"_prefill_attn_fp32(?:_|$)", base):
        return "prefill_attn_fp32"
    if re.search(r"_FFN_PF(?:_|$)", base):
        return "FFN_PF"
    if re.search(r"_FFN(?:_|$)", base):
        return "FFN"
    if re.search(r"_prefill(?:_|$)", base):
        return "prefill"
    return None


def _derive_kind_base(base: str, target_kind: str) -> Optional[str]:
    m = KIND_RE.match(base)
    if not m:
        return None
    return f"{m.group('prefix')}_{target_kind}{m.group('suffix') or ''}"


def _format_fn(template: str, context: int) -> str:
    try:
        return str(template).format(context=int(context))
    except Exception:
        return str(template)


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML object in {path}")
    return data


def _scan_kind_bases(
    *,
    model_dir: Path,
    chunk_idx: int,
    num_chunks: int,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    pattern = f"*_chunk_{chunk_idx:02d}of{num_chunks:02d}.mlpackage"
    for pkg in sorted(model_dir.glob(pattern)):
        stem = _strip_model_ext(pkg.name)
        m = CHUNK_RE.match(stem)
        if not m:
            continue
        base = m.group(1)
        kind = _classify_kind(base)
        if not kind:
            continue
        out.setdefault(kind, []).append(base)
    return out


def _resolve_source_path(
    *,
    export: ContextExport,
    chunk_idx: int,
    num_chunks: int,
    kind: str,
    role: str,
) -> Tuple[Path, str]:
    if role not in ("infer", "prefill"):
        raise ValueError(f"Invalid role: {role}")

    base_for_role = export.ffn_base if role == "infer" else (export.ffn_prefill_base or export.ffn_base)
    candidates: List[str] = []

    if base_for_role:
        if _classify_kind(base_for_role) == kind:
            candidates.append(base_for_role)
        derived = _derive_kind_base(base_for_role, kind)
        if derived:
            candidates.append(derived)

    if export.ffn_base not in candidates and _classify_kind(export.ffn_base) == kind:
        candidates.append(export.ffn_base)
    if export.ffn_prefill_base and export.ffn_prefill_base not in candidates:
        if _classify_kind(export.ffn_prefill_base) == kind:
            candidates.append(export.ffn_prefill_base)

    seen = set()
    unique_candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    for base in unique_candidates:
        path = export.model_dir / f"{base}_chunk_{chunk_idx:02d}of{num_chunks:02d}.mlpackage"
        if path.exists():
            return path, base

    scanned = _scan_kind_bases(model_dir=export.model_dir, chunk_idx=chunk_idx, num_chunks=num_chunks)
    for base in scanned.get(kind, []):
        path = export.model_dir / f"{base}_chunk_{chunk_idx:02d}of{num_chunks:02d}.mlpackage"
        if path.exists():
            return path, base

    tried = [
        str(export.model_dir / f"{base}_chunk_{chunk_idx:02d}of{num_chunks:02d}.mlpackage")
        for base in unique_candidates
    ]
    raise FileNotFoundError(
        "Missing source model for context={} role={} kind={} chunk={:02d}of{:02d}\n"
        "Tried:\n  - {}\n"
        "Scanned {} candidates for this chunk: {}".format(
            export.context,
            role,
            kind,
            chunk_idx,
            num_chunks,
            "\n  - ".join(tried) if tried else "<none>",
            export.model_dir,
            sorted(scanned.keys()),
        )
    )


def _make_descriptor_sources(
    *,
    path_fn_pairs: Iterable[Tuple[Path, str]],
) -> List[Tuple[str, str, str]]:
    sources: List[Tuple[str, str, str]] = []
    for path, target_fn in path_fn_pairs:
        sources.append((str(path), "main", target_fn))
    return sources


def _save_multifunction(
    *,
    sources: List[Tuple[str, str, str]],
    output_path: Path,
    dedup_weights: bool,
    verbose: bool,
) -> None:
    if not sources:
        raise ValueError(f"No sources for {output_path}")

    prepare_dedup_sources = None
    if dedup_weights:
        try:
            from anemll.utils.dedup_weights import prepare_dedup_sources as _prepare

            prepare_dedup_sources = _prepare
        except Exception:
            print("Warning: anemll.utils.dedup_weights unavailable; continuing without dedup.")
            dedup_weights = False

    ctx = (
        prepare_dedup_sources(sources, verbose=verbose)
        if dedup_weights and prepare_dedup_sources is not None and len(sources) > 1
        else nullcontext(sources)
    )

    with ctx as deduped_sources:
        resolved_sources = list(deduped_sources)
        if not resolved_sources:
            raise ValueError(f"No resolved sources for {output_path}")
        desc = ct.utils.MultiFunctionDescriptor()
        for path, src_fn, target_fn in resolved_sources:
            desc.add_function(path, src_fn, target_fn)
        desc.default_function_name = resolved_sources[0][2]
        ct.utils.save_multifunction(desc, str(output_path))


def _compile_mlpackage(
    *,
    package_path: Path,
    output_dir: Path,
    force_mlprogram: bool = True,
) -> None:
    compiled_path = output_dir / f"{package_path.stem}.mlmodelc"
    if compiled_path.exists():
        shutil.rmtree(compiled_path)

    cmd = ["xcrun", "coremlcompiler", "compile", str(package_path), str(output_dir)]
    if force_mlprogram:
        cmd += ["--add-mlprogram-if-eligible", "force"]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to compile {package_path.name}\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr}"
        )


def _infer_model_prefix(ffn_base: str, fallback: Optional[str]) -> str:
    if fallback:
        return str(fallback)
    m = KIND_RE.match(ffn_base)
    if m:
        return m.group("prefix")
    return ffn_base.split("_")[0]


def _copy_shared_artifacts(
    *,
    source_export: ContextExport,
    output_dir: Path,
    copy_hf_dist: bool,
) -> None:
    params = source_export.params
    src = source_export.model_dir

    for name in COMMON_FILES_TO_COPY:
        p = src / name
        if p.exists():
            _copy_path(p, output_dir / name)

    for key in ("embeddings", "lm_head"):
        raw = str(params.get(key, "")).strip()
        if not raw:
            continue
        stem = _strip_model_ext(raw)
        for ext in (".mlmodelc", ".mlpackage"):
            p = src / f"{stem}{ext}"
            if p.exists():
                _copy_path(p, output_dir / p.name)

    if copy_hf_dist:
        hf_dist = src / "hf_dist"
        if hf_dist.exists():
            _copy_path(hf_dist, output_dir / "hf_dist")


def _state_tag_suffix(state_tag: str) -> str:
    tag = str(state_tag).strip().strip("_")
    return f"_{tag}" if tag else ""


def _build_context_exports(
    *,
    contexts: Sequence[int],
    context_dirs: Dict[int, Path],
) -> Dict[int, ContextExport]:
    exports: Dict[int, ContextExport] = {}
    for ctx in contexts:
        model_dir = context_dirs[ctx]
        meta_path = model_dir / "meta.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.yaml: {meta_path}")
        meta = _load_yaml(meta_path)
        params = meta.get("model_info", {}).get("parameters", {})
        if not isinstance(params, dict):
            raise ValueError(f"Invalid model_info.parameters in {meta_path}")

        ctx_meta = int(params.get("context_length", ctx))
        if ctx_meta != int(ctx):
            raise ValueError(
                f"Context mismatch for {model_dir}: requested {ctx}, meta says {ctx_meta}"
            )

        ffn_raw = str(params.get("ffn", "")).strip()
        if not ffn_raw:
            raise ValueError(f"Missing 'ffn' in {meta_path}")
        ffn_base, num_chunks = _split_chunk_stem(ffn_raw)

        ffn_prefill_raw = str(params.get("ffn_prefill", "")).strip()
        ffn_prefill_base: Optional[str] = None
        if ffn_prefill_raw:
            try:
                parsed_prefill_base, prefill_chunks = _split_chunk_stem(ffn_prefill_raw)
                if prefill_chunks == num_chunks:
                    ffn_prefill_base = parsed_prefill_base
            except Exception:
                # Keep robust when optional metadata is malformed.
                ffn_prefill_base = None

        exports[ctx] = ContextExport(
            context=ctx,
            model_dir=model_dir,
            meta=meta,
            params=params,
            ffn_base=ffn_base,
            ffn_prefill_base=ffn_prefill_base,
            num_chunks=int(params.get("num_chunks", num_chunks)),
            batch_size=int(params.get("batch_size", 32)),
            model_prefix=str(params.get("model_prefix", "")).strip() or None,
        )
    return exports


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-context chunk exports into state-transition multifunction models."
    )
    parser.add_argument(
        "--contexts",
        required=True,
        help='Infer contexts, comma/space-separated (e.g. "512 1024 2048 4096")',
    )
    parser.add_argument(
        "--prefill-contexts",
        default=None,
        help="Optional prefill contexts (defaults to --contexts).",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Context used as default infer/prefill context and base metadata source (default: max of --contexts).",
    )
    parser.add_argument(
        "--context-root",
        default="/Volumes/Models/ANE",
        help="Root directory containing per-context model folders.",
    )
    parser.add_argument(
        "--context-name-template",
        default=None,
        help='Directory template under --context-root (must include "{context}").',
    )
    parser.add_argument(
        "--context-dir",
        action="append",
        default=[],
        help="Explicit context mapping entry N=/path. Repeat for multiple contexts.",
    )
    parser.add_argument("--output", required=True, help="Output directory for combined state-transition model.")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Override number of chunks (default: inferred from meta.yaml).",
    )
    parser.add_argument("--model-prefix", default=None, help="Override model prefix in output names/meta.")
    parser.add_argument("--model-name", default=None, help="Override model_info.name in output meta.yaml.")
    parser.add_argument(
        "--state-tag",
        default="statex",
        help="Suffix tag inserted in output chunk base names (default: statex).",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Explicit output base for non-split chunks (default: <prefix>_FFN_PF_<state-tag>).",
    )
    parser.add_argument(
        "--output-base-infer",
        default=None,
        help="Explicit infer output base for --split-infer-prefill (default: <prefix>_FFN_<state-tag>).",
    )
    parser.add_argument(
        "--output-base-prefill",
        default=None,
        help="Explicit prefill output base for --split-infer-prefill (default: <prefix>_prefill_<state-tag>).",
    )
    parser.add_argument(
        "--infer-kind",
        default="FFN",
        help="Source kind used for infer functions (default: FFN).",
    )
    parser.add_argument(
        "--prefill-kind",
        default="prefill",
        help="Source kind used for prefill functions (default: prefill).",
    )
    parser.add_argument(
        "--infer-chunk1-kind",
        default=None,
        help="Optional extra chunk1 infer stage kind (e.g. FFN_attn_fp32).",
    )
    parser.add_argument(
        "--prefill-chunk1-kind",
        default=None,
        help="Optional extra chunk1 prefill stage kind (e.g. prefill_attn_fp32).",
    )
    parser.add_argument(
        "--infer-function-template",
        default="infer_ctx{context}",
        help='Infer function naming template (default: "infer_ctx{context}").',
    )
    parser.add_argument(
        "--prefill-function-template",
        default="prefill_ctx{context}",
        help='Prefill function naming template (default: "prefill_ctx{context}").',
    )
    parser.add_argument(
        "--prefill-default-context",
        type=int,
        default=None,
        help="Default prefill context for fallback when not all contexts have prefill function.",
    )
    parser.add_argument(
        "--split-infer-prefill",
        action="store_true",
        help="Write separate infer and prefill chunk files instead of a single FFN_PF file.",
    )
    parser.add_argument(
        "--no-alias-functions",
        action="store_true",
        help='Do not add alias functions "infer"/"prefill"; keep only ctx-scoped names.',
    )
    parser.add_argument("--no-compile", action="store_true", help="Skip compilation to .mlmodelc.")
    parser.add_argument("--copy-hf-dist", action="store_true", help="Copy hf_dist directory from max-context export.")
    parser.add_argument(
        "--skip-anemll-dedup",
        action="store_true",
        help="Disable anemll-dedup before save_multifunction.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan only.")
    parser.add_argument("--force", action="store_true", help="Delete existing output directory.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    contexts = _parse_context_list(args.contexts)
    if not contexts:
        raise ValueError("No contexts parsed from --contexts")

    prefill_contexts = _parse_context_list(args.prefill_contexts) if args.prefill_contexts else list(contexts)
    missing_prefill = [c for c in prefill_contexts if c not in contexts]
    if missing_prefill:
        raise ValueError(f"--prefill-contexts has values not in --contexts: {missing_prefill}")

    max_context = int(args.max_context if args.max_context is not None else max(contexts))
    if max_context not in contexts:
        raise ValueError(f"--max-context {max_context} must be included in --contexts {contexts}")

    prefill_default_context = int(
        args.prefill_default_context if args.prefill_default_context is not None else max_context
    )
    if prefill_default_context not in contexts:
        raise ValueError(
            f"--prefill-default-context {prefill_default_context} must be in --contexts {contexts}"
        )

    explicit_context_dirs = _parse_context_dir_entries(args.context_dir)
    context_root = Path(args.context_root).expanduser().resolve()
    if not explicit_context_dirs and not args.context_name_template:
        raise ValueError("Provide --context-name-template (or explicit --context-dir mappings).")

    context_dirs = _resolve_context_dirs(
        contexts=contexts,
        context_root=context_root,
        name_template=str(args.context_name_template or ""),
        explicit_context_dirs=explicit_context_dirs,
    )
    exports = _build_context_exports(contexts=contexts, context_dirs=context_dirs)
    max_export = exports[max_context]

    num_chunks = int(args.num_chunks if args.num_chunks is not None else max_export.num_chunks)
    for ctx in contexts:
        if exports[ctx].num_chunks != num_chunks:
            raise ValueError(
                f"num_chunks mismatch: ctx{ctx} meta={exports[ctx].num_chunks}, expected={num_chunks}"
            )

    model_prefix = _infer_model_prefix(max_export.ffn_base, args.model_prefix or max_export.model_prefix)
    state_suffix = _state_tag_suffix(args.state_tag)

    infer_kind = str(args.infer_kind).strip()
    prefill_kind = str(args.prefill_kind).strip()
    infer_chunk1_kind = str(args.infer_chunk1_kind).strip() if args.infer_chunk1_kind else infer_kind
    prefill_chunk1_kind = str(args.prefill_chunk1_kind).strip() if args.prefill_chunk1_kind else prefill_kind

    infer_chunk1_enabled = infer_chunk1_kind != infer_kind
    prefill_chunk1_enabled = prefill_chunk1_kind != prefill_kind
    infer_chunk1_fp32_enabled = infer_chunk1_enabled and ("fp32" in infer_chunk1_kind.lower())
    prefill_chunk1_fp32_enabled = prefill_chunk1_enabled and ("fp32" in prefill_chunk1_kind.lower())

    if args.output_base:
        output_base = str(args.output_base)
    else:
        output_base = f"{model_prefix}_FFN_PF{state_suffix}"
    output_base_infer = str(args.output_base_infer) if args.output_base_infer else f"{model_prefix}_FFN{state_suffix}"
    output_base_prefill = (
        str(args.output_base_prefill) if args.output_base_prefill else f"{model_prefix}_prefill{state_suffix}"
    )

    chunk1_infer_base = f"{model_prefix}_{infer_chunk1_kind}{state_suffix}" if infer_chunk1_enabled else None
    chunk1_prefill_base = f"{model_prefix}_{prefill_chunk1_kind}{state_suffix}" if prefill_chunk1_enabled else None

    output_dir = Path(args.output).expanduser().resolve()

    print("[plan] contexts:", contexts)
    print("[plan] prefill_contexts:", prefill_contexts)
    print("[plan] max_context:", max_context)
    print("[plan] prefill_default_context:", prefill_default_context)
    print("[plan] output:", output_dir)
    print("[plan] num_chunks:", num_chunks)
    print("[plan] model_prefix:", model_prefix)
    print("[plan] infer_kind/prefill_kind:", infer_kind, "/", prefill_kind)
    if infer_chunk1_enabled:
        print("[plan] chunk1 infer stage:", infer_chunk1_kind, "->", chunk1_infer_base)
    if prefill_chunk1_enabled:
        print("[plan] chunk1 prefill stage:", prefill_chunk1_kind, "->", chunk1_prefill_base)

    if args.dry_run:
        print("[dry-run] no files will be written.")
    else:
        if output_dir.exists():
            if args.force:
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(f"Output exists: {output_dir} (use --force)")
        output_dir.mkdir(parents=True, exist_ok=True)
        _copy_shared_artifacts(source_export=max_export, output_dir=output_dir, copy_hf_dist=args.copy_hf_dist)

    build_infos: List[ChunkBuildInfo] = []
    generated_packages: List[Path] = []

    for chunk_idx in range(1, num_chunks + 1):
        infer_path_fn_pairs: List[Tuple[Path, str]] = []
        prefill_path_fn_pairs: List[Tuple[Path, str]] = []
        infer_sources_manifest: Dict[str, str] = {}
        prefill_sources_manifest: Dict[str, str] = {}
        infer_functions: List[str] = []
        prefill_functions: List[str] = []

        infer_alias_path: Optional[Path] = None
        prefill_alias_path: Optional[Path] = None

        for ctx in contexts:
            export = exports[ctx]
            infer_path, _ = _resolve_source_path(
                export=export,
                chunk_idx=chunk_idx,
                num_chunks=num_chunks,
                kind=infer_kind,
                role="infer",
            )
            infer_fn = _format_fn(args.infer_function_template, ctx)
            infer_path_fn_pairs.append((infer_path, infer_fn))
            infer_sources_manifest[str(ctx)] = str(infer_path)
            infer_functions.append(infer_fn)

            if ctx == max_context:
                infer_alias_path = infer_path

            if ctx in prefill_contexts:
                prefill_path, _ = _resolve_source_path(
                    export=export,
                    chunk_idx=chunk_idx,
                    num_chunks=num_chunks,
                    kind=prefill_kind,
                    role="prefill",
                )
                prefill_fn = _format_fn(args.prefill_function_template, ctx)
                prefill_path_fn_pairs.append((prefill_path, prefill_fn))
                prefill_sources_manifest[str(ctx)] = str(prefill_path)
                prefill_functions.append(prefill_fn)
                if ctx == prefill_default_context:
                    prefill_alias_path = prefill_path

        if prefill_alias_path is None:
            export = exports[prefill_default_context]
            prefill_alias_path, _ = _resolve_source_path(
                export=export,
                chunk_idx=chunk_idx,
                num_chunks=num_chunks,
                kind=prefill_kind,
                role="prefill",
            )

        if not args.no_alias_functions:
            if infer_alias_path is None:
                raise RuntimeError("Internal error: infer alias path unresolved")
            infer_path_fn_pairs.append((infer_alias_path, "infer"))
            prefill_path_fn_pairs.append((prefill_alias_path, "prefill"))

        if args.split_infer_prefill:
            out_infer = output_dir / f"{output_base_infer}_chunk_{chunk_idx:02d}of{num_chunks:02d}.mlpackage"
            out_prefill = output_dir / f"{output_base_prefill}_chunk_{chunk_idx:02d}of{num_chunks:02d}.mlpackage"
            build_infos.append(
                ChunkBuildInfo(
                    chunk=chunk_idx,
                    infer_sources=infer_sources_manifest,
                    prefill_sources=prefill_sources_manifest,
                    infer_functions=infer_functions,
                    prefill_functions=prefill_functions,
                    output=None,
                    output_infer=out_infer.name,
                    output_prefill=out_prefill.name,
                )
            )
            if not args.dry_run:
                _save_multifunction(
                    sources=_make_descriptor_sources(path_fn_pairs=infer_path_fn_pairs),
                    output_path=out_infer,
                    dedup_weights=not args.skip_anemll_dedup,
                    verbose=args.verbose,
                )
                _save_multifunction(
                    sources=_make_descriptor_sources(path_fn_pairs=prefill_path_fn_pairs),
                    output_path=out_prefill,
                    dedup_weights=not args.skip_anemll_dedup,
                    verbose=args.verbose,
                )
            generated_packages.extend([out_infer, out_prefill])
            print(
                f"[chunk {chunk_idx:02d}] split outputs: {out_infer.name} (infer) + {out_prefill.name} (prefill)"
            )
        else:
            out_combined = output_dir / f"{output_base}_chunk_{chunk_idx:02d}of{num_chunks:02d}.mlpackage"
            build_infos.append(
                ChunkBuildInfo(
                    chunk=chunk_idx,
                    infer_sources=infer_sources_manifest,
                    prefill_sources=prefill_sources_manifest,
                    infer_functions=infer_functions,
                    prefill_functions=prefill_functions,
                    output=out_combined.name,
                    output_infer=None,
                    output_prefill=None,
                )
            )
            if not args.dry_run:
                combined_pairs = infer_path_fn_pairs + prefill_path_fn_pairs
                _save_multifunction(
                    sources=_make_descriptor_sources(path_fn_pairs=combined_pairs),
                    output_path=out_combined,
                    dedup_weights=not args.skip_anemll_dedup,
                    verbose=args.verbose,
                )
            generated_packages.append(out_combined)
            print(f"[chunk {chunk_idx:02d}] output: {out_combined.name}")

    chunk1_infer_output_name: Optional[str] = None
    chunk1_prefill_output_name: Optional[str] = None

    if infer_chunk1_enabled:
        infer_chunk1_pairs: List[Tuple[Path, str]] = []
        alias_path: Optional[Path] = None
        for ctx in contexts:
            export = exports[ctx]
            source_path, _ = _resolve_source_path(
                export=export,
                chunk_idx=1,
                num_chunks=num_chunks,
                kind=infer_chunk1_kind,
                role="infer",
            )
            fn = _format_fn(args.infer_function_template, ctx)
            infer_chunk1_pairs.append((source_path, fn))
            if ctx == max_context:
                alias_path = source_path
        if not args.no_alias_functions and alias_path is not None:
            infer_chunk1_pairs.append((alias_path, "infer"))

        chunk1_out = output_dir / f"{chunk1_infer_base}_chunk_01of{num_chunks:02d}.mlpackage"
        chunk1_infer_output_name = chunk1_out.name
        if not args.dry_run:
            _save_multifunction(
                sources=_make_descriptor_sources(path_fn_pairs=infer_chunk1_pairs),
                output_path=chunk1_out,
                dedup_weights=not args.skip_anemll_dedup,
                verbose=args.verbose,
            )
        generated_packages.append(chunk1_out)
        print(f"[chunk1 infer] output: {chunk1_out.name}")

    if prefill_chunk1_enabled:
        prefill_chunk1_pairs: List[Tuple[Path, str]] = []
        alias_path: Optional[Path] = None
        for ctx in prefill_contexts:
            export = exports[ctx]
            source_path, _ = _resolve_source_path(
                export=export,
                chunk_idx=1,
                num_chunks=num_chunks,
                kind=prefill_chunk1_kind,
                role="prefill",
            )
            fn = _format_fn(args.prefill_function_template, ctx)
            prefill_chunk1_pairs.append((source_path, fn))
            if ctx == prefill_default_context:
                alias_path = source_path
        if not args.no_alias_functions and alias_path is not None:
            prefill_chunk1_pairs.append((alias_path, "prefill"))

        chunk1_out = output_dir / f"{chunk1_prefill_base}_chunk_01of{num_chunks:02d}.mlpackage"
        chunk1_prefill_output_name = chunk1_out.name
        if not args.dry_run:
            _save_multifunction(
                sources=_make_descriptor_sources(path_fn_pairs=prefill_chunk1_pairs),
                output_path=chunk1_out,
                dedup_weights=not args.skip_anemll_dedup,
                verbose=args.verbose,
            )
        generated_packages.append(chunk1_out)
        print(f"[chunk1 prefill] output: {chunk1_out.name}")

    if (not args.no_compile) and (not args.dry_run):
        for pkg in generated_packages:
            print(f"[compile] {pkg.name}")
            _compile_mlpackage(package_path=pkg, output_dir=output_dir, force_mlprogram=True)

    # Build output meta.yaml from max-context meta as base.
    output_meta = copy.deepcopy(max_export.meta)
    model_info = output_meta.setdefault("model_info", {})
    params = model_info.setdefault("parameters", {})

    compiled_ext = "mlmodelc" if (not args.no_compile) else "mlpackage"

    if args.split_infer_prefill:
        params["ffn"] = f"{output_base_infer}_chunk_01of{num_chunks:02d}.{compiled_ext}"
        params["ffn_prefill"] = f"{output_base_prefill}_chunk_01of{num_chunks:02d}.{compiled_ext}"
    else:
        params["ffn"] = f"{output_base}_chunk_01of{num_chunks:02d}.{compiled_ext}"
        params.pop("ffn_prefill", None)

    params["context_length"] = int(max_context)
    params["num_chunks"] = int(num_chunks)
    params["batch_size"] = int(max_export.batch_size)
    params["model_prefix"] = model_prefix
    params["state_transition_infer_contexts"] = list(contexts)
    params["state_transition_infer_function_template"] = str(args.infer_function_template)
    params["state_transition_infer_default_function"] = _format_fn(args.infer_function_template, max_context)
    params["state_transition_prefill_context"] = int(prefill_default_context)
    params["state_transition_prefill_default_function"] = _format_fn(
        args.prefill_function_template, prefill_default_context
    )
    params["state_transition_prefill_contexts"] = list(prefill_contexts)
    params["state_transition_prefill_function_template"] = str(args.prefill_function_template)
    params["state_transition_all_context_prefill"] = bool(
        prefill_contexts and (
            len(prefill_contexts) > 1 or prefill_contexts[0] != int(prefill_default_context)
        )
    )
    params["state_transition_combined_functions_layout"] = (
        "infer_ctx|prefill_ctx (split)" if args.split_infer_prefill else "infer_ctx+prefill_ctx"
    )
    params["state_transition_no_alias_functions"] = bool(args.no_alias_functions)
    params["state_transition_split_infer_prefill"] = bool(args.split_infer_prefill)
    params["state_transition_chunk1_infer_kind"] = infer_chunk1_kind
    params["state_transition_chunk1_infer_kinds"] = {str(c): infer_chunk1_kind for c in contexts}
    params["state_transition_chunk1_fp32_kind"] = infer_chunk1_kind
    params["state_transition_chunk1_fp32_contexts"] = list(contexts) if infer_chunk1_fp32_enabled else []
    params["state_transition_chunk1_fp32_enabled"] = bool(infer_chunk1_fp32_enabled)
    params["state_transition_prefill_chunk1_kind"] = prefill_chunk1_kind
    params["state_transition_prefill_chunk1_kinds"] = {str(c): prefill_chunk1_kind for c in contexts}
    params["state_transition_prefill_chunk1_fp32_contexts"] = (
        list(prefill_contexts) if prefill_chunk1_fp32_enabled else []
    )
    params["state_transition_prefill_chunk1_fp32_enabled"] = bool(prefill_chunk1_fp32_enabled)
    params["state_transition_chunk1_fp32_prefill_mismatch"] = bool(
        infer_chunk1_fp32_enabled and (not prefill_chunk1_fp32_enabled)
    )

    if args.model_name:
        model_info["name"] = str(args.model_name)
    elif "name" in model_info:
        current_name = str(model_info.get("name", "")).strip()
        if "state-transition" not in current_name:
            model_info["name"] = f"{current_name}-state-transition"

    model_info["description"] = (
        f"State-transition chunks with infer_ctx for contexts {contexts}; "
        f"prefill_ctx for contexts {prefill_contexts}; "
        f"{'no alias functions' if args.no_alias_functions else 'alias infer/prefill enabled'}"
    )

    manifest = {
        "contexts": list(contexts),
        "prefill_contexts": list(prefill_contexts),
        "max_context": int(max_context),
        "prefill_default_context": int(prefill_default_context),
        "num_chunks": int(num_chunks),
        "infer_kind": infer_kind,
        "infer_chunk1_kind": infer_chunk1_kind,
        "prefill_kind": prefill_kind,
        "prefill_chunk1_kind": prefill_chunk1_kind,
        "split_infer_prefill": bool(args.split_infer_prefill),
        "no_alias_functions": bool(args.no_alias_functions),
        "dedup_enabled": bool(not args.skip_anemll_dedup),
        "compiled": bool(not args.no_compile),
        "output_base": (None if args.split_infer_prefill else output_base),
        "output_base_infer": (output_base_infer if args.split_infer_prefill else None),
        "output_base_prefill": (output_base_prefill if args.split_infer_prefill else None),
        "chunk1_infer_output": chunk1_infer_output_name,
        "chunk1_prefill_output": chunk1_prefill_output_name,
        "chunks": [
            {
                "chunk": item.chunk,
                "infer_sources": item.infer_sources,
                "prefill_sources": item.prefill_sources,
                "infer_functions": item.infer_functions,
                "prefill_functions": item.prefill_functions,
                "output": item.output,
                "output_infer": item.output_infer,
                "output_prefill": item.output_prefill,
            }
            for item in build_infos
        ],
    }

    if args.dry_run:
        print("[dry-run] meta preview fields:")
        print("  params.ffn =", params.get("ffn"))
        if "ffn_prefill" in params:
            print("  params.ffn_prefill =", params.get("ffn_prefill"))
        print("  state_transition_infer_contexts =", params.get("state_transition_infer_contexts"))
        print("  state_transition_prefill_contexts =", params.get("state_transition_prefill_contexts"))
        print("[dry-run] done.")
        return 0

    meta_path = output_dir / "meta.yaml"
    manifest_path = output_dir / "state_transition_manifest.yaml"
    with meta_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(output_meta, fh, sort_keys=False, allow_unicode=False)
    with manifest_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False, allow_unicode=False)

    print("[done] output:", output_dir)
    print("[done] meta:", meta_path)
    print("[done] manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
