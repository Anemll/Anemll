#!/usr/bin/env python3
"""Build state-transition chunk packages from already-converted context exports.

This script combines per-context chunk models into new multi-function chunk packages
without re-running conversion or LUT palettization.

Per output chunk:
- Adds one infer function per context: infer_ctx{N}
- Adds compatibility alias: infer (points to max-context infer)
- Adds one prefill function from max-context model only: prefill

Typical usage:
  python tests/dev/export_state_transition_chunks.py \
    --contexts \
      512=/Volumes/Models/ANE/vibethinker_1.5b_ctx0512_fp16_hybrid \
      1024=/Volumes/Models/ANE/vibethinker_1.5b_ctx1024_fp16_hybrid \
      2048=/Volumes/Models/ANE/vibethinker_1.5b_ctx2048_fp16_hybrid \
      3072=/Volumes/Models/ANE/vibethinker_1.5b_ctx3072_fp16_hybrid \
      4096=/Volumes/Models/ANE/vibethinker_1.5b_ctx4096_fp16_hybrid \
    --output-dir /Volumes/Models/ANE/vibethinker_1.5b_state_transition
"""

from __future__ import annotations

import argparse
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
            "coremltools is required to export state-transition chunk packages. "
            "Install it in your active environment first."
        ) from exc
    return ct


def _require_coreml_optimize():
    try:
        import coremltools.optimize as cto  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "coremltools.optimize is required for post-combine LUT quantization."
        ) from exc
    return cto


def _require_yaml():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read/write meta.yaml for state-transition export."
        ) from exc
    return yaml


def _parse_context_entries(entries: Iterable[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"Invalid --contexts entry '{raw}'. Expected N=/path/to/model_dir")
        lhs, rhs = raw.split("=", 1)
        context = int(lhs)
        model_dir = Path(rhs).expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(model_dir)
        out[context] = model_dir

    if not out:
        raise ValueError("No contexts provided")
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _load_meta_params(model_dir: Path) -> tuple[dict, dict]:
    yaml = _require_yaml()
    meta_path = model_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.yaml in {model_dir}")

    meta = yaml.safe_load(meta_path.read_text())
    params = meta.get("model_info", {}).get("parameters", {})
    if not isinstance(params, dict) or not params:
        raise ValueError(f"meta.yaml is missing model_info.parameters in {model_dir}")

    return meta, params


def _split_chunk_stem(ffn_value: str) -> tuple[str, int]:
    stem = re.sub(r"\.(mlmodelc|mlpackage)$", "", str(ffn_value))
    m = re.match(r"^(.+)_chunk_(\d+)of(\d+)$", stem)
    if not m:
        raise ValueError(
            "Could not parse chunk naming from ffn='{}' (expected ..._chunk_01ofNN)".format(ffn_value)
        )
    base = m.group(1)
    total_chunks = int(m.group(3))
    return base, total_chunks


def _candidate_ml_paths(model_dir: Path, stem: str) -> list[Path]:
    return [
        model_dir / f"{stem}.mlpackage",
        model_dir / f"{stem}.mlmodelc",
    ]


def _resolve_model_path(model_dir: Path, stem: str) -> Path:
    for p in _candidate_ml_paths(model_dir, stem):
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find model for stem '{stem}' in {model_dir} (.mlpackage/.mlmodelc)"
    )


def _available_functions(model_path: Path, ct_module) -> list[str]:
    model = ct_module.models.MLModel(str(model_path))
    spec = model.get_spec()

    names: list[str] = []

    desc = getattr(spec, "description", None)
    if desc is not None and hasattr(desc, "functions"):
        try:
            for f in desc.functions:
                if getattr(f, "name", None):
                    names.append(f.name)
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
    names = _available_functions(model_path, ct_module)
    if preferred in names:
        return preferred
    if "main" in names:
        return "main"
    if len(names) == 1:
        return names[0]
    raise RuntimeError(
        f"Could not resolve source function '{preferred}' from {model_path}. Available: {names}"
    )


def _parse_lut_arg(raw: str | None) -> tuple[int | None, int]:
    if raw is None:
        return None, 8
    s = str(raw).strip().lower()
    if s in ("", "none", "no", "false"):
        return None, 8
    if "," in s:
        lhs, rhs = s.split(",", 1)
        bits = int(lhs)
        rhs = rhs.strip().lower()
        if rhs in ("tensor", "t", "0"):
            per_channel = 0
        else:
            per_channel = int(rhs)
        return bits, per_channel
    return int(s), 8


def _quantize_combined_model(model, lut_bits: int, per_channel: int, cto_module):
    use_per_tensor = per_channel <= 0
    if use_per_tensor:
        config = cto_module.coreml.OptimizationConfig(
            global_config=cto_module.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=lut_bits,
                granularity="per_tensor",
                num_kmeans_workers=1,
            ),
        )
    else:
        config = cto_module.coreml.OptimizationConfig(
            global_config=cto_module.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=lut_bits,
                granularity="per_grouped_channel",
                group_size=per_channel,
                num_kmeans_workers=1,
            ),
        )
    return cto_module.coreml.palettize_weights(model, config)


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


def _copy_shared_assets(max_ctx_dir: Path, out_dir: Path, params: dict) -> None:
    optional_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    ]

    for fname in optional_files:
        src = max_ctx_dir / fname
        if src.exists():
            _copy_path(src, out_dir / fname)

    for key in ("embeddings", "lm_head"):
        val = params.get(key)
        if not val:
            continue
        stem = re.sub(r"\.(mlmodelc|mlpackage)$", "", str(val))
        src = _resolve_model_path(max_ctx_dir, stem)
        _copy_path(src, out_dir / src.name)


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


def _update_meta(
    out_dir: Path,
    template_meta: dict,
    output_ffn_base: str,
    num_chunks: int,
    contexts: list[int],
    max_context: int,
    compiled_chunks: bool,
    lut2_bits: int | None,
    lut2_per_channel: int,
) -> None:
    yaml = _require_yaml()
    meta = template_meta
    params = meta.setdefault("model_info", {}).setdefault("parameters", {})

    ffn_ext = "mlmodelc" if compiled_chunks else "mlpackage"
    params["ffn"] = f"{output_ffn_base}_chunk_01of{num_chunks:02d}.{ffn_ext}"
    params["state_transition_infer_contexts"] = [int(c) for c in contexts]
    params["state_transition_infer_function_template"] = "infer_ctx{context}"
    params["state_transition_prefill_context"] = int(max_context)
    if lut2_bits is not None:
        params["lut_ffn"] = int(lut2_bits)
        params["lut_ffn_per_channel"] = int(lut2_per_channel)

    desc = meta["model_info"].get("description", "")
    line = (
        f"State-transition functions: infer_ctx{{{', '.join(str(c) for c in contexts)}}}, "
        f"prefill_ctx{max_context}"
    )
    if isinstance(desc, str) and desc:
        if line not in desc:
            meta["model_info"]["description"] = f"{desc.rstrip()} | {line}"
    else:
        meta["model_info"]["description"] = line

    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Combine chunked models from multiple context exports into state-transition "
            "multi-function chunk packages."
        )
    )
    ap.add_argument(
        "--contexts",
        nargs="+",
        required=True,
        help="List of context model dirs in form N=/path/to/model_dir",
    )
    ap.add_argument("--output-dir", required=True, help="Directory for exported multifunction chunks")
    ap.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Context to source prefill from (default: highest provided context)",
    )
    ap.add_argument(
        "--output-suffix",
        default="_statex",
        help="Suffix appended to FFN base for new chunk names (default: _statex)",
    )
    ap.add_argument(
        "--output-base",
        default=None,
        help="Override output FFN base name (without _chunk_XXofYY)",
    )
    ap.add_argument(
        "--infer-fn",
        default="infer",
        help="Source infer function name in chunk packages (default: infer)",
    )
    ap.add_argument(
        "--prefill-fn",
        default="prefill",
        help="Source prefill function name in max-context chunk package (default: prefill)",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="Compile output .mlpackage chunks to .mlmodelc",
    )
    ap.add_argument(
        "--lut2",
        type=str,
        default=None,
        help=(
            "Post-combine LUT quantization for FFN chunks. "
            "Format: 'bits' or 'bits,per_channel' (e.g. '6' or '6,4'). "
            "Use 'bits,0' for per-tensor. Default: none."
        ),
    )
    ap.add_argument(
        "--no-copy-shared",
        action="store_true",
        help="Do not copy tokenizer/embedding/lm_head/meta assets from max-context directory",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing output directory",
    )
    args = ap.parse_args()
    lut2_bits, lut2_per_channel = _parse_lut_arg(args.lut2)
    ct = _require_coremltools()
    yaml = _require_yaml()
    cto = _require_coreml_optimize() if lut2_bits is not None else None

    context_dirs = _parse_context_entries(args.contexts)
    context_list = list(context_dirs.keys())

    max_context = args.max_context if args.max_context is not None else max(context_list)
    if max_context not in context_dirs:
        raise ValueError(f"max_context={max_context} not found in --contexts")

    max_ctx_dir = context_dirs[max_context]
    max_meta, max_params = _load_meta_params(max_ctx_dir)

    source_ffn = max_params.get("ffn")
    if not source_ffn:
        raise ValueError(f"Missing 'ffn' in {max_ctx_dir}/meta.yaml")

    source_ffn_base, num_chunks = _split_chunk_stem(str(source_ffn))
    output_ffn_base = args.output_base or f"{source_ffn_base}{args.output_suffix}"

    out_dir = Path(args.output_dir).expanduser().resolve()
    if out_dir.exists() and not args.force:
        raise FileExistsError(
            f"Output directory already exists: {out_dir}\n"
            "Use --force to reuse it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_copy_shared:
        _copy_shared_assets(max_ctx_dir, out_dir, max_params)

    manifest = {
        "contexts": context_list,
        "max_context": max_context,
        "num_chunks": num_chunks,
        "source_ffn_base": source_ffn_base,
        "output_ffn_base": output_ffn_base,
        "chunks": [],
    }

    for chunk_idx in range(1, num_chunks + 1):
        chunk_suffix = f"_chunk_{chunk_idx:02d}of{num_chunks:02d}"
        infer_sources: list[tuple[int, Path]] = []

        for ctx, model_dir in context_dirs.items():
            source_stem = f"{source_ffn_base}{chunk_suffix}"
            source_path = _resolve_model_path(model_dir, source_stem)
            infer_sources.append((ctx, source_path))

        max_infer_path = next(path for (ctx, path) in infer_sources if ctx == max_context)

        out_pkg = out_dir / f"{output_ffn_base}{chunk_suffix}.mlpackage"
        tmp_pkg = out_dir / f"tmp_{out_pkg.name}"
        if tmp_pkg.exists():
            shutil.rmtree(tmp_pkg)
        if out_pkg.exists():
            shutil.rmtree(out_pkg)

        desc = ct.utils.MultiFunctionDescriptor()

        # Add one infer function per state size/context.
        for ctx, model_path in infer_sources:
            src_fn = _resolve_source_function(model_path, args.infer_fn, ct)
            desc.add_function(str(model_path), src_fn, f"infer_ctx{ctx}")

        # Compatibility aliases for existing loaders.
        infer_src_fn = _resolve_source_function(max_infer_path, args.infer_fn, ct)
        prefill_src_fn = _resolve_source_function(max_infer_path, args.prefill_fn, ct)
        desc.add_function(str(max_infer_path), infer_src_fn, "infer")
        desc.add_function(str(max_infer_path), prefill_src_fn, "prefill")
        desc.default_function_name = "infer"

        ct.utils.save_multifunction(desc, str(tmp_pkg))
        model = ct.models.MLModel(str(tmp_pkg))
        if lut2_bits is not None:
            print(
                f"  Quantizing combined chunk with LUT {lut2_bits},"
                f"{lut2_per_channel} ..."
            )
            model = _quantize_combined_model(model, lut2_bits, lut2_per_channel, cto)
        model.save(str(out_pkg))
        shutil.rmtree(tmp_pkg, ignore_errors=True)

        if args.compile:
            _compile_mlpackage(out_pkg, out_dir)

        manifest["chunks"].append(
            {
                "chunk": chunk_idx,
                "output": out_pkg.name,
                "infer_functions": [f"infer_ctx{c}" for c in context_list],
                "prefill_function": "prefill",
            }
        )

        print(f"Built chunk {chunk_idx}/{num_chunks}: {out_pkg.name}")

    if not args.no_copy_shared:
        _update_meta(
            out_dir=out_dir,
            template_meta=max_meta,
            output_ffn_base=output_ffn_base,
            num_chunks=num_chunks,
            contexts=context_list,
            max_context=max_context,
            compiled_chunks=args.compile,
            lut2_bits=lut2_bits,
            lut2_per_channel=lut2_per_channel,
        )

    (out_dir / "state_transition_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )

    print("\nDone")
    print(f"Output dir: {out_dir}")
    print(f"Contexts: {context_list}")
    print(f"Prefill source context: {max_context}")
    print(f"FFN base: {output_ffn_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
