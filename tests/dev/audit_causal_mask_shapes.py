#!/usr/bin/env python3
"""Audit causal_mask input shapes across per-context or combined model directories.

Verifies that every CoreML chunk artifact's causal_mask input dimension matches
the expected context_length from meta.yaml.

Usage examples:

  # Audit per-context directories (auto-discovers contexts from template)
  python3 tests/dev/audit_causal_mask_shapes.py \
    --contexts-root /Volumes/Models/ANE \
    --name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
    --contexts 512 1024 2048 3072 4096

  # Audit a single combined xstates directory
  python3 tests/dev/audit_causal_mask_shapes.py \
    --meta /Volumes/Models/ANE/vibethinker_1.5b_xstates_split_noalias/meta.yaml

  # Audit specific directories
  python3 tests/dev/audit_causal_mask_shapes.py \
    --dirs /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
           /Volumes/Models/ANE/vibethinker_1.5b_ctx1024_L6_4_hybrid

  # Only check chunk01 artifacts
  python3 tests/dev/audit_causal_mask_shapes.py \
    --contexts-root /Volumes/Models/ANE \
    --name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
    --contexts 512 1024 2048 \
    --chunk1-only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def _load_meta_context(model_dir: Path) -> Optional[int]:
    """Read context_length from meta.yaml if present."""
    meta_path = model_dir / "meta.yaml"
    if not meta_path.exists():
        return None
    try:
        meta = yaml.safe_load(meta_path.read_text())
        params = meta.get("model_info", {}).get("parameters", {})
        return int(params.get("context_length", 0)) or None
    except Exception:
        return None


def _get_causal_mask_shape(pkg_path: Path) -> Optional[List[int]]:
    """Extract causal_mask input shape from an mlpackage or mlmodelc spec."""
    import coremltools as ct

    try:
        if pkg_path.suffix == ".mlpackage":
            m = ct.models.MLModel(str(pkg_path))
        elif pkg_path.suffix == ".mlmodelc":
            # For compiled models, try loading spec from the weight dir
            # But compiled models don't expose spec easily; skip
            return None
        else:
            return None

        spec = m.get_spec()
        for inp in spec.description.input:
            if inp.name == "causal_mask":
                return list(inp.type.multiArrayType.shape)
    except Exception:
        return None
    return None


def _find_chunk_packages(model_dir: Path, chunk1_only: bool = False) -> List[Path]:
    """Find all chunk .mlpackage files in a model directory."""
    packages = []
    for p in sorted(model_dir.glob("*.mlpackage")):
        # Match chunk patterns like *_chunk_01of03.mlpackage
        if re.search(r"_chunk_\d+of\d+\.mlpackage$", p.name):
            if chunk1_only and not re.search(r"_chunk_01of\d+\.mlpackage$", p.name):
                continue
            packages.append(p)
    return packages


def _classify_artifact(name: str) -> str:
    """Classify artifact type from filename."""
    stem = re.sub(r"\.mlpackage$", "", name)
    if "_attn_fp32_" in stem:
        if "_prefill_" in stem or stem.startswith("prefill_") or "_prefill_attn" in stem:
            return "prefill_attn_fp32"
        return "FFN_attn_fp32"
    if "_prefill_" in stem or stem.startswith("prefill_"):
        return "prefill_lut"
    if "_FFN_" in stem:
        return "FFN_lut"
    return "other"


def audit_directory(
    model_dir: Path,
    expected_context: Optional[int],
    chunk1_only: bool = False,
) -> Tuple[int, int, List[str]]:
    """Audit a single model directory. Returns (pass_count, fail_count, messages)."""
    passes = 0
    fails = 0
    messages: List[str] = []

    meta_ctx = _load_meta_context(model_dir)
    if expected_context is None:
        expected_context = meta_ctx

    if expected_context is None:
        messages.append(f"  SKIP: no expected context (meta.yaml missing or no context_length)")
        return 0, 0, messages

    if meta_ctx is not None and meta_ctx != expected_context:
        messages.append(
            f"  WARN: meta.yaml says context_length={meta_ctx} but expected {expected_context}"
        )

    packages = _find_chunk_packages(model_dir, chunk1_only=chunk1_only)
    if not packages:
        messages.append(f"  WARN: no chunk .mlpackage files found")
        return 0, 0, messages

    for pkg in packages:
        shape = _get_causal_mask_shape(pkg)
        if shape is None:
            messages.append(f"  SKIP: {pkg.name} (no causal_mask or load error)")
            continue

        actual_ctx = shape[-1]
        # For prefill, the mask is [1, 1, batch_size, context] so last dim is context
        # For infer, the mask is [1, 1, 1, context] so last dim is context
        if actual_ctx == expected_context:
            passes += 1
            messages.append(f"  PASS: {pkg.name}: causal_mask={shape}")
        else:
            fails += 1
            messages.append(
                f"  FAIL: {pkg.name}: causal_mask={shape} "
                f"(expected last dim={expected_context}, got {actual_ctx})"
            )

    return passes, fails, messages


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit causal_mask shapes in CoreML model artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--contexts-root",
        default=None,
        help="Root directory containing per-context model folders.",
    )
    ap.add_argument(
        "--name-template",
        default="vibethinker_1.5b_ctx{context}_L6_4_hybrid",
        help="Folder name template with {context} placeholder.",
    )
    ap.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=None,
        help="Context sizes to audit (e.g. 512 1024 2048).",
    )
    ap.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        help="Explicit model directories to audit (alternative to --contexts-root).",
    )
    ap.add_argument(
        "--meta",
        default=None,
        help="Combined meta.yaml path (audits single directory, reads contexts from meta).",
    )
    ap.add_argument(
        "--chunk1-only",
        action="store_true",
        help="Only audit chunk_01 artifacts.",
    )
    args = ap.parse_args()

    # Collect directories to audit: list of (label, path, expected_context)
    audit_targets: List[Tuple[str, Path, Optional[int]]] = []

    if args.meta:
        meta_path = Path(args.meta).expanduser().resolve()
        if not meta_path.exists():
            print(f"ERROR: meta.yaml not found: {meta_path}", file=sys.stderr)
            return 1
        model_dir = meta_path.parent
        meta = yaml.safe_load(meta_path.read_text())
        params = meta.get("model_info", {}).get("parameters", {})
        ctx = int(params.get("context_length", 0)) or None
        audit_targets.append((f"combined (ctx={ctx})", model_dir, ctx))

    elif args.dirs:
        for d in args.dirs:
            p = Path(d).expanduser().resolve()
            if not p.is_dir():
                print(f"WARNING: directory not found: {p}", file=sys.stderr)
                continue
            ctx = _load_meta_context(p)
            audit_targets.append((f"ctx{ctx or '?'} ({p.name})", p, ctx))

    elif args.contexts_root and args.contexts:
        root = Path(args.contexts_root).expanduser().resolve()
        for ctx in sorted(args.contexts):
            name = args.name_template.format(context=ctx)
            p = root / name
            if not p.is_dir():
                print(f"WARNING: directory not found: {p}", file=sys.stderr)
                continue
            audit_targets.append((f"ctx{ctx}", p, ctx))
    else:
        print(
            "ERROR: provide one of: --meta, --dirs, or --contexts-root + --contexts",
            file=sys.stderr,
        )
        return 1

    if not audit_targets:
        print("ERROR: no directories to audit", file=sys.stderr)
        return 1

    total_passes = 0
    total_fails = 0

    for label, model_dir, expected_ctx in audit_targets:
        print(f"\n== {label} ==")
        print(f"   dir: {model_dir}")
        passes, fails, messages = audit_directory(
            model_dir, expected_ctx, chunk1_only=args.chunk1_only
        )
        for msg in messages:
            print(msg)
        total_passes += passes
        total_fails += fails

    print(f"\n{'=' * 50}")
    print(f"Total: {total_passes} passed, {total_fails} failed")
    if total_fails > 0:
        print("STATUS: FAIL")
        return 1
    else:
        print("STATUS: PASS")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
