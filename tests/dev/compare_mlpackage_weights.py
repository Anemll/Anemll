#!/usr/bin/env python3
"""Compare weight constants between two CoreML .mlpackage files.

Loads both models, iterates over all const ops (weights), matches them by name,
and reports whether they are byte-identical, numerically close, or divergent.

Usage:
  python3 tests/dev/compare_mlpackage_weights.py \
    /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid/qwen25_FFN_lut6_chunk_01of03.mlpackage \
    /Volumes/Models/ANE/vibethinker_1.5b_ctx2048_L6_4_hybrid/qwen25_FFN_lut6_chunk_01of03.mlpackage

  # Compare all matching chunks between two context dirs
  python3 tests/dev/compare_mlpackage_weights.py \
    --dir-a /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
    --dir-b /Volumes/Models/ANE/vibethinker_1.5b_ctx2048_L6_4_hybrid

  # Only compare chunk01
  python3 tests/dev/compare_mlpackage_weights.py \
    --dir-a /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
    --dir-b /Volumes/Models/ANE/vibethinker_1.5b_ctx2048_L6_4_hybrid \
    --chunk1-only

  # Show per-weight details (verbose)
  python3 tests/dev/compare_mlpackage_weights.py \
    --dir-a /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
    --dir-b /Volumes/Models/ANE/vibethinker_1.5b_ctx2048_L6_4_hybrid \
    --verbose
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _extract_weights(mlpackage_path: str) -> Dict[str, np.ndarray]:
    """Extract all const weight arrays from an mlpackage via MIL spec."""
    import coremltools as ct

    model = ct.models.MLModel(mlpackage_path)
    spec = model.get_spec()

    weights = {}

    # Walk the MIL program to find const ops
    if spec.HasField("mlProgram"):
        prog = spec.mlProgram
        for func in prog.functions.values():
            for block in func.block_specializations.values():
                for op in block.operations:
                    if op.type == "const":
                        name = op.outputs[0].name
                        # Get the value from the op attributes
                        if "val" in op.attributes:
                            val_attr = op.attributes["val"]
                            arr = _extract_value(val_attr)
                            if arr is not None and arr.size > 1:
                                weights[name] = arr
    return weights


def _extract_value(val_attr) -> Optional[np.ndarray]:
    """Extract numpy array from a MIL Value proto attribute."""
    try:
        if val_attr.HasField("blobFileValue"):
            # Large weights stored as blob files - we can't easily read these
            # from the spec alone. Fall back to the weight loading approach.
            return None
        if val_attr.HasField("immediateValue"):
            imm = val_attr.immediateValue
            if imm.HasField("tensor"):
                tensor = imm.tensor
                if tensor.HasField("floats"):
                    shape = list(tensor.floats.dimensions)
                    return np.array(tensor.floats.values, dtype=np.float32).reshape(shape)
                if tensor.HasField("ints"):
                    shape = list(tensor.ints.dimensions)
                    return np.array(tensor.ints.values, dtype=np.int32).reshape(shape)
        return None
    except Exception:
        return None


def _extract_weights_via_mil(mlpackage_path: str) -> Dict[str, np.ndarray]:
    """Extract weights by loading model into MIL and reading const op values."""
    import coremltools as ct

    model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    weights = {}

    try:
        from coremltools.converters.mil.frontend.milproto.load import load as mil_load

        prog = mil_load(
            spec,
            specification_version=spec.specificationVersion,
            file_weights_dir=model.weights_dir,
        )

        for func_name, func in prog.functions.items():
            for op in func.find_ops(op_type="const"):
                name = op.name
                try:
                    val = op.val
                    if val is None:
                        continue
                    # val is a MIL Value object; .val gives numpy array
                    arr = val.val if hasattr(val, 'val') else val
                    if isinstance(arr, np.ndarray) and arr.size > 1:
                        weights[f"{func_name}/{name}"] = arr
                    elif isinstance(arr, (np.generic, int, float)):
                        # Skip scalars
                        continue
                except Exception:
                    continue
    except Exception as e:
        print(f"  WARNING: MIL load failed ({e}), falling back to spec-only extraction")
        weights = _extract_weights(mlpackage_path)

    return weights


def _normalize_key(key: str) -> str:
    """Normalize a MIL const op name for fuzzy matching.

    Strips trailing _N suffix (auto-generated counter that differs across traces)
    while preserving semantic parts like _palettized_indices, _palettized_lut.
    E.g. 'main/layers_0_gate_proj_weight_palettized_lut_0' -> 'main/layers_0_gate_proj_weight_palettized_lut'
    But keeps the layer index: 'main/model_model_layers_5_mlp_...' stays as-is for the _5_ part.
    """
    # Only strip trailing _N where N is a single digit at the very end
    return re.sub(r"_(\d+)$", "", key)


def _compare_pair(
    a: np.ndarray, b: np.ndarray, key: str, verbose: bool,
) -> Tuple[str, Optional[str]]:
    """Compare two arrays. Returns (category, message_or_None)."""
    if a.shape != b.shape:
        msg = f"  SHAPE: {key}: A={a.shape} vs B={b.shape}" if verbose else None
        return "shape_mismatch", msg

    if np.array_equal(a, b):
        msg = f"  IDENTICAL: {key} shape={a.shape} dtype={a.dtype}" if verbose else None
        return "identical", msg

    max_diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
    mean_diff = np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64)))
    rel_max = max_diff / (np.max(np.abs(a.astype(np.float64))) + 1e-30)

    if np.allclose(a, b, rtol=0.0, atol=6e-8):
        msg = (
            f"  CLOSE (fp16 tol): {key} shape={a.shape} "
            f"max_diff={max_diff:.2e} mean_diff={mean_diff:.2e}"
        ) if verbose else None
        return "close", msg

    msg = (
        f"  DIFFERENT: {key} shape={a.shape} dtype={a.dtype} "
        f"max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} rel_max={rel_max:.2e}"
    )
    return "different", msg


def compare_weights(
    weights_a: Dict[str, np.ndarray],
    weights_b: Dict[str, np.ndarray],
    label_a: str = "A",
    label_b: str = "B",
    verbose: bool = False,
) -> Tuple[int, int, int, int, List[str]]:
    """Compare two weight dictionaries.

    Uses exact key matching first, then fuzzy matching (strip trailing _N suffix)
    for unmatched entries. This handles MIL auto-generated name differences across traces.

    Returns (identical, close, different, shape_mismatch, messages).
    """
    identical = 0
    close = 0
    different = 0
    shape_mismatch = 0
    only_a = 0
    only_b = 0
    fuzzy_matched = 0
    messages: List[str] = []

    # Phase 1: exact key matching
    exact_matched_keys = set(weights_a.keys()) & set(weights_b.keys())
    remaining_a = {k: v for k, v in weights_a.items() if k not in exact_matched_keys}
    remaining_b = {k: v for k, v in weights_b.items() if k not in exact_matched_keys}

    for key in sorted(exact_matched_keys):
        cat, msg = _compare_pair(weights_a[key], weights_b[key], key, verbose)
        if cat == "identical":
            identical += 1
        elif cat == "close":
            close += 1
        elif cat == "different":
            different += 1
        elif cat == "shape_mismatch":
            shape_mismatch += 1
        if msg:
            messages.append(msg)

    # Phase 2: fuzzy matching on remaining (strip trailing _N suffix)
    norm_b: Dict[str, List[str]] = {}
    for k in remaining_b:
        nk = _normalize_key(k)
        norm_b.setdefault(nk, []).append(k)

    still_only_a = []
    matched_b_keys = set()
    for key_a in sorted(remaining_a.keys()):
        nk = _normalize_key(key_a)
        candidates = norm_b.get(nk, [])
        # Find best match by shape
        matched = False
        for key_b in candidates:
            if key_b in matched_b_keys:
                continue
            a = remaining_a[key_a]
            b = remaining_b[key_b]
            if a.shape == b.shape:
                fuzzy_matched += 1
                matched_b_keys.add(key_b)
                label = f"{key_a} <~> {key_b}" if key_a != key_b else key_a
                cat, msg = _compare_pair(a, b, label, verbose=True)  # always show fuzzy
                if cat == "identical":
                    identical += 1
                elif cat == "close":
                    close += 1
                elif cat == "different":
                    different += 1
                elif cat == "shape_mismatch":
                    shape_mismatch += 1
                if msg:
                    messages.append(f"  [fuzzy] {msg.strip()}")
                matched = True
                break
        if not matched:
            still_only_a.append(key_a)

    still_only_b = [k for k in remaining_b if k not in matched_b_keys]
    only_a = len(still_only_a)
    only_b = len(still_only_b)

    if verbose:
        for k in still_only_a:
            messages.append(f"  ONLY-{label_a}: {k} shape={weights_a[k].shape} dtype={weights_a[k].dtype}")
        for k in still_only_b:
            messages.append(f"  ONLY-{label_b}: {k} shape={weights_b[k].shape} dtype={weights_b[k].dtype}")

    total_matched = len(exact_matched_keys) + fuzzy_matched
    messages.insert(0,
        f"  Keys: {len(exact_matched_keys)} exact + {fuzzy_matched} fuzzy = {total_matched} matched | "
        f"Only-{label_a}: {only_a} | Only-{label_b}: {only_b}"
    )

    return identical, close, different, shape_mismatch, messages


def compare_packages(
    pkg_a: str,
    pkg_b: str,
    label_a: str = "A",
    label_b: str = "B",
    verbose: bool = False,
) -> Tuple[int, int, int, int]:
    """Compare two mlpackage files. Returns (identical, close, different, shape_mismatch)."""
    print(f"\n{'='*70}")
    print(f"  {label_a}: {os.path.basename(pkg_a)}")
    print(f"  {label_b}: {os.path.basename(pkg_b)}")
    print(f"{'='*70}")

    print(f"  Loading {label_a}...")
    weights_a = _extract_weights_via_mil(pkg_a)
    print(f"  Loading {label_b}...")
    weights_b = _extract_weights_via_mil(pkg_b)

    print(f"  {label_a}: {len(weights_a)} weight tensors")
    print(f"  {label_b}: {len(weights_b)} weight tensors")

    identical, close, different, shape_mismatch, messages = compare_weights(
        weights_a, weights_b, label_a, label_b, verbose=verbose
    )

    for msg in messages:
        print(msg)

    # Summary
    total = identical + close + different + shape_mismatch
    print(f"\n  Summary: {total} matched weights")
    print(f"    IDENTICAL (byte-equal):    {identical:4d}  ({100*identical/max(total,1):.1f}%)")
    print(f"    CLOSE (fp16 tol 6e-8):     {close:4d}  ({100*close/max(total,1):.1f}%)")
    print(f"    DIFFERENT:                 {different:4d}  ({100*different/max(total,1):.1f}%)")
    print(f"    SHAPE MISMATCH:            {shape_mismatch:4d}  ({100*shape_mismatch/max(total,1):.1f}%)")

    if total > 0:
        dedup_eligible = identical + close
        print(f"\n    Dedup-eligible (identical+close): {dedup_eligible}/{total} = {100*dedup_eligible/total:.1f}%")
        if different > 0:
            print(f"    >>> {different} weights have values that DIFFER beyond fp16 tolerance <<<")
            print(f"    >>> These prevent save_multifunction dedup from sharing blobs <<<")

    return identical, close, different, shape_mismatch


def find_matching_packages(dir_a: Path, dir_b: Path, chunk1_only: bool = False) -> List[Tuple[Path, Path, str]]:
    """Find mlpackage files that exist in both directories with matching names."""
    pairs = []
    for pkg_a in sorted(dir_a.glob("*.mlpackage")):
        # Skip non-chunk files for focused comparison
        if not re.search(r"_chunk_\d+of\d+\.mlpackage$", pkg_a.name):
            continue
        if chunk1_only and not re.search(r"_chunk_01of\d+\.mlpackage$", pkg_a.name):
            continue
        pkg_b = dir_b / pkg_a.name
        if pkg_b.exists():
            pairs.append((pkg_a, pkg_b, pkg_a.stem))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare weight constants between CoreML mlpackage files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Positional: two packages directly
    ap.add_argument("packages", nargs="*", help="Two mlpackage paths to compare directly.")

    # Or: two directories
    ap.add_argument("--dir-a", default=None, help="First model directory.")
    ap.add_argument("--dir-b", default=None, help="Second model directory.")
    ap.add_argument("--chunk1-only", action="store_true", help="Only compare chunk_01 artifacts.")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show per-weight details.")

    args = ap.parse_args()

    if args.packages and len(args.packages) == 2:
        # Direct two-package comparison
        pkg_a, pkg_b = args.packages
        identical, close, different, shape_mismatch = compare_packages(
            pkg_a, pkg_b,
            label_a="A", label_b="B",
            verbose=args.verbose,
        )
        return 1 if different > 0 else 0

    elif args.dir_a and args.dir_b:
        dir_a = Path(args.dir_a).expanduser().resolve()
        dir_b = Path(args.dir_b).expanduser().resolve()

        if not dir_a.is_dir():
            print(f"ERROR: not a directory: {dir_a}", file=sys.stderr)
            return 1
        if not dir_b.is_dir():
            print(f"ERROR: not a directory: {dir_b}", file=sys.stderr)
            return 1

        pairs = find_matching_packages(dir_a, dir_b, chunk1_only=args.chunk1_only)
        if not pairs:
            print("ERROR: no matching mlpackage files found", file=sys.stderr)
            return 1

        label_a = dir_a.name
        label_b = dir_b.name

        print(f"Comparing {len(pairs)} matching packages between:")
        print(f"  A: {dir_a}")
        print(f"  B: {dir_b}")

        totals = {"identical": 0, "close": 0, "different": 0, "shape_mismatch": 0}
        for pkg_a, pkg_b, stem in pairs:
            ident, cls, diff, shp = compare_packages(
                str(pkg_a), str(pkg_b),
                label_a=label_a, label_b=label_b,
                verbose=args.verbose,
            )
            totals["identical"] += ident
            totals["close"] += cls
            totals["different"] += diff
            totals["shape_mismatch"] += shp

        # Grand total
        total = sum(totals.values())
        print(f"\n{'='*70}")
        print(f"GRAND TOTAL across {len(pairs)} package pairs: {total} weights")
        print(f"  IDENTICAL:      {totals['identical']:5d}  ({100*totals['identical']/max(total,1):.1f}%)")
        print(f"  CLOSE:          {totals['close']:5d}  ({100*totals['close']/max(total,1):.1f}%)")
        print(f"  DIFFERENT:      {totals['different']:5d}  ({100*totals['different']/max(total,1):.1f}%)")
        print(f"  SHAPE MISMATCH: {totals['shape_mismatch']:5d}  ({100*totals['shape_mismatch']/max(total,1):.1f}%)")

        dedup_eligible = totals["identical"] + totals["close"]
        print(f"\n  Dedup-eligible: {dedup_eligible}/{total} = {100*dedup_eligible/max(total,1):.1f}%")

        return 1 if totals["different"] > 0 else 0

    else:
        print("Usage: provide two mlpackage paths, or --dir-a and --dir-b", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
