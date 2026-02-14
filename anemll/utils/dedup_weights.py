#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

"""Surgical weight deduplication for CoreML multifunction models.

Before calling ct.utils.save_multifunction(), this utility replaces palettized
weight blobs (LUT + indices) in non-anchor models with the anchor model's blobs
when the dequantized values are semantically identical. This forces byte-identical
const blobs so CoreMLTools' dedup pass can share them.

Background:
  MIL optimization passes (add_fp16_cast, constant folding) produce microscopically
  different fp16 representations depending on graph shape (context length, seq_len).
  K-means then converges to different LUT centroids + index assignments that encode
  the *same* dequantized values (verified: cosine similarity = 1.0, mean_diff ~3e-8).
  CoreMLTools dedup compares raw bytes, so these semantically-identical weights are
  not shared — causing ~15-40% size bloat depending on the function combination.

Usage in combine pipeline:
  from anemll.utils.dedup_weights import prepare_dedup_sources

  # Instead of:
  #   desc.add_function(ffn_path, "main", "infer")
  #   desc.add_function(prefill_path, "main", "prefill")
  #   ct.utils.save_multifunction(desc, output_path)
  #
  # Use:
  #   sources = [(ffn_path, "main", "infer"), (prefill_path, "main", "prefill")]
  #   with prepare_dedup_sources(sources) as deduped:
  #       desc = ct.utils.MultiFunctionDescriptor()
  #       for path, src_fn, tgt_fn in deduped:
  #           desc.add_function(path, src_fn, tgt_fn)
  #       desc.default_function_name = "infer"
  #       ct.utils.save_multifunction(desc, output_path)

Standalone testing:
  python3 -m anemll.utils.dedup_weights \\
    --anchor /path/to/infer.mlpackage \\
    --target /path/to/prefill.mlpackage \\
    --source-fn-anchor main --source-fn-target main \\
    --dry-run
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Replacement reason codes
# ---------------------------------------------------------------------------

class ReplacementReason(Enum):
    """Reason code for each weight pair evaluation."""
    IDENTICAL = "identical"           # Already byte-identical, no action needed
    DEQ_CLOSE = "deq_close"           # Dequantized values match within thresholds
    REPLACED_NO_VERIFY = "replaced_no_verify"  # Replaced without dequant verification
    REJECTED_SHAPE = "rejected_shape"          # Shape mismatch
    REJECTED_THRESHOLD = "rejected_threshold"  # Failed acceptance thresholds
    REJECTED_DEQ_FAIL = "rejected_deq_fail"    # Dequantization failed
    LAYOUT_UNSUPPORTED = "layout_unsupported"  # Non-palettized or unsupported layout


@dataclass
class ReplacementDiag:
    """Per-replacement diagnostic record."""
    base_name: str
    reason: ReplacementReason
    tensor_class: str = ""          # e.g. "palettized_lut", "fp16_const"
    bytes_saved: int = 0            # Estimated bytes saved (idx + lut)
    cos_sim: float = 0.0
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    anchor_shape: Tuple = ()
    target_shape: Tuple = ()


# ---------------------------------------------------------------------------
# Tensor class classification (pluggable matching)
# ---------------------------------------------------------------------------

def _classify_tensor(nk: str) -> str:
    """Classify a normalized key into a tensor class for matching.

    Returns a class label used to group weights for targeted matching logic.
    Avoids a single global regex for all const names.
    """
    if "_palettized_indices" in nk:
        return "palettized_indices"
    if "_palettized_lut" in nk:
        return "palettized_lut"
    if "_weight" in nk and "_palettized" not in nk:
        return "dense_weight"
    if "_bias" in nk:
        return "bias"
    if "causal_mask" in nk or "mask" in nk:
        return "mask"
    if "kv_cache" in nk or "cache" in nk:
        return "cache"
    return "other"


def _is_palettized_pair_key(nk: str) -> bool:
    """Check if a key is part of a palettized weight pair."""
    return "_palettized_indices" in nk or "_palettized_lut" in nk


# ---------------------------------------------------------------------------
# Preflight compatibility checks
# ---------------------------------------------------------------------------

class PreflightError(Exception):
    """Raised when preflight compatibility checks fail."""
    pass


def _preflight_check_io_signature(anchor_prog, target_prog,
                                  anchor_fn: str = "main",
                                  target_fn: str = "main"):
    """Verify anchor and target have compatible I/O signatures.

    Checks that input/output names and shapes match, which ensures the models
    are from the same architecture and chunk configuration.
    """
    a_func = anchor_prog.functions.get(anchor_fn)
    t_func = target_prog.functions.get(target_fn)

    if a_func is None:
        raise PreflightError(f"Anchor has no function '{anchor_fn}'")
    if t_func is None:
        raise PreflightError(f"Target has no function '{target_fn}'")

    # Compare input names and shapes
    a_inputs = {inp.name: tuple(inp.shape) for inp in a_func.inputs.values()
                if hasattr(inp, 'shape')}
    t_inputs = {inp.name: tuple(inp.shape) for inp in t_func.inputs.values()
                if hasattr(inp, 'shape')}

    a_names = set(a_inputs.keys())
    t_names = set(t_inputs.keys())

    if a_names != t_names:
        diff = a_names.symmetric_difference(t_names)
        raise PreflightError(
            f"I/O signature mismatch: input names differ. "
            f"Anchor-only: {a_names - t_names}, Target-only: {t_names - a_names}"
        )

    # Check shapes (allow sequence length dimension to differ for infer vs prefill)
    for name in a_names:
        a_shape = a_inputs[name]
        t_shape = t_inputs[name]
        if len(a_shape) != len(t_shape):
            raise PreflightError(
                f"I/O signature mismatch: input '{name}' rank differs: "
                f"{len(a_shape)} vs {len(t_shape)}"
            )


def _preflight_check_weight_counts(anchor_weights, target_weights):
    """Verify anchor and target have similar weight tensor counts.

    A large discrepancy suggests different model architectures or chunk configurations.
    """
    a_count = len(anchor_weights)
    t_count = len(target_weights)

    if a_count == 0 or t_count == 0:
        raise PreflightError(
            f"Empty weight set: anchor has {a_count}, target has {t_count} tensors"
        )

    # Allow up to 20% difference (different graph optimizations may add/remove constants)
    ratio = min(a_count, t_count) / max(a_count, t_count)
    if ratio < 0.8:
        raise PreflightError(
            f"Weight count mismatch: anchor has {a_count}, target has {t_count} tensors "
            f"(ratio {ratio:.2f} < 0.8 threshold). "
            f"This suggests different model architectures or chunk configurations."
        )


def _preflight_check_palettized_config(anchor_weights, target_weights):
    """Verify LUT configurations are compatible (same bits, group sizes).

    Checks that palettized weight pairs have matching LUT dimensions, which
    encode the number of centroids (2^bits) and group size.
    """
    def _get_lut_configs(weights):
        configs = {}
        for nk, arr in weights.items():
            if "_palettized_lut" in nk:
                base = nk.replace("_palettized_lut", "")
                # LUT shape encodes: (num_groups, 1, 1, 1, num_centroids, 1) or (G, C)
                squeezed = arr.squeeze()
                if squeezed.ndim == 2:
                    configs[base] = (squeezed.shape[0], squeezed.shape[1])
                elif squeezed.ndim == 1:
                    configs[base] = (1, squeezed.shape[0])
        return configs

    a_configs = _get_lut_configs(anchor_weights)
    t_configs = _get_lut_configs(target_weights)

    mismatched = []
    for base in a_configs:
        if base in t_configs and a_configs[base] != t_configs[base]:
            mismatched.append(
                f"  {base}: anchor={a_configs[base]} vs target={t_configs[base]}"
            )

    if mismatched:
        raise PreflightError(
            f"LUT configuration mismatch for {len(mismatched)} weight(s):\n"
            + "\n".join(mismatched[:5])
            + ("\n  ..." if len(mismatched) > 5 else "")
        )


# ---------------------------------------------------------------------------
# MIL weight extraction
# ---------------------------------------------------------------------------

def _load_mil_program(mlpackage_path: str):
    """Load an mlpackage into a MIL program with weight data resolved."""
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load as mil_load

    model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    prog = mil_load(
        spec,
        specification_version=spec.specificationVersion,
        file_weights_dir=model.weights_dir,
    )
    return prog, model


def _normalize_key(name: str) -> str:
    """Strip trailing auto-generated _N suffix for cross-trace matching.

    E.g. 'layers_0_gate_proj_weight_palettized_lut_0' ->
         'layers_0_gate_proj_weight_palettized_lut'
    """
    return re.sub(r"_(\d+)$", "", name)


def _extract_const_weights(prog, func_name: str = "main") -> Dict[str, np.ndarray]:
    """Extract {normalized_key: numpy_array} for all non-scalar const ops."""
    weights = {}
    func = prog.functions.get(func_name)
    if func is None:
        # Try first available function
        func = next(iter(prog.functions.values()), None)
    if func is None:
        return weights

    for op in func.find_ops(op_type="const"):
        val = op.val
        if val is None:
            continue
        arr = val.val if hasattr(val, "val") else val
        if isinstance(arr, np.ndarray) and arr.size > 1:
            nk = _normalize_key(op.name)
            weights[nk] = arr
    return weights


# ---------------------------------------------------------------------------
# Dequantization for verification
# ---------------------------------------------------------------------------

def _dequantize_lut(indices: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Dequantize palettized weight: value = lut[group, index].

    Handles CoreML's multi-dimensional tensor layouts:
      - indices: (O, I, 1, 1) or (O, I) — uint8 bin assignments
      - lut: (G, 1, 1, 1, C, 1) or (G, C) — fp16 palette per group

    Squeezes to 2D before dequantizing, then uses vectorized numpy indexing.

    Returns:
        Dequantized array with same number of elements as indices, or None on failure.
    """
    # Squeeze to essential dimensions
    idx = indices.squeeze()
    lt = lut.squeeze()

    # Ensure 2D
    if idx.ndim == 1:
        idx = idx.reshape(-1, 1)
    if idx.ndim != 2:
        return None

    if lt.ndim == 1:
        # Single group, all centroids in one row
        lt = lt.reshape(1, -1)
    if lt.ndim != 2:
        return None

    O, I = idx.shape
    G, C = lt.shape
    if G == 0 or O == 0 or C == 0:
        return None
    group_size = max(1, O // G)

    # Vectorized dequantization: group_ids[o] = o // group_size
    group_ids = np.minimum(np.arange(O) // group_size, G - 1)  # (O,)
    # result[o, i] = lt[group_ids[o], idx[o, i]]
    result = lt[group_ids[:, None], idx]  # (O, I)
    return result


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a_flat = a.astype(np.float64).ravel()
    b_flat = b.astype(np.float64).ravel()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-30 or norm_b < 1e-30:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Core: find and verify replaceable weight pairs
# ---------------------------------------------------------------------------

def find_replaceable_weights(
    anchor_weights: Dict[str, np.ndarray],
    target_weights: Dict[str, np.ndarray],
    cos_threshold: float = 0.9999,
    max_abs_threshold: Optional[float] = None,
    mean_abs_threshold: float = 0.001,
    verify_dequant: bool = True,
    verbose: bool = False,
    diagnostics: Optional[List[ReplacementDiag]] = None,
) -> Dict[str, str]:
    """Find target weight keys that can be safely replaced with anchor values.

    Matches palettized weight pairs (_palettized_indices + _palettized_lut)
    and verifies that dequantized values are semantically identical.

    Acceptance modes:
      - Relaxed (default, max_abs_threshold=None): cos >= T AND mean_abs <= M
        Appropriate when original weights are identical and differences are
        purely k-means centroid placement artifacts.
      - Strict (max_abs_threshold=float): cos >= T AND max_abs <= A AND mean_abs <= M

    Args:
        anchor_weights: Extracted weights from the anchor model
        target_weights: Extracted weights from the target model
        cos_threshold: Minimum cosine similarity (default 0.9999)
        max_abs_threshold: Maximum absolute difference allowed (None=relaxed, no max_abs gate)
        mean_abs_threshold: Maximum mean absolute difference allowed (default 0.001)
        verify_dequant: If True, verify via dequantization before replacing
        verbose: Print per-weight replacement details
        diagnostics: If provided, append ReplacementDiag records for each weight pair

    Returns:
        Dict mapping target normalized_key -> anchor normalized_key for all
        weights that should be replaced.
    """
    replacements: Dict[str, str] = {}  # target_nk -> anchor_nk

    # Group weights by base name using tensor class classification
    def _base_name(nk: str) -> Optional[str]:
        tc = _classify_tensor(nk)
        if tc == "palettized_indices":
            return nk.replace("_palettized_indices", "")
        if tc == "palettized_lut":
            return nk.replace("_palettized_lut", "")
        return None

    # Find all palettized weight bases in anchor
    anchor_bases = set()
    for nk in anchor_weights:
        base = _base_name(nk)
        if base is not None:
            anchor_bases.add(base)

    # For each base that exists in both anchor and target
    replaced_count = 0
    skipped_count = 0
    already_identical = 0
    total_bytes_saved = 0

    for base in sorted(anchor_bases):
        idx_key = f"{base}_palettized_indices"
        lut_key = f"{base}_palettized_lut"

        # Check both components exist in both
        if idx_key not in anchor_weights or lut_key not in anchor_weights:
            continue
        if idx_key not in target_weights or lut_key not in target_weights:
            continue

        a_idx = anchor_weights[idx_key]
        a_lut = anchor_weights[lut_key]
        t_idx = target_weights[idx_key]
        t_lut = target_weights[lut_key]

        # Shape must match
        if a_idx.shape != t_idx.shape or a_lut.shape != t_lut.shape:
            if verbose:
                print(f"    SKIP (shape): {base}  "
                      f"idx: {a_idx.shape} vs {t_idx.shape}, "
                      f"lut: {a_lut.shape} vs {t_lut.shape}")
            skipped_count += 1
            if diagnostics is not None:
                diagnostics.append(ReplacementDiag(
                    base_name=base,
                    reason=ReplacementReason.REJECTED_SHAPE,
                    tensor_class="palettized",
                    anchor_shape=a_idx.shape,
                    target_shape=t_idx.shape,
                ))
            continue

        # Already identical?
        if np.array_equal(a_idx, t_idx) and np.array_equal(a_lut, t_lut):
            already_identical += 1
            if diagnostics is not None:
                diagnostics.append(ReplacementDiag(
                    base_name=base,
                    reason=ReplacementReason.IDENTICAL,
                    tensor_class="palettized",
                    anchor_shape=a_idx.shape,
                    target_shape=t_idx.shape,
                ))
            continue

        # Estimate bytes that would be saved
        pair_bytes = a_idx.nbytes + a_lut.nbytes

        # Verify via dequantization if requested
        if verify_dequant:
            a_deq = _dequantize_lut(a_idx, a_lut)
            t_deq = _dequantize_lut(t_idx, t_lut)

            if a_deq is None or t_deq is None:
                if verbose:
                    print(f"    SKIP (deq failed): {base}")
                skipped_count += 1
                if diagnostics is not None:
                    diagnostics.append(ReplacementDiag(
                        base_name=base,
                        reason=ReplacementReason.REJECTED_DEQ_FAIL,
                        tensor_class="palettized",
                        anchor_shape=a_idx.shape,
                        target_shape=t_idx.shape,
                    ))
                continue

            # Multi-metric acceptance
            # Strict:  cos >= T AND max_abs <= A AND mean_abs <= M
            # Relaxed: cos >= T AND mean_abs <= M (no max_abs gate)
            #
            # Relaxed is appropriate when the original weights are identical and
            # only differ due to k-means centroid placement artifacts. A single
            # outlier max_abs of 0.01-0.014 in a multi-million element tensor
            # is meaningless LUT noise when cos=1.0 and mean_abs ~3e-8.
            cos = _cosine_similarity(a_deq, t_deq)
            diff = np.abs(a_deq.astype(np.float64) - t_deq.astype(np.float64))
            max_abs = float(np.max(diff))
            mean_abs = float(np.mean(diff))

            if max_abs_threshold is None:
                # Relaxed mode: cos + mean_abs only
                accepted = (cos >= cos_threshold
                            and mean_abs <= mean_abs_threshold)
            else:
                # Strict mode: all three metrics
                accepted = (cos >= cos_threshold
                            and max_abs <= max_abs_threshold
                            and mean_abs <= mean_abs_threshold)

            if not accepted:
                if verbose:
                    reject_reasons = []
                    if cos < cos_threshold:
                        reject_reasons.append(f"cos={cos:.6f}<{cos_threshold}")
                    if max_abs_threshold is not None and max_abs > max_abs_threshold:
                        reject_reasons.append(f"max_abs={max_abs:.2e}>{max_abs_threshold}")
                    if mean_abs > mean_abs_threshold:
                        reject_reasons.append(f"mean_abs={mean_abs:.2e}>{mean_abs_threshold}")
                    print(f"    SKIP ({', '.join(reject_reasons)}): {base}")
                skipped_count += 1
                if diagnostics is not None:
                    diagnostics.append(ReplacementDiag(
                        base_name=base,
                        reason=ReplacementReason.REJECTED_THRESHOLD,
                        tensor_class="palettized",
                        cos_sim=cos,
                        max_abs_diff=max_abs,
                        mean_abs_diff=mean_abs,
                        anchor_shape=a_idx.shape,
                        target_shape=t_idx.shape,
                    ))
                continue

            if verbose:
                print(f"    REPLACE (cos={cos:.6f}, max_abs={max_abs:.2e}, "
                      f"mean_abs={mean_abs:.2e}, ~{pair_bytes/1024:.0f}KB): {base}")

            if diagnostics is not None:
                diagnostics.append(ReplacementDiag(
                    base_name=base,
                    reason=ReplacementReason.DEQ_CLOSE,
                    tensor_class="palettized",
                    bytes_saved=pair_bytes,
                    cos_sim=cos,
                    max_abs_diff=max_abs,
                    mean_abs_diff=mean_abs,
                    anchor_shape=a_idx.shape,
                    target_shape=t_idx.shape,
                ))
        else:
            if verbose:
                print(f"    REPLACE (no verify, ~{pair_bytes/1024:.0f}KB): {base}")
            if diagnostics is not None:
                diagnostics.append(ReplacementDiag(
                    base_name=base,
                    reason=ReplacementReason.REPLACED_NO_VERIFY,
                    tensor_class="palettized",
                    bytes_saved=pair_bytes,
                    anchor_shape=a_idx.shape,
                    target_shape=t_idx.shape,
                ))

        replacements[idx_key] = idx_key
        replacements[lut_key] = lut_key
        replaced_count += 1
        total_bytes_saved += pair_bytes

    if verbose or replaced_count > 0:
        print(f"  Dedup summary: {replaced_count} weight pairs to replace "
              f"(~{total_bytes_saved / 1e6:.1f} MB), "
              f"{already_identical} already identical, {skipped_count} skipped")

    return replacements


# ---------------------------------------------------------------------------
# Apply replacements to MIL program and save modified mlpackage
# ---------------------------------------------------------------------------

def _apply_replacements_to_mlpackage(
    source_path: str,
    anchor_weights: Dict[str, np.ndarray],
    replacements: Dict[str, str],
    output_path: str,
    src_func_name: str = "main",
    verbose: bool = False,
) -> int:
    """Load source mlpackage, replace specified const ops with anchor values, save to output_path.

    Returns number of ops replaced.
    """
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load as mil_load
    from coremltools.converters.mil import mil as _mil

    model = ct.models.MLModel(source_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    prog = mil_load(
        spec,
        specification_version=spec.specificationVersion,
        file_weights_dir=model.weights_dir,
    )

    func = prog.functions.get(src_func_name)
    if func is None:
        func = next(iter(prog.functions.values()), None)
    if func is None:
        print(f"  WARNING: no function found in {source_path}")
        return 0

    replaced = 0
    for op in func.find_ops(op_type="const"):
        nk = _normalize_key(op.name)
        if nk not in replacements:
            continue
        anchor_nk = replacements[nk]
        if anchor_nk not in anchor_weights:
            continue

        anchor_arr = anchor_weights[anchor_nk]

        # Get current numpy array from the const op's output variable
        out_var = op.outputs[0]
        current_arr = out_var.val
        if current_arr is None:
            continue
        if not isinstance(current_arr, np.ndarray):
            continue
        if current_arr.shape != anchor_arr.shape:
            continue

        # Already identical — skip
        if np.array_equal(current_arr, anchor_arr):
            continue

        # Replace the value in-place via _sym_val.val (the writable path)
        # Var.val is read-only, but _sym_val.val has a setter
        out_var._sym_val.val = anchor_arr.copy().reshape(current_arr.shape)
        replaced += 1

    if replaced > 0:
        # Save modified program back to mlpackage using EMPTY pipeline
        # to avoid re-running optimization passes that would undo our replacements
        from coremltools.converters.mil.converter import mil_convert as _mil_convert

        # Ensure default_function_name points to an existing function
        # (multi-function models loaded via milproto may default to "main"
        # which doesn't exist when functions are named e.g. "infer", "prefill")
        if prog.default_function_name not in prog.functions:
            prog.default_function_name = src_func_name

        # Map spec version to deployment target to preserve LUT op support
        _spec_to_target = {
            7: ct.target.iOS16,
            8: ct.target.iOS17,
            9: ct.target.iOS18,
        }
        deploy_target = _spec_to_target.get(spec.specificationVersion, ct.target.iOS18)

        mlmodel = _mil_convert(
            prog,
            convert_from="milinternal",
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            skip_model_load=True,
            pass_pipeline=ct.PassPipeline.EMPTY,
            minimum_deployment_target=deploy_target,
        )
        mlmodel.save(output_path)

        if verbose:
            print(f"    Saved modified model ({replaced} ops replaced) -> {output_path}")
    else:
        # No changes needed — just copy
        if source_path != output_path:
            shutil.copytree(source_path, output_path)
        if verbose:
            print(f"    No replacements needed, copied as-is -> {output_path}")

    return replaced


# ---------------------------------------------------------------------------
# High-level API: prepare deduped sources for save_multifunction
# ---------------------------------------------------------------------------

@contextmanager
def prepare_dedup_sources(
    sources: List[Tuple[str, str, str]],
    cos_threshold: float = 0.9999,
    max_abs_threshold: Optional[float] = None,
    mean_abs_threshold: float = 0.001,
    verify_dequant: bool = True,
    verbose: bool = False,
    temp_dir: Optional[str] = None,
    preflight: bool = True,
    diagnostics: Optional[List[ReplacementDiag]] = None,
):
    """Context manager that prepares dedup-optimized source mlpackages.

    Takes a list of (mlpackage_path, src_function_name, target_function_name) tuples.
    The first entry is treated as the anchor. For all subsequent entries, palettized
    weights that are semantically identical to the anchor's are replaced with the
    anchor's byte-exact blobs.

    Yields a list of (path, src_fn, tgt_fn) tuples where paths may point to
    temporary modified mlpackages. Temp files are cleaned up on context exit.

    Args:
        sources: List of (mlpackage_path, src_function_name, target_function_name)
        cos_threshold: Minimum cosine similarity for replacement (default 0.9999)
        max_abs_threshold: Maximum absolute difference allowed (None=relaxed, no max_abs gate)
        mean_abs_threshold: Maximum mean absolute difference allowed (default 0.001)
        verify_dequant: If True, verify via dequantization before replacing
        verbose: Print per-weight replacement details
        temp_dir: Override temp directory (default: system temp)
        preflight: If True, run strict compatibility checks before dedup
        diagnostics: If provided, append per-replacement diagnostic records

    Yields:
        List of (path, src_fn, tgt_fn) ready for MultiFunctionDescriptor
    """
    if len(sources) < 2:
        yield sources
        return

    tmp_root = tempfile.mkdtemp(prefix="dedup_weights_", dir=temp_dir)
    result = []
    total_replaced = 0

    try:
        t0 = time.time()

        # Step 1: Extract anchor weights
        anchor_path, anchor_src_fn, anchor_tgt_fn = sources[0]
        print(f"[dedup] Loading anchor: {os.path.basename(anchor_path)}")
        anchor_prog, _ = _load_mil_program(anchor_path)
        anchor_weights = _extract_const_weights(anchor_prog, anchor_src_fn)
        print(f"[dedup] Anchor has {len(anchor_weights)} weight tensors")

        # Anchor is used as-is
        result.append((anchor_path, anchor_src_fn, anchor_tgt_fn))

        # Step 2: For each non-anchor source, find and apply replacements
        for i, (src_path, src_fn, tgt_fn) in enumerate(sources[1:], 1):
            print(f"[dedup] Processing target {i}/{len(sources)-1}: {os.path.basename(src_path)} -> {tgt_fn}")

            # Extract target weights
            target_prog, _ = _load_mil_program(src_path)
            target_weights = _extract_const_weights(target_prog, src_fn)

            # Run preflight checks
            if preflight:
                try:
                    _preflight_check_io_signature(
                        anchor_prog, target_prog, anchor_src_fn, src_fn)
                    _preflight_check_weight_counts(anchor_weights, target_weights)
                    _preflight_check_palettized_config(anchor_weights, target_weights)
                    if verbose:
                        print(f"[dedup]   Preflight checks passed")
                except PreflightError as e:
                    print(f"[dedup]   WARNING: Preflight check failed: {e}")
                    print(f"[dedup]   Skipping dedup for this source (using original)")
                    result.append((src_path, src_fn, tgt_fn))
                    continue

            # Find replaceable weights with multi-metric acceptance
            replacements = find_replaceable_weights(
                anchor_weights, target_weights,
                cos_threshold=cos_threshold,
                max_abs_threshold=max_abs_threshold,
                mean_abs_threshold=mean_abs_threshold,
                verify_dequant=verify_dequant,
                verbose=verbose,
                diagnostics=diagnostics,
            )

            if not replacements:
                print(f"[dedup]   No replacements needed")
                result.append((src_path, src_fn, tgt_fn))
                continue

            # Apply replacements and save to temp
            temp_pkg = os.path.join(tmp_root, f"dedup_{i}_{os.path.basename(src_path)}")
            n_replaced = _apply_replacements_to_mlpackage(
                src_path, anchor_weights, replacements, temp_pkg,
                src_func_name=src_fn, verbose=verbose,
            )
            total_replaced += n_replaced

            if n_replaced > 0:
                result.append((temp_pkg, src_fn, tgt_fn))
                print(f"[dedup]   Replaced {n_replaced} const ops")
            else:
                result.append((src_path, src_fn, tgt_fn))

        elapsed = time.time() - t0
        print(f"[dedup] Done in {elapsed:.1f}s: {total_replaced} total ops replaced across "
              f"{len(sources)-1} non-anchor sources")

        # Print diagnostics summary if collected
        if diagnostics is not None and len(diagnostics) > 0:
            _print_diagnostics_summary(diagnostics)

        yield result

    finally:
        # Cleanup temp directory
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)


def _print_diagnostics_summary(diagnostics: List[ReplacementDiag]):
    """Print a summary table of per-replacement diagnostics."""
    by_reason = {}
    for d in diagnostics:
        by_reason.setdefault(d.reason.value, []).append(d)

    total_saved = sum(d.bytes_saved for d in diagnostics)

    print(f"\n[dedup] Diagnostics summary ({len(diagnostics)} weight pairs evaluated):")
    for reason, items in sorted(by_reason.items()):
        saved = sum(d.bytes_saved for d in items)
        print(f"  {reason}: {len(items)} pairs"
              + (f" (~{saved / 1e6:.1f} MB saved)" if saved > 0 else ""))

    if total_saved > 0:
        print(f"  Total estimated savings: ~{total_saved / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Standalone CLI for testing
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Surgical weight dedup between two mlpackages.\n\n"
                    "Replaces palettized weights in --target with --anchor's blobs\n"
                    "where dequantized values are verified identical via multi-metric\n"
                    "acceptance (cosine, max_abs, mean_abs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--anchor", required=True, help="Anchor mlpackage path")
    ap.add_argument("--target", required=True, help="Target mlpackage path to compare/dedup")
    ap.add_argument("--output", default=None,
                    help="Output path for deduped target (omit for dry-run)")
    ap.add_argument("--source-fn-anchor", default="main",
                    help="Source function name in anchor model (default: main)")
    ap.add_argument("--source-fn-target", default="main",
                    help="Source function name in target model (default: main)")
    ap.add_argument("--cos-threshold", type=float, default=0.9999,
                    help="Cosine similarity threshold (default 0.9999)")
    ap.add_argument("--max-abs-threshold", type=float, default=None,
                    help="Maximum absolute difference threshold (default: None/relaxed). "
                         "Set to e.g. 0.01 for strict mode.")
    ap.add_argument("--mean-abs-threshold", type=float, default=0.001,
                    help="Mean absolute difference threshold (default 0.001)")
    ap.add_argument("--strict", action="store_true",
                    help="Enable strict mode: require max_abs <= 0.01 in addition to cos+mean_abs. "
                         "Default is relaxed (cos+mean_abs only), which is safe because "
                         "max_abs outliers are k-means centroid artifacts, not real weight divergence.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip dequantization verification (faster but less safe)")
    ap.add_argument("--no-preflight", action="store_true",
                    help="Skip preflight compatibility checks")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Print per-replacement diagnostics (tensor bytes saved, reason code)")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Just report what would be replaced, don't save")

    args = ap.parse_args()

    # --strict sets max_abs_threshold=0.01 if not explicitly provided
    if args.strict and args.max_abs_threshold is None:
        args.max_abs_threshold = 0.01

    mode = "strict" if args.max_abs_threshold is not None else "relaxed"
    print(f"Anchor: {args.anchor}")
    print(f"Target: {args.target}")
    print(f"Source functions: anchor={args.source_fn_anchor}, target={args.source_fn_target}")
    print(f"Mode: {mode}" + (f" (max_abs<={args.max_abs_threshold})" if args.max_abs_threshold else ""))

    # Load anchor
    print("\nLoading anchor...")
    anchor_prog, _ = _load_mil_program(args.anchor)
    anchor_weights = _extract_const_weights(anchor_prog, args.source_fn_anchor)
    print(f"  {len(anchor_weights)} weight tensors")

    # Load target
    print("Loading target...")
    target_prog, _ = _load_mil_program(args.target)
    target_weights = _extract_const_weights(target_prog, args.source_fn_target)
    print(f"  {len(target_weights)} weight tensors")

    # Run preflight checks
    if not args.no_preflight:
        print("\nRunning preflight checks...")
        try:
            _preflight_check_io_signature(
                anchor_prog, target_prog,
                args.source_fn_anchor, args.source_fn_target)
            _preflight_check_weight_counts(anchor_weights, target_weights)
            _preflight_check_palettized_config(anchor_weights, target_weights)
            print("  All preflight checks passed")
        except PreflightError as e:
            print(f"  PREFLIGHT FAILED: {e}")
            print("  Use --no-preflight to bypass (at your own risk)")
            return 1

    # Collect diagnostics if requested
    diag_list: Optional[List[ReplacementDiag]] = [] if args.diagnostics else None

    # Find replacements with multi-metric acceptance
    print("\nFinding replaceable weights...")
    replacements = find_replaceable_weights(
        anchor_weights, target_weights,
        cos_threshold=args.cos_threshold,
        max_abs_threshold=args.max_abs_threshold,
        mean_abs_threshold=args.mean_abs_threshold,
        verify_dequant=not args.no_verify,
        verbose=args.verbose,
        diagnostics=diag_list,
    )

    n_pairs = len(replacements) // 2  # indices + lut = 2 keys per pair
    print(f"\nResult: {n_pairs} weight pairs ({len(replacements)} const ops) can be replaced")

    # Print diagnostics if collected
    if diag_list is not None and len(diag_list) > 0:
        _print_diagnostics_summary(diag_list)
        if args.verbose:
            print("\nDetailed diagnostics:")
            for d in diag_list:
                line = f"  [{d.reason.value:>20s}] {d.base_name}"
                if d.bytes_saved > 0:
                    line += f"  ({d.bytes_saved / 1024:.0f} KB saved)"
                if d.cos_sim > 0:
                    line += f"  cos={d.cos_sim:.6f}"
                if d.max_abs_diff > 0:
                    line += f"  max_abs={d.max_abs_diff:.2e}"
                if d.mean_abs_diff > 0:
                    line += f"  mean_abs={d.mean_abs_diff:.2e}"
                print(line)

    if args.dry_run or not args.output:
        print("(dry-run mode — no output written)")
        return 0

    # Apply and save
    print(f"\nApplying replacements and saving to: {args.output}")
    n_replaced = _apply_replacements_to_mlpackage(
        args.target, anchor_weights, replacements, args.output,
        src_func_name=args.source_fn_target, verbose=args.verbose,
    )
    print(f"Done: {n_replaced} ops replaced")

    return 0


if __name__ == "__main__":
    sys.exit(main())
