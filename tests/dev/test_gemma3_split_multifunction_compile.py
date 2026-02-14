#!/usr/bin/env python3
"""
Build two-function Gemma3 packages per chunk (infer+infer_rotate, prefill+prefill_rotate),
compile to .mlmodelc, and validate function loading on compiled models.
"""

import argparse
import os
import subprocess
from pathlib import Path

import coremltools as ct


def _find(path: Path) -> Path | None:
    return path if path.exists() else None


def _pick_model(base_dir: Path, prefix: str, name: str, lut: str | None, chunk: int, num_chunks: int) -> Path | None:
    if lut:
        cand = base_dir / f"{prefix}_{name}_lut{lut}_chunk_{chunk:02d}of{num_chunks:02d}.mlpackage"
        hit = _find(cand)
        if hit:
            return hit
    cand = base_dir / f"{prefix}_{name}_chunk_{chunk:02d}of{num_chunks:02d}.mlpackage"
    return _find(cand)


def _compile(model_path: Path, out_dir: Path) -> None:
    cmd = [
        "xcrun",
        "coremlcompiler",
        "compile",
        str(model_path),
        str(out_dir),
        "--add-mlprogram-if-eligible",
        "force",
    ]
    subprocess.run(cmd, check=True)


def _load_compiled(modelc_path: Path, function_name: str) -> None:
    cm = ct.models.CompiledMLModel(str(modelc_path), function_name=function_name)
    _ = cm  # force construction


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split Gemma3 multi-function chunk into infer/prefill packages and test compiled loading."
    )
    parser.add_argument("--dir", required=True, help="Directory with chunk mlpackages")
    parser.add_argument("--prefix", default="gemma3", help="Model prefix")
    parser.add_argument("--chunk", type=int, default=1, help="Chunk index (1-based)")
    parser.add_argument("--num-chunks", type=int, default=2, help="Total chunks")
    parser.add_argument("--lut", default=None, help="LUT bits (e.g., 4)")
    args = parser.parse_args()

    base_dir = Path(args.dir).resolve()
    chunk = args.chunk
    num_chunks = args.num_chunks
    prefix = args.prefix
    lut = args.lut

    ffn = _pick_model(base_dir, prefix, "FFN", lut, chunk, num_chunks)
    ffn_rot = _pick_model(base_dir, prefix, "FFN_rotate", lut, chunk, num_chunks)
    pf = _pick_model(base_dir, prefix, "prefill", lut, chunk, num_chunks)
    pf_rot = _pick_model(base_dir, prefix, "prefill_rotate", lut, chunk, num_chunks)

    if not all([ffn, ffn_rot, pf, pf_rot]):
        print("Missing input models:")
        print(f"  FFN: {ffn}")
        print(f"  FFN_rotate: {ffn_rot}")
        print(f"  prefill: {pf}")
        print(f"  prefill_rotate: {pf_rot}")
        return 1

    infer_out = base_dir / f"{prefix}_FFN_INF_chunk_{chunk:02d}of{num_chunks:02d}.mlpackage"
    prefill_out = base_dir / f"{prefix}_PREFILL_chunk_{chunk:02d}of{num_chunks:02d}.mlpackage"

    for out_path, left, right, lname, rname in [
        (infer_out, ffn, ffn_rot, "infer", "infer_rotate"),
        (prefill_out, pf, pf_rot, "prefill", "prefill_rotate"),
    ]:
        tmp = base_dir / f"tmp_{out_path.name}"
        if tmp.exists():
            subprocess.run(["rm", "-rf", str(tmp)], check=True)
        desc = ct.utils.MultiFunctionDescriptor()
        desc.add_function(str(left), "main", lname)
        desc.add_function(str(right), "main", rname)
        desc.default_function_name = lname
        ct.utils.save_multifunction(desc, str(tmp))
        model = ct.models.MLModel(str(tmp))
        model.save(str(out_path))
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
        print(f"Saved {out_path.name}")

    # Compile and test function loading
    for pkg, fn_a, fn_b in [
        (infer_out, "infer", "infer_rotate"),
        (prefill_out, "prefill", "prefill_rotate"),
    ]:
        _compile(pkg, base_dir)
        modelc = pkg.with_suffix(".mlmodelc")
        print(f"Compiled: {modelc.name}")
        try:
            _load_compiled(modelc, fn_a)
            _load_compiled(modelc, fn_b)
            print(f"Compiled functions OK: {fn_a}, {fn_b}")
        except Exception as e:
            print(f"Compiled function load failed for {modelc.name}: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
