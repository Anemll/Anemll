#!/usr/bin/env python3
"""
Calculate weight sizes and recommend chunk splits for HuggingFace models.

Reads config.json from a local HF model (or HF cache path) and computes:
- Embedding / LM head sizes
- Per-layer weight sizes (attention + MLP)
- Total FFN body size
- Chunk split options with sizes at FP16 and various LUT quantization levels
- Recommended --chunk value for a target max chunk size

Supports: LLaMA, Qwen2, Qwen3, Gemma, Gemma2, Gemma3, DeepSeek, and
any standard HF decoder-only model with config.json.

Usage:
    # Interactive report
    python anemll/utils/calc_chunk_split.py <model_path_or_hf_id>
    python anemll/utils/calc_chunk_split.py --max-chunk-mb 950 --lut 6 <model_path>

    # Scripting (returns single integer for --chunk)
    python anemll/utils/calc_chunk_split.py --auto --lut 6 --max-chunk-mb 950 <model_path>

    # Used by convert_model.sh --chunk auto
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_config(model_path: str) -> Path:
    """Find config.json from model path, HF cache, or HF model ID."""
    p = Path(model_path)

    # Direct path to config.json
    if p.name == "config.json" and p.is_file():
        return p

    # Directory containing config.json
    if p.is_dir():
        cfg = p / "config.json"
        if cfg.is_file():
            return cfg
        # Check snapshots subdirectory (HF cache layout)
        snapshots = p / "snapshots"
        if snapshots.is_dir():
            for snap in sorted(snapshots.iterdir(), reverse=True):
                cfg = snap / "config.json"
                if cfg.is_file():
                    return cfg

    # Try HF cache: ~/.cache/huggingface/hub/models--org--name
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.is_dir():
        # Try direct model ID (e.g., "Qwen/Qwen3-4B" -> "models--Qwen--Qwen3-4B")
        cache_name = "models--" + model_path.replace("/", "--")
        cache_dir = hf_cache / cache_name
        if cache_dir.is_dir():
            snapshots = cache_dir / "snapshots"
            if snapshots.is_dir():
                for snap in sorted(snapshots.iterdir(), reverse=True):
                    cfg = snap / "config.json"
                    if cfg.is_file():
                        return cfg

    raise FileNotFoundError(
        f"Cannot find config.json at '{model_path}'. "
        f"Provide a path to a directory containing config.json, "
        f"an HF cache path, or a HuggingFace model ID (e.g., Qwen/Qwen3-4B)."
    )


def load_text_config(config_path: Path) -> Dict[str, Any]:
    """Load config.json and extract text model config (handles nested multimodal configs)."""
    with open(config_path) as f:
        cfg = json.load(f)

    # Multimodal models (Gemma3, Gemma3n) nest text config
    if "text_config" in cfg and "hidden_size" not in cfg:
        text_cfg = cfg["text_config"]
        for key in ["architectures", "model_type", "tie_word_embeddings"]:
            if key in cfg and key not in text_cfg:
                text_cfg[key] = cfg[key]
        return text_cfg

    # Some multimodal Gemma3 have hidden_size at top level from text_config
    if "text_config" in cfg and "hidden_size" in cfg.get("text_config", {}):
        text_cfg = cfg["text_config"]
        for key in ["architectures", "model_type", "tie_word_embeddings"]:
            if key in cfg and key not in text_cfg:
                text_cfg[key] = cfg[key]
        return text_cfg

    return cfg


def get_int(cfg: Dict, key: str, default: Optional[int] = None) -> int:
    val = cfg.get(key, default)
    if val is None:
        raise KeyError(f"Missing required config field: '{key}'")
    return int(val)


def get_head_dim(cfg: Dict) -> int:
    """Get head_dim, explicit or computed."""
    if "head_dim" in cfg:
        return int(cfg["head_dim"])
    return get_int(cfg, "hidden_size") // get_int(cfg, "num_attention_heads")


def get_intermediate_size(cfg: Dict, layer_idx: int = 0) -> int:
    """Get intermediate_size (may be per-layer list in Gemma3n)."""
    val = cfg.get("intermediate_size")
    if isinstance(val, list):
        return int(val[layer_idx])
    return int(val)


def has_attention_bias(cfg: Dict) -> bool:
    """Check if model uses bias in attention projections."""
    model_type = cfg.get("model_type", "")
    # Qwen2 defaults to bias=True if field is absent
    if model_type == "qwen2" and "attention_bias" not in cfg:
        return True
    return cfg.get("attention_bias", False)


def has_mlp_bias(cfg: Dict) -> bool:
    """Check if model uses bias in MLP projections."""
    return cfg.get("mlp_bias", False)


def calc_layer_weights(cfg: Dict, layer_idx: int = 0) -> Dict[str, int]:
    """Calculate weight parameter counts for a single transformer layer."""
    hidden = get_int(cfg, "hidden_size")
    head_dim = get_head_dim(cfg)
    n_heads = get_int(cfg, "num_attention_heads")
    n_kv_heads = get_int(cfg, "num_key_value_heads", n_heads)
    intermediate = get_intermediate_size(cfg, layer_idx)
    attn_bias = has_attention_bias(cfg)
    mlp_bias = has_mlp_bias(cfg)

    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    weights = {}

    # Attention projections
    weights["q_proj"] = hidden * q_dim + (q_dim if attn_bias else 0)
    weights["k_proj"] = hidden * kv_dim + (kv_dim if attn_bias else 0)
    weights["v_proj"] = hidden * kv_dim + (kv_dim if attn_bias else 0)
    weights["o_proj"] = q_dim * hidden + (hidden if attn_bias else 0)

    # QK norms (Qwen3, Gemma3)
    model_type = cfg.get("model_type", "")
    if model_type in ("qwen3", "gemma3_text", "gemma3", "gemma3n_text"):
        weights["q_norm"] = head_dim
        weights["k_norm"] = head_dim

    # MLP (gate/up/down for SiLU-gated models)
    weights["gate_proj"] = hidden * intermediate + (intermediate if mlp_bias else 0)
    weights["up_proj"] = hidden * intermediate + (intermediate if mlp_bias else 0)
    weights["down_proj"] = intermediate * hidden + (hidden if mlp_bias else 0)

    # Layer norms (RMSNorm weights)
    weights["input_layernorm"] = hidden
    weights["post_attention_layernorm"] = hidden

    # Gemma3 has pre/post feedforward layernorms
    if model_type in ("gemma3_text", "gemma3", "gemma3n_text"):
        weights["pre_feedforward_layernorm"] = hidden
        weights["post_feedforward_layernorm"] = hidden

    return weights


def calc_model_weights(cfg: Dict) -> Dict[str, Any]:
    """Calculate all weight sizes for the model."""
    hidden = get_int(cfg, "hidden_size")
    vocab = get_int(cfg, "vocab_size")
    n_layers = get_int(cfg, "num_hidden_layers")
    tied = cfg.get("tie_word_embeddings", False)

    result = {
        "config": {
            "model_type": cfg.get("model_type", "unknown"),
            "architectures": cfg.get("architectures", []),
            "hidden_size": hidden,
            "intermediate_size": cfg.get("intermediate_size"),
            "num_hidden_layers": n_layers,
            "num_attention_heads": get_int(cfg, "num_attention_heads"),
            "num_key_value_heads": get_int(cfg, "num_key_value_heads", get_int(cfg, "num_attention_heads")),
            "head_dim": get_head_dim(cfg),
            "vocab_size": vocab,
            "tie_word_embeddings": tied,
        },
    }

    # Embeddings
    embed_params = vocab * hidden
    result["embeddings"] = {"params": embed_params, "bytes_fp16": embed_params * 2}

    # LM head
    if tied:
        result["lm_head"] = {"params": 0, "bytes_fp16": 0, "note": "tied with embeddings"}
    else:
        lm_params = vocab * hidden
        result["lm_head"] = {"params": lm_params, "bytes_fp16": lm_params * 2}

    # Per-layer weights
    layers = []
    for i in range(n_layers):
        lw = calc_layer_weights(cfg, i)
        total_params = sum(lw.values())
        layers.append({
            "layer_idx": i,
            "weights": lw,
            "total_params": total_params,
            "bytes_fp16": total_params * 2,
        })
    result["layers"] = layers

    # Final norm
    result["final_norm"] = {"params": hidden, "bytes_fp16": hidden * 2}

    # Totals
    total_layer_params = sum(l["total_params"] for l in layers)
    total_params = embed_params + (0 if tied else vocab * hidden) + total_layer_params + hidden
    result["totals"] = {
        "embeddings_bytes": embed_params * 2,
        "lm_head_bytes": 0 if tied else vocab * hidden * 2,
        "layers_bytes": total_layer_params * 2,
        "final_norm_bytes": hidden * 2,
        "total_bytes_fp16": total_params * 2,
        "total_params": total_params,
    }

    return result


def calc_chunk_splits(
    layers: List[Dict], n_chunks: int
) -> List[Tuple[int, int, int]]:
    """Calculate balanced chunk splits. Returns list of (start, end, bytes_fp16)."""
    n_layers = len(layers)
    base = n_layers // n_chunks
    rem = n_layers % n_chunks

    chunks = []
    for c in range(n_chunks):
        start = c * base + min(c, rem)
        end = start + base + (1 if c < rem else 0)
        chunk_bytes = sum(layers[i]["bytes_fp16"] for i in range(start, end))
        chunks.append((start, end, chunk_bytes))
    return chunks


def lut_ratio(bits: int) -> float:
    """Approximate compression ratio for LUT quantization vs FP16."""
    return bits / 16.0


def format_bytes(b: int) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024**3:.1f} GB"
    elif b >= 1024 ** 2:
        return f"{b / 1024**2:.1f} MB"
    elif b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def recommend_chunks(
    layers: List[Dict],
    max_chunk_bytes: int,
    lut_bits: Optional[int] = None,
    overhead: float = 1.0,
) -> int:
    """Find minimum number of chunks where largest chunk <= max_chunk_bytes.

    Args:
        layers: Per-layer weight info from calc_model_weights()
        max_chunk_bytes: Maximum allowed bytes per chunk
        lut_bits: LUT quantization bits (None or 16 = FP16)
        overhead: Safety multiplier (e.g., 1.10 for 10% margin)
    """
    n_layers = len(layers)
    ratio = lut_ratio(lut_bits) if lut_bits and lut_bits < 16 else 1.0

    for n in range(1, n_layers + 1):
        splits = calc_chunk_splits(layers, n)
        largest = max(s[2] for s in splits)
        if int(largest * ratio * overhead) <= max_chunk_bytes:
            return n
    return n_layers


def print_report(result: Dict, max_chunk_mb: Optional[int], lut_bits: Optional[int], overhead: float = 1.0):
    cfg = result["config"]

    print("=" * 70)
    print(f"Model Weight Size Calculator")
    print("=" * 70)
    print(f"  Architecture:    {', '.join(cfg['architectures'])}")
    print(f"  Model type:      {cfg['model_type']}")
    print(f"  Hidden size:     {cfg['hidden_size']}")
    isize = cfg['intermediate_size']
    if isinstance(isize, list):
        print(f"  Intermediate:    {min(isize)}-{max(isize)} (per-layer)")
    else:
        print(f"  Intermediate:    {isize}")
    print(f"  Layers:          {cfg['num_hidden_layers']}")
    print(f"  Attention heads: {cfg['num_attention_heads']} (KV: {cfg['num_key_value_heads']})")
    print(f"  Head dim:        {cfg['head_dim']}")
    print(f"  Vocab size:      {cfg['vocab_size']}")
    print(f"  Tied embeddings: {cfg['tie_word_embeddings']}")
    if overhead != 1.0:
        print(f"  Overhead:        {overhead:.0%}")
    print()

    # Component sizes
    t = result["totals"]
    print("Component Sizes (FP16):")
    print(f"  Embeddings:      {format_bytes(t['embeddings_bytes']):>10}")
    if t["lm_head_bytes"] > 0:
        print(f"  LM Head:         {format_bytes(t['lm_head_bytes']):>10}")
    else:
        print(f"  LM Head:         (tied with embeddings)")
    print(f"  FFN Body:        {format_bytes(t['layers_bytes']):>10}  ({cfg['num_hidden_layers']} layers)")
    print(f"  Final Norm:      {format_bytes(t['final_norm_bytes']):>10}")
    print(f"  ────────────────────────────")
    print(f"  Total:           {format_bytes(t['total_bytes_fp16']):>10}  ({t['total_params'] / 1e9:.2f}B params)")
    print()

    # Per-layer breakdown (first layer)
    layer0 = result["layers"][0]
    print(f"Per-Layer Breakdown (layer 0, FP16):")
    for name, params in layer0["weights"].items():
        print(f"  {name:35s} {format_bytes(params * 2):>10}")
    print(f"  {'─' * 35} {'─' * 10}")
    print(f"  {'Total per layer':35s} {format_bytes(layer0['bytes_fp16']):>10}")

    # Check if layers vary in size
    layer_sizes = [l["bytes_fp16"] for l in result["layers"]]
    if min(layer_sizes) != max(layer_sizes):
        print(f"  Note: Layer sizes vary ({format_bytes(min(layer_sizes))} - {format_bytes(max(layer_sizes))})")
    print()

    # Quantization sizes
    lut_options = [16, 8, 6, 4]
    layers = result["layers"]

    print("Chunk Split Options:")
    if overhead != 1.0:
        print(f"  (sizes include {overhead:.0%} overhead)")
    print()

    # Header
    header = f"  {'Chunks':>6}  {'Layout':>20}"
    for bits in lut_options:
        label = "FP16" if bits == 16 else f"LUT{bits}"
        header += f"  {label + ' max':>10}"
    print(header)
    print(f"  {'─' * 6}  {'─' * 20}", end="")
    for _ in lut_options:
        print(f"  {'─' * 10}", end="")
    print()

    for n_chunks in range(1, min(13, cfg["num_hidden_layers"] + 1)):
        splits = calc_chunk_splits(layers, n_chunks)
        layout = "+".join(str(e - s) for s, e, _ in splits)
        largest_fp16 = max(b for _, _, b in splits)

        row = f"  {n_chunks:>6}  {layout:>20}"
        for bits in lut_options:
            ratio = lut_ratio(bits)
            size = int(largest_fp16 * ratio * overhead)
            row += f"  {format_bytes(size):>10}"
        print(row)
    print()

    # Recommendation
    if max_chunk_mb:
        max_bytes = max_chunk_mb * 1024 * 1024
        print(f"Recommendation (max chunk: {max_chunk_mb} MB, overhead: {overhead:.0%}):")
        for bits in lut_options:
            label = "FP16" if bits == 16 else f"LUT{bits}"
            n = recommend_chunks(layers, max_bytes, bits, overhead)
            splits = calc_chunk_splits(layers, n)
            largest = max(b for _, _, b in splits)
            actual = int(largest * lut_ratio(bits) * overhead)
            layout = "+".join(str(e - s) for s, e, _ in splits)
            print(f"  {label:>5}: --chunk {n:>2}  ({layout})  largest = {format_bytes(actual)}")
        print()

    if lut_bits:
        label = f"LUT{lut_bits}"
        ratio = lut_ratio(lut_bits)
        target_mb = max_chunk_mb or 1024
        target_bytes = target_mb * 1024 * 1024
        print(f"Target: {label}, max chunk ≤ {target_mb} MB (overhead: {overhead:.0%})")
        n = recommend_chunks(layers, target_bytes, lut_bits, overhead)
        splits = calc_chunk_splits(layers, n)
        largest = max(b for _, _, b in splits)
        actual = int(largest * ratio * overhead)
        layout = "+".join(str(e - s) for s, e, _ in splits)
        print(f"  → --chunk {n}  ({layout})  largest chunk = {format_bytes(actual)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate weight sizes and recommend chunk splits for HuggingFace models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive report
  python anemll/utils/calc_chunk_split.py Qwen/Qwen3-4B
  python anemll/utils/calc_chunk_split.py --max-chunk-mb 950 --lut 6 google/gemma-3-1b-it

  # Scripting (for convert_model.sh --chunk auto)
  python anemll/utils/calc_chunk_split.py --auto --lut 6 --max-chunk-mb 950 Qwen/Qwen3-4B
  python anemll/utils/calc_chunk_split.py --auto --lut 6 --overhead 1.10 google/gemma-3-1b-it
        """,
    )
    parser.add_argument("model", help="Path to model directory, HF cache path, or HF model ID")
    parser.add_argument(
        "--max-chunk-mb",
        type=int,
        default=950,
        help="Target max chunk size in MB (default: 950)",
    )
    parser.add_argument(
        "--lut",
        type=int,
        choices=[4, 6, 8, 16],
        default=None,
        help="LUT quantization bits (4, 6, 8, or 16 for FP16). Default: show all",
    )
    parser.add_argument(
        "--overhead",
        type=float,
        default=1.10,
        help="Safety overhead multiplier (default: 1.10 = 10%%)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Output only the recommended chunk count (single integer, for scripting)",
    )

    args = parser.parse_args()

    try:
        config_path = find_config(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    cfg = load_text_config(config_path)
    result = calc_model_weights(cfg)

    if args.auto:
        # Scripting mode: output only the chunk count
        lut_bits = args.lut  # None means FP16
        max_bytes = args.max_chunk_mb * 1024 * 1024
        n = recommend_chunks(result["layers"], max_bytes, lut_bits, args.overhead)
        print(n)
    else:
        # Interactive report mode
        print(f"Config: {config_path}")
        print_report(result, args.max_chunk_mb, args.lut, args.overhead)


if __name__ == "__main__":
    main()
