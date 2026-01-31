#!/usr/bin/env python3
"""
Benchmark SDPA chunking strategies on ANE.

Tests different chunk sizes for sliding window and global attention.
Converts only a single attention block, not the full model.

Usage:
    # Test sliding window (1024 KV length) with different Q chunk sizes
    python tests/dev/benchmark_sdpa_chunking.py \
        --seq-len 1024 \
        --kv-len 1024 \
        --chunk-sizes 0 512 256 \
        --iterations 50

    # Test global attention (4096 KV length)
    python tests/dev/benchmark_sdpa_chunking.py \
        --seq-len 1024 \
        --kv-len 4096 \
        --chunk-sizes 0 512 256 128 \
        --iterations 50

    # Gemma3 4B config (prefill batch=64)
    python tests/dev/benchmark_sdpa_chunking.py \
        --seq-len 64 \
        --kv-len 1024 \
        --hidden-size 3072 \
        --num-heads 24 \
        --head-dim 128 \
        --chunk-sizes 0 32 16
"""

import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from pathlib import Path
import tempfile
import shutil


class SDPABlock(nn.Module):
    """Single attention block for benchmarking SDPA chunking."""

    def __init__(self, hidden_size=2048, num_heads=16, head_dim=128,
                 seq_len=1024, kv_len=1024, chunk_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.kv_len = kv_len
        self.chunk_size = chunk_size
        self.scale = 1.0 / math.sqrt(head_dim)

        # Q, K, V projections as Conv2D for ANE compatibility
        self.q_proj = nn.Conv2d(hidden_size, num_heads * head_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(hidden_size, num_heads * head_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(hidden_size, num_heads * head_dim, 1, bias=False)
        self.o_proj = nn.Conv2d(num_heads * head_dim, hidden_size, 1, bias=False)

    def forward(self, x, k_cache, v_cache):
        """
        x: [B, hidden, 1, seq_len] - input (query source)
        k_cache: [B, heads*head_dim, 1, kv_len] - key cache
        v_cache: [B, heads*head_dim, 1, kv_len] - value cache
        """
        B = x.shape[0]

        # Project Q from input
        Q = self.q_proj(x)  # [B, heads*dim, 1, seq_len]

        # Use cached K, V
        K = k_cache
        V = v_cache

        # Reshape for attention: [B, heads, seq, head_dim]
        Q = Q.view(B, self.num_heads, self.head_dim, self.seq_len).permute(0, 1, 3, 2)
        K = K.view(B, self.num_heads, self.head_dim, self.kv_len).permute(0, 1, 3, 2)
        V = V.view(B, self.num_heads, self.head_dim, self.kv_len).permute(0, 1, 3, 2)

        if self.chunk_size is None or self.chunk_size == 0 or self.seq_len <= self.chunk_size:
            # Standard SDPA
            output = self._sdpa(Q, K, V)
        else:
            # Chunked SDPA
            output = self._sdpa_chunked(Q, K, V)

        # Reshape back: [B, heads, seq, dim] -> [B, heads*dim, 1, seq]
        output = output.permute(0, 1, 3, 2).reshape(B, -1, 1, self.seq_len)
        output = self.o_proj(output)

        return output

    def _sdpa(self, Q, K, V):
        """Standard scaled dot-product attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, V)

    def _sdpa_chunked(self, Q, K, V):
        """Chunked SDPA - process Q in chunks for memory efficiency."""
        outputs = []
        for i in range(0, self.seq_len, self.chunk_size):
            end = min(i + self.chunk_size, self.seq_len)
            q_chunk = Q[:, :, i:end, :]

            scores = torch.matmul(q_chunk, K.transpose(-2, -1)) * self.scale
            probs = F.softmax(scores, dim=-1)
            chunk_out = torch.matmul(probs, V)
            outputs.append(chunk_out)

        return torch.cat(outputs, dim=2)


class SDPAChunkedKV(nn.Module):
    """Attention block with chunked K/V for single-token decode.

    Uses online softmax (Flash Attention style) to correctly normalize
    attention scores when processing K/V in chunks.
    """

    def __init__(self, hidden_size=2048, num_heads=16, head_dim=128,
                 seq_len=1, kv_len=1024, chunk_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len  # Should be 1 for decode
        self.kv_len = kv_len
        self.chunk_size = chunk_size
        self.scale = 1.0 / math.sqrt(head_dim)

        # Projections
        self.q_proj = nn.Conv2d(hidden_size, num_heads * head_dim, 1, bias=False)
        self.o_proj = nn.Conv2d(num_heads * head_dim, hidden_size, 1, bias=False)

    def forward(self, x, k_cache, v_cache):
        """
        x: [B, hidden, 1, 1] - single token input
        k_cache: [B, heads*head_dim, 1, kv_len]
        v_cache: [B, heads*head_dim, 1, kv_len]
        """
        B = x.shape[0]

        # Project Q
        Q = self.q_proj(x)  # [B, heads*dim, 1, 1]

        # Reshape for attention
        Q = Q.view(B, self.num_heads, self.head_dim, self.seq_len).permute(0, 1, 3, 2)
        K = k_cache.view(B, self.num_heads, self.head_dim, self.kv_len).permute(0, 1, 3, 2)
        V = v_cache.view(B, self.num_heads, self.head_dim, self.kv_len).permute(0, 1, 3, 2)

        if self.chunk_size is None or self.chunk_size == 0 or self.kv_len <= self.chunk_size:
            output = self._sdpa(Q, K, V)
        else:
            output = self._sdpa_chunked_kv(Q, K, V)

        output = output.permute(0, 1, 3, 2).reshape(B, -1, 1, self.seq_len)
        output = self.o_proj(output)
        return output

    def _sdpa(self, Q, K, V):
        """Standard SDPA."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, V)

    def _sdpa_chunked_kv(self, Q, K, V):
        """Chunked K/V with online softmax for correct normalization."""
        B, H, S, D = Q.shape  # S should be 1 for decode

        # For tracing, we need to unroll the loop
        # Process K/V in chunks with online softmax
        num_chunks = (self.kv_len + self.chunk_size - 1) // self.chunk_size

        # First chunk - initialize
        k0 = K[:, :, 0:self.chunk_size, :]
        v0 = V[:, :, 0:self.chunk_size, :]
        scores0 = torch.matmul(Q, k0.transpose(-2, -1)) * self.scale

        running_max = scores0.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores0 - running_max)
        running_sum = exp_scores.sum(dim=-1, keepdim=True)
        output = torch.matmul(exp_scores, v0)

        # Process remaining chunks
        if num_chunks >= 2:
            start = self.chunk_size
            end = min(start + self.chunk_size, self.kv_len)
            k1 = K[:, :, start:end, :]
            v1 = V[:, :, start:end, :]
            scores1 = torch.matmul(Q, k1.transpose(-2, -1)) * self.scale

            chunk_max = scores1.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(running_max, chunk_max)

            scale_old = torch.exp(running_max - new_max)
            scale_new = torch.exp(scores1 - new_max)

            running_sum = running_sum * scale_old + scale_new.sum(dim=-1, keepdim=True)
            output = output * scale_old + torch.matmul(scale_new, v1)
            running_max = new_max

        if num_chunks >= 3:
            start = self.chunk_size * 2
            end = min(start + self.chunk_size, self.kv_len)
            k2 = K[:, :, start:end, :]
            v2 = V[:, :, start:end, :]
            scores2 = torch.matmul(Q, k2.transpose(-2, -1)) * self.scale

            chunk_max = scores2.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(running_max, chunk_max)

            scale_old = torch.exp(running_max - new_max)
            scale_new = torch.exp(scores2 - new_max)

            running_sum = running_sum * scale_old + scale_new.sum(dim=-1, keepdim=True)
            output = output * scale_old + torch.matmul(scale_new, v2)
            running_max = new_max

        if num_chunks >= 4:
            start = self.chunk_size * 3
            end = min(start + self.chunk_size, self.kv_len)
            k3 = K[:, :, start:end, :]
            v3 = V[:, :, start:end, :]
            scores3 = torch.matmul(Q, k3.transpose(-2, -1)) * self.scale

            chunk_max = scores3.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(running_max, chunk_max)

            scale_old = torch.exp(running_max - new_max)
            scale_new = torch.exp(scores3 - new_max)

            running_sum = running_sum * scale_old + scale_new.sum(dim=-1, keepdim=True)
            output = output * scale_old + torch.matmul(scale_new, v3)
            running_max = new_max

        # Final normalization
        output = output / running_sum
        return output


class SDPASlidingWindow(nn.Module):
    """Attention block with sliding window support."""

    def __init__(self, hidden_size=2048, num_heads=16, head_dim=128,
                 seq_len=1024, sliding_window=1024, chunk_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.sliding_window = sliding_window
        self.chunk_size = chunk_size
        self.scale = 1.0 / math.sqrt(head_dim)

        # Projections
        self.q_proj = nn.Conv2d(hidden_size, num_heads * head_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(hidden_size, num_heads * head_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(hidden_size, num_heads * head_dim, 1, bias=False)
        self.o_proj = nn.Conv2d(num_heads * head_dim, hidden_size, 1, bias=False)

    def forward(self, x, k_cache, v_cache):
        """
        Sliding window attention with optional chunking.

        x: [B, hidden, 1, seq_len]
        k_cache: [B, heads*head_dim, 1, sliding_window]
        v_cache: [B, heads*head_dim, 1, sliding_window]
        """
        B = x.shape[0]

        Q = self.q_proj(x)
        K = k_cache
        V = v_cache

        # Reshape: [B, heads, seq/kv, head_dim]
        Q = Q.view(B, self.num_heads, self.head_dim, self.seq_len).permute(0, 1, 3, 2)
        K = K.view(B, self.num_heads, self.head_dim, self.sliding_window).permute(0, 1, 3, 2)
        V = V.view(B, self.num_heads, self.head_dim, self.sliding_window).permute(0, 1, 3, 2)

        if self.chunk_size is None or self.chunk_size == 0 or self.seq_len <= self.chunk_size:
            output = self._sdpa(Q, K, V)
        else:
            output = self._sdpa_chunked(Q, K, V)

        output = output.permute(0, 1, 3, 2).reshape(B, -1, 1, self.seq_len)
        output = self.o_proj(output)

        return output

    def _sdpa(self, Q, K, V):
        """Standard SDPA with sliding window KV."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, V)

    def _sdpa_chunked(self, Q, K, V):
        """Chunked Q with sliding window KV."""
        outputs = []
        for i in range(0, self.seq_len, self.chunk_size):
            end = min(i + self.chunk_size, self.seq_len)
            q_chunk = Q[:, :, i:end, :]

            # For sliding window, K and V are already windowed
            scores = torch.matmul(q_chunk, K.transpose(-2, -1)) * self.scale
            probs = F.softmax(scores, dim=-1)
            chunk_out = torch.matmul(probs, V)
            outputs.append(chunk_out)

        return torch.cat(outputs, dim=2)


def convert_to_coreml(model, seq_len, kv_len, hidden_size, output_path):
    """Convert SDPA block to CoreML."""
    model.eval()

    # Sample inputs in FP16
    x = torch.randn(1, hidden_size, 1, seq_len, dtype=torch.float16)
    k_cache = torch.randn(1, hidden_size, 1, kv_len, dtype=torch.float16)
    v_cache = torch.randn(1, hidden_size, 1, kv_len, dtype=torch.float16)

    # Trace
    with torch.no_grad():
        traced = torch.jit.trace(model.half(), (x, k_cache, v_cache))

    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
            ct.TensorType(name="k_cache", shape=k_cache.shape, dtype=np.float16),
            ct.TensorType(name="v_cache", shape=v_cache.shape, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS16,
    )

    mlmodel.save(output_path)
    print(f"  Saved: {output_path}")
    return output_path


def benchmark_model(model_path, iterations=50, warmup=10):
    """Benchmark a CoreML model on ANE."""
    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)

    # Get input shapes from spec
    spec = model.get_spec()
    inputs = {}
    for inp in spec.description.input:
        shape = list(inp.type.multiArrayType.shape)
        inputs[inp.name] = np.random.randn(*shape).astype(np.float16)

    # Warmup
    for _ in range(warmup):
        model.predict(inputs)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.time()
        model.predict(inputs)
        times.append((time.time() - start) * 1000)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SDPA chunking on ANE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sliding window 1024
  python benchmark_sdpa_chunking.py --seq-len 1024 --kv-len 1024 --chunk-sizes 0 512 256

  # Global attention 4096
  python benchmark_sdpa_chunking.py --seq-len 1024 --kv-len 4096 --chunk-sizes 0 512 256 128

  # Gemma3 4B prefill (batch=64)
  python benchmark_sdpa_chunking.py --seq-len 64 --kv-len 1024 --hidden-size 3072 --num-heads 24
        """
    )
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Query sequence length (default: 1024)")
    parser.add_argument("--kv-len", type=int, default=1024,
                        help="KV cache length / sliding window (default: 1024)")
    parser.add_argument("--hidden-size", type=int, default=2048,
                        help="Hidden size (default: 2048)")
    parser.add_argument("--num-heads", type=int, default=16,
                        help="Number of attention heads (default: 16)")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Head dimension (default: 128)")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[0, 512, 256, 128],
                        help="Chunk sizes to test, 0=no chunking (default: 0 512 256 128)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Benchmark iterations (default: 50)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save models (default: temp dir)")
    parser.add_argument("--keep-models", action="store_true",
                        help="Keep converted models after benchmark")
    parser.add_argument("--sliding-window", action="store_true",
                        help="Use sliding window attention model")
    parser.add_argument("--mode", type=str, default="chunk-q", choices=["chunk-q", "chunk-kv"],
                        help="Chunking mode: chunk-q (prefill) or chunk-kv (decode with online softmax)")
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        cleanup = False
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="sdpa_bench_"))
        cleanup = not args.keep_models

    print(f"\n{'='*70}")
    print(f"SDPA Chunking Benchmark")
    print(f"{'='*70}")
    print(f"Config:")
    print(f"  Seq length:    {args.seq_len}")
    print(f"  KV length:     {args.kv_len}")
    print(f"  Hidden size:   {args.hidden_size}")
    print(f"  Num heads:     {args.num_heads}")
    print(f"  Head dim:      {args.head_dim}")
    print(f"  Sliding window: {args.sliding_window}")
    print(f"  Mode:          {args.mode}")
    print(f"  Chunk sizes:   {args.chunk_sizes}")
    print(f"  Output dir:    {output_dir}")
    print(f"{'='*70}")

    # Memory estimate
    mem_full = args.seq_len * args.kv_len * args.num_heads * 2  # FP16
    print(f"\nAttention score memory (full): {mem_full / 1024 / 1024:.2f} MB")

    results = []

    for chunk_size in args.chunk_sizes:
        chunk_name = "none" if chunk_size == 0 else str(chunk_size)
        name = f"sdpa_seq{args.seq_len}_kv{args.kv_len}_chunk{chunk_name}"
        print(f"\n--- Testing chunk_size={chunk_name} ---")

        # Memory estimate for this chunk
        effective_chunk = args.seq_len if chunk_size == 0 else min(chunk_size, args.seq_len)
        mem_chunk = effective_chunk * args.kv_len * args.num_heads * 2
        print(f"  Attention score memory: {mem_chunk / 1024 / 1024:.2f} MB")

        # Create model based on mode
        if args.mode == "chunk-kv":
            # Chunked K/V for single-token decode
            model = SDPAChunkedKV(
                hidden_size=args.hidden_size,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                seq_len=args.seq_len,  # Should be 1 for decode
                kv_len=args.kv_len,
                chunk_size=chunk_size if chunk_size > 0 else None,
            )
        elif args.sliding_window:
            model = SDPASlidingWindow(
                hidden_size=args.hidden_size,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                seq_len=args.seq_len,
                sliding_window=args.kv_len,
                chunk_size=chunk_size if chunk_size > 0 else None,
            )
        else:
            model = SDPABlock(
                hidden_size=args.hidden_size,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                seq_len=args.seq_len,
                kv_len=args.kv_len,
                chunk_size=chunk_size if chunk_size > 0 else None,
            )

        # Convert to CoreML
        output_path = output_dir / f"{name}.mlpackage"
        try:
            convert_to_coreml(
                model,
                args.seq_len,
                args.kv_len,
                args.hidden_size,
                str(output_path)
            )
        except Exception as e:
            print(f"  Conversion failed: {e}")
            continue

        # Benchmark
        try:
            result = benchmark_model(str(output_path), args.iterations, args.warmup)
            result['chunk_size'] = chunk_size
            result['name'] = name
            result['mem_mb'] = mem_chunk / 1024 / 1024
            results.append(result)

            print(f"  Mean: {result['mean_ms']:.2f}ms, "
                  f"P50: {result['p50_ms']:.2f}ms, "
                  f"P95: {result['p95_ms']:.2f}ms")
        except Exception as e:
            print(f"  Benchmark failed: {e}")
            continue

    # Summary table
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Chunk':<10} {'Mem (MB)':<10} {'Mean':<10} {'P50':<10} {'P95':<10} {'Speedup':<10}")
        print("-" * 80)

        baseline = results[0]['mean_ms']
        for r in results:
            speedup = baseline / r['mean_ms']
            chunk_str = "None" if r['chunk_size'] == 0 else str(r['chunk_size'])
            print(f"{chunk_str:<10} {r['mem_mb']:<10.2f} {r['mean_ms']:<10.2f} "
                  f"{r['p50_ms']:<10.2f} {r['p95_ms']:<10.2f} {speedup:<10.2f}x")

        # Find best
        best = min(results, key=lambda x: x['mean_ms'])
        print(f"\nBest: chunk_size={best['chunk_size'] or 'None'} "
              f"({best['mean_ms']:.2f}ms, {baseline/best['mean_ms']:.2f}x vs baseline)")

    # Cleanup
    if cleanup:
        print(f"\nCleaning up temp directory: {output_dir}")
        shutil.rmtree(output_dir)
    else:
        print(f"\nModels saved in: {output_dir}")


if __name__ == "__main__":
    main()
