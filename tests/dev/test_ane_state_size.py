#!/usr/bin/env python3
"""
Test ANE compatibility with different state sizes.
Creates minimal CoreML models with state operations to identify ANE limits.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from coremltools.converters.mil import Builder as mb

OUTPUT_DIR = '/Volumes/Models/ANE/test_state_size'


class SimpleStateModel(nn.Module):
    """
    Simple model that mimics KV cache state operations:
    - Read from state
    - Slice state
    - Update state
    """

    def __init__(self, state_shape, batch_size=1):
        super().__init__()
        self.state_shape = state_shape
        self.batch_size = batch_size
        self.hidden_dim = state_shape[-1]

        # Simple projection to simulate attention output
        self.proj = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, hidden_states, position, kv_cache):
        """
        hidden_states: [1, batch, hidden_dim]
        position: [1]
        kv_cache: [layers, heads, seq_len, kv_dim]
        """
        # Read from cache (slice)
        # This mimics: cache[:, :, :position, :]
        pos = position[0].item() if position.numel() > 0 else 0

        # Slice the cache up to current position
        cache_slice = kv_cache[:, :, :pos+self.batch_size, :]

        # Simple computation on cache
        output = cache_slice.mean(dim=-1, keepdim=True)

        # Update cache (slice_update)
        # This would update kv_cache[:, :, pos:pos+batch, :] = new_values

        return output, kv_cache


class MinimalPrefillModel(nn.Module):
    """Minimal model to reproduce prefill state issues."""

    def __init__(self, num_layers, num_heads, context_len, kv_dim, batch_size):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_len = context_len
        self.kv_dim = kv_dim
        self.batch_size = batch_size

        # Simple layer
        self.norm = nn.LayerNorm(kv_dim)

    def forward(self, hidden_states, position):
        """
        hidden_states: [1, batch, kv_dim]
        position: [1]

        Returns: [1, batch, kv_dim]
        """
        # Normalize
        out = self.norm(hidden_states)
        return out


def create_and_test_model(name, num_layers, num_heads, context_len, kv_dim, batch_size):
    """Create a model, convert to CoreML, compile, and test loading."""

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  Layers: {num_layers}, Heads: {num_heads}")
    print(f"  Context: {context_len}, KV dim: {kv_dim}")
    print(f"  Batch: {batch_size}")

    state_shape = (num_layers, num_heads, context_len, kv_dim)
    state_size_mb = np.prod(state_shape) * 2 / (1024 * 1024)
    print(f"  State shape: {state_shape}")
    print(f"  State size: {state_size_mb:.2f} MB")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Create PyTorch model
        model = MinimalPrefillModel(num_layers, num_heads, context_len, kv_dim, batch_size)
        model.eval()

        # Sample inputs
        hidden_states = torch.randn(1, batch_size, kv_dim, dtype=torch.float16)
        position = torch.tensor([0], dtype=torch.int32)

        # Trace
        traced = torch.jit.trace(model, (hidden_states, position))

        # Convert to CoreML with state
        state_shape_ct = ct.StateType(
            wrapped_type=ct.TensorType(shape=state_shape, dtype=np.float16),
            name="kv_cache"
        )

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(shape=(1, batch_size, kv_dim), dtype=np.float16, name="hidden_states"),
                ct.TensorType(shape=(1,), dtype=np.int32, name="position"),
            ],
            states=[state_shape_ct],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS18,
        )

        # Save
        pkg_path = os.path.join(OUTPUT_DIR, f"test_{name}.mlpackage")
        mlmodel.save(pkg_path)
        print(f"  Saved: {pkg_path}")

        # Compile
        import subprocess
        compiled_path = pkg_path.replace('.mlpackage', '.mlmodelc')
        cmd = ['xcrun', 'coremlcompiler', 'compile', pkg_path, OUTPUT_DIR, '--add-mlprogram-if-eligible', 'force']
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  Compiled: SUCCESS")

            # Test CPU_AND_NE
            try:
                model_ne = ct.models.CompiledMLModel(compiled_path, ct.ComputeUnit.CPU_AND_NE)
                print(f"  CPU_AND_NE load: SUCCESS")
                return (name, state_shape, batch_size, "CPU_AND_NE", "SUCCESS")
            except Exception as e:
                print(f"  CPU_AND_NE load: FAILED - {str(e)[:60]}")
                return (name, state_shape, batch_size, "CPU_AND_NE", "FAILED")
        else:
            print(f"  Compile: FAILED - {result.stderr[:80]}")
            return (name, state_shape, batch_size, "compile", "FAILED")

    except Exception as e:
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return (name, state_shape if 'state_shape' in dir() else None, batch_size, "convert", "ERROR")


def test_without_state():
    """Test simple models WITHOUT state to see if the issue is state-specific."""

    print(f"\n{'='*60}")
    print("Testing simple models WITHOUT state")
    print(f"{'='*60}")

    configs = [
        # (name, hidden_dim, batch_size)
        ("simple_infer", 2560, 1),
        ("simple_prefill_64", 2560, 64),
        ("simple_prefill_128", 2560, 128),
    ]

    results = []

    for name, hidden_dim, batch_size in configs:
        print(f"\n--- {name} ---")

        model = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        model.eval()

        x = torch.randn(1, batch_size, hidden_dim, dtype=torch.float16)

        try:
            traced = torch.jit.trace(model, x)

            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(shape=(1, batch_size, hidden_dim), dtype=np.float16, name="x")],
                compute_units=ct.ComputeUnit.ALL,
                minimum_deployment_target=ct.target.iOS18,
            )

            pkg_path = os.path.join(OUTPUT_DIR, f"test_{name}.mlpackage")
            mlmodel.save(pkg_path)

            import subprocess
            compiled_path = pkg_path.replace('.mlpackage', '.mlmodelc')
            cmd = ['xcrun', 'coremlcompiler', 'compile', pkg_path, OUTPUT_DIR]
            subprocess.run(cmd, capture_output=True, text=True)

            # Test loading
            model_ne = ct.models.CompiledMLModel(compiled_path, ct.ComputeUnit.CPU_AND_NE)
            print(f"  CPU_AND_NE: SUCCESS")
            results.append((name, "SUCCESS"))

        except Exception as e:
            print(f"  Error: {str(e)[:80]}")
            results.append((name, "FAILED"))

    return results


def test_with_state():
    """Test models WITH state operations."""

    print(f"\n{'='*60}")
    print("Testing models WITH state (KV cache)")
    print(f"{'='*60}")

    # Gemma3 4B sizes:
    # Local cache per chunk (2 chunks): [58/2, 4, 1024, 256] = [29, 4, 1024, 256]
    # Global cache per chunk (2 chunks): [10/2, 4, 4096, 256] = [5, 4, 4096, 256]

    configs = [
        # Test with 2 chunks (current config)
        ("2chunk_local_infer", 29, 4, 1024, 256, 1),
        ("2chunk_local_prefill", 29, 4, 1024, 256, 64),

        # Test with 4 chunks (proposed fix)
        ("4chunk_local_infer", 15, 4, 1024, 256, 1),
        ("4chunk_local_prefill", 15, 4, 1024, 256, 64),

        # Test with 8 chunks
        ("8chunk_local_infer", 8, 4, 1024, 256, 1),
        ("8chunk_local_prefill", 8, 4, 1024, 256, 64),
    ]

    results = []
    for name, num_layers, num_heads, context_len, kv_dim, batch_size in configs:
        result = create_and_test_model(name, num_layers, num_heads, context_len, kv_dim, batch_size)
        results.append(result)

    return results


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # First test without state
    results_no_state = test_without_state()

    # Then test with state
    results_with_state = test_with_state()

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    print("\nWithout state:")
    for name, status in results_no_state:
        print(f"  {name:25s} -> {status}")

    print("\nWith state:")
    for r in results_with_state:
        if r:
            name, shape, batch, cu, status = r
            if shape:
                size_mb = np.prod(shape) * 2 / (1024 * 1024)
                print(f"  {name:25s} shape={shape} batch={batch:2d} size={size_mb:5.1f}MB -> {status}")
            else:
                print(f"  {name:25s} -> {status}")
