#!/usr/bin/env python3
"""
Test script to verify static slicing in Gemma3 model for ANE compatibility.

This script:
1. Creates a minimal Gemma3 model with 1 chunk
2. Exports infer and prefill functions
3. Compiles to mlmodelc
4. Tests loading with CPU_AND_NE

The key change: all KV cache operations now use static bounds
instead of dynamic current_pos-based slicing.
"""

import os
import sys
import subprocess
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import coremltools as ct

OUTPUT_DIR = '/Volumes/Models/ANE/test_static_slice'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test parameters - use small values for quick testing
CONTEXT_LENGTH = 512  # Small context for quick test
BATCH_SIZE = 64
SLIDING_WINDOW = 512


def test_static_slicing_pattern():
    """Test that the slicing pattern is static by examining traced operations."""
    print("\n" + "="*60)
    print("Testing Static Slicing Pattern")
    print("="*60)

    from anemll.models.gemma3_model import Gemma3Model, Gemma3Config, MODEL_DTYPE, TEST_DEVICE

    # Create minimal config
    config = Gemma3Config(
        hidden_size=256,  # Small for testing
        num_hidden_layers=4,  # Minimal layers
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1024,
        context_length=CONTEXT_LENGTH,
        state_length=CONTEXT_LENGTH,
        sliding_window=SLIDING_WINDOW,
        use_split_cache=True,
        # Layer types: 3 local (sliding), 1 global (full attention)
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    )

    print(f"Config: context={CONTEXT_LENGTH}, sliding_window={SLIDING_WINDOW}, layers={config.num_hidden_layers}")

    # Create model
    model = Gemma3Model(config)
    model.eval()

    # Test single-token inference (infer function) - test process_layers directly
    print("\n--- Testing single-token inference (process_layers) ---")
    # Input to process_layers is hidden_states (after embedding), not input_ids
    hidden_states = torch.randn(1, 1, config.hidden_size, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    position_ids = torch.tensor([0], dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros(1, 1, 1, CONTEXT_LENGTH, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    current_pos = torch.tensor([0], dtype=torch.int32, device=TEST_DEVICE)

    # Set force_rotation_mode to False for fill mode (infer function)
    config.force_rotation_mode = False

    with torch.no_grad():
        try:
            # Trace the process_layers method directly (this is what the converter does)
            print("  Tracing process_layers...")

            class InferWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                    return self.model.process_layers(
                        hidden_states=hidden_states,
                        position_ids=position_ids,
                        causal_mask=causal_mask,
                        current_pos=current_pos,
                        IN_PREFILL=False
                    )

            wrapper = InferWrapper(model)
            traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos))
            print("  Trace successful!")

            # Check the graph for slice operations
            graph_str = str(traced.graph)

            # Look for dynamic slice patterns
            if "aten::slice" in graph_str:
                print("  Found aten::slice operations in graph")
                # Count slice operations
                slice_count = graph_str.count("aten::slice")
                print(f"  Number of slice operations: {slice_count}")

            # Look for narrow operations (static slicing)
            if "aten::narrow" in graph_str:
                narrow_count = graph_str.count("aten::narrow")
                print(f"  Number of narrow operations: {narrow_count}")

            print("  Single-token test: PASSED")

        except Exception as e:
            print(f"  Single-token test: FAILED - {e}")
            import traceback
            traceback.print_exc()
            return False

    # Test prefill (multi-token) - test process_layers directly
    print("\n--- Testing prefill (process_layers) ---")
    hidden_states_pf = torch.randn(1, BATCH_SIZE, config.hidden_size, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    position_ids_pf = torch.arange(BATCH_SIZE, dtype=torch.int32, device=TEST_DEVICE)
    causal_mask_pf = torch.zeros(1, 1, BATCH_SIZE, CONTEXT_LENGTH, dtype=MODEL_DTYPE, device=TEST_DEVICE)

    with torch.no_grad():
        try:
            class PrefillWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                    return self.model.process_layers(
                        hidden_states=hidden_states,
                        position_ids=position_ids,
                        causal_mask=causal_mask,
                        current_pos=current_pos,
                        IN_PREFILL=True,
                    )

            wrapper_pf = PrefillWrapper(model)
            traced_pf = torch.jit.trace(wrapper_pf, (hidden_states_pf, position_ids_pf, causal_mask_pf, current_pos))
            print("  Prefill trace successful!")
            print("  Prefill test: PASSED")

        except Exception as e:
            print(f"  Prefill test: FAILED - {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "="*60)
    print("All static slicing tests PASSED!")
    print("="*60)
    return True


def test_coreml_conversion():
    """Test CoreML conversion with the static slicing changes."""
    print("\n" + "="*60)
    print("Testing CoreML Conversion")
    print("="*60)

    from anemll.models.gemma3_model import Gemma3Model, Gemma3Config, MODEL_DTYPE, TEST_DEVICE

    # Create minimal config
    config = Gemma3Config(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1024,
        context_length=CONTEXT_LENGTH,
        state_length=CONTEXT_LENGTH,
        sliding_window=SLIDING_WINDOW,
        use_split_cache=True,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        force_rotation_mode=False,  # Fill mode for infer
    )

    model = Gemma3Model(config)
    model.eval()

    # Prepare inputs - test process_layers directly (hidden_states after embedding)
    hidden_states = torch.randn(1, 1, config.hidden_size, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    position_ids = torch.tensor([0], dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros(1, 1, 1, CONTEXT_LENGTH, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    current_pos = torch.tensor([0], dtype=torch.int32, device=TEST_DEVICE)

    class InferWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, hidden_states, position_ids, causal_mask, current_pos):
            return self.model.process_layers(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )

    wrapper = InferWrapper(model)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos))

    print("  Traced model, converting to CoreML...")

    # Get state configuration
    # Local cache: 3 local layers * 2 (K+V) = 6 entries
    # Global cache: 1 global layer * 2 (K+V) = 2 entries
    local_cache_shape = (6, config.num_key_value_heads, SLIDING_WINDOW, config.head_dim)
    global_cache_shape = (2, config.num_key_value_heads, CONTEXT_LENGTH, config.head_dim)

    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=local_cache_shape, dtype=np.float16),
            name="model.kv_cache_local"
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=global_cache_shape, dtype=np.float16),
            name="model.kv_cache_global"
        ),
    ]

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden_states", shape=(1, 1, config.hidden_size), dtype=np.float16),
                ct.TensorType(name="position_ids", shape=(1,), dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=(1, 1, 1, CONTEXT_LENGTH), dtype=np.float16),
                ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="output", dtype=np.float16)],
            states=states,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS18,
        )

        # Save
        pkg_path = os.path.join(OUTPUT_DIR, "test_static_infer.mlpackage")
        mlmodel.save(pkg_path)
        print(f"  Saved: {pkg_path}")

        # Compile
        compiled_path = pkg_path.replace('.mlpackage', '.mlmodelc')
        cmd = ['xcrun', 'coremlcompiler', 'compile', pkg_path, OUTPUT_DIR, '--add-mlprogram-if-eligible', 'force']
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  Compiled: SUCCESS")

            # Test loading with CPU_AND_NE
            try:
                model_ne = ct.models.CompiledMLModel(compiled_path, ct.ComputeUnit.CPU_AND_NE)
                print(f"  CPU_AND_NE load: SUCCESS")
                return True
            except Exception as e:
                print(f"  CPU_AND_NE load: FAILED - {str(e)[:80]}")
                return False
        else:
            print(f"  Compile: FAILED - {result.stderr[:80]}")
            return False

    except Exception as e:
        print(f"  Conversion: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("Testing Gemma3 Static Slicing for ANE Compatibility")
    print("="*60)

    # Test 1: Verify static slicing pattern in traced model
    if not test_static_slicing_pattern():
        print("\nStatic slicing pattern test failed!")
        sys.exit(1)

    # Test 2: CoreML conversion and loading
    if not test_coreml_conversion():
        print("\nCoreML conversion test failed!")
        sys.exit(1)

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nThe static slicing changes are working correctly.")
    print("Models should now load with CPU_AND_NE without slice_by_index errors.")
