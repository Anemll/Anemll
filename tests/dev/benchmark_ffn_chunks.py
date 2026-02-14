#!/usr/bin/env python3
"""
Benchmark FFN chunk inference timing on ANE.

Compares actual inference time between different models/quantizations.
"""

import argparse
import time
import numpy as np
import coremltools as ct
from pathlib import Path


def get_model_inputs(model_path: str):
    """Load model and get input specs."""
    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    spec = model.get_spec()

    inputs = {}
    input_info = []

    for input_desc in spec.description.input:
        name = input_desc.name

        if input_desc.type.HasField('multiArrayType'):
            shape = list(input_desc.type.multiArrayType.shape)
            # Replace -1 with reasonable defaults
            shape = [s if s > 0 else 1 for s in shape]

            dtype_map = {
                65568: np.float16,
                65600: np.float32,
                131104: np.int32,
            }
            dtype = dtype_map.get(
                input_desc.type.multiArrayType.dataType,
                np.float32
            )
            inputs[name] = np.random.randn(*shape).astype(dtype)
            input_info.append((name, shape, dtype))

    return model, inputs, input_info


def benchmark_model(model_path: str, iterations: int = 50, warmup: int = 10, function_name: str = None):
    """Benchmark a single model."""
    print(f"\n{'='*70}")
    print(f"Model: {Path(model_path).name}")
    print(f"{'='*70}")

    # Load model
    print(f"Loading model...")
    load_start = time.time()
    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    load_time = (time.time() - load_start) * 1000
    print(f"Load time: {load_time:.1f}ms")

    # Get spec and list functions
    spec = model.get_spec()

    # Check for ML Program functions
    if spec.WhichOneof('Type') == 'mlProgram':
        functions = list(spec.mlProgram.functions.keys())
        print(f"Functions: {functions}")

        if function_name is None:
            # Default to 'infer' if available
            function_name = 'infer' if 'infer' in functions else functions[0]
        print(f"Testing function: {function_name}")

    # Get inputs for the function
    inputs = {}
    print("\nInputs:")
    for input_desc in spec.description.input:
        name = input_desc.name
        if input_desc.type.HasField('multiArrayType'):
            shape = list(input_desc.type.multiArrayType.shape)
            shape = [max(1, s) for s in shape]  # Replace -1 with 1

            dtype_map = {
                65568: np.float16,
                65600: np.float32,
                131104: np.int32,
            }
            dtype = dtype_map.get(
                input_desc.type.multiArrayType.dataType,
                np.float32
            )

            # Create appropriate input
            if 'position' in name.lower() or 'pos' in name.lower():
                inputs[name] = np.array([0], dtype=np.int32)
            elif 'mask' in name.lower():
                inputs[name] = np.zeros(shape, dtype=dtype)
            elif 'ids' in name.lower():
                inputs[name] = np.ones(shape, dtype=np.int32)
            else:
                inputs[name] = np.random.randn(*shape).astype(dtype)

            print(f"  {name}: {shape} ({dtype.__name__})")

    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for i in range(warmup):
        try:
            if function_name:
                model.predict(inputs, function_name=function_name)
            else:
                model.predict(inputs)
        except Exception as e:
            print(f"Warmup failed: {e}")
            return None

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        start = time.time()
        if function_name:
            model.predict(inputs, function_name=function_name)
        else:
            model.predict(inputs)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{iterations}: {elapsed:.2f}ms (avg: {np.mean(times):.2f}ms)")

    # Results
    results = {
        'model': Path(model_path).name,
        'function': function_name,
        'load_time_ms': load_time,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
    }

    print(f"\n--- Results ---")
    print(f"  Mean:   {results['mean_ms']:.2f}ms")
    print(f"  Std:    {results['std_ms']:.2f}ms")
    print(f"  Min:    {results['min_ms']:.2f}ms")
    print(f"  Max:    {results['max_ms']:.2f}ms")
    print(f"  P50:    {results['p50_ms']:.2f}ms")
    print(f"  P95:    {results['p95_ms']:.2f}ms")
    print(f"  P99:    {results['p99_ms']:.2f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark FFN chunks")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths to benchmark")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--function", type=str, default=None, help="Function name (infer, prefill, etc)")
    args = parser.parse_args()

    all_results = []

    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            continue

        result = benchmark_model(
            model_path,
            iterations=args.iterations,
            warmup=args.warmup,
            function_name=args.function
        )
        if result:
            all_results.append(result)

    # Summary comparison
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Model':<50} {'Mean':>10} {'P50':>10} {'P95':>10}")
        print("-"*80)

        for r in all_results:
            print(f"{r['model']:<50} {r['mean_ms']:>10.2f} {r['p50_ms']:>10.2f} {r['p95_ms']:>10.2f}")

        # Speedup calculation
        if len(all_results) == 2:
            speedup = all_results[1]['mean_ms'] / all_results[0]['mean_ms']
            faster = "first" if speedup > 1 else "second"
            ratio = max(speedup, 1/speedup)
            print(f"\n{faster.capitalize()} model is {ratio:.2f}x faster")


if __name__ == "__main__":
    main()
