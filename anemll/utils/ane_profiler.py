#!/usr/bin/env python3
"""
ANE Profiler - Automated CoreML/ANE Debugging Without Xcode

Uses CoreMLTools 9.0+ MLComputePlan API to:
1. Analyze which ops run on ANE vs CPU vs GPU
2. Identify ops that fall back from ANE
3. Profile estimated costs per operation
4. Test predictions across compute units
5. Generate compatibility reports

Usage:
    python ane_profiler.py --model model.mlpackage --report
    python ane_profiler.py --model model.mlpackage --test-prediction
    python ane_profiler.py --model model.mlpackage --compare-compute-units
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class OpProfile:
    """Profile for a single operation."""
    name: str
    op_type: str
    device: str  # "ane", "gpu", "cpu", "unknown"
    estimated_cost: Optional[float] = None
    supported_on_ane: bool = False
    fallback_reason: Optional[str] = None


@dataclass
class ANEReport:
    """Full ANE compatibility report."""
    model_path: str
    total_ops: int = 0
    ane_ops: int = 0
    gpu_ops: int = 0
    cpu_ops: int = 0
    unknown_ops: int = 0

    ane_percentage: float = 0.0

    # Ops by type
    ops_by_type: Dict[str, List[str]] = field(default_factory=dict)
    ops_by_device: Dict[str, List[str]] = field(default_factory=dict)

    # Fallback analysis
    fallback_ops: List[OpProfile] = field(default_factory=list)

    # Timing
    load_time_ms: float = 0.0
    prediction_time_ms: float = 0.0

    # Per compute unit results
    compute_unit_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ANEProfiler:
    """Profile CoreML models for ANE compatibility."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.report = ANEReport(model_path=str(model_path))
        self._model = None
        self._compute_plan = None
        self._model_structure = None

    def _load_model(self):
        """Load CoreML model."""
        import coremltools as ct

        if self._model is None:
            start = time.time()
            self._model = ct.models.MLModel(str(self.model_path))
            self.report.load_time_ms = (time.time() - start) * 1000

        return self._model

    def _load_compute_plan(self):
        """Load MLComputePlan for the model."""
        from coremltools.models.compute_plan import MLComputePlan

        if self._compute_plan is None:
            # Need compiled model path
            compiled_path = self.model_path

            if self.model_path.suffix == '.mlpackage':
                # Use a fresh temp directory for each compilation to avoid permission issues
                # coremlcompiler compile is read-only on the source, writes to output dir
                import subprocess
                import tempfile
                import hashlib

                # Create unique temp dir based on model path hash
                model_hash = hashlib.md5(str(self.model_path).encode()).hexdigest()[:8]
                tmpdir = Path(tempfile.gettempdir()) / f'ane_profiler_{model_hash}'

                # Clean up any existing directory (may have stale permissions)
                if tmpdir.exists():
                    import shutil
                    try:
                        shutil.rmtree(tmpdir)
                    except Exception:
                        pass  # Ignore cleanup errors

                tmpdir.mkdir(exist_ok=True)

                # Compile (read-only on source, writes to tmpdir)
                result = subprocess.run(
                    ['xcrun', 'coremlcompiler', 'compile',
                     str(self.model_path), str(tmpdir)],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    print(f"Warning: Compilation failed: {result.stderr}")
                    return None

                # Find the compiled model
                compiled_models = list(tmpdir.glob('*.mlmodelc'))
                if not compiled_models:
                    print(f"Warning: No compiled model found in {tmpdir}")
                    return None
                compiled_path = compiled_models[0]

            try:
                self._compute_plan = MLComputePlan.load_from_path(str(compiled_path))
            except Exception as e:
                print(f"Warning: Could not load compute plan: {e}")
                return None

        return self._compute_plan

    def _load_model_structure(self):
        """Load model structure for op enumeration."""
        # Model structure comes from compute plan
        compute_plan = self._load_compute_plan()
        if compute_plan is not None:
            self._model_structure = compute_plan.model_structure
        return self._model_structure

    def analyze_compute_plan(self, verbose: bool = True) -> ANEReport:
        """Analyze which ops run on which compute units."""
        import coremltools as ct

        if verbose:
            print("\n" + "=" * 70)
            print("ANE COMPUTE PLAN ANALYSIS")
            print("=" * 70)
            print(f"Model: {self.model_path}")

        # Load model structure
        model_structure = self._load_model_structure()
        if model_structure is None:
            print("Could not load model structure")
            return self.report

        # Try to load compute plan
        compute_plan = self._load_compute_plan()

        # Enumerate operations
        ops_by_device = defaultdict(list)
        ops_by_type = defaultdict(list)

        # Check if it's an ML Program
        program = model_structure.program
        if program is not None:
            if verbose:
                print("\nModel type: ML Program")

            # Iterate through functions
            for func_name, func in program.functions.items():
                if verbose:
                    print(f"\nFunction: {func_name}")

                # Iterate through operations
                for op in func.block.operations:
                    op_type = op.operator_name if hasattr(op, 'operator_name') else 'unknown'
                    op_outputs = op.outputs if hasattr(op, 'outputs') else []
                    op_name = op_outputs[0].name if op_outputs else op_type

                    self.report.total_ops += 1
                    ops_by_type[op_type].append(op_name)

                    # Get device usage if compute plan available
                    device = "none"  # Default for const/metadata ops
                    if compute_plan is not None:
                        try:
                            # API takes just the operation, not function name
                            usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(op)
                            if usage is not None:
                                # Parse device from usage repr
                                usage_str = repr(usage).lower()
                                if 'ane' in usage_str or 'neural' in usage_str:
                                    device = "ane"
                                    self.report.ane_ops += 1
                                elif 'gpu' in usage_str or 'mps' in usage_str:
                                    device = "gpu"
                                    self.report.gpu_ops += 1
                                elif 'cpu' in usage_str:
                                    device = "cpu"
                                    self.report.cpu_ops += 1
                                else:
                                    device = "unknown"
                                    self.report.unknown_ops += 1
                            else:
                                # None means no device (const ops, etc.)
                                pass
                        except Exception as e:
                            self.report.unknown_ops += 1
                    else:
                        self.report.unknown_ops += 1

                    ops_by_device[device].append(f"{op_type}:{op_name}")

        # Check neural network
        elif model_structure.neural_network is not None:
            if verbose:
                print("\nModel type: Neural Network")

            for layer in model_structure.neural_network.layers:
                layer_name = layer.name if hasattr(layer, 'name') else str(layer)
                layer_type = layer.type if hasattr(layer, 'type') else 'unknown'

                self.report.total_ops += 1
                ops_by_type[layer_type].append(layer_name)

                # Get device usage
                device = "unknown"
                if compute_plan is not None:
                    try:
                        usage = compute_plan.get_compute_device_usage_for_neuralnetwork_layer(layer)
                        if usage is not None:
                            usage_str = str(usage).lower()
                            if 'ane' in usage_str or 'neural' in usage_str:
                                device = "ane"
                                self.report.ane_ops += 1
                            elif 'gpu' in usage_str:
                                device = "gpu"
                                self.report.gpu_ops += 1
                            elif 'cpu' in usage_str:
                                device = "cpu"
                                self.report.cpu_ops += 1
                    except Exception:
                        self.report.unknown_ops += 1
                else:
                    self.report.unknown_ops += 1

                ops_by_device[device].append(f"{layer_type}:{layer_name}")

        # Calculate percentages (excluding const/none ops which are just data)
        executable_ops = self.report.ane_ops + self.report.gpu_ops + self.report.cpu_ops
        if executable_ops > 0:
            self.report.ane_percentage = (self.report.ane_ops / executable_ops) * 100

        self.report.ops_by_device = dict(ops_by_device)
        self.report.ops_by_type = dict(ops_by_type)

        # Count const/metadata ops
        const_ops = len(ops_by_device.get('none', []))

        if verbose:
            print(f"\nTotal operations: {self.report.total_ops}")
            print(f"  Const/metadata: {const_ops}")
            print(f"  Executable ops: {executable_ops}")
            print(f"    ANE: {self.report.ane_ops} ({self.report.ane_percentage:.1f}%)")
            print(f"    GPU: {self.report.gpu_ops}")
            print(f"    CPU: {self.report.cpu_ops}")
            print(f"  Unknown: {self.report.unknown_ops}")

            if ops_by_type:
                print("\nOperations by type:")
                for op_type, ops in sorted(ops_by_type.items(), key=lambda x: -len(x[1])):
                    print(f"  {op_type}: {len(ops)}")

            # Show GPU fallback ops
            gpu_ops = ops_by_device.get('gpu', [])
            if gpu_ops:
                print(f"\n--- GPU Fallback Operations ({len(gpu_ops)}) ---")
                for op in gpu_ops[:20]:
                    print(f"  - {op}")
                if len(gpu_ops) > 20:
                    print(f"  ... and {len(gpu_ops) - 20} more")

            # Show CPU ops if any
            cpu_ops = ops_by_device.get('cpu', [])
            if cpu_ops:
                print(f"\n--- CPU-Only Operations ({len(cpu_ops)}) ---")
                for op in cpu_ops[:20]:
                    print(f"  - {op}")
                if len(cpu_ops) > 20:
                    print(f"  ... and {len(cpu_ops) - 20} more")

        return self.report

    def test_prediction(
        self,
        compute_unit: str = "ALL",
        sample_inputs: Optional[Dict[str, np.ndarray]] = None,
        verbose: bool = True
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Test model prediction on specified compute unit.

        Args:
            compute_unit: "ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"
            sample_inputs: Optional input dict, will generate random if None
            verbose: Print details

        Returns:
            (success, time_ms, error_message)
        """
        import coremltools as ct

        compute_units = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        }

        if compute_unit not in compute_units:
            return False, 0, f"Invalid compute unit: {compute_unit}"

        if verbose:
            print(f"\n--- Testing prediction with {compute_unit} ---")

        try:
            # Load model with specified compute unit
            start = time.time()
            model = ct.models.MLModel(
                str(self.model_path),
                compute_units=compute_units[compute_unit]
            )
            load_time = (time.time() - start) * 1000

            if verbose:
                print(f"Load time: {load_time:.1f}ms")

            # Generate sample inputs if not provided
            if sample_inputs is None:
                sample_inputs = {}
                spec = model.get_spec()

                for input_desc in spec.description.input:
                    name = input_desc.name

                    # Handle different input types
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
                        sample_inputs[name] = np.random.randn(*shape).astype(dtype)

                    elif input_desc.type.HasField('imageType'):
                        # Skip image inputs for now
                        pass

            if verbose:
                print(f"Inputs: {list(sample_inputs.keys())}")

            # Run prediction
            start = time.time()
            outputs = model.predict(sample_inputs)
            pred_time = (time.time() - start) * 1000

            if verbose:
                print(f"Prediction time: {pred_time:.1f}ms")
                print(f"Outputs: {list(outputs.keys())}")

                # Check for NaN/Inf
                for name, value in outputs.items():
                    if isinstance(value, np.ndarray):
                        has_nan = np.isnan(value).any()
                        has_inf = np.isinf(value).any()
                        if has_nan or has_inf:
                            print(f"  WARNING: {name} has NaN={has_nan}, Inf={has_inf}")
                        else:
                            print(f"  {name}: range=[{value.min():.4f}, {value.max():.4f}]")

            return True, pred_time, None

        except Exception as e:
            error_msg = str(e)
            if verbose:
                print(f"FAILED: {error_msg}")
            return False, 0, error_msg

    def compare_compute_units(
        self,
        sample_inputs: Optional[Dict[str, np.ndarray]] = None,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Compare prediction across all compute units."""
        if verbose:
            print("\n" + "=" * 70)
            print("COMPUTE UNIT COMPARISON")
            print("=" * 70)

        results = {}

        for unit in ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]:
            success, time_ms, error = self.test_prediction(
                compute_unit=unit,
                sample_inputs=sample_inputs,
                verbose=verbose
            )
            results[unit] = {
                "success": success,
                "time_ms": time_ms,
                "error": error
            }

        self.report.compute_unit_results = results

        if verbose:
            print("\n--- Summary ---")
            for unit, result in results.items():
                status = "✓ PASS" if result["success"] else "✗ FAIL"
                time_str = f"{result['time_ms']:.1f}ms" if result["success"] else result["error"][:50]
                print(f"  {unit:<15} {status}  {time_str}")

        return results

    def estimate_costs(self, verbose: bool = True) -> Dict[str, Any]:
        """Estimate operation costs using compute plan."""
        compute_plan = self._load_compute_plan()
        if compute_plan is None:
            if verbose:
                print("Could not load compute plan for cost estimation")
            return {}

        model_structure = self._load_model_structure()
        if model_structure is None or model_structure.program is None:
            return {}

        costs = {
            'total_estimated_cost': 0.0,
            'ane_cost': 0.0,
            'gpu_cost': 0.0,
            'cpu_cost': 0.0,
            'ops_with_cost': [],
        }

        if verbose:
            print("\n" + "=" * 70)
            print("OPERATION COST ESTIMATION")
            print("=" * 70)

        program = model_structure.program
        for func_name, func in program.functions.items():
            for op in func.block.operations:
                try:
                    cost_info = compute_plan.get_estimated_cost_for_mlprogram_operation(op)
                    if cost_info is not None:
                        # Get weight from cost info
                        weight = getattr(cost_info, 'weight', 0.0)
                        if weight > 0:
                            op_name = op.outputs[0].name if op.outputs else op.operator_name
                            costs['total_estimated_cost'] += weight
                            costs['ops_with_cost'].append({
                                'name': op_name,
                                'type': op.operator_name,
                                'weight': weight
                            })
                except Exception:
                    pass

        # Sort by cost
        costs['ops_with_cost'].sort(key=lambda x: -x['weight'])

        if verbose:
            print(f"\nTotal estimated cost weight: {costs['total_estimated_cost']:.2f}")
            print("\nTop 10 most expensive operations:")
            for op_info in costs['ops_with_cost'][:10]:
                pct = (op_info['weight'] / costs['total_estimated_cost'] * 100) if costs['total_estimated_cost'] > 0 else 0
                print(f"  {op_info['type']:<20} {op_info['name']:<40} weight={op_info['weight']:.4f} ({pct:.1f}%)")

        return costs

    def benchmark(
        self,
        compute_unit: str = "CPU_AND_NE",
        iterations: int = 10,
        warmup: int = 3,
        sample_inputs: Optional[Dict[str, np.ndarray]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark model performance with actual timing in milliseconds.

        Args:
            compute_unit: Which compute unit to benchmark
            iterations: Number of timed iterations
            warmup: Number of warmup iterations (not timed)
            sample_inputs: Optional inputs, generates random if None
            verbose: Print details

        Returns:
            Dict with timing stats in milliseconds
        """
        import coremltools as ct

        compute_units = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        }

        if compute_unit not in compute_units:
            return {"error": f"Invalid compute unit: {compute_unit}"}

        if verbose:
            print("\n" + "=" * 70)
            print(f"BENCHMARK: {compute_unit}")
            print("=" * 70)

        results = {
            "compute_unit": compute_unit,
            "iterations": iterations,
            "warmup": warmup,
            "load_time_ms": 0.0,
            "times_ms": [],
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
            "error": None,
        }

        try:
            # Load model
            if verbose:
                print(f"Loading model with {compute_unit}...")
            start = time.time()
            model = ct.models.MLModel(
                str(self.model_path),
                compute_units=compute_units[compute_unit]
            )
            results["load_time_ms"] = (time.time() - start) * 1000
            if verbose:
                print(f"Load time: {results['load_time_ms']:.1f}ms")

            # Generate sample inputs if needed
            if sample_inputs is None:
                sample_inputs = {}
                spec = model.get_spec()
                for input_desc in spec.description.input:
                    name = input_desc.name
                    if input_desc.type.HasField('multiArrayType'):
                        shape = list(input_desc.type.multiArrayType.shape)
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
                        sample_inputs[name] = np.random.randn(*shape).astype(dtype)

            if verbose:
                print(f"Inputs: {list(sample_inputs.keys())}")

            # Warmup runs
            if verbose:
                print(f"Warming up ({warmup} iterations)...")
            for _ in range(warmup):
                try:
                    model.predict(sample_inputs)
                except Exception as e:
                    results["error"] = str(e)
                    if verbose:
                        print(f"Warmup failed: {e}")
                    return results

            # Timed runs
            if verbose:
                print(f"Benchmarking ({iterations} iterations)...")
            times = []
            for i in range(iterations):
                start = time.time()
                model.predict(sample_inputs)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
                if verbose and (i + 1) % 5 == 0:
                    print(f"  Iteration {i + 1}/{iterations}: {elapsed:.2f}ms")

            results["times_ms"] = times
            results["mean_ms"] = np.mean(times)
            results["min_ms"] = np.min(times)
            results["max_ms"] = np.max(times)
            results["std_ms"] = np.std(times)

            if verbose:
                print(f"\n--- Results ---")
                print(f"  Mean:   {results['mean_ms']:.2f}ms")
                print(f"  Min:    {results['min_ms']:.2f}ms")
                print(f"  Max:    {results['max_ms']:.2f}ms")
                print(f"  Std:    {results['std_ms']:.2f}ms")
                print(f"  Load:   {results['load_time_ms']:.1f}ms")

        except Exception as e:
            results["error"] = str(e)
            if verbose:
                print(f"Benchmark failed: {e}")

        return results

    def benchmark_all_units(
        self,
        iterations: int = 10,
        warmup: int = 3,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark all compute units and compare."""
        if verbose:
            print("\n" + "=" * 70)
            print("BENCHMARK ALL COMPUTE UNITS")
            print("=" * 70)

        all_results = {}
        for unit in ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]:
            result = self.benchmark(
                compute_unit=unit,
                iterations=iterations,
                warmup=warmup,
                verbose=verbose
            )
            all_results[unit] = result

        # Summary comparison
        if verbose:
            print("\n" + "=" * 70)
            print("TIMING COMPARISON (milliseconds)")
            print("=" * 70)
            print(f"{'Compute Unit':<15} {'Mean':>10} {'Min':>10} {'Max':>10} {'Load':>10} {'Status'}")
            print("-" * 70)

            for unit, result in all_results.items():
                if result.get("error"):
                    print(f"{unit:<15} {'--':>10} {'--':>10} {'--':>10} {result['load_time_ms']:>10.1f} ✗ {result['error'][:20]}")
                else:
                    print(f"{unit:<15} {result['mean_ms']:>10.2f} {result['min_ms']:>10.2f} {result['max_ms']:>10.2f} {result['load_time_ms']:>10.1f} ✓")

            # Find fastest
            valid_results = {k: v for k, v in all_results.items() if not v.get("error")}
            if valid_results:
                fastest = min(valid_results.items(), key=lambda x: x[1]["mean_ms"])
                print(f"\nFastest: {fastest[0]} ({fastest[1]['mean_ms']:.2f}ms)")

        return all_results

    def identify_ane_blockers(self, verbose: bool = True) -> List[str]:
        """Identify operations that prevent ANE execution."""
        blockers = []

        # Known ANE-incompatible ops
        known_blockers = {
            # Dynamic ops
            'while_loop': 'Dynamic control flow',
            'cond': 'Dynamic control flow',
            'select': 'Dynamic control flow',
            # Unsupported data types
            'cast': 'May cast to unsupported dtype',
            # Shape ops with dynamic behavior
            'reshape': 'May have dynamic shape',
            'gather': 'May have dynamic indices',
            'scatter': 'May have dynamic indices',
            # Large tensors
            'conv': 'Channel count may exceed ANE limit',
        }

        if self.report.ops_by_type:
            for op_type, ops in self.report.ops_by_type.items():
                if op_type in known_blockers:
                    blockers.append(f"{op_type}: {known_blockers.get(op_type, 'Known ANE blocker')} ({len(ops)} instances)")

        # Check for CPU-only ops
        if self.report.ops_by_device.get('cpu'):
            cpu_ops = self.report.ops_by_device['cpu']
            if verbose:
                print(f"\nCPU-only operations ({len(cpu_ops)}):")
                for op in cpu_ops[:10]:
                    print(f"  - {op}")
                if len(cpu_ops) > 10:
                    print(f"  ... and {len(cpu_ops) - 10} more")

        return blockers

    def generate_report(self, output_path: Optional[str] = None, verbose: bool = True):
        """Generate full compatibility report."""
        if verbose:
            print("\n" + "=" * 70)
            print("ANE COMPATIBILITY REPORT")
            print("=" * 70)

        # Run all analyses
        self.analyze_compute_plan(verbose=verbose)
        self.compare_compute_units(verbose=verbose)
        blockers = self.identify_ane_blockers(verbose=verbose)

        # Determine overall status
        ane_works = self.report.compute_unit_results.get('CPU_AND_NE', {}).get('success', False)

        if verbose:
            print("\n" + "=" * 70)
            print("FINAL VERDICT")
            print("=" * 70)

            if ane_works and self.report.ane_percentage > 90:
                print("✓ Model is ANE-compatible (>90% ops on ANE)")
            elif ane_works:
                print(f"⚠ Model runs on ANE but only {self.report.ane_percentage:.1f}% ops accelerated")
            else:
                print("✗ Model fails on ANE")
                if blockers:
                    print("\nPotential blockers:")
                    for b in blockers:
                        print(f"  - {b}")

        # Save report
        if output_path:
            report_dict = asdict(self.report)
            # Convert non-serializable items
            report_dict['fallback_ops'] = [asdict(op) for op in self.report.fallback_ops]

            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)

            if verbose:
                print(f"\nReport saved to: {output_path}")

        return self.report


def main():
    parser = argparse.ArgumentParser(
        description='ANE Profiler - Automated CoreML/ANE Debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full report
    python ane_profiler.py --model model.mlpackage --report

    # Quick prediction test
    python ane_profiler.py --model model.mlpackage --test CPU_AND_NE

    # Compare all compute units
    python ane_profiler.py --model model.mlpackage --compare

    # Analyze compute plan
    python ane_profiler.py --model model.mlpackage --analyze
"""
    )

    parser.add_argument('--model', '-m', required=True, help='Path to .mlpackage or .mlmodelc')
    parser.add_argument('--report', '-r', action='store_true', help='Generate full report')
    parser.add_argument('--analyze', '-a', action='store_true', help='Analyze compute plan')
    parser.add_argument('--test', '-t', metavar='UNIT', help='Test prediction (ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE)')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare all compute units')
    parser.add_argument('--costs', action='store_true', help='Estimate operation costs')
    parser.add_argument('--benchmark', '-b', metavar='UNIT', help='Benchmark specific compute unit (CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, ALL)')
    parser.add_argument('--benchmark-all', '-B', action='store_true', help='Benchmark all compute units')
    parser.add_argument('--iterations', '-n', type=int, default=10, help='Benchmark iterations (default: 10)')
    parser.add_argument('--warmup', '-w', type=int, default=3, help='Warmup iterations (default: 3)')
    parser.add_argument('--output', '-o', help='Output JSON report path')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    profiler = ANEProfiler(args.model)
    verbose = not args.quiet

    if args.report:
        profiler.generate_report(output_path=args.output, verbose=verbose)
    elif args.analyze:
        profiler.analyze_compute_plan(verbose=verbose)
    elif args.costs:
        profiler.analyze_compute_plan(verbose=verbose)
        profiler.estimate_costs(verbose=verbose)
    elif args.test:
        success, time_ms, error = profiler.test_prediction(
            compute_unit=args.test, verbose=verbose
        )
        sys.exit(0 if success else 1)
    elif args.compare:
        profiler.compare_compute_units(verbose=verbose)
    elif args.benchmark:
        profiler.benchmark(
            compute_unit=args.benchmark,
            iterations=args.iterations,
            warmup=args.warmup,
            verbose=verbose
        )
    elif args.benchmark_all:
        profiler.benchmark_all_units(
            iterations=args.iterations,
            warmup=args.warmup,
            verbose=verbose
        )
    else:
        # Default: full report
        profiler.generate_report(output_path=args.output, verbose=verbose)


if __name__ == "__main__":
    main()
