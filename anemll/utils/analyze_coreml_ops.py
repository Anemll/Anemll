#!/usr/bin/env python3
"""
CoreML Model Operation Analyzer

This utility analyzes CoreML models using decoreml to provide a breakdown
of operation types, backend distribution, and runtime estimates.

Usage:
    python analyze_coreml_ops.py                    # Auto-find latest analytics.mil
    python analyze_coreml_ops.py --list             # List all available analytics files
    python analyze_coreml_ops.py --select 2         # Analyze the 2nd most recent file
    python analyze_coreml_ops.py --file analytics.mil
    python analyze_coreml_ops.py --perf model.mlperf  # Analyze Xcode performance report
"""

import re
import os
import glob
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import datetime


def find_all_analytics_files() -> List[Tuple[str, datetime.datetime]]:
    """Find all analytics.mil files in the CoreML cache, sorted by modification time."""
    search_path = os.path.expanduser(
        "~/Library/Caches/com.apple.dt.DTMLModelRunnerService/com.apple.e5rt.e5bundlecache/"
    )
    analytics_files = glob.glob(
        os.path.join(search_path, "**", "analytics.mil"), recursive=True
    )
    if not analytics_files:
        return []

    # Sort by modification time (newest first)
    files_with_time = []
    for f in analytics_files:
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        files_with_time.append((f, mtime))

    return sorted(files_with_time, key=lambda x: x[1], reverse=True)


def find_latest_analytics_file() -> str:
    """Find the most recently modified analytics.mil file in the CoreML cache."""
    files = find_all_analytics_files()
    if not files:
        raise FileNotFoundError(
            "No analytics.mil files found. Run a CoreML model first to generate analytics."
        )
    return files[0][0]


def list_analytics_files():
    """List all available analytics files with details."""
    files = find_all_analytics_files()
    if not files:
        print("No analytics.mil files found.")
        print("\nTo generate analytics.mil:")
        print("  1. Load and run your CoreML model in Xcode or via coremltools")
        print("  2. The analytics.mil will be created in the CoreML cache")
        return

    print(f"\nFound {len(files)} analytics.mil file(s):\n")
    print(f"{'#':>3}  {'Modified':20}  {'Path'}")
    print("-" * 80)

    for i, (path, mtime) in enumerate(files, 1):
        # Extract model name from path if possible
        parts = path.split('/')
        bundle_name = ""
        for part in parts:
            if part.endswith('.bundle'):
                bundle_name = part.replace('.bundle', '')
                break

        time_str = mtime.strftime('%Y-%m-%d %H:%M:%S')
        # Shorten path for display
        short_path = path
        if len(path) > 60:
            short_path = "..." + path[-57:]

        print(f"{i:3}  {time_str:20}  {short_path}")

    print("\nUse --select N to analyze a specific file (e.g., --select 1 for latest)")
    print("Use --file PATH to analyze a specific file by path")


def extract_operation(tensor_operation: str) -> Dict:
    """Extract operation details from a tensor operation string."""
    result = {
        'operation': 'unknown',
        'name': 'unknown',
        'selected_backend': 'unknown',
        'runtimes': {},
        'validation_messages': {},
        'tensor_shape': None,
        'tensor_dtype': None,
        'tensor_elements': 0
    }

    # Extract tensor output type and shape: tensor<dtype, [dim1, dim2, ...]>
    tensor_match = re.match(r'\s*tensor<(\w+),\s*\[([^\]]+)\]>', tensor_operation)
    if tensor_match:
        result['tensor_dtype'] = tensor_match.group(1)
        dims_str = tensor_match.group(2)
        try:
            dims = [int(d.strip()) for d in dims_str.split(',')]
            result['tensor_shape'] = dims
            result['tensor_elements'] = 1
            for d in dims:
                result['tensor_elements'] *= d
        except ValueError:
            pass

    # Extract operation type
    # First check for operations at start of line (e.g., write_state, read_state)
    start_op_match = re.match(r"(write_state|read_state)\(", tensor_operation)
    if start_op_match:
        result['operation'] = start_op_match.group(1)
    else:
        # Handles "tensor<...> name = operation(" format
        operation_match = re.search(r" = (\w+)\(", tensor_operation)
        if operation_match:
            result['operation'] = operation_match.group(1)

    # Extract name
    name_match = re.search(r'name = string\("(.+?)"\)', tensor_operation)
    if name_match:
        result['name'] = name_match.group(1)

    # Extract selected backend
    selected_backend_match = re.search(
        r'SelectedBackend = string\("(.+?)"\)', tensor_operation
    )
    if selected_backend_match:
        result['selected_backend'] = selected_backend_match.group(1)

    # Extract runtimes (handles scientific notation like 7.23e-06)
    runtimes_match = re.search(
        r"EstimatedRuntime = dict<string, fp64>\(\{\{(.+?)\}\}\)", tensor_operation
    )
    if runtimes_match:
        runtimes_str = runtimes_match.group(1)
        # Match scientific notation: 1.23e-06, 7.5E+03, etc.
        runtimes_pairs = re.findall(r'"(.+?)", ([0-9.eE\-+]+)', runtimes_str)
        result['runtimes'] = {backend: float(runtime) for backend, runtime in runtimes_pairs}

    # Extract validation messages
    validation_match = re.search(
        r"ValidationMessage = dict<string, string>\(\{\{(.+?)\}\}\)", tensor_operation
    )
    if validation_match:
        validation_str = validation_match.group(1)
        validation_pairs = re.findall(r'"(.+?)", "(.+?)"', validation_str)
        result['validation_messages'] = {
            backend: msg.replace('\\"', '"') for backend, msg in validation_pairs
        }

    return result


def categorize_operation(op: Dict) -> str:
    """Categorize an operation into a higher-level category based on name and type."""
    name = op['name'].lower()
    op_type = op['operation'].lower()

    # KV Cache operations - check op_type first for state operations
    if op_type in ['read_state', 'write_state']:
        return 'KV Cache Read' if op_type == 'read_state' else 'KV Cache Update'
    if op_type == 'slice_update':
        return 'KV Cache Update'
    if op_type == 'slice_by_index':
        return 'KV Cache slice_by_index'

    # KV Cache operations by name
    if 'kv' in name or 'cache' in name:
        if 'update' in name or 'write' in name or op_type == 'slice_update':
            return 'KV Cache Update'
        elif 'read' in name or 'gather' in name:
            return 'KV Cache Read'
        return 'KV Cache'

    # Attention operations
    if 'attention' in name or 'attn' in name:
        if 'q_proj' in name or 'query' in name:
            return 'Attention Q Projection'
        elif 'k_proj' in name or 'key' in name:
            return 'Attention K Projection'
        elif 'v_proj' in name or 'value' in name:
            return 'Attention V Projection'
        elif 'o_proj' in name or 'out' in name:
            return 'Attention Output Projection'
        elif 'softmax' in name:
            return 'Attention Softmax'
        elif 'matmul' in name or 'mm' in name:
            return 'Attention MatMul'
        return 'Attention'

    # FFN/MLP operations
    if 'ffn' in name or 'mlp' in name or 'feed_forward' in name:
        if 'gate' in name:
            return 'FFN Gate'
        elif 'up' in name:
            return 'FFN Up'
        elif 'down' in name:
            return 'FFN Down'
        return 'FFN'

    # Normalization - only actual norm operations, not slices of normalized values
    if op_type in ['layer_norm', 'batch_norm', 'instance_norm', 'rms_norm']:
        return 'Normalization'
    if ('norm' in name or 'rmsnorm' in name) and op_type not in ['slice_by_index', 'reshape', 'transpose']:
        return 'Normalization'

    # Embedding operations
    if 'embed' in name or 'embedding' in name:
        return 'Embedding'

    # LM Head
    if 'lm_head' in name or 'output' in name and 'proj' in name:
        return 'LM Head'

    # Quantization/Dequantization
    if 'dequant' in name or 'quantize' in name:
        return 'Dequantization'
    if op_type == 'constexpr_lut_to_dense':
        return 'LUT Dequantization'
    if op_type == 'constexpr_affine_dequantize':
        return 'Affine Dequantization'
    if 'lut' in op_type or 'palettize' in op_type:
        return 'LUT Operations'

    # Activation functions
    if op_type in ['silu', 'gelu', 'relu', 'swish', 'sigmoid', 'tanh']:
        return 'Activation'

    # Basic operations by type
    op_type_categories = {
        'conv': 'Convolution',
        'linear': 'Linear Layer',
        'matmul': 'MatMul',
        'add': 'Element-wise Add',
        'mul': 'Element-wise Mul',
        'sub': 'Element-wise Sub',
        'div': 'Element-wise Div',
        'reshape': 'Reshape',
        'transpose': 'Transpose',
        'split': 'Split',
        'concat': 'Concat',
        'gather': 'Gather',
        'scatter': 'Scatter',
        'reduce': 'Reduce',
        'softmax': 'Softmax',
        'cast': 'Cast',
        'const': 'Constant',
        'expand_dims': 'Expand Dims',
        'squeeze': 'Squeeze',
    }

    for key, category in op_type_categories.items():
        if key in op_type:
            return category

    return f'Other ({op_type})'


def parse_mlperf_report(perf_path: str) -> Dict:
    """Parse an Xcode .mlperf performance report and return measured timing data."""
    report_json = os.path.join(perf_path, 'report.json')
    if not os.path.exists(report_json):
        raise FileNotFoundError(f"report.json not found in {perf_path}")

    with open(report_json, 'r') as f:
        data = json.load(f)

    result = {
        'model_name': os.path.basename(perf_path).replace('.mlperf', ''),
        'device_info': data.get('deviceInfo', {}),
        'compute_unit': data.get('computeUnit', 'unknown'),
        'predict': {},
        'load': {},
        'compile': {},
        'operations': []
    }

    # Extract prediction timing (actual measured runtime)
    if 'deviceResults' in data and 'predict' in data['deviceResults']:
        pred = data['deviceResults']['predict']
        samples = pred.get('samples', [])
        if samples:
            sorted_samples = sorted(samples)
            n = len(sorted_samples)
            result['predict'] = {
                'num_samples': n,
                'median_ms': sorted_samples[n // 2] * 1000,
                'mean_ms': sum(samples) / n * 1000,
                'min_ms': min(samples) * 1000,
                'max_ms': max(samples) * 1000,
                'p95_ms': sorted_samples[int(n * 0.95)] * 1000 if n > 20 else None,
                'num_operations': pred.get('numOperations', 0)
            }

    # Extract load timing
    if 'deviceResults' in data and 'load' in data['deviceResults']:
        load = data['deviceResults']['load']
        samples = load.get('samples', [])
        if samples:
            sorted_samples = sorted(samples)
            n = len(sorted_samples)
            result['load'] = {
                'median_ms': sorted_samples[n // 2] * 1000,
                'mean_ms': sum(samples) / n * 1000,
            }

    # Extract compile timing
    if 'deviceResults' in data and 'compile' in data['deviceResults']:
        comp = data['deviceResults']['compile']
        samples = comp.get('samples', [])
        if samples:
            sorted_samples = sorted(samples)
            n = len(sorted_samples)
            result['compile'] = {
                'median_ms': sorted_samples[n // 2] * 1000,
                'mean_ms': sum(samples) / n * 1000,
            }

    # Extract per-operation data if available
    if 'deviceResults' in data and 'modelStructure' in data['deviceResults']:
        ms = data['deviceResults']['modelStructure']
        if 'program' in ms and '_0' in ms['program'] and 'functions' in ms['program']['_0']:
            funcs = ms['program']['_0']['functions']
            for func_name, func_data in funcs.items():
                if 'block' in func_data and 'operations' in func_data['block']:
                    for op in func_data['block']['operations']:
                        if 'cost' in op and 'deviceUsage' in op:
                            op_info = {
                                'function': func_name,
                                'operator': op.get('operatorName', 'unknown'),
                                'cost_weight': op.get('cost', {}).get('weight', 0),
                                'device': 'unknown'
                            }
                            # Get preferred device
                            preferred = op.get('deviceUsage', {}).get('preferred', {})
                            if 'deviceType' in preferred:
                                if 'neuralEngine' in preferred['deviceType']:
                                    op_info['device'] = 'ane'
                                elif 'cpu' in preferred['deviceType']:
                                    op_info['device'] = 'cpu'
                                elif 'gpu' in preferred['deviceType']:
                                    op_info['device'] = 'gpu'
                            # Get output name
                            outputs = op.get('outputs', [])
                            if outputs and 'name' in outputs[0]:
                                op_info['name'] = outputs[0]['name']
                            result['operations'].append(op_info)

    return result


def print_perf_report(perf_data: Dict):
    """Print a formatted performance report from .mlperf data."""
    print("\n" + "=" * 80)
    print("COREML PERFORMANCE REPORT (Measured)")
    print("=" * 80)
    print(f"\nModel: {perf_data['model_name']}")

    # Compute unit mapping
    compute_units = {0: 'CPU Only', 1: 'CPU and GPU', 2: 'All', 3: 'CPU and Neural Engine'}
    cu = perf_data.get('compute_unit', 'unknown')
    cu_name = compute_units.get(cu, f'Unknown ({cu})')
    print(f"Compute Unit: {cu_name}")

    # Device info
    dev = perf_data.get('device_info', {})
    if dev:
        model = dev.get('modelName', dev.get('displayName', 'unknown'))
        os_ver = dev.get('osNameAndVersionWithoutBuildNumber', 'unknown')
        print(f"Device: {model}")
        print(f"OS: {os_ver}")

    # Prediction timing
    pred = perf_data.get('predict', {})
    if pred:
        print("\n" + "-" * 40)
        print("PREDICTION TIMING (Actual Measured)")
        print("-" * 40)
        print(f"  Samples:    {pred.get('num_samples', 0)}")
        print(f"  Median:     {pred.get('median_ms', 0):.2f}ms")
        print(f"  Mean:       {pred.get('mean_ms', 0):.2f}ms")
        print(f"  Min:        {pred.get('min_ms', 0):.2f}ms")
        print(f"  Max:        {pred.get('max_ms', 0):.2f}ms")
        if pred.get('p95_ms'):
            print(f"  P95:        {pred.get('p95_ms', 0):.2f}ms")
        print(f"  Operations: {pred.get('num_operations', 0)}")

    # Load timing
    load = perf_data.get('load', {})
    if load:
        print("\n" + "-" * 40)
        print("LOAD TIMING")
        print("-" * 40)
        print(f"  Median:     {load.get('median_ms', 0):.2f}ms")

    # Compile timing
    comp = perf_data.get('compile', {})
    if comp:
        print("\n" + "-" * 40)
        print("COMPILE TIMING")
        print("-" * 40)
        print(f"  Median:     {comp.get('median_ms', 0):.2f}ms")

    # Operation breakdown by device
    ops = perf_data.get('operations', [])
    if ops:
        print("\n" + "-" * 40)
        print("OPERATION DEVICE DISTRIBUTION")
        print("-" * 40)
        device_counts = defaultdict(int)
        for op in ops:
            device_counts[op.get('device', 'unknown')] += 1

        total = len(ops)
        device_names = {
            'ane': 'Neural Engine (ANE)',
            'cpu': 'CPU',
            'gpu': 'GPU',
            'unknown': 'Unknown'
        }
        for device, count in sorted(device_counts.items(), key=lambda x: -x[1]):
            pct = (count / total) * 100 if total > 0 else 0
            name = device_names.get(device, device)
            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
            print(f"  {name:25} {count:6} ({pct:5.1f}%) {bar}")

    # Calculate per-op timing from normalized cost weights
    total_ms = pred.get('median_ms', 0) if pred else 0
    if ops and total_ms > 0:
        total_weight = sum(op.get('cost_weight', 0) for op in ops)
        if total_weight > 0:
            scale = total_ms / total_weight  # ms per weight unit

            # Aggregate by operator type
            op_type_time = defaultdict(float)
            op_type_count = defaultdict(int)
            for op in ops:
                op_name = op.get('operator', 'unknown')
                time_ms = op.get('cost_weight', 0) * scale
                op_type_time[op_name] += time_ms
                op_type_count[op_name] += 1

            # KV Cache operations to group
            kv_cache_ops = ['ios18.slice_update', 'ios18.slice_by_index', 'ios18.read_state', 'ios18.write_state']
            kv_cache_time = sum(op_type_time.get(op, 0) for op in kv_cache_ops)
            kv_cache_count = sum(op_type_count.get(op, 0) for op in kv_cache_ops)

            print("\n" + "-" * 60)
            print("RUNTIME BY OPERATION TYPE (Calculated from measured total)")
            print("-" * 60)

            # Print KV Cache total first if it exists
            printed_kv = False
            for op_name, time_ms in sorted(op_type_time.items(), key=lambda x: -x[1]):
                # Group KV Cache operations
                if op_name in kv_cache_ops:
                    if not printed_kv and kv_cache_time > 0:
                        kv_pct = (kv_cache_time / total_ms) * 100 if total_ms > 0 else 0
                        kv_avg = (kv_cache_time / kv_cache_count * 1000) if kv_cache_count > 0 else 0
                        print(f"  {'KV Cache (total)':35} {kv_cache_time:8.2f}ms ({kv_pct:5.1f}%)  [{kv_cache_count} ops]  avg: {kv_avg:.0f}μs")
                        # Print sub-categories indented, sorted by time
                        kv_sub = [(op, op_type_time.get(op, 0)) for op in kv_cache_ops if op in op_type_time]
                        for kv_op, kv_time in sorted(kv_sub, key=lambda x: -x[1]):
                            kv_op_pct = (kv_time / total_ms) * 100 if total_ms > 0 else 0
                            kv_op_count = op_type_count.get(kv_op, 0)
                            kv_op_avg = (kv_time / kv_op_count * 1000) if kv_op_count > 0 else 0
                            print(f"    - {kv_op:33} {kv_time:8.2f}ms ({kv_op_pct:5.1f}%)  [{kv_op_count} ops]  avg: {kv_op_avg:.0f}μs")
                        printed_kv = True
                    continue

                count = op_type_count[op_name]
                pct = (time_ms / total_ms) * 100 if total_ms > 0 else 0
                avg_us = (time_ms / count * 1000) if count > 0 else 0
                print(f"  {op_name:35} {time_ms:8.2f}ms ({pct:5.1f}%)  [{count} ops]  avg: {avg_us:.0f}μs")

    print("\n" + "-" * 60)
    print("NOTE: Tensor shapes not available in .mlperf reports.")
    print("Use analytics.mil for tensor size info (run without --perf)")
    print("=" * 80 + "\n")


def parse_analytics_file(file_path: str) -> List[Dict]:
    """Parse an analytics.mil file and return list of operations."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract all operations with SelectedBackend (executable ops only)
    # This includes tensor<...> ops and write_state ops
    all_ops = [
        line.strip() for line in content.split(";")
        if "SelectedBackend" in line and ("tensor" in line or "write_state" in line or "read_state" in line)
    ]

    operations = []
    for op_str in all_ops:
        op = extract_operation(op_str)
        op['category'] = categorize_operation(op)
        operations.append(op)

    return operations


def analyze_operations(operations: List[Dict]) -> Dict:
    """Analyze operations and generate statistics."""
    stats = {
        'total_ops': len(operations),
        'by_type': defaultdict(int),
        'by_category': defaultdict(int),
        'by_backend': defaultdict(int),
        'by_category_backend': defaultdict(lambda: defaultdict(int)),
        'runtime_by_backend': defaultdict(float),
        'runtime_by_category': defaultdict(float),
        'ane_failures': defaultdict(list),
        'tensor_sizes': defaultdict(list),  # Track tensor sizes by op type
    }

    for op in operations:
        op_type = op['operation']
        category = op['category']
        backend = op['selected_backend']

        stats['by_type'][op_type] += 1
        stats['by_category'][category] += 1
        stats['by_backend'][backend] += 1
        stats['by_category_backend'][category][backend] += 1

        # Sum up runtimes for selected backend
        if backend in op['runtimes']:
            runtime = op['runtimes'][backend]
            stats['runtime_by_backend'][backend] += runtime
            stats['runtime_by_category'][category] += runtime

        # Track ANE failures
        if backend != 'ane' and 'ane' in op.get('validation_messages', {}):
            reason = op['validation_messages']['ane']
            stats['ane_failures'][reason].append(op['name'])

        # Track tensor sizes for specific operations
        if op['tensor_shape'] and op_type in ['slice_update', 'softmax', 'slice_by_index', 'read_state', 'write_state']:
            stats['tensor_sizes'][op_type].append({
                'name': op['name'],
                'shape': op['tensor_shape'],
                'elements': op['tensor_elements'],
                'dtype': op['tensor_dtype']
            })

    return stats


def print_report(stats: Dict, file_path: str, verbose: bool = False):
    """Print a formatted analysis report."""
    print("\n" + "=" * 80)
    print("COREML OPERATION ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nAnalytics file: {file_path}")
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
    print(f"Last modified: {mtime}")
    total_runtime = sum(stats['runtime_by_backend'].values())
    print(f"\nTotal operations: {stats['total_ops']}")
    print(f"Total estimated runtime: {total_runtime:.2f}ms")

    total = stats['total_ops']
    backend_names = {
        'ane': 'Neural Engine (ANE)',
        'mps_graph': 'GPU (MPS)',
        'classic_cpu': 'CPU',
        'bnns': 'CPU (BNNS)',
    }

    # Category breakdown by RUNTIME
    print("\n" + "-" * 60)
    print("RUNTIME BY CATEGORY (ms)")
    print("-" * 60)

    # Calculate KV Cache total runtime
    kv_cache_categories = ['KV Cache', 'KV Cache Read', 'KV Cache Update', 'KV Cache slice_by_index']
    kv_cache_runtime = sum(stats['runtime_by_category'].get(cat, 0) for cat in kv_cache_categories)
    kv_cache_count = sum(stats['by_category'].get(cat, 0) for cat in kv_cache_categories)

    # Print categories by runtime, grouping KV Cache entries
    printed_kv_total = False
    for category, runtime in sorted(stats['runtime_by_category'].items(), key=lambda x: -x[1]):
        # Skip individual KV Cache categories, we'll print the total
        if category in kv_cache_categories:
            if not printed_kv_total and kv_cache_runtime > 0:
                pct = (kv_cache_runtime / total_runtime) * 100 if total_runtime > 0 else 0
                kv_avg = (kv_cache_runtime / kv_cache_count * 1000) if kv_cache_count > 0 else 0
                print(f"  {'KV Cache (total)':30} {kv_cache_runtime:10.2f}ms ({pct:5.1f}%)  [{kv_cache_count} ops]  avg: {kv_avg:.0f}μs")
                # Print sub-categories indented, sorted by runtime
                kv_sub = [(cat, stats['runtime_by_category'].get(cat, 0)) for cat in kv_cache_categories if cat in stats['runtime_by_category']]
                for kv_cat, kv_rt in sorted(kv_sub, key=lambda x: -x[1]):
                    kv_pct = (kv_rt / total_runtime) * 100 if total_runtime > 0 else 0
                    kv_ops = stats['by_category'].get(kv_cat, 0)
                    kv_avg_sub = (kv_rt / kv_ops * 1000) if kv_ops > 0 else 0
                    print(f"    - {kv_cat:28} {kv_rt:10.2f}ms ({kv_pct:5.1f}%)  [{kv_ops} ops]  avg: {kv_avg_sub:.0f}μs")
                printed_kv_total = True
            continue

        count = stats['by_category'].get(category, 0)
        pct = (runtime / total_runtime) * 100 if total_runtime > 0 else 0
        avg = (runtime / count * 1000) if count > 0 else 0
        print(f"  {category:30} {runtime:10.2f}ms ({pct:5.1f}%)  [{count} ops]  avg: {avg:.0f}μs")

    # Tensor sizes for key operations
    if stats.get('tensor_sizes'):
        print("\n" + "-" * 60)
        print("TENSOR SIZES (Largest by element count)")
        print("-" * 60)

        def format_shape(shape):
            return '[' + ', '.join(str(d) for d in shape) + ']'

        def format_size(elements):
            if elements >= 1_000_000:
                return f"{elements / 1_000_000:.1f}M"
            elif elements >= 1_000:
                return f"{elements / 1_000:.1f}K"
            return str(elements)

        for op_type in ['slice_update', 'softmax', 'slice_by_index', 'read_state', 'write_state']:
            if op_type in stats['tensor_sizes']:
                tensors = stats['tensor_sizes'][op_type]
                # Sort by element count, get largest
                sorted_tensors = sorted(tensors, key=lambda x: -x['elements'])
                largest = sorted_tensors[0]
                # Check if all are same size
                unique_shapes = set(tuple(t['shape']) for t in tensors)
                if len(unique_shapes) == 1:
                    print(f"  {op_type:20} {format_shape(largest['shape']):25} = {format_size(largest['elements']):>8} elements  ({len(tensors)} ops, all same)")
                else:
                    print(f"  {op_type:20} {format_shape(largest['shape']):25} = {format_size(largest['elements']):>8} elements  (max of {len(tensors)} ops)")
                    # Show range if varied
                    smallest = sorted_tensors[-1]
                    if smallest['elements'] != largest['elements']:
                        print(f"    (min: {format_shape(smallest['shape'])} = {format_size(smallest['elements'])} elements)")

    # Operation type breakdown (verbose)
    if verbose:
        print("\n" + "-" * 40)
        print("OPERATIONS BY TYPE (raw)")
        print("-" * 40)
        for op_type, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
            pct = (count / total) * 100 if total > 0 else 0
            print(f"  {op_type:30} {count:5} ({pct:5.1f}%)")

    # ANE failures
    if stats['ane_failures']:
        print("\n" + "-" * 40)
        print("ANE FAILURE REASONS")
        print("-" * 40)
        for reason, ops in sorted(stats['ane_failures'].items(), key=lambda x: -len(x[1])):
            print(f"\n  Reason: {reason}")
            print(f"  Affected ops: {len(ops)}")
            if verbose and len(ops) <= 10:
                for op_name in ops[:10]:
                    print(f"    - {op_name}")
            elif len(ops) > 10:
                for op_name in ops[:5]:
                    print(f"    - {op_name}")
                print(f"    ... and {len(ops) - 5} more")

    # Backend distribution (moved to end)
    print("\n" + "-" * 40)
    print("BACKEND DISTRIBUTION")
    print("-" * 40)
    for backend, count in sorted(stats['by_backend'].items(), key=lambda x: -x[1]):
        pct = (count / total) * 100 if total > 0 else 0
        name = backend_names.get(backend, backend)
        bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        print(f"  {name:25} {count:6} ({pct:5.1f}%) {bar}")

    # Runtime by backend (moved to end)
    total_runtime = sum(stats['runtime_by_backend'].values())
    if total_runtime > 0:
        print("\n" + "-" * 40)
        print("ESTIMATED RUNTIME BY BACKEND")
        print("-" * 40)
        for backend, runtime in sorted(stats['runtime_by_backend'].items(), key=lambda x: -x[1]):
            pct = (runtime / total_runtime) * 100
            name = backend_names.get(backend, backend)
            print(f"  {name:25} {runtime:10.4f}ms ({pct:5.1f}%)")

    # Summary
    ane_count = stats['by_backend'].get('ane', 0)
    ane_pct = (ane_count / total) * 100 if total > 0 else 0
    print("\n" + "=" * 80)
    print(f"SUMMARY: {ane_pct:.1f}% of operations run on Neural Engine ({ane_count}/{total})")
    if ane_pct < 90:
        print("⚠️  Consider optimizing operations not running on ANE for better performance")
    else:
        print("✓  Good ANE coverage")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CoreML model operations and backend distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_coreml_ops.py                    # Auto-find latest analytics.mil
  python analyze_coreml_ops.py --list             # List all available analytics files
  python analyze_coreml_ops.py --select 2         # Analyze 2nd most recent file
  python analyze_coreml_ops.py --file analytics.mil
  python analyze_coreml_ops.py -v                 # Verbose output with raw op types
  python analyze_coreml_ops.py --ane-only         # Only show ANE operations
  python analyze_coreml_ops.py model.mlperf       # Analyze Xcode performance report
  python analyze_coreml_ops.py --perf model.mlperf  # Same as above
        """
    )
    parser.add_argument(
        'input_path',
        nargs='?',
        help='Path to .mlperf folder or analytics.mil file'
    )
    parser.add_argument(
        '--file', '-f',
        help='Path to analytics.mil file (auto-detects if not provided)'
    )
    parser.add_argument(
        '--perf', '-p',
        help='Path to .mlperf performance report folder'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available analytics files'
    )
    parser.add_argument(
        '--select', '-s',
        type=int,
        help='Select analytics file by index (1=latest, 2=second latest, etc.)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed operation type breakdown'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--ane-only',
        action='store_true',
        help='Only include operations that run on ANE (Neural Engine)'
    )

    args = parser.parse_args()

    # Handle --list option
    if args.list:
        list_analytics_files()
        return 0

    # Check for .mlperf input (positional arg, --perf, or --file with .mlperf)
    perf_path = args.perf or args.input_path
    if perf_path and (perf_path.endswith('.mlperf') or os.path.isdir(perf_path) and 'report.json' in os.listdir(perf_path)):
        # Handle .mlperf performance report
        if not os.path.exists(perf_path):
            print(f"Error: Path not found: {perf_path}")
            return 1
        try:
            perf_data = parse_mlperf_report(perf_path)
            print_perf_report(perf_data)
            if args.json:
                print(json.dumps(perf_data, indent=2, default=str))
            return 0
        except Exception as e:
            print(f"Error parsing .mlperf: {e}")
            return 1

    # Find analytics file
    file_path = None
    if args.file:
        file_path = args.file
    elif args.input_path:
        file_path = args.input_path
    elif args.select:
        files = find_all_analytics_files()
        if not files:
            print("Error: No analytics.mil files found.")
            return 1
        if args.select < 1 or args.select > len(files):
            print(f"Error: Invalid selection. Choose 1-{len(files)}")
            return 1
        file_path = files[args.select - 1][0]
        print(f"Selected file #{args.select}: {file_path}")
    else:
        try:
            file_path = find_latest_analytics_file()
            print(f"Using latest analytics file: {file_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nTo generate analytics.mil:")
            print("  1. Load and run your CoreML model in Xcode or via coremltools")
            print("  2. The analytics.mil will be created in the CoreML cache")
            return 1

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return 1

    # Parse and analyze
    operations = parse_analytics_file(file_path)

    # Filter to ANE-only if requested
    if args.ane_only:
        operations = [op for op in operations if op['selected_backend'] == 'ane']
        print(f"Filtered to ANE-only: {len(operations)} operations")

    stats = analyze_operations(operations)

    if args.json:
        import json
        # Convert defaultdicts to regular dicts for JSON serialization
        output = {
            'file_path': file_path,
            'total_ops': stats['total_ops'],
            'by_type': dict(stats['by_type']),
            'by_category': dict(stats['by_category']),
            'by_backend': dict(stats['by_backend']),
            'by_category_backend': {k: dict(v) for k, v in stats['by_category_backend'].items()},
            'runtime_by_backend': dict(stats['runtime_by_backend']),
            'runtime_by_category': dict(stats['runtime_by_category']),
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(stats, file_path, verbose=args.verbose)

    return 0


if __name__ == '__main__':
    exit(main())
