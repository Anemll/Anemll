# ANE Profiler

Automated CoreML/ANE profiling and analysis without Xcode.

## Overview

`ane_profiler.py` uses CoreMLTools 9.0+ MLComputePlan API to:
- Analyze which operations run on ANE vs GPU vs CPU
- Benchmark inference timing across compute units
- Identify operations that fall back from ANE
- Estimate operation costs
- Generate compatibility reports

## Requirements

- macOS 15+ (Sequoia)
- CoreMLTools 9.0+
- Xcode Command Line Tools (for `coremlcompiler`)

## Quick Start

```bash
# Full ANE compatibility report
python anemll/utils/ane_profiler.py --model model.mlpackage --report

# Analyze compute plan (which ops run on ANE/GPU/CPU)
python anemll/utils/ane_profiler.py --model model.mlpackage --analyze

# Benchmark all compute units with timing
python anemll/utils/ane_profiler.py --model model.mlpackage --benchmark-all
```

## Commands

### `--analyze` / `-a`
Analyze compute plan to see ANE/GPU/CPU operation distribution.

```bash
python ane_profiler.py -m model.mlpackage --analyze
```

**Output:**
```
Total operations: 4630
  Const/metadata: 2996
  Executable ops: 1634
    ANE: 1622 (99.3%)
    GPU: 12
    CPU: 0

--- GPU Fallback Operations (12) ---
  - ios18.cast:current_pos_to_int16
  - select:select_0
  ...
```

### `--benchmark` / `-b UNIT`
Benchmark a specific compute unit with millisecond timing.

```bash
# Benchmark ANE only
python ane_profiler.py -m model.mlpackage --benchmark CPU_AND_NE

# Benchmark CPU only
python ane_profiler.py -m model.mlpackage --benchmark CPU_ONLY

# Benchmark GPU
python ane_profiler.py -m model.mlpackage --benchmark CPU_AND_GPU
```

**Compute Units:**
| Unit | Description |
|------|-------------|
| `CPU_ONLY` | CPU execution only |
| `CPU_AND_GPU` | CPU + GPU (Metal) |
| `CPU_AND_NE` | CPU + Neural Engine (ANE) |
| `ALL` | Let CoreML decide best backend |

### `--benchmark-all` / `-B`
Benchmark all compute units and compare timing.

```bash
python ane_profiler.py -m model.mlpackage --benchmark-all --iterations 10 --warmup 3
```

**Output:**
```
TIMING COMPARISON (milliseconds)
======================================================================
Compute Unit          Mean        Min        Max       Load Status
----------------------------------------------------------------------
CPU_ONLY             32.41      32.22      32.95     4199.3 ✓
CPU_AND_GPU          25.72       9.64      97.99     2695.3 ✓
CPU_AND_NE            6.25       6.20       6.31     4013.1 ✓
ALL                   6.25       6.13       6.50     3181.3 ✓

Fastest: CPU_AND_NE (6.25ms)
```

### `--compare` / `-c`
Compare prediction results across compute units (load time only for stateful models).

```bash
python ane_profiler.py -m model.mlpackage --compare
```

### `--costs`
Estimate operation costs using compute plan weights.

```bash
python ane_profiler.py -m model.mlpackage --costs
```

**Output:**
```
OPERATION COST ESTIMATION
======================================================================
Total estimated cost weight: 0.85

Top 10 most expensive operations:
  ios18.slice_update   kv_cache_update_1    weight=0.0079 (0.9%)
  ios18.slice_update   kv_cache_update_2    weight=0.0079 (0.9%)
  ...
```

### `--report` / `-r`
Generate full compatibility report (combines all analyses).

```bash
python ane_profiler.py -m model.mlpackage --report --output report.json
```

### `--test` / `-t UNIT`
Test prediction on a specific compute unit (exit code 0=success, 1=fail).

```bash
python ane_profiler.py -m model.mlpackage --test CPU_AND_NE
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Path to .mlpackage or .mlmodelc | Required |
| `--iterations`, `-n` | Number of benchmark iterations | 10 |
| `--warmup`, `-w` | Warmup iterations (not timed) | 3 |
| `--output`, `-o` | Save JSON report to file | None |
| `--quiet`, `-q` | Minimal output | False |

## Interpreting Results

### ANE Coverage

| ANE % | Status | Action |
|-------|--------|--------|
| >95% | Excellent | Ready for deployment |
| 80-95% | Good | Review GPU fallback ops |
| 50-80% | Mixed | May need optimization |
| <50% | Poor | Model needs restructuring |

### Common GPU Fallback Operations

| Operation | Reason |
|-----------|--------|
| `select` | Dynamic control flow |
| `ios18.cast` | Integer type casts |
| `ios18.gather` | Dynamic indexing |
| `ios18.greater_equal` | Comparison ops |
| `while_loop` | Dynamic iteration |

### Load Time vs Inference Time

- **Load time** includes compilation - ANE compilation is slower but runs faster
- **Inference time** is what matters for production
- ANE typically: slower load, faster inference
- GPU typically: faster load, variable inference

## Examples

### Example 1: Analyze Gemma3 FFN Chunk

```bash
python ane_profiler.py \
  -m /path/to/gemma3_FFN_lut4_chunk_01of02.mlpackage \
  --analyze
```

### Example 2: Benchmark LM Head Performance

```bash
python ane_profiler.py \
  -m /path/to/model_lm_head.mlpackage \
  --benchmark-all \
  --iterations 20 \
  --warmup 5
```

### Example 3: Full Report with JSON Output

```bash
python ane_profiler.py \
  -m /path/to/model.mlpackage \
  --report \
  --output model_ane_report.json
```

### Example 4: Run All Backends (Find Best Performance)

Sometimes `ALL` or a specific backend gives better results. Test all to find optimal:

```bash
# Benchmark all backends with timing comparison
python ane_profiler.py -m model.mlpackage --benchmark-all --iterations 20

# Quick comparison (load time + single prediction)
python ane_profiler.py -m model.mlpackage --compare

# Test with ALL (let CoreML pick best backend)
python ane_profiler.py -m model.mlpackage --benchmark ALL
```

**Typical results:**
```
TIMING COMPARISON (milliseconds)
======================================================================
Compute Unit          Mean        Min        Max       Load Status
----------------------------------------------------------------------
CPU_ONLY             32.41      32.22      32.95     4199.3 ✓
CPU_AND_GPU          25.72       9.64      97.99     2695.3 ✓  ← High variance
CPU_AND_NE            6.25       6.20       6.31     4013.1 ✓  ← Fastest, consistent
ALL                   6.25       6.13       6.50     3181.3 ✓  ← Picks ANE

Fastest: CPU_AND_NE (6.25ms)
```

### Example 5: CI/CD Integration

```bash
# Check ANE compatibility (exit 0 if works)
python ane_profiler.py -m model.mlpackage --test CPU_AND_NE || exit 1

# Or test ALL backends pass
for unit in CPU_ONLY CPU_AND_GPU CPU_AND_NE ALL; do
  python ane_profiler.py -m model.mlpackage --test $unit || echo "$unit FAILED"
done

# Verify >90% ANE coverage
python ane_profiler.py -m model.mlpackage --analyze --quiet | grep "ANE:"
```

## Stateful Models (KV Cache)

Models with MLState inputs (KV cache) will show prediction failures during benchmarking:
```
FAILED: The input feature for model_model_kv_cache_global must be an MLState
```

This is expected - the profiler uses random inputs which don't include MLState. However:
- **Compute plan analysis still works** (--analyze)
- **Load time comparison still works** (--compare)
- **Cost estimation still works** (--costs)

For actual inference benchmarking of stateful models, use the full inference pipeline.

## Comparison with Xcode Instruments

| Feature | ane_profiler.py | Xcode Instruments |
|---------|-----------------|-------------------|
| Op device assignment | Yes | Yes |
| Per-op timing | Estimated weights | Actual timing |
| Inference benchmarks | Yes | Yes |
| Memory profiling | No | Yes |
| GPU Frame Capture | No | Yes |
| Requires Xcode | No | Yes |
| CI/CD friendly | Yes | Limited |
| JSON export | Yes | Limited |

## Troubleshooting

### "Compilation failed"
Ensure Xcode Command Line Tools are installed:
```bash
xcode-select --install
xcrun --find coremlcompiler
```

### "Could not load compute plan"
- Check model path is correct
- Ensure model is a valid .mlpackage or .mlmodelc
- Try with a pre-compiled .mlmodelc

### High GPU variance in benchmarks
GPU execution can have high variance due to:
- Context switching with system GPU tasks
- Metal shader compilation
- Memory transfer overhead

Run more iterations (`--iterations 50`) for more stable averages.

## See Also

- [fp16_compatibility_check.py](./fp16_compatibility_check.py) - FP16 overflow detection
- [GEMMA3_FP16_SCALING.md](../../docs/GEMMA3_FP16_SCALING.md) - FP16 scaling for Gemma3
- CoreMLTools documentation: https://coremltools.readme.io/
