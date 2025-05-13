# ANE/CoreML Model Evaluation System PRD

## Overview
This document outlines the requirements for an evaluation system for ANE/CoreML models, similar to the existing `run_all_evaluations.sh` for MLX models but optimized for Apple Neural Engine acceleration.

## Goals
1. Create a comprehensive evaluation pipeline for ANE/CoreML models
2. Minimize model reloading between evaluation tasks for performance
3. Support both rapid prototyping and production-grade evaluation
4. Allow comparison of models with different quantization strategies
5. Leverage existing MLX evaluation infrastructure where possible

## Key Requirements

### Core Functionality
- **Multiple Benchmark Support**: Support standard LLM benchmarks (ARC, BoolQ, HellaSwag, etc.)
- **Perplexity Calculation**: Efficient perplexity evaluation on customizable text samples
- **Performance Metrics**: Track inference speed, memory usage, and energy consumption
- **Quantization Analysis**: Compare performance between different quantization approaches
- **Batch Mode**: Run all tasks with a single model load when possible

### Implementation Components

#### 1. Prototype Implementation (`evaluate/ane/evaluate_ane.py`)
- Python script for model evaluation
- Loads model once for all evaluations
- Uses CoreML for model inference with KV cache support
- Basic reporting of results
- Primary focus on functionality validation

#### 2. Full Evaluation Implementation (Future expansion)
- Comprehensive evaluation pipeline
- Optimized for performance and reliability
- Proper model caching between tasks
- Detailed metrics tracking
- Flexible configuration via config files

#### 3. Bash Runner Script (`evaluate/run_ane_evaluations.sh`)
- Similar interface to existing `run_all_evaluations.sh`
- Support for both prototype and full implementation
- Command-line argument handling
- Environment setup and validation

## Technical Specifications

### Model Loading
```python
def load_model(model_path, use_compiled=True):
    """Load CoreML model with appropriate settings for ANE"""
    compute_unit = ct.ComputeUnit.CPU_AND_NE
    
    if use_compiled and model_path.suffix == '.mlmodelc':
        return ct.models.CompiledMLModel(str(model_path), compute_unit)
    else:
        return ct.models.MLModel(str(model_path))
```

### Performance Optimization
- Load model components (embedding, FFN, etc.) once at startup
- Maintain KV cache between evaluation samples when possible
- Utilize batching for compatible benchmarks
- Implement proper cleanup to avoid memory leaks
- Support chunked models for memory efficiency

### Benchmark Support
- Direct ANE implementation of standard LLM benchmarks
- Option to use MLX code for evaluation tasks with ANE model inference
- Custom benchmarks for ANE-specific optimizations

### Reporting
- Standardized output format compatible with existing reporting
- Timing information for each benchmark
- Perplexity and accuracy metrics
- Resource utilization statistics

## Implementation Plan

### Phase 1: Prototype
1. Create initial `evaluate/ane/evaluate_ane.py` script for basic functionality
2. Implement perplexity calculation for CoreML models with KV cache support
3. Add support for key benchmarks (ARC, BoolQ, HellaSwag, etc.)
4. Leverage existing test code from `tests/chat.py` for efficient inference

### Phase 2: Full Implementation
1. Expand `evaluate_ane.py` with more sophisticated evaluation techniques
2. Add comprehensive benchmark support with proper KV cache management
3. Implement performance monitoring and optimization features
4. Create detailed reporting with benchmark comparisons

### Phase 3: Integration
1. Enhance bash runner script capabilities
2. Ensure compatibility with existing infrastructure
3. Add comparison capabilities with MLX models
4. Add support for different quantization strategies

## Success Criteria
1. Successfully run all standard benchmarks on ANE/CoreML models
2. Match or exceed MLX evaluation speed for equivalent tasks
3. Provide accurate metrics for model comparison
4. Support different quantization strategies (LUT, linear, etc.)
5. Enable easy integration into existing workflows

## Implementation Notes
1. KV cache management: We're leveraging the existing implementation from `tests/chat.py` for efficient token generation and context processing.
2. Dataset usage: All evaluations use the HuggingFace datasets from local cache at `~/.cache/huggingface/datasets/`.
3. For models that exceed ANE memory limits, we can use chunked models or fall back to CPU-only execution.
4. When measuring performance, we track both throughput (tokens/sec) and latency metrics for a complete picture of model capabilities. 