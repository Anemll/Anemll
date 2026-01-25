# Segmented Evaluation with simple_evaluate_segmented

This document describes the standalone copy of `lm_eval.simple_evaluate` with dataset segmentation support.

## Overview

The `simple_evaluate_with_segmentation.py` file contains a complete copy of the `simple_evaluate` function from lm-evaluation-harness v0.4.9, with additional parameters to support dataset segmentation for parallel evaluation across multiple processes or machines.

## Key Features

### 1. Complete Preprocessing Pipeline Preservation
- **Identical tokenization**: Uses the same tokenizer and preprocessing as the original harness
- **Identical prompt formatting**: Preserves chat templates, system instructions, and fewshot formatting
- **Identical request building**: Maintains the same request creation and caching logic
- **Identical metric calculation**: Uses the same metrics and aggregation methods

### 2. Segmentation Support
Three new parameters enable dataset segmentation:
- `segment_start`: Starting index of the segment (0-based)
- `segment_size`: Number of examples to evaluate in this segment  
- `total_dataset_size`: Total size of complete dataset (for metadata)

### 3. Seamless Integration
- Works as a drop-in replacement for `lm_eval.simple_evaluate`
- Converts segmentation parameters to the existing `samples` mechanism internally
- Maintains all original parameters and behaviors

## Usage

### Basic Segmentation
```python
from simple_evaluate_with_segmentation import simple_evaluate_segmented

# Evaluate examples 0-99 out of 1000 total
results = simple_evaluate_segmented(
    model=my_model,
    tasks=["boolq"],
    segment_start=0,
    segment_size=100,
    total_dataset_size=1000,
    batch_size=1,
    random_seed=42
)
```

### Parallel Evaluation Example
```python
# Process 1: Examples 0-249
results_1 = simple_evaluate_segmented(
    model=model, tasks=["boolq"],
    segment_start=0, segment_size=250, total_dataset_size=1000
)

# Process 2: Examples 250-499  
results_2 = simple_evaluate_segmented(
    model=model, tasks=["boolq"],
    segment_start=250, segment_size=250, total_dataset_size=1000
)

# Process 3: Examples 500-749
results_3 = simple_evaluate_segmented(
    model=model, tasks=["boolq"],
    segment_start=500, segment_size=250, total_dataset_size=1000
)

# Process 4: Examples 750-999
results_4 = simple_evaluate_segmented(
    model=model, tasks=["boolq"],
    segment_start=750, segment_size=250, total_dataset_size=1000
)
```

### Integration with Existing ANE Evaluation
```python
# Use with your existing ANELM model
from simple_evaluate_with_segmentation import simple_evaluate_segmented

# Initialize your ANE model (same as before)
lm = ANELM(
    model_path="/path/to/model",
    verbose_output=True,
    log_incorrect_answers=True
)

# Run segmented evaluation
results = simple_evaluate_segmented(
    model=lm,  # Pass your ANE model directly
    tasks=["boolq"],
    segment_start=100,
    segment_size=200,
    batch_size=1,
    num_fewshot=0,
    random_seed=123
)
```

## Implementation Details

### Conversion to Samples Format
The segmentation parameters are converted internally to the `samples` format that lm-evaluation-harness already supports:

```python
# Segmentation: segment_start=100, segment_size=50
# Becomes: samples={"boolq": [100, 101, 102, ..., 149]}
```

This ensures compatibility with the existing evaluation pipeline.

### Per-Task Segmentation
The function automatically handles different dataset sizes per task:
- Queries each task's dataset size
- Calculates appropriate segment indices for each task
- Handles cases where segment extends beyond dataset size

### Metadata Preservation
Segmentation information is preserved in the results:
```python
results["config"]["segment_start"] = 100
results["config"]["segment_size"] = 50  
results["config"]["total_dataset_size"] = 1000

results["segmentation"] = {
    "segment_start": 100,
    "segment_size": 50, 
    "total_dataset_size": 1000
}
```

## File Structure

```
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev/
├── simple_evaluate_with_segmentation.py  # Main segmented evaluation function
└── README_segmentation.md                # This documentation
```

## Dependencies

The segmented evaluation function requires the same dependencies as the original lm-evaluation-harness:

- `lm_eval` package (v0.4.9 tested)
- `torch`
- `numpy` 
- All other lm-evaluation-harness dependencies

## Key Functions

### `simple_evaluate_segmented()`
Main function that provides segmented evaluation with the same interface as `lm_eval.simple_evaluate` plus segmentation parameters.

### `evaluate_segmented()`
Internal function that handles the core evaluation logic with segmentation metadata.

### `request_caching_arg_to_dict()`
Helper function for cache request argument processing (copied from original).

## Validation and Error Handling

- Validates that `segment_start` and `segment_size` are provided together
- Ensures `segment_start >= 0` and `segment_size > 0`
- Handles cases where segment extends beyond dataset size
- Provides informative logging about segment boundaries per task

## Benefits

1. **Parallel Processing**: Enable evaluation across multiple processes/machines
2. **Memory Management**: Evaluate large datasets in smaller chunks
3. **Progress Tracking**: Monitor progress through large evaluations
4. **Debugging**: Test on small segments before full evaluation
5. **Reproducibility**: Maintains exact same preprocessing and evaluation logic

## Compatibility

- Fully compatible with existing ANE evaluation scripts
- Works with all lm-evaluation-harness task types
- Supports all original simple_evaluate parameters
- Can be used as a drop-in replacement

## Example Integration Script

```bash
#!/bin/bash
# Parallel BoolQ evaluation across 4 processes

python -c "
from simple_evaluate_with_segmentation import simple_evaluate_segmented
from your_model_loader import load_ane_model

model = load_ane_model('/path/to/model')

# Segment 1: 0-249
results = simple_evaluate_segmented(
    model=model,
    tasks=['boolq'], 
    segment_start=0,
    segment_size=250,
    total_dataset_size=1000,
    batch_size=1
)
print('Segment 1 accuracy:', results['results']['boolq']['acc'])
" &

# Similar for segments 2, 3, 4...
wait  # Wait for all processes to complete
```

This segmentation capability enables efficient parallel evaluation while preserving the exact preprocessing pipeline of lm-evaluation-harness.