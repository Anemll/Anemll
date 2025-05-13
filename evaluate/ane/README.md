# ANE Model Evaluation

This directory contains tools for evaluating Apple Neural Engine (ANE) models using lm-evaluation-harness.

## Core Components

1. **ane_model.py**: A clean abstraction for ANE/CoreML model operations
   - Handles model loading, state management, and tensor conversions
   - Provides a consistent interface for prefill and predict operations
   - Ensures proper batch size and data type handling for CoreML

2. **anelm_harness.py**: Integration with lm-evaluation-harness using the ANE_Model abstraction
   - Implements the ANELM model class required by lm-evaluation-harness
   - Properly manages state between prefill and predict operations
   - Ensures each prompt is evaluated independently with a clean state

3. **evaluate_with_harness.py**: Original implementation (older version)
   - Legacy implementation with direct CoreML model management
   - Has issues with CoreML tensor conversions in some cases

## Known Issues and Solutions

### CoreML "value type not convertible" errors

The most common error when running ANE model evaluation is "value type not convertible".
This occurs when:

1. The CoreML model expects inputs with a specific batch size (typically 64)
2. We try to feed inputs with a different batch size (typically 1)
3. The tensor data type doesn't match what the model expects (needs int32)

Our solution:
- Use batch_size=1 for the evaluation harness (strictly serial evaluation)
- Properly pad tensors to the model's compiled batch size in the ANE_Model class
- Ensure all inputs are of the correct data type (int32)
- Reset state between prompts to avoid state contamination

### ANE Resource Busy errors

The Apple Neural Engine is a shared resource that can become busy when accessed concurrently.

Our solution:
- Set environment variables to force CoreML to use single-threaded mode
- Use a batch size of 1 with the harness for strictly serial execution
- Reset state between each prompt to ensure clean evaluation

## Usage

### New Implementation (Recommended)

```bash
./run_new_eval.sh --model /path/to/your/model --tasks boolq,arc_easy,hellaswag
```

### Original Implementation

```bash
./run_serial_eval.sh --model /path/to/your/model --tasks boolq,arc_easy,hellaswag
```

### Available Options

- `--model PATH`: Path to model directory
- `--tasks LIST`: Comma-separated list of tasks to evaluate
- `--num-shots N`: Number of few-shot examples (default: 0)
- `--batch-size N`: Batch size for evaluation (default: 1, recommended for ANE)
- `--output-dir DIR`: Directory to save results (default: results)
- `--limit N`: Limit number of examples per task
- `--max-tokens N`: Maximum number of tokens to generate
- `--seed N`: Random seed (default: 123)
- `--apply-chat-template`: Apply chat template to prompts

## Recommended Model Directory Structure

Your model directory should contain:
- embeddings.mlmodelc or embeddings.mlpackage
- lm_head.mlmodelc or lm_head.mlpackage
- FFN_*.mlmodelc or FFN_*.mlpackage files (may be chunked)
- tokenizer files (tokenizer.json, tokenizer_config.json, etc.)

## Troubleshooting

If you encounter "value type not convertible" errors:
1. Ensure you're using batch_size=1 for the harness
2. Verify that CoreML single-threading environment variables are set
3. Check that your model's compiled batch size matches what's in the model's metadata

## Credits

This implementation is based on MLX-LM's approach to model evaluation, adapted for CoreML and the Apple Neural Engine.

## How the ANE Model Implementation Works

The ANE_Model class provides a clean abstraction over CoreML models compiled for Apple Neural Engine. It handles two main operations:

1. **Prefill** - Takes a sequence of tokens and runs the full attention mechanism
   - Uses the compiled batch size (e.g., 64)
   - Pads input tensors to match the compiled batch size
   - Run on full context (multiple tokens at once)

2. **Predict** - Generates a single next token based on the current state
   - Uses a single token input of shape [1, 1] (not padded to batch size)
   - Does not pass batch_size parameter to the embedding model
   - Emulates a "single token" auto-regressive generation

The key insight is that CoreML models have different functions optimized for different input shapes:
- The prefill function expects batch_size=64 tensors
- The predict function expects a single token tensor [1, 1]

This implementation allows the LM-evaluation-harness to work with ANE models by:
1. Maintaining proper state between prompts
2. Enforcing serial execution with batch_size=1
3. Properly handling the tensor shapes and data types expected by CoreML

### Common Issues

- "Value type not convertible" errors usually indicate tensor shape or dtype mismatches
- Make sure to reset state between different evaluation prompts
- Set environment variables for single-threaded CoreML execution 