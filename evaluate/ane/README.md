# ANE Model Evaluation

This directory contains tools for evaluating Apple Neural Engine (ANE) models using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

We evaluate models directly on Apple Neural Engine (ANE) hardware. Since ANE is a black box with no available emulators, one must run models on actual ANE hardware to get true performance measurements.

## Quick Start

### Direct Harness Integration

```bash
python evaluate_with_harness.py --model /path/to/your/model --tasks boolq,arc_easy,hellaswag
```

## Core Components

1. **evaluate_with_harness.py**: Complete harness integration for ANE models
   - Provides full lm-evaluation-harness integration for ANEMLL models
   - Implements ANELM model class with direct CoreML model handling
   - Supports most standard evaluation tasks (BoolQ, ARC, HellaSwag, etc.)
   - Includes advanced features like incorrect answer logging and verbose output
   - Tested with lm-evaluation-harness version 0.4.9
   - Self-contained implementation with direct CoreML integration


## Available Options

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
3. Check that your ANEMLL model's compiled batch size matches what's in the model's metadata

## Credits

This implementation is based on MLX-LM's approach to model evaluation, adapted for CoreML and the Apple Neural Engine.




## How the Harness Integration Works

The `evaluate_with_harness.py` script provides complete lm-evaluation-harness integration for ANE models by:

1. **Direct CoreML Integration** - Manages ANEMLL models directly without abstraction layers
2. **ANELM Model Class** - Implements the required interface for lm-evaluation-harness
3. **State Management** - Properly handles state between evaluation prompts
4. **Tensor Handling** - Ensures correct tensor shapes and data types for CoreML

### Key Features

- **Serial Execution** - Uses batch_size=1 for strictly serial evaluation
- **Advanced Logging** - Optional verbose output and incorrect answer logging
- **Flexible Model Loading** - Supports various model file naming conventions
- **Chat Template Support** - Optional chat template application for instruction-tuned models

### Common Issues

- "Value type not convertible" errors usually indicate tensor shape or dtype mismatches
- Make sure to reset state between different evaluation prompts
- Set environment variables for single-threaded CoreML execution 