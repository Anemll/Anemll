# ANE/CoreML Model Evaluation

This directory contains tools for evaluating ANE/CoreML models on standard LLM benchmarks and performance metrics.

## Directory Structure

```
evaluate/
├── ane/                      # ANE-specific evaluation code
│   ├── evaluate_ane.py       # Standalone evaluator script for ANE models
│   ├── ane_model.py          # ANE model abstraction class
│   ├── anelm_harness.py      # LM evaluation harness integration
│   └── run_eval.sh           # Runner script for harness-based evaluation
├── configs/                  # Configuration files
│   ├── config.sh             # User configuration (create from template)
│   ├── config.sh.template    # Configuration template
│   └── sample_text.txt       # Sample text for perplexity evaluation
├── results/                  # Evaluation results (created during evaluation)
└── run_ane_evaluations.sh    # Original runner script for standalone evaluation
```

## Evaluation Approaches

This codebase provides two approaches for evaluating ANE models:

### 1. Harness-Based Evaluation (Recommended)

Uses the lm-evaluation-harness framework with ANE model integration:

```bash
# Run with the harness-based approach
./evaluate/ane/run_eval.sh --tasks "arc_easy,hellaswag" --model /path/to/model
```

#### Features:
- Standard metrics consistent with other models in lm-evaluation-harness
- Support for multiple-choice task evaluation with normalized probabilities
- Debug mode for troubleshooting with verbose output
- Control over max tokens, batch size, and other parameters

### 2. Standalone Evaluation

Original self-contained evaluation script:

```bash
# Run with the standalone approach
./evaluate/run_ane_evaluations.sh --tasks="arc_easy hellaswag" --model=/path/to/model
```

## Quick Start (Harness-Based Approach)

```bash
# Run evaluation with default settings
./evaluate/ane/run_eval.sh

# Run with a specific model
./evaluate/ane/run_eval.sh --model /path/to/your/model

# Run specific tasks
./evaluate/ane/run_eval.sh --tasks "arc_easy,boolq,hellaswag"

# Run with debug output enabled
./evaluate/ane/run_eval.sh --tasks "hellaswag" --debug

# Limit the number of examples (useful for testing)
./evaluate/ane/run_eval.sh --tasks "boolq" --limit 10

# Run perplexity evaluation
./evaluate/ane/run_eval.sh --perplexity default                 # Using default sample text
./evaluate/ane/run_eval.sh --perplexity path/to/text/file.txt   # Using custom text file
./evaluate/ane/run_eval.sh --perplexity                         # Using wikitext dataset
```

## Available Tasks

- `arc_easy`: AI2 Reasoning Challenge (Easy)
- `arc_challenge`: AI2 Reasoning Challenge (Challenge)
- `boolq`: Boolean Questions
- `hellaswag`: HellaSwag 
- `winogrande`: Winogrande
- `openbookqa`: OpenBookQA
- `piqa`: Physical Interaction QA
- And many more from lm-evaluation-harness

## Prerequisites

- Python 3.9 or higher
- CoreML Tools (`pip install coremltools`)
- HuggingFace Transformers (`pip install transformers`) 
- lm-evaluation-harness (`pip install lm-evaluation-harness`)
- PyTorch (`pip install torch`) for harness-based evaluation
- Compiled CoreML model(s)

## Detailed Usage (Harness-Based)

### Command Line Arguments

```bash
./evaluate/ane/run_eval.sh --help
```

Key arguments include:

- `--model PATH`: Path to the model directory
- `--tasks LIST`: Comma-separated list of tasks to evaluate
- `--num-shots N`: Number of few-shot examples (default: 0)
- `--batch-size N`: Batch size for evaluation (default and recommended: 1)
- `--output-dir DIR`: Directory to save results
- `--limit N`: Limit number of examples per task
- `--max-tokens N`: Maximum number of tokens to generate
- `--seed N`: Random seed for reproducibility
- `--debug`: Enable verbose debug output
- `--perplexity [FILE]`: Run perplexity evaluation (optional path to text file)

### Detailed Usage (Standalone)

- `--model=PATH`: Path to the model directory
- `--output-dir=PATH`: Directory to save results
- `--tasks=TASKS`: Space-separated list of tasks to run
- `--perplexity-text=PATH`: Path to text file for perplexity evaluation (or "default")

## Model Requirements

The evaluation scripts expect a model directory with the following components:

- Token embedding model (e.g., `embeddings.mlpackage` or `embeddings.mlmodelc`)
- Language model head (e.g., `lm_head.mlpackage` or `lm_head.mlmodelc`)
- FFN model(s) (e.g., `FFN_PF_chunk_01of04.mlpackage` for chunked models)

Both approaches will automatically detect and load compiled models (`.mlmodelc`) if available.

## Results

### Harness-Based Results

```
results/
├── eval_[model_name]_[num_shots]shot_[tasks].json   # Full results in JSON format
├── perplexity_results.txt                           # Perplexity results (if requested)
```

### Standalone Results

```
results/
├── perplexity_results.txt        # Perplexity evaluation results
├── arc_easy_results.json         # ARC Easy results
├── boolq_results.json            # BoolQ results
├── ...                           # Other task results
└── summary.json                  # Overall summary of all results
```

## Implementation Notes

- The harness-based approach (anelm_harness.py) provides better integration with standard benchmarks
- The standalone approach (evaluate_ane.py) offers more custom metrics and self-contained evaluation
- Both systems are designed to be extensible
- ANE models should always be evaluated with batch_size=1 for optimal performance 