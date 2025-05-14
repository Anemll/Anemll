# ANE/CoreML Model Evaluation

This directory contains tools for evaluating ANE/CoreML models on standard LLM benchmarks and performance metrics.

## Directory Structure

```
evaluate/
├── ane/                      # ANE-specific evaluation code
│   ├── ane_model.py          # ANE model abstraction class
│   ├── anelm_harness.py      # LM evaluation harness integration
│   ├── run_eval.sh           # Runner script for harness-based evaluation
│   ├── full_perplexity.py    # Full perplexity evaluation script
│   └── evaluate_with_harness.py # Evaluation with harness script
├── configs/                  # Configuration files
│   └── config.json           # Configuration file for evaluation
└── results/                  # Evaluation results (created during evaluation)
```

## Evaluation Approaches

This codebase provides an approach for evaluating ANE models with lm-evaluation-harness:

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

### Evaluating Multiple Tasks and Combined Reports

To evaluate multiple tasks in a single run and generate a consolidated JSON report, provide a space-separated list of task names to the `--tasks` argument. The script will process each task sequentially.

**Example:**

```bash
./evaluate/ane/run_eval.sh --tasks "winogrande boolq arc_challenge arc_easy hellaswag openbookqa piqa social_iqa" --limit 100 --model ~/Models/ANE/anemll-llama3.2-1B-LUT4-b128-ctx1024
```

This will:
1.  Run evaluations for `winogrande`, `boolq`, `arc_challenge`, `arc_easy`, `hellaswag`, `openbookqa`, `piqa`, and `social_iqa` with a limit of 100 samples per task.
2.  Generate an individual JSON result file for each task (e.g., `winogrande_MODELNAME_DATE.json`, `boolq_MODELNAME_DATE.json`, etc.) in the specified output directory.
3.  Create a **combined JSON file** named `all_tasks_MODELNAME_DATE_TIME.json` in the output directory. This file contains a top-level `metadata` object with details about the overall `run_eval.sh` execution (like the full command line, start/end times, total duration) and a `results` object that aggregates the individual JSON outputs from each task.

This combined report is useful for getting a comprehensive overview of the model's performance across multiple benchmarks from a single execution.

## Available Tasks

- `arc_easy`: AI2 Reasoning Challenge (Easy)
- `arc_challenge`: AI2 Reasoning Challenge (Challenge)
- `boolq`: Boolean Questions
- `hellaswag`: HellaSwag 
- `winogrande`: Winogrande
- `openbookqa`: OpenBookQA
- `piqa`: Physical Interaction QA
- `mmlu`: Massive Multitask Language Understanding (various subjects)
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



## Model Requirements

The evaluation scripts expect a model directory with the following components:

- Token embedding model (e.g., `embeddings.mlpackage` or `embeddings.mlmodelc`)
- Language model head (e.g., `lm_head.mlpackage` or `lm_head.mlmodelc`)
- FFN model(s) (e.g., `FFN_PF_chunk_01of04.mlpackage` for chunked models)

Both approaches will automatically detect and load compiled models (`.mlmodelc`) if available.

## Results

### Harness-Based Results

Individual task results are saved as JSON files, typically named `TASKNAME_MODELNAME_DATE.json`.
If perplexity evaluation is run, results might be in `perplexity_results.txt` or a task-specific JSON (e.g., `wikitext_MODELNAME_DATE.json`).

When multiple tasks are specified in a single `run_eval.sh` command (see "Evaluating Multiple Tasks and Combined Reports"), the script also generates a **combined JSON file**:

```
results/
├── TASK1_MODELNAME_DATE.json          # Individual result for Task1
├── TASK2_MODELNAME_DATE.json          # Individual result for Task2
...
├── all_tasks_MODELNAME_DATE_TIME.json # Combined results for all tasks in the run
├── perplexity_results.txt             # Perplexity results (if requested via older methods)
```

The `all_tasks_MODELNAME_DATE_TIME.json` file includes:
- Top-level `metadata` about the overall `run_eval.sh` script execution.
- A `results` object containing the merged results from each individual task processed during that run. Each task's original metrics and metadata (including its specific duration, sample counts, etc., as generated by `anelm_harness.py`) are preserved under its respective key within this `results` object.

## Implementation Notes

- The harness-based approach (anelm_harness.py) provides better integration with standard benchmarks

- Both systems are designed to be extensible
- ANE models should always be evaluated with batch_size=1 for optimal performance 