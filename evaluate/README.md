# ANE/CoreML Model Evaluation

This directory contains tools for evaluating ANE/CoreML models on standard LLM benchmarks and performance metrics.

## Directory Structure

```
evaluate/
├── ane/                  # ANE-specific evaluation code
│   └── evaluate_ane.py   # Main evaluator script for ANE models
├── configs/              # Configuration files
│   ├── config.sh         # User configuration (create from template)
│   ├── config.sh.template # Configuration template
│   └── sample_text.txt   # Sample text for perplexity evaluation
├── results/              # Evaluation results (created during evaluation)
└── run_ane_evaluations.sh # Main runner script
```

## Quick Start

```bash
# Run evaluation with default settings
./evaluate/run_ane_evaluations.sh

# Run with a specific model
./evaluate/run_ane_evaluations.sh --model=/path/to/model

# Run specific tasks
./evaluate/run_ane_evaluations.sh --tasks="arc_easy boolq hellaswag"

# Run with perplexity evaluation
./evaluate/run_ane_evaluations.sh --perplexity-text=default
```

## Configuration

Copy the template configuration file and customize it:

```bash
cp evaluate/configs/config.sh.template evaluate/configs/config.sh
# Edit config.sh with your custom settings
```

## Available Tasks

- `arc_easy`: AI2 Reasoning Challenge (Easy)
- `arc_challenge`: AI2 Reasoning Challenge (Challenge)
- `boolq`: Boolean Questions
- `hellaswag`: HellaSwag 
- `winogrande`: Winogrande
- `openbookqa`: OpenBookQA
- `piqa`: Physical Interaction QA
- `perplexity`: Perplexity evaluation

## Prerequisites

- Python 3.9, 3.10, or 3.11
- CoreML Tools
- HuggingFace Datasets
- Compiled CoreML model(s)

## Detailed Usage

### Command Line Arguments

- `--model=PATH`: Path to the model directory
- `--output-dir=PATH`: Directory to save results
- `--tasks=TASKS`: Space-separated list of tasks to run
- `--perplexity-text=PATH`: Path to text file for perplexity evaluation (or "default")

### Model Requirements

The evaluation script expects the following files in the model directory:

- `embeddings.mlpackage`: Token embedding model
- `lm_head.mlpackage`: Language model head
- `llama_FFN_PF_chunk_01of01.mlpackage`: FFN model (or multiple chunks)

## Architecture

The evaluation system is based on the following components:

1. **ANE Evaluator Script** (`evaluate_ane.py`): Python script for evaluating ANE/CoreML models
2. **Bash Runner Script** (`run_ane_evaluations.sh`): Shell script for running evaluations
3. **Configuration** (`configs/config.sh`): Settings for model paths and tasks

## Extending

To add a new benchmark:

1. Add a new evaluation method to the `ANEModelEvaluator` class in `evaluate_ane.py`
2. Update the `run_evaluations` method to handle the new task
3. Add the new task to the documentation and configuration

## Results

Results are saved to the specified output directory with the following structure:

```
results/
├── perplexity_results.txt        # Perplexity evaluation results
├── arc_easy_results.json         # ARC Easy results
├── boolq_results.json            # BoolQ results
├── ...                           # Other task results
└── summary.json                  # Overall summary of all results
```

## Implementation Notes

- The system is designed to reuse existing code from the `tests` directory
- The evaluator avoids reloading models between tasks for better performance
- Results can be compared between different quantization strategies
- The system is designed to be extensible for adding new benchmarks 