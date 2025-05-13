# ANE/CoreML Model Evaluation

This directory contains tools for evaluating ANE/CoreML models on standard LLM benchmarks using the Apple Neural Engine.

## Quick Start

```bash
# Run evaluation with default settings
./evaluate/ane/run_eval.sh

# Run with a specific model
./evaluate/ane/run_eval.sh --model /path/to/your/model

# Run specific tasks
./evaluate/ane/run_eval.sh --tasks "arc_easy,boolq,hellaswag"

# Run with debug output enabled
./evaluate/ane/run_eval.sh --tasks "hellaswag" --debug

# Run perplexity evaluation with default sample text
./evaluate/ane/run_eval.sh --perplexity default

# Run perplexity evaluation with custom text file
./evaluate/ane/run_eval.sh --perplexity /path/to/text/file.txt

# Run perplexity using wikitext dataset (via harness)
./evaluate/ane/run_eval.sh --perplexity

# Limit the number of samples for faster evaluation
./evaluate/ane/run_eval.sh --tasks "boolq,arc_easy" --limit 10

#perplxity (chunk 200 seems to be good for estimnation)
python ./evaluate/ane/full_perplexity.py --dataset wikitext --chunk-size 200 --max-chunks 10 --subset-size 200
```

## Implementation Overview

The evaluation is implemented using two main components:

1. **ANE_Model** (`evaluate/ane/ane_model.py`): Class that abstracts CoreML model loading, state handling, and token probability computation
2. **ANELM Harness** (`evaluate/ane/anelm_harness.py`): Implementation of the lm-evaluation-harness API for ANE models

These components provide a standardized way to evaluate ANE models within the lm-evaluation-harness framework.

## Available Tasks

The implementation supports most tasks available in lm-evaluation-harness, including:

- `arc_easy`: AI2 Reasoning Challenge (Easy)
- `arc_challenge`: AI2 Reasoning Challenge (Challenge)
- `boolq`: Boolean Questions
- `hellaswag`: HellaSwag 
- `winogrande`: Winogrande
- `openbookqa`: OpenBookQA
- `piqa`: Physical Interaction QA
- `mmlu`: Massive Multitask Language Understanding (configurable subjects)
- And many more from lm-evaluation-harness

## Prerequisites

- Python 3.9 or higher
- CoreML Tools (`pip install coremltools`)
- HuggingFace Transformers (`pip install transformers`)
- lm-evaluation-harness (`pip install lm-evaluation-harness`)
- PyTorch (`pip install torch`)
- Compiled CoreML model(s)

## Detailed Usage

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
- `--apply-chat-template`: Apply chat template to prompts (if available)
- `--debug`: Enable verbose debug output
- `--perplexity [FILE]`: Run perplexity evaluation (with optional text file path)

### Direct Python Usage

You can also use the Python script directly:

```bash
python evaluate/ane/anelm_harness.py \
  --model /path/to/your/model \
  --tasks arc_easy,hellaswag \
  --output-dir results
```

### Debugging Issues

If you encounter issues with the evaluation, the debug flag can help identify problems:

```bash
./evaluate/ane/run_eval.sh --tasks "hellaswag" --debug
```

This will show detailed information about:
- Model loading
- Tokenization
- Token probability computation
- Score normalization
- Multiple-choice selection

## Model Requirements

The evaluation script expects a model directory with the following components:

- Token embedding model (e.g., `embeddings.mlpackage` or `embeddings.mlmodelc`)
- Language model head (e.g., `lm_head.mlpackage` or `lm_head.mlmodelc`)
- FFN model(s) (e.g., `FFN_PF_chunk_01of04.mlpackage` for chunked models)

The script will automatically detect and load compiled models (`.mlmodelc`) if available.

## Results

Results are saved to the specified output directory with the following structure:

```
output_dir/
├── eval_[model_name]_[num_shots]shot_[tasks].json   # Full results in JSON format
```

## Performance Considerations

- ANE models should always be evaluated with `--batch-size 1` to ensure proper serial execution
- For multiple-choice tasks, the implementation uses a normalized log-likelihood approach
- CoreML is configured to use single-threaded execution for optimal performance on ANE 