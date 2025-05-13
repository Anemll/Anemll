#!/bin/bash

# Script to run evaluation tasks for ANE/CoreML models
# Similar to run_all_evaluations.sh but for ANE models

# Parse command line arguments
PERPLEXITY_TEXT=""
MODEL_PATH=""
OUTPUT_DIR=""
TASKS=""

# Parse command line arguments with both --arg=value and --arg value formats
for arg in "$@"; do
    case $arg in
        --perplexity-text=*)
            PERPLEXITY_TEXT="${arg#*=}"
            shift
            ;;
        --model=*)
            MODEL_PATH="${arg#*=}"
            shift
            ;;
        --output-dir=*)
            OUTPUT_DIR="${arg#*=}"
            shift
            ;;
        --tasks=*)
            TASKS="${arg#*=}"
            shift
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --perplexity-text)
            PERPLEXITY_TEXT="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
    esac
done

# Start timing
start_time=$(date +%s)

# Try to load config from JSON first (more consistent with Python)
CONFIG_JSON="./evaluate/configs/config.json"
if [ -f "$CONFIG_JSON" ]; then
    echo "Loading configuration from $CONFIG_JSON..."
    # Use Python to extract values from JSON
    if command -v python3 >/dev/null 2>&1; then
        DEFAULT_MODEL_PATH=$(python3 -c "import json; f=open('$CONFIG_JSON'); config=json.load(f); print(config.get('default_model_path', '/Users/anemll/Models/ANE/models/latest'))")
        DEFAULT_OUTPUT_DIR=$(python3 -c "import json; f=open('$CONFIG_JSON'); config=json.load(f); print(config.get('default_output_dir', './evaluate/results'))")
        DEFAULT_TASKS=$(python3 -c "import json; f=open('$CONFIG_JSON'); config=json.load(f); print(' '.join(config.get('default_tasks', ['boolq', 'arc_easy', 'hellaswag'])))")
    else
        # Fallback values if Python is not available
        DEFAULT_MODEL_PATH="/Users/anemll/Models/ANE/models/latest"
        DEFAULT_OUTPUT_DIR="./evaluate/results"
        DEFAULT_TASKS="boolq arc_easy hellaswag"
    fi
else
    # Fall back to shell config
    CONFIG_PATH="./evaluate/configs/config.sh"
    if [ -f "$CONFIG_PATH" ]; then
        echo "Loading configuration from $CONFIG_PATH..."
        source "$CONFIG_PATH"
    else
        echo "Warning: No config file found, using default values"
        DEFAULT_MODEL_PATH="/Users/anemll/Models/ANE/models/latest"
        DEFAULT_OUTPUT_DIR="./evaluate/results"
        DEFAULT_TASKS="boolq arc_easy hellaswag"
    fi
fi

# Use config variables or command line args or fallback to default
MODEL_PATH=${MODEL_PATH:-${DEFAULT_MODEL_PATH:-"/Users/anemll/Models/ANE/models/latest"}}
OUTPUT_DIR=${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR:-"./evaluate/results"}}

# Handle task format - parse tasks from quoted string or use defaults
if [ -n "$TASKS" ]; then
    # Remove quotes if they exist
    TASKS="${TASKS//\"/}"
    TASKS="${TASKS//\'/}"
    
    # Split into array based on spaces
    read -ra TASKS_ARR <<< "$TASKS"
else
    # Use default tasks if none specified
    read -ra TASKS_ARR <<< "$DEFAULT_TASKS"
fi

# Set default perplexity text file if requested but not provided
if [ "$PERPLEXITY_TEXT" = "default" ]; then
    PERPLEXITY_TEXT="./evaluate/configs/sample_text.txt"
    # Check if sample_text.txt exists, if not use a default text
    if [ ! -f "$PERPLEXITY_TEXT" ]; then
        echo "$PERPLEXITY_TEXT not found, creating a default text file for perplexity evaluation"
        mkdir -p $(dirname "$PERPLEXITY_TEXT")
        cat > "$PERPLEXITY_TEXT" << EOF
The artificial intelligence revolution has transformed how we live and work. Machine learning models can now perform tasks that were once thought to require human intelligence, from language translation to medical diagnosis. 

Large language models, trained on vast corpora of text, have demonstrated remarkable capabilities in generating coherent and contextually relevant text. However, these models still face challenges such as hallucinations, bias, and ethical concerns.

As researchers continue to push the boundaries of what's possible, it's important to develop robust evaluation methods to measure model performance. Perplexity is one such metric, measuring how well a probability model predicts a sample. Lower perplexity indicates better prediction of the text.

The future of AI depends on addressing these challenges while leveraging the technology's potential to solve important problems in healthcare, education, climate science, and other domains critical to human welfare.
EOF
    fi
fi

# Ensure any remaining tilde paths are expanded
MODEL_PATH=$(eval echo "${MODEL_PATH}")
OUTPUT_DIR=$(eval echo "${OUTPUT_DIR}")

echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Tasks: ${TASKS_ARR[*]}"
if [ -n "$PERPLEXITY_TEXT" ]; then
    echo "Perplexity text file: $PERPLEXITY_TEXT"
    # Add perplexity to tasks if not already there
    if [[ ! " ${TASKS_ARR[*]} " =~ " perplexity " ]]; then
        TASKS_ARR+=("perplexity")
    fi
fi

# Activate virtual environment if it exists
if [ -d "env-anemll" ]; then
    echo "Activating virtual environment..."
    source env-anemll/bin/activate
elif [ -d "anemll-env" ]; then
    echo "Activating virtual environment..."
    source anemll-env/bin/activate
else
    echo "Warning: No virtual environment found. Continuing with system Python."
fi

# Verify required packages
echo "Checking required packages..."
python -c "import coremltools" 2>/dev/null || { 
    echo "Error: coremltools not found. Please install it with: pip install coremltools" 
    exit 1
}

python -c "import datasets" 2>/dev/null || {
    echo "Warning: datasets not found. Installing..."
    pip install datasets
}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create a temporary file to store task times
TIMES_FILE=$(mktemp)

# Build command line arguments for the Python evaluator
CMD_ARGS="--model \"$MODEL_PATH\" --output-dir \"$OUTPUT_DIR\""

# Add tasks
if [ ${#TASKS_ARR[@]} -gt 0 ]; then
    # Use quoted tasks array to preserve spaces and handle task names properly
    TASK_LIST=$(printf "\"%s\" " "${TASKS_ARR[@]}")
    CMD_ARGS="$CMD_ARGS --tasks $TASK_LIST"
fi

# Add perplexity text if provided
if [ -n "$PERPLEXITY_TEXT" ]; then
    CMD_ARGS="$CMD_ARGS --perplexity-text \"$PERPLEXITY_TEXT\""
fi

# Run the evaluation
echo "Running: python evaluate/ane/evaluate_ane.py $CMD_ARGS"
eval "python evaluate/ane/evaluate_ane.py $CMD_ARGS"
EVAL_EXIT_CODE=$?

# Check if evaluation exited with an error
if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "Error: Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

# Extract results for summary if available
if [ -f "$OUTPUT_DIR/summary.json" ]; then
    echo "Results available in $OUTPUT_DIR/summary.json"
    
    # Try to extract total time if jq is available
    if command -v jq >/dev/null 2>&1; then
        total_duration=$(jq '.total_duration' "$OUTPUT_DIR/summary.json")
        echo "Total evaluation time: $total_duration seconds"
    fi
fi

# Calculate end time and duration
end_time=$(date +%s)
duration=$((end_time - start_time))

# Print timing summary
echo ""
echo "===================================="
echo "Evaluation Timing Summary"
echo "===================================="
echo "Total script time: $(($duration / 3600)) hours, $((($duration % 3600) / 60)) minutes, $(($duration % 60)) seconds"
echo "===================================="

# Clean up temporary file
rm -f $TIMES_FILE

echo "All evaluations completed" 