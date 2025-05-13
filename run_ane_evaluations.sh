#!/bin/bash

# Script to run evaluation tasks for ANE/CoreML models
# Similar to run_all_evaluations.sh but for ANE models

# Parse command line arguments
USE_PROTOTYPE=true
PERPLEXITY_TEXT=""
MODEL_PATH=""
OUTPUT_DIR=""
TASKS=""

for arg in "$@"; do
    case $arg in
        --full)
            USE_PROTOTYPE=false
            shift
            ;;
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
    esac
done

# Start timing
start_time=$(date +%s)

# Load configuration
if [ -f "config.sh" ]; then
    echo "Loading configuration from config.sh..."
    source config.sh
else
    echo "Warning: config.sh not found, using default values"
    DEFAULT_MODEL_PATH="/Volumes/Models/CoreML/llama-3.2-1B"
    DEFAULT_OUTPUT_DIR="./ane_evaluation_results"
    DEFAULT_TASKS="winogrande boolq arc_challenge arc_easy hellaswag openbookqa piqa"
fi

# Use config variables or command line args or fallback to default
MODEL_PATH=${MODEL_PATH:-${DEFAULT_MODEL_PATH:-"/Volumes/Models/CoreML/llama-3.2-1B"}}
OUTPUT_DIR=${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR:-"./ane_evaluation_results"}}
TASKS_ARR=(${TASKS:-${DEFAULT_TASKS:-"winogrande boolq arc_challenge arc_easy hellaswag"}})

# Set default perplexity text file if requested but not provided
if [ "$PERPLEXITY_TEXT" = "default" ]; then
    PERPLEXITY_TEXT="sample_text.txt"
    # Check if sample_text.txt exists, if not use a default text
    if [ ! -f "$PERPLEXITY_TEXT" ]; then
        echo "sample_text.txt not found, creating a default text file for perplexity evaluation"
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
mkdir -p $OUTPUT_DIR

# Create a temporary file to store task times
TIMES_FILE=$(mktemp)

# Run evaluations using the specified approach
if [ "$USE_PROTOTYPE" = true ]; then
    echo "Using prototype implementation (run.py)..."
    
    # Build command line arguments
    CMD_ARGS="--model \"$MODEL_PATH\" --output-dir \"$OUTPUT_DIR\""
    
    # Add tasks
    if [ ${#TASKS_ARR[@]} -gt 0 ]; then
        CMD_ARGS="$CMD_ARGS --tasks ${TASKS_ARR[*]}"
    fi
    
    # Add perplexity text if provided
    if [ -n "$PERPLEXITY_TEXT" ]; then
        CMD_ARGS="$CMD_ARGS --perplexity-text \"$PERPLEXITY_TEXT\""
    fi
    
    # Run the evaluation
    echo "Running: python python/run.py $CMD_ARGS"
    eval "python python/run.py $CMD_ARGS"
    
    # Extract results for summary if available
    if [ -f "$OUTPUT_DIR/summary.json" ]; then
        echo "Results available in $OUTPUT_DIR/summary.json"
        
        # Try to extract total time if jq is available
        if command -v jq >/dev/null 2>&1; then
            total_duration=$(jq '.total_duration' "$OUTPUT_DIR/summary.json")
            echo "Total evaluation time: $total_duration seconds"
        fi
    fi
else
    echo "Using full implementation (evaluate_ane_models.py)..."
    echo "This implementation is not yet available."
    exit 1
    
    # This would be implemented in the future to use the full evaluation system
    # Similar to the prototype but with more advanced features
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