#!/bin/bash
# run_new_eval.sh - Run ANE model evaluation with the new harness implementation
# Uses the abstracted ANE_Model class for better state handling

# Set default values
MODEL_PATH="/Users/anemll/Models/ANE/anemll-Llama-3.2-1B-FP16-b64-ctx1024"
TASKS="hellaswag"
NUM_SHOTS=0
BATCH_SIZE=1  # Always use batch_size=1 for ANE models
OUTPUT_DIR="results"
LIMIT=""
MAX_TOKENS=""
CHAT_TEMPLATE=""
SEED=123

print_header() {
    echo "=================== ANE MODEL EVALUATION (NEW) ==================="
    echo "Model path:   ${MODEL_PATH}"
    echo "Tasks:        ${TASKS}"
    echo "Num shots:    ${NUM_SHOTS}"
    echo "Batch size:   ${BATCH_SIZE} (for strict serial execution)"
    echo "Output dir:   ${OUTPUT_DIR}"
    if [ -n "$LIMIT" ]; then
        echo "Example limit: ${LIMIT}"
    fi
    echo "==========================================================="
}

# Parse command line arguments
while (( "$#" )); do
    case "$1" in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --num-shots)
            NUM_SHOTS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --apply-chat-template)
            CHAT_TEMPLATE="--apply-chat-template"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model PATH             Path to model directory"
            echo "  --tasks LIST             Comma-separated list of tasks to evaluate"
            echo "  --num-shots N            Number of few-shot examples (default: 0)"
            echo "  --batch-size N           Batch size for evaluation (default: 1, recommended for ANE)"
            echo "  --output-dir DIR         Directory to save results (default: results)"
            echo "  --limit N                Limit number of examples per task"
            echo "  --max-tokens N           Maximum number of tokens to generate"
            echo "  --seed N                 Random seed (default: 123)"
            echo "  --apply-chat-template    Apply chat template to prompts"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
print_header

# Build command
CMD="python anelm_harness.py --model \"${MODEL_PATH}\" --tasks ${TASKS} --batch-size ${BATCH_SIZE} --output-dir \"${OUTPUT_DIR}\" --num-shots ${NUM_SHOTS} --seed ${SEED}"

if [ -n "$LIMIT" ]; then
    CMD="${CMD} --limit ${LIMIT}"
fi

if [ -n "$MAX_TOKENS" ]; then
    CMD="${CMD} --max-tokens ${MAX_TOKENS}"
fi

if [ -n "$CHAT_TEMPLATE" ]; then
    CMD="${CMD} ${CHAT_TEMPLATE}"
fi

# Run evaluation
echo "Running evaluation command:"
echo "${CMD}"
echo "==========================================================="
eval "${CMD}"

# Print completion message
echo "==========================================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"