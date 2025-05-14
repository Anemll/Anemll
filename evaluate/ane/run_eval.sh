#!/bin/bash
# Main evaluation script for ANE/CoreML models using anelm_harness.py

# Strict mode
set -euo pipefail

# Helper function to format duration
format_duration() {
    local seconds=$1
    local h=$((seconds / 3600))
    local m=$(((seconds % 3600) / 60))
    local s=$((seconds % 60))
    printf "%02dh:%02dm:%02ds" $h $m $s
}

# Defaults
MODEL_PATH=""
TASK_LIST=""
NUM_SHOTS=0 # Default to 0-shot
BATCH_SIZE=1 # Default and recommended for harness serial execution
OUTPUT_DIR_BASE="evaluate/results"
DEBUG_MODE=false
LIMIT=""
SAFETY_MARGIN=100 # Default safety margin for context length
DOWNLOAD_TIMEOUT=120 # Default download timeout for datasets (seconds)
MAX_RETRIES=5    # Default max retries for dataset downloads
SEED=123

# Store the original command line
OVERALL_CMD_LINE_STR=$(printf "'%s' " "$0" "$@")
SCRIPT_START_TIME_SECONDS=$(date +%s)
DATE_STR=$(date +"%Y%m%d")
TIME_STR_FILENAME=$(date +"%H%M%S")
TIME_STR_METADATA_START=$(date +"%I:%M:%S%p")

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
        MODEL_PATH="$2"
        shift; shift
        ;;
        --tasks)
        TASK_LIST="$2"
        shift; shift
        ;;
        --num-shots)
        NUM_SHOTS="$2"
        shift; shift
        ;;
        --batch-size)
        BATCH_SIZE="$2"
        shift; shift
        ;;
        --output-dir)
        OUTPUT_DIR_BASE="$2"
        shift; shift
        ;;
        --debug)
        DEBUG_MODE=true
        shift
        ;;
        --limit)
        LIMIT="$2"
        shift; shift
        ;;
        --safety-margin)
        SAFETY_MARGIN="$2"
        shift; shift
        ;;
        --download-timeout)
        DOWNLOAD_TIMEOUT="$2"
        shift; shift
        ;;
        --max-retries)
        MAX_RETRIES="$2"
        shift; shift
        ;;
        --seed)
        SEED="$2"
        shift; shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model is required."
    exit 1
fi
if [ -z "$TASK_LIST" ]; then
    echo "Error: --tasks is required."
    exit 1
fi

# Expand ~ in MODEL_PATH
MODEL_PATH=$(eval echo "$MODEL_PATH")

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

# Ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install jq to continue."
    echo "On macOS: brew install jq"
    echo "On Debian/Ubuntu: sudo apt-get install jq"
    exit 1
fi


# Script directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS_SCRIPT="${SCRIPT_DIR}/anelm_harness.py"
OUTPUT_DIR="${OUTPUT_DIR_BASE}" # Use the base name, results will be in e.g. evaluate/results/
mkdir -p "$OUTPUT_DIR"

# Prepare model name for filenames
MODEL_NAME_FULL=$(basename "$MODEL_PATH")
MODEL_NAME_SAFE=$(echo "$MODEL_NAME_FULL" | tr '.' '_' | tr '/' '_') # Replace . and / with _

# Define and initialize combined JSON output file
COMBINED_OUTPUT_FILENAME="${OUTPUT_DIR}/all_tasks_${MODEL_NAME_SAFE}_${DATE_STR}_${TIME_STR_FILENAME}.json"
LIMIT_FOR_METADATA=${LIMIT:-"all"}

echo "{}" | jq \
  --arg overall_cmd_line "$OVERALL_CMD_LINE_STR" \
  --arg model_path "$MODEL_PATH" \
  --arg date_str "$DATE_STR" \
  --arg time_start "$TIME_STR_METADATA_START" \
  --arg limit_val "$LIMIT_FOR_METADATA" \
  --arg num_shots_val "$NUM_SHOTS" \
  --arg batch_size_val "$BATCH_SIZE" \
  --arg seed_val "$SEED" \
  '.metadata.overall_cmd_line = $overall_cmd_line |
   .metadata.model_path = $model_path |
   .metadata.date = $date_str |
   .metadata.time_start_run_eval = $time_start |
   .metadata.limit_per_task = $limit_val |
   .metadata.num_shots = $num_shots_val |
   .metadata.batch_size = $batch_size_val |
   .metadata.seed = $seed_val |
   .results = {}' > "$COMBINED_OUTPUT_FILENAME"


# Display run parameters
echo "=================== ANE MODEL EVALUATION ==================="
echo "Model path:   $MODEL_PATH"
echo "Tasks:        $TASK_LIST"
echo "Num shots:    $NUM_SHOTS"
echo "Batch size:   $BATCH_SIZE (for strict serial execution)"
echo "Output dir:   $OUTPUT_DIR"
[ "$DEBUG_MODE" = true ] && echo "Debug mode:   ENABLED"
[ -n "$LIMIT" ] && echo "Limit:        $LIMIT samples"
echo "Safety margin: $SAFETY_MARGIN tokens"
echo "Seed:         $SEED"
echo "Combined results will be saved to: $COMBINED_OUTPUT_FILENAME"
echo "=========================================================="


echo "Running LM evaluation with ANE_Model on $TASK_LIST..."

# Loop through each task and run evaluation
for TASK_NAME in $TASK_LIST; do
    echo ""
    echo "Running evaluation for task: $TASK_NAME"
    # Define task-specific output path for anelm_harness.py
    TASK_OUTPUT_PATH="${OUTPUT_DIR}/${TASK_NAME}_${MODEL_NAME_SAFE}_${DATE_STR}.json"
    echo "Results will be saved to: $TASK_OUTPUT_PATH by the Python script."

    PYTHON_CMD_ARGS=(
        "--model" "$MODEL_PATH"
        "--tasks" "$TASK_NAME" # Pass one task at a time
        "--num-shots" "$NUM_SHOTS"
        "--batch-size" "$BATCH_SIZE"
        "--safety-margin" "$SAFETY_MARGIN"
        "--download-timeout" "$DOWNLOAD_TIMEOUT"
        "--max-retries" "$MAX_RETRIES"
        "--seed" "$SEED"
        "--output-dir" "$OUTPUT_DIR" # anelm_harness uses this if output-path is not set
        "--output-path" "$TASK_OUTPUT_PATH" # Explicit output path for the Python script
    )

    if [ "$DEBUG_MODE" = true ]; then
        PYTHON_CMD_ARGS+=("--debug")
    fi

    if [ -n "$LIMIT" ]; then
        PYTHON_CMD_ARGS+=("--limit" "$LIMIT")
    fi
    
    # Pass the overall run_eval.sh command line to anelm_harness.py for its metadata
    PYTHON_CMD_ARGS+=("--cmd-line-str" "$OVERALL_CMD_LINE_STR")


    echo "Executing: python $HARNESS_SCRIPT ... (see full command below)"
    if [ "$DEBUG_MODE" = true ]; then
        set -x # Print command before execution
    fi
    python "$HARNESS_SCRIPT" "${PYTHON_CMD_ARGS[@]}"
    PYTHON_EXIT_CODE=$?
    if [ "$DEBUG_MODE" = true ]; then
        set +x
    fi

    if [ $PYTHON_EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_NAME completed. Results saved to $TASK_OUTPUT_PATH by the Python script."
        if [ -f "$TASK_OUTPUT_PATH" ]; then
            # Merge this task's result into the combined JSON file
            # The content of TASK_OUTPUT_PATH is like {"task_name": {metrics_and_metadata...}}
            # We want to add this directly to .results in COMBINED_OUTPUT_FILENAME
            jq --slurpfile task_data "$TASK_OUTPUT_PATH" \
               '.results += $task_data[0]' "$COMBINED_OUTPUT_FILENAME" > tmp_combined.json && mv tmp_combined.json "$COMBINED_OUTPUT_FILENAME"
            echo "Merged results for $TASK_NAME into $COMBINED_OUTPUT_FILENAME"
            # Optionally, remove individual task file: rm "$TASK_OUTPUT_PATH"
        else
            echo "Warning: Task output file $TASK_OUTPUT_PATH not found for merging, though Python script exited successfully."
        fi
    else
        echo "Error: Evaluation for task $TASK_NAME failed with exit code $PYTHON_EXIT_CODE."
        # Optionally, decide if the whole script should exit on first task failure
    fi
done

# Calculate and print total script duration
SCRIPT_END_TIME_SECONDS=$(date +%s)
TOTAL_SCRIPT_DURATION_SECONDS=$((SCRIPT_END_TIME_SECONDS - SCRIPT_START_TIME_SECONDS))
TOTAL_SCRIPT_DURATION_FORMATTED=$(format_duration $TOTAL_SCRIPT_DURATION_SECONDS)
TIME_STR_METADATA_END=$(date +"%I:%M:%S%p")

# Finalize combined JSON with overall script duration and end time
jq --arg end_time "$TIME_STR_METADATA_END" \
   --arg duration "$TOTAL_SCRIPT_DURATION_FORMATTED" \
   '.metadata.time_end_run_eval = $end_time |
    .metadata.total_duration_run_eval = $duration' "$COMBINED_OUTPUT_FILENAME" > tmp_combined.json && mv tmp_combined.json "$COMBINED_OUTPUT_FILENAME"

echo ""
echo "=========================================================="
echo "Evaluation script run complete!"
echo "Combined results for all tasks saved to: $COMBINED_OUTPUT_FILENAME"
echo "Total script duration: $TOTAL_SCRIPT_DURATION_FORMATTED"
echo "==========================================================" 