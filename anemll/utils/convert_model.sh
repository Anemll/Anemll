#!/bin/bash

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Auto-activate a local virtual environment if none is active.
# - If you already activated a venv, we leave it alone.
# - You can override with ANEMLL_VENV (venv dir) or ANEMLL_VENV_ACTIVATE (activate script).
# - Disable with ANEMLL_AUTO_VENV=0.
if [ -z "${VIRTUAL_ENV:-}" ] && [ "${ANEMLL_AUTO_VENV:-1}" != "0" ]; then
    ACTIVATE_CANDIDATES=()
    if [ -n "${ANEMLL_VENV_ACTIVATE:-}" ]; then
        ACTIVATE_CANDIDATES+=("${ANEMLL_VENV_ACTIVATE}")
    fi
    if [ -n "${ANEMLL_VENV:-}" ]; then
        ACTIVATE_CANDIDATES+=("${ANEMLL_VENV}/bin/activate")
    fi
    ACTIVATE_CANDIDATES+=(
        "${PROJECT_ROOT}/env-anemll/bin/activate"
        "${PROJECT_ROOT}/anemll-env/bin/activate"
        "${PROJECT_ROOT}/.venv/bin/activate"
        "${PROJECT_ROOT}/venv/bin/activate"
    )

    for ACTIVATE in "${ACTIVATE_CANDIDATES[@]}"; do
        if [ -f "${ACTIVATE}" ]; then
            # shellcheck disable=SC1090
            source "${ACTIVATE}"
            echo "Activated Python environment: ${ACTIVATE}"
            break
        fi
    done
fi

# Default values
CONTEXT_LENGTH=512
BATCH_SIZE=64
LUT_PART1=""  # No LUT for embeddings
LUT_PART2=4   # FFN and prefill
LUT_PART3=6   # LM head
RESTART_STEP=1
ONLY_STEP=""  # Run only this step if set
PREFIX="llama"  # Default prefix for model names
MODEL_PATH=""
OUTPUT_DIR=""
NUM_CHUNKS=2   # Default number of chunks

# Initialize SKIP_CHECK before parsing arguments
SKIP_CHECK=false
FORCE_MLPROGRAM_COMPILE=false
ALLOW_MISSING_WEIGHTS=false
ARGMAX_IN_MODEL=false
SPLIT_ROTATE=false
FP16_SCALE=""  # FP16 residual scaling for Gemma3 models
CLAMP=""  # Runtime residual clamping for FP16 overflow prevention

# Default converter; may be overridden after parsing config.json
CONVERTER="python3 -m anemll.ane_converter.llama_converter"

# Function to print usage
print_usage() {
    echo "Usage: $0 --model <path_to_model> --output <output_directory> [options]"
    echo "Options:"
    echo "  --model         Path to the model directory (required)"
    echo "  --output        Output directory for converted models (required)"
    echo "  --context       Context length (default: 512)"
    echo "  --batch         Batch size (default: 64)"
    echo "  --lut1          LUT bits for embeddings (default: none)"
    echo "                  Format: 'bits' or 'bits,per_channel' (e.g., '6' or '6,4')"
    echo "                  Use 'bits,0' or 'bits,tensor' for per-tensor quantization"
    echo "                  Default per_channel is 8 if not specified"
    echo "  --lut2          LUT bits for FFN/prefill (default: 4)"
    echo "                  Format: 'bits' or 'bits,per_channel' (e.g., '4,8' or '6,4')"
    echo "                  Use 'bits,0' or 'bits,tensor' for per-tensor quantization"
    echo "                  Default per_channel is 8 if not specified"
    echo "  --lut3          LUT bits for LM head (default: 6)"
    echo "                  Format: 'bits' or 'bits,per_channel' (e.g., '6' or '6,4')"
    echo "                  Use 'bits,0' or 'bits,tensor' for per-tensor quantization"
    echo "                  Default per_channel is 8 if not specified"
    echo "  --no-lut        Disable all LUT quantization (FP16 only)"
    echo "  --restart       Restart from specific step (1-8, default: 1)"
    echo "  --only          Run only specified step and exit (1-8)"
    echo "  --prefix        Prefix for model names (default: llama)"
    echo "  --chunk         Number of chunks to split FFN/prefill (default: 2)"
    echo "  --skip-check    Skip the dependency check step"
    echo "  --force-mlprogram-compile  Force ML Program when compiling .mlpackage models"
    echo "  --allow-missing-weights  Continue conversion even if some weights are missing"
    echo "  --argmax        Compute argmax inside LM head (outputs idx+val pairs instead of logits)"
    echo "  --split-rotate  Combine rotate/non-rotate into two files per chunk (FFN + PF)"
    echo "  --fp16-scale    FP16 residual scaling for Gemma3 (e.g., 'auto', '0.1875')"
    echo "                  Recommended: 0.48 for 270M, 0.82 for 1B, 0.1875 for 4B QAT"
    echo "  --clamp         Runtime residual clamping value (e.g., 55000)"
    echo "                  Alternative to --fp16-scale for overflow prevention"
    echo ""
    echo "Examples:"
    echo "  # Use default per_channel (8) for all parts"
    echo "  $0 --model ./model --output ./output --lut2 4 --lut3 6"
    echo ""
    echo "  # Specify custom per_channel values"
    echo "  $0 --model ./model --output ./output --lut2 4,16 --lut3 6,4"
    echo ""
    echo "  # Use per-tensor quantization (no channel grouping)"
    echo "  $0 --model ./model --output ./output --lut2 4,0 --lut3 6,0"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --context)
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lut1)
            LUT_PART1="$2"
            shift 2
            ;;
        --lut2)
            LUT_PART2="$2"
            shift 2
            ;;
        --lut3)
            LUT_PART3="$2"
            shift 2
            ;;
        --restart)
            RESTART_STEP="$2"
            shift 2
            ;;
        --only)
            ONLY_STEP="$2"
            shift 2
            ;;
        --chunk)
            NUM_CHUNKS="$2"
            shift 2
            ;;
        --skip-check)
            SKIP_CHECK=true
            shift
            ;;
        --force-mlprogram-compile)
            FORCE_MLPROGRAM_COMPILE=true
            shift
            ;;
        --allow-missing-weights)
            ALLOW_MISSING_WEIGHTS=true
            shift
            ;;
        --argmax)
            ARGMAX_IN_MODEL=true
            shift
            ;;
        --split-rotate)
            SPLIT_ROTATE=true
            shift
            ;;
        --fp16-scale)
            FP16_SCALE="$2"
            shift 2
            ;;
        --clamp)
            CLAMP="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Model path and output directory are required"
    print_usage
fi

# Allow continuing when some weights are missing (use with caution)
if [ "$ALLOW_MISSING_WEIGHTS" = true ]; then
    export ANEMLL_ALLOW_MISSING_WEIGHTS=1
    echo "Warning: ANEMLL_ALLOW_MISSING_WEIGHTS=1 (missing weights will be ignored)"
fi

# Check if MODEL_PATH looks like a HuggingFace model name (e.g., "google/gemma-3-1b-it")
# HuggingFace names contain "/" but don't start with "/" or "~" or "."
if [[ "$MODEL_PATH" == *"/"* ]] && [[ ! "$MODEL_PATH" =~ ^[/~.] ]]; then
    echo "Detected HuggingFace model name: $MODEL_PATH"

    # Convert model name to cache path format (e.g., google/gemma-3-1b-it -> models--google--gemma-3-1b-it)
    HF_CACHE_NAME="models--$(echo "$MODEL_PATH" | sed 's|/|--|g')"
    HF_CACHE_DIR="$HOME/.cache/huggingface/hub/$HF_CACHE_NAME"

    if [ -d "$HF_CACHE_DIR/snapshots" ]; then
        # Model exists in cache, find the latest snapshot
        SNAPSHOT_DIR=$(find "$HF_CACHE_DIR/snapshots" -maxdepth 1 -type d | tail -1)
        if [ -d "$SNAPSHOT_DIR" ] && [ "$SNAPSHOT_DIR" != "$HF_CACHE_DIR/snapshots" ]; then
            echo "Found cached model at: $SNAPSHOT_DIR"
            MODEL_PATH="$SNAPSHOT_DIR"
        else
            echo "Cache directory exists but no snapshots found. Downloading model..."
            huggingface-cli download "$MODEL_PATH"
            SNAPSHOT_DIR=$(find "$HF_CACHE_DIR/snapshots" -maxdepth 1 -type d | tail -1)
            MODEL_PATH="$SNAPSHOT_DIR"
        fi
    else
        echo "Model not in cache. Downloading from HuggingFace..."
        huggingface-cli download "$MODEL_PATH"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to download model from HuggingFace"
            echo "Make sure you're logged in: huggingface-cli login"
            exit 1
        fi
        SNAPSHOT_DIR=$(find "$HF_CACHE_DIR/snapshots" -maxdepth 1 -type d | tail -1)
        MODEL_PATH="$SNAPSHOT_DIR"
    fi

    echo "Using model path: $MODEL_PATH"
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory does not exist: $MODEL_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create output directory"
        exit 1
    fi
fi

# Convert paths to absolute paths
MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)" || {
    # If output directory doesn't exist, get absolute path another way
    OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")"
}

# Detect architecture from config.json
CONFIG_FILE="$MODEL_PATH/config.json"
if [ -f "$CONFIG_FILE" ]; then
    ARCH=$(jq -r '.model_type // (.architectures[0] // "")' "$CONFIG_FILE" | tr '[:upper:]' '[:lower:]')
    # Check for Qwen2 (which is Qwen 2.5) or Qwen2ForCausalLM architecture
    if [[ "$ARCH" == "qwen2" ]] || [[ "$ARCH" == *"qwen2forcausallm"* ]]; then
        CONVERTER="python3 -m anemll.ane_converter.qwen2_5_converter"
        # Use "qwen25" as default prefix for Qwen 2.5 models unless explicitly set
        if [ "$PREFIX" = "llama" ]; then
            PREFIX="qwen25"
        fi
    elif [[ "$ARCH" == qwen* ]]; then
        CONVERTER="python3 -m anemll.ane_converter.qwen_converter"
        # Use "qwen" as default prefix for Qwen models unless explicitly set
        if [ "$PREFIX" = "llama" ]; then
            PREFIX="qwen"
        fi
    elif [[ "$ARCH" == "gemma3_text"* ]] || [[ "$ARCH" == "gemma3"* ]]; then
        CONVERTER="python3 -m anemll.ane_converter.gemma3_converter"
        # Use "gemma3" as default prefix for Gemma3 models unless explicitly set
        if [ "$PREFIX" = "llama" ]; then
            PREFIX="gemma3"
        fi
    else
        CONVERTER="python3 -m anemll.ane_converter.llama_converter"
    fi

    # Extract sliding_window from config.json (used for Gemma3 rotation checks)
    # Gemma3 models have it nested in .text_config.sliding_window
    # Default: 512 for 1B models, 1024 for 4B models
    SLIDING_WINDOW=$(jq -r '.text_config.sliding_window // .sliding_window // 512' "$CONFIG_FILE")
    echo "Detected sliding_window: $SLIDING_WINDOW"
fi

# Step 0: Check dependencies
if [ "$SKIP_CHECK" = false ]; then
    "$SCRIPT_DIR/check_dependencies.sh" --model "$MODEL_PATH" --output "$OUTPUT_DIR" "$@"
    if [ $? -ne 0 ]; then
        echo "Dependency check failed. Aborting."
        exit 1
    fi
fi

# Create initial meta.yaml with conversion config (for progress monitoring)
INITIAL_META="$OUTPUT_DIR/meta_progress.yaml"
cat > "$INITIAL_META" << EOF
# Conversion in progress - this file is for monitoring only
# Final meta.yaml will be created at step 7
conversion:
  status: in_progress
  start_time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
  model_path: $MODEL_PATH
  output_dir: $OUTPUT_DIR
  context_length: $CONTEXT_LENGTH
  batch_size: $BATCH_SIZE
  num_chunks: $NUM_CHUNKS
  prefix: $PREFIX
  architecture: ${ARCH:-unknown}
  lut_part1: ${LUT_PART1:-none}
  lut_part2: ${LUT_PART2:-none}
  lut_part3: ${LUT_PART3:-none}
  fp16_scale: ${FP16_SCALE:-none}
  argmax: $ARGMAX_IN_MODEL
  split_rotate: $SPLIT_ROTATE
steps:
  - name: embeddings
    part: 1
    status: pending
  - name: lm_head
    part: 3
    status: pending
  - name: ffn
    part: 2
    status: pending
  - name: prefill
    part: 2_prefill
    status: pending
  - name: ffn_rotate
    part: 2_rotate
    status: pending
    gemma3_only: true
  - name: prefill_rotate
    part: 2_prefill_rotate
    status: pending
    gemma3_only: true
  - name: combine
    part: 5
    status: pending
  - name: compile
    part: 6
    status: pending
  - name: tokenizer
    part: 7
    status: pending
  - name: test
    part: 8
    status: pending
EOF
echo "Created progress tracking file: $INITIAL_META"

# Function to run step if restart_step is less than or equal to step number
run_step() {
    local step=$1
    local description=$2
    local cmd=$3
    
    if [ $RESTART_STEP -le $step ]; then
        # Skip if ONLY_STEP is set and doesn't match current step
        if [ ! -z "$ONLY_STEP" ] && [ "$ONLY_STEP" != "$step" ]; then
            return
        fi
        
        echo "Step $step: $description"
        bash -c "$cmd"
        if [ $? -ne 0 ]; then
            echo "Error in step $step: $description"
            exit 1
        fi
        # Exit after running the only step if specified
        if [ ! -z "$ONLY_STEP" ] && [ "$ONLY_STEP" = "$step" ]; then
            echo "Completed step $step (--only mode)"
            exit 0
        fi
    else
        # Only show skip message if not in ONLY_STEP mode
        if [ -z "$ONLY_STEP" ]; then
            echo "Skipping step $step: $description"
        fi
    fi
}

# Prepare FP16 scaling parameter for Gemma3 models (must be before any converter steps)
FP16_SCALE_PARAM=""
if [ ! -z "$FP16_SCALE" ]; then
    FP16_SCALE_PARAM="--fp16-scale $FP16_SCALE"
    echo "Using FP16 residual scaling: $FP16_SCALE"
fi

# Prepare clamp parameter for runtime overflow prevention
CLAMP_PARAM=""
if [ ! -z "$CLAMP" ]; then
    CLAMP_PARAM="--clamp $CLAMP"
    echo "Using residual clamping at: $CLAMP"
fi

# Step 1: Convert Embeddings (Part 1)
LUT1_PARAM=""
if [ ! -z "$LUT_PART1" ]; then
    LUT1_PARAM="--lut $LUT_PART1"
fi

if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "1" ]; then
    run_step 1 "Converting Embeddings" "$CONVERTER \
        --part 1 \
        $LUT1_PARAM \
        $FP16_SCALE_PARAM $CLAMP_PARAM \
        --context-length $CONTEXT_LENGTH \
        --batch-size $BATCH_SIZE \
        --context-length $CONTEXT_LENGTH \
        --batch-size $BATCH_SIZE \
        --prefix \"$PREFIX\" \
        --model \"$MODEL_PATH\" \
        --output \"$OUTPUT_DIR\""
else
    echo "Skipping step 1: Converting Embeddings"
fi

# Step 2: Convert LM Head (Part 3)
LUT3_PARAM=""
if [ ! -z "$LUT_PART3" ]; then
    LUT3_PARAM="--lut $LUT_PART3"
fi

# Prepare argmax parameter for LM head
ARGMAX_PARAM=""
if [ "$ARGMAX_IN_MODEL" = true ]; then
    ARGMAX_PARAM="--argmax"
fi

if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "2" ]; then
    run_step 2 "Converting LM Head" "$CONVERTER \
        --part 3 \
        $LUT3_PARAM \
        $ARGMAX_PARAM \
        $FP16_SCALE_PARAM $CLAMP_PARAM \
        --context-length $CONTEXT_LENGTH \
        --context-length $CONTEXT_LENGTH \
        --prefix \"$PREFIX\" \
        --model \"$MODEL_PATH\" \
        --output \"$OUTPUT_DIR\""
else
    echo "Skipping step 2: Converting LM Head"
fi

# Step 3: Convert FFN (Part 2)
LUT2_PARAM=""
if [ ! -z "$LUT_PART2" ]; then
    LUT2_PARAM="--lut $LUT_PART2"
fi

if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "3" ]; then
    run_step 3 "Converting FFN" "$CONVERTER \
        --part 2 \
        $LUT2_PARAM \
        $FP16_SCALE_PARAM $CLAMP_PARAM \
        --chunk $NUM_CHUNKS \
        --context-length $CONTEXT_LENGTH \
        --batch-size $BATCH_SIZE \
        --prefix \"$PREFIX\" \
        --model \"$MODEL_PATH\" \
        --output \"$OUTPUT_DIR\""
else
    echo "Skipping step 3: Converting FFN"
fi

if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "4" ]; then
    run_step 4 "Converting Prefill" "$CONVERTER \
        --part 2_prefill \
        $LUT2_PARAM \
        $FP16_SCALE_PARAM $CLAMP_PARAM \
        --chunk $NUM_CHUNKS \
        --context-length $CONTEXT_LENGTH \
        --batch-size $BATCH_SIZE \
        --prefix \"$PREFIX\" \
        --model \"$MODEL_PATH\" \
        --output \"$OUTPUT_DIR\""
else
    echo "Skipping step 4: Converting Prefill"
fi

# Step 4a: Convert FFN Rotate (Part 2_rotate) - Gemma3 only
# For Gemma3 models with context > sliding_window (512), we need rotation versions
if [[ "$ARCH" == "gemma3_text"* ]] || [[ "$ARCH" == "gemma3"* ]]; then
    if [ $CONTEXT_LENGTH -gt $SLIDING_WINDOW ]; then
        if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "4" ]; then
            run_step 4 "Converting FFN Rotate" "$CONVERTER \
                --part 2_rotate \
                $LUT2_PARAM \
                $FP16_SCALE_PARAM $CLAMP_PARAM \
                --chunk $NUM_CHUNKS \
                --context-length $CONTEXT_LENGTH \
                --batch-size $BATCH_SIZE \
                --prefix \"$PREFIX\" \
                --model \"$MODEL_PATH\" \
                --output \"$OUTPUT_DIR\""
        fi

        # Step 4b: Convert Prefill Rotate (Part 2_prefill_rotate) - Gemma3 only
        if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "4" ]; then
            run_step 4 "Converting Prefill Rotate" "$CONVERTER \
                --part 2_prefill_rotate \
                $LUT2_PARAM \
                $FP16_SCALE_PARAM $CLAMP_PARAM \
                --chunk $NUM_CHUNKS \
                --context-length $CONTEXT_LENGTH \
                --batch-size $BATCH_SIZE \
                --prefix \"$PREFIX\" \
                --model \"$MODEL_PATH\" \
                --output \"$OUTPUT_DIR\""
        fi
    fi
fi

# Step 5: Combine Models
# For Gemma3 with context > 512, use --gemma3 flag to combine 4 functions
GEMMA3_FLAG=""
SPLIT_ROTATE_FLAG=""
if [[ "$ARCH" == "gemma3_text"* ]] || [[ "$ARCH" == "gemma3"* ]]; then
    if [ $CONTEXT_LENGTH -gt $SLIDING_WINDOW ]; then
        GEMMA3_FLAG="--gemma3"
    fi
fi
if [ "$SPLIT_ROTATE" = true ]; then
    SPLIT_ROTATE_FLAG="--split-rotate"
    GEMMA3_FLAG=""
fi

if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "5" ]; then
    if [ ! -z "$LUT_PART2" ]; then
        run_step 5 "Combining Models" "python3 \"$PROJECT_ROOT/anemll/utils/combine_models.py\" \
            --chunk $NUM_CHUNKS \
            $LUT2_PARAM \
            $SPLIT_ROTATE_FLAG \
            $GEMMA3_FLAG \
            --prefix \"$PREFIX\" \
            --input \"$OUTPUT_DIR\" \
            --output \"$OUTPUT_DIR\""
    else
        run_step 5 "Combining Models" "python3 \"$PROJECT_ROOT/anemll/utils/combine_models.py\" \
            --chunk $NUM_CHUNKS \
            $SPLIT_ROTATE_FLAG \
            $GEMMA3_FLAG \
            --prefix \"$PREFIX\" \
            --input \"$OUTPUT_DIR\" \
            --output \"$OUTPUT_DIR\""
    fi
else
    echo "Skipping step 5: Combining Models"
fi

# Step 6: Compile Models - Always run compilation for all parts that have LUT specified
FORCE_MLPROG_FLAG=""
if [ "$FORCE_MLPROGRAM_COMPILE" = true ]; then
    FORCE_MLPROG_FLAG="--force-mlprogram"
fi
run_step 6 "Compiling Models Part 1" "python3 \"$PROJECT_ROOT/anemll/utils/compile_models.py\" 1 ${LUT_PART1:+--lut $LUT_PART1} $FORCE_MLPROG_FLAG --prefix \"$PREFIX\" --input \"$OUTPUT_DIR\" --output \"$OUTPUT_DIR\""
run_step 6 "Compiling Models Part 3" "python3 \"$PROJECT_ROOT/anemll/utils/compile_models.py\" 3 ${LUT_PART3:+--lut $LUT_PART3} $FORCE_MLPROG_FLAG --prefix \"$PREFIX\" --input \"$OUTPUT_DIR\" --output \"$OUTPUT_DIR\""
if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "6" ]; then
    run_step 6 "Compiling Models Part 2" "python3 \"$PROJECT_ROOT/anemll/utils/compile_models.py\" 2 ${LUT_PART2:+--lut $LUT_PART2} $FORCE_MLPROG_FLAG $SPLIT_ROTATE_FLAG --chunk $NUM_CHUNKS --prefix \"$PREFIX\" --input \"$OUTPUT_DIR\" --output \"$OUTPUT_DIR\""
fi

# Step 7: Copy tokenizer files and create meta.yaml
if [ "$MODEL_PATH" != "$OUTPUT_DIR" ]; then
    # Detect HuggingFace cache path and extract proper model name
    if [[ "$MODEL_PATH" =~ \.cache/huggingface/hub/models--([^/]+)--([^/]+)/snapshots/ ]]; then
        # Extract org and model name from HF cache path
        HF_ORG="${BASH_REMATCH[1]}"
        HF_MODEL="${BASH_REMATCH[2]}"
        MODEL_NAME="${HF_ORG}-${HF_MODEL}"
    else
        MODEL_NAME=$(basename "$MODEL_PATH")
    fi
    run_step 7 "Copying tokenizer files and creating meta.yaml" "
        # Copy tokenizer files if they exist
        (cp \"$MODEL_PATH/tokenizer.json\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/tokenizer_config.json\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/tokenizer.model\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/vocab.json\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/merges.txt\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/chat_template.jinja\" \"$OUTPUT_DIR/\" || true) && \
        
        # Create config.json if it doesn't exist
        if [ ! -f \"$OUTPUT_DIR/config.json\" ]; then
            echo \"Creating config.json for iOS tokenizer...\" && \
            if [[ \"$ARCH\" == \"qwen2\" ]] || [[ \"$ARCH\" == *\"qwen2forcausallm\"* ]]; then
                # Create Qwen 2.5-specific config.json
                cat > \"$OUTPUT_DIR/config.json\" <<'EOF_CONFIG'
{
  \"tokenizer_class\": \"Qwen2Tokenizer\",
  \"model_type\": \"qwen2\"
}
EOF_CONFIG
            elif [[ \"$ARCH\" == qwen* ]]; then
                # Create Qwen-specific config.json
                cat > \"$OUTPUT_DIR/config.json\" <<'EOF_CONFIG'
{
  \"tokenizer_class\": \"Qwen2Tokenizer\",
  \"model_type\": \"qwen3\"
}
EOF_CONFIG
            else
                python3 -m anemll.ane_converter.create_config_json --output \"$OUTPUT_DIR/config.json\"
            fi
        fi && \
        
        # Create meta.yaml with correct LUT values based on actual file existence
        ARGMAX_META_FLAG=\"\"
        if [ \"$ARGMAX_IN_MODEL\" = true ]; then
            ARGMAX_META_FLAG=\"--argmax\"
        fi
        SPLIT_ROTATE_META_FLAG=\"\"
        if [ \"$SPLIT_ROTATE\" = true ]; then
            SPLIT_ROTATE_META_FLAG=\"--split-rotate\"
        fi
        # Add sliding_window flag for Gemma3 models
        SLIDING_WINDOW_FLAG=\"\"
        if [[ \"$ARCH\" == \"gemma3\"* ]] && [ -n \"$SLIDING_WINDOW\" ]; then
            SLIDING_WINDOW_FLAG=\"--sliding-window $SLIDING_WINDOW\"
        fi
        python3 \"$PROJECT_ROOT/anemll/utils/generate_meta_yaml.py\" \
            \"$MODEL_NAME\" \"$CONTEXT_LENGTH\" \"$BATCH_SIZE\" \
            \"${LUT_PART1:-none}\" \"${LUT_PART2:-none}\" \"${LUT_PART3:-none}\" \
            $NUM_CHUNKS \"$PREFIX\" \"$ARCH\" \"$OUTPUT_DIR\" \$ARGMAX_META_FLAG \$SPLIT_ROTATE_META_FLAG \$SLIDING_WINDOW_FLAG
    "
fi


# Step 8: Test with chat.py
run_step 8 "Testing with chat.py" "python3 \"$PROJECT_ROOT/tests/chat.py\" \
    --meta \"$OUTPUT_DIR/meta.yaml\" \
    --prompt \"Who are you ?\" \
    --max-tokens 100"

# Print chat.py command for reference
echo -e "\nTo chat with the model, use:"
echo -e "\nOption 1 - Using meta.yaml (recommended):"
echo "python3 $PROJECT_ROOT/tests/chat.py \\"
echo "    --meta \"$OUTPUT_DIR/meta.yaml\""
echo -e "\nOr for full conversation mode:"
echo "python3 $PROJECT_ROOT/tests/chat_full.py \\"
echo "    --meta \"$OUTPUT_DIR/meta.yaml\""

echo -e "\nOption 2 - Manual configuration:"
EMBEDDINGS_NAME="${PREFIX}_embeddings${LUT_PART1:+_lut$LUT_PART1}"
LMHEAD_NAME="${PREFIX}_lm_head${LUT_PART3:+_lut$LUT_PART3}"

echo "python3 $PROJECT_ROOT/tests/chat.py \\"
echo "    --embed $EMBEDDINGS_NAME \\"
echo "    --lmhead $LMHEAD_NAME \\"
if [ "$SPLIT_ROTATE" = true ]; then
    FFN_BASE="${PREFIX}_FFN_PF${LUT_PART2:+_lut$LUT_PART2}"
    echo "    --ffn ${FFN_BASE}_chunk_01of$(printf "%02d" $NUM_CHUNKS) \\"
    echo "    --pf ${FFN_BASE}_chunk_01of$(printf "%02d" $NUM_CHUNKS)_rot \\"
    echo "    --split-rotate \\"
else
    FFN_BASE="${PREFIX}_FFN_PF${LUT_PART2:+_lut$LUT_PART2}"
    echo "    --ffn ${FFN_BASE}_chunk_01of$(printf "%02d" $NUM_CHUNKS) \\"
fi
echo "    --tokenizer \"$OUTPUT_DIR\" \\"
echo "    --context-length $CONTEXT_LENGTH \\"
echo "    --d \"$OUTPUT_DIR\""

echo -e "\nOption 3 - Using Swift CLI (requires building anemll-swift-cli):"
echo "cd $PROJECT_ROOT/anemll-swift-cli && swift run anemllcli \\"
echo "    --meta \"$OUTPUT_DIR/meta.yaml\""

echo -e "\nTo prepare model for HuggingFace upload:"
echo "# For standard distribution:"
echo "./anemll/utils/prepare_hf.sh --input \"$OUTPUT_DIR\""
echo ""
echo "# For iOS-ready version (with unzipped MLMODELC files):"
echo "./anemll/utils/prepare_hf.sh --input \"$OUTPUT_DIR\" --ios"

echo -e "\nConversion completed successfully!" 
