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
NUM_CHUNKS=2   # Default number of chunks (or 'auto')
MAX_CHUNK_MB=950  # Max chunk size in MB for --chunk auto

# Initialize SKIP_CHECK before parsing arguments
SKIP_CHECK=false
FORCE_MLPROGRAM_COMPILE=false
ALLOW_MISSING_WEIGHTS=false
ARGMAX_IN_MODEL=false
SPLIT_ROTATE=false
FP16_SCALE=""  # FP16 residual scaling for Gemma3 models
CLAMP=""  # Runtime residual clamping for FP16 overflow prevention
DYNAMIC_PREFILL_SLICE=true
STATIC_PREFILL_SLICE=false
SINGLE_CACHE=false
CHUNK_NO=""
SKIP_ANEMLL_DEDUP=false

# Default converter; may be overridden after parsing config.json
CONVERTER="python3 -m anemll.ane_converter.llama_converter"

# Capture exact invocation for troubleshooting metadata.
ORIGINAL_ARGS=("$@")
CONVERSION_COMMAND="$(printf '%q ' "$0" "${ORIGINAL_ARGS[@]}")"
CONVERSION_COMMAND="${CONVERSION_COMMAND% }"

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
    echo "  --dynamic-prefill-slice  Use dynamic slicing for prefill KV writes (default ON)"
    echo "  --static-prefill-slice   Disable dynamic slicing for prefill KV writes"
    echo "  --single-cache           Use unified KV cache (Gemma3 only; disables rotate)"
    echo "  --chunk-no      Convert only this chunk number (1-based) for chunked parts"
    echo "                  Format: 'bits' or 'bits,per_channel' (e.g., '6' or '6,4')"
    echo "                  Use 'bits,0' or 'bits,tensor' for per-tensor quantization"
    echo "                  Default per_channel is 8 if not specified"
    echo "  --no-lut        Disable all LUT quantization (FP16 only)"
    echo "  --restart       Restart from specific step (1-8, default: 1)"
    echo "  --only          Run only specified step and exit (1-8)"
    echo "  --prefix        Prefix for model names (default: llama)"
    echo "  --chunk         Number of chunks or 'auto' (default: 2)"
    echo "  --max-chunk-mb  Max chunk size in MB for --chunk auto (default: 950)"
    echo "  --skip-check    Skip the dependency check step"
    echo "  --force-mlprogram-compile  Force ML Program when compiling .mlpackage models"
    echo "  --allow-missing-weights  Continue conversion even if some weights are missing"
    echo "  --argmax        Compute argmax inside LM head (outputs idx+val pairs instead of logits)"
    echo "  --split-rotate  Combine rotate/non-rotate into two files per chunk (FFN + PF)"
    echo "  --fp16-scale    FP16 residual scaling for Gemma3 (e.g., 'auto', '0.1875')"
    echo "                  Recommended: 0.48 for 270M, 0.82 for 1B, 0.1875 for 4B QAT"
    echo "  --clamp         Runtime residual clamping value (e.g., 55000)"
    echo "                  Alternative to --fp16-scale for overflow prevention"
    echo "  --skip-anemll-dedup Disable anemll-dedup weight dedup in combine step (default: enabled)"
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
        --max-chunk-mb)
            MAX_CHUNK_MB="$2"
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
        --dynamic-prefill-slice)
            DYNAMIC_PREFILL_SLICE=true
            shift
            ;;
        --static-prefill-slice)
            DYNAMIC_PREFILL_SLICE=false
            STATIC_PREFILL_SLICE=true
            shift
            ;;
        --single-cache)
            SINGLE_CACHE=true
            shift
            ;;
        --chunk-no)
            CHUNK_NO="$2"
            shift 2
            ;;
        --skip-anemll-dedup)
            SKIP_ANEMLL_DEDUP=true
            shift
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
ARCH="llama"
SLIDING_WINDOW=""
HAS_SWA=false
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

    # Detect whether this architecture actually uses sliding-window attention.
    USE_SLIDING_WINDOW=$(jq -r '.text_config.use_sliding_window // .use_sliding_window // empty' "$CONFIG_FILE")
    SLIDING_WINDOW_RAW=$(jq -r '.text_config.sliding_window // .sliding_window // empty' "$CONFIG_FILE")
    if [[ "$ARCH" == "gemma3_text"* ]] || [[ "$ARCH" == "gemma3"* ]]; then
        HAS_SWA=true
        if [[ "$SLIDING_WINDOW_RAW" =~ ^[0-9]+$ ]] && [ "$SLIDING_WINDOW_RAW" -gt 0 ]; then
            SLIDING_WINDOW="$SLIDING_WINDOW_RAW"
        else
            SLIDING_WINDOW=512
        fi
    elif [[ "$USE_SLIDING_WINDOW" == "true" ]] && [[ "$SLIDING_WINDOW_RAW" =~ ^[0-9]+$ ]] && [ "$SLIDING_WINDOW_RAW" -gt 0 ]; then
        HAS_SWA=true
        SLIDING_WINDOW="$SLIDING_WINDOW_RAW"
    fi

    if [ "$HAS_SWA" = true ]; then
        echo "Detected sliding_window: $SLIDING_WINDOW (SWA enabled)"
    else
        echo "Detected SWA: disabled"
    fi
fi

# Optional argmax layout metadata for runtimes that need exact local->global mapping.
VOCAB_SIZE_JSON="null"
LM_HEAD_CHUNK_SIZES_JSON="null"
SPLIT_LM_HEAD=8
if [[ "$ARCH" == qwen* ]] || [[ "$ARCH" == gemma* ]]; then
    SPLIT_LM_HEAD=16
fi
if [ -f "$CONFIG_FILE" ]; then
    VOCAB_SIZE_RAW=$(jq -r '.vocab_size // .text_config.vocab_size // empty' "$CONFIG_FILE")
    if [[ "$VOCAB_SIZE_RAW" =~ ^[0-9]+$ ]] && [ "$VOCAB_SIZE_RAW" -gt 0 ]; then
        VOCAB_SIZE_JSON="$VOCAB_SIZE_RAW"
        base=$((VOCAB_SIZE_RAW / SPLIT_LM_HEAD))
        rem=$((VOCAB_SIZE_RAW % SPLIT_LM_HEAD))
        sizes="["
        for ((i=0; i<SPLIT_LM_HEAD; i++)); do
            size=$base
            if [ $i -lt $rem ]; then
                size=$((size + 1))
            fi
            if [ $i -gt 0 ]; then
                sizes+=","
            fi
            sizes+="$size"
        done
        sizes+="]"
        LM_HEAD_CHUNK_SIZES_JSON="$sizes"
        echo "Detected vocab_size: $VOCAB_SIZE_RAW, split_lm_head: $SPLIT_LM_HEAD"
        echo "Derived lm_head_chunk_sizes: $LM_HEAD_CHUNK_SIZES_JSON"
    fi
fi

# Resolve --chunk auto
if [ "$NUM_CHUNKS" = "auto" ]; then
    echo ""
    echo "Auto-detecting optimal chunk count..."
    # Extract LUT bits from LUT_PART2 (e.g., "6" or "6,4" → "6")
    LUT_BITS=$(echo "$LUT_PART2" | cut -d',' -f1)
    if [ -z "$LUT_BITS" ] || [ "$LUT_BITS" = "none" ] || [ "$LUT_BITS" = "no" ] || [ "$LUT_BITS" = "false" ]; then
        LUT_BITS=16  # FP16 if no LUT
    fi

    NUM_CHUNKS=$(python3 "$SCRIPT_DIR/calc_chunk_split.py" \
        --auto --max-chunk-mb "$MAX_CHUNK_MB" --lut "$LUT_BITS" \
        --overhead 1.10 "$MODEL_PATH" 2>&1)

    if [ $? -ne 0 ] || [ -z "$NUM_CHUNKS" ] || ! [[ "$NUM_CHUNKS" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Failed to auto-detect chunk count: $NUM_CHUNKS"
        echo "Please specify --chunk manually."
        exit 1
    fi

    LUT_LABEL="LUT${LUT_BITS}"
    if [ "$LUT_BITS" = "16" ]; then
        LUT_LABEL="FP16"
    fi
    echo "Auto-detected: --chunk $NUM_CHUNKS ($LUT_LABEL, max ${MAX_CHUNK_MB}MB per chunk with 10% overhead)"
    echo ""
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

# Dynamic prefill slice (default ON)
DYNAMIC_PREFILL_SLICE_PARAM=""
if [ "$STATIC_PREFILL_SLICE" = true ]; then
    DYNAMIC_PREFILL_SLICE_PARAM="--static-prefill-slice"
elif [ "$DYNAMIC_PREFILL_SLICE" = true ]; then
    DYNAMIC_PREFILL_SLICE_PARAM="--dynamic-prefill-slice"
fi

SINGLE_CACHE_PARAM=""
if [ "$SINGLE_CACHE" = true ]; then
    if [[ "$ARCH" == "gemma3_text"* ]] || [[ "$ARCH" == "gemma3"* ]]; then
        SINGLE_CACHE_PARAM="--single-cache"
    else
        echo "Warning: --single-cache is only supported for Gemma3; ignoring."
    fi
fi
CHUNK_NO_PARAM=""
if [ -n "$CHUNK_NO" ]; then
    CHUNK_NO_PARAM="--chunk-no $CHUNK_NO"
fi

if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "2" ]; then
    run_step 2 "Converting LM Head" "$CONVERTER \
        --part 3 \
        $LUT3_PARAM \
        $ARGMAX_PARAM \
        $FP16_SCALE_PARAM $CLAMP_PARAM \
        $SINGLE_CACHE_PARAM \
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
        $SINGLE_CACHE_PARAM \
        $CHUNK_NO_PARAM \
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
        $DYNAMIC_PREFILL_SLICE_PARAM \
        $FP16_SCALE_PARAM $CLAMP_PARAM \
        $SINGLE_CACHE_PARAM \
        $CHUNK_NO_PARAM \
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
        if [ "$SINGLE_CACHE" = true ]; then
            echo "Skipping rotate conversions: --single-cache disables rotate/prefill_rotate."
        else
        if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "4" ]; then
            run_step 4 "Converting FFN Rotate" "$CONVERTER \
                --part 2_rotate \
                $LUT2_PARAM \
                $FP16_SCALE_PARAM $CLAMP_PARAM \
                $SINGLE_CACHE_PARAM \
                $CHUNK_NO_PARAM \
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
                $DYNAMIC_PREFILL_SLICE_PARAM \
                $FP16_SCALE_PARAM $CLAMP_PARAM \
                $SINGLE_CACHE_PARAM \
                $CHUNK_NO_PARAM \
                --chunk $NUM_CHUNKS \
                --context-length $CONTEXT_LENGTH \
                --batch-size $BATCH_SIZE \
                --prefix \"$PREFIX\" \
                --model \"$MODEL_PATH\" \
                --output \"$OUTPUT_DIR\""
        fi
        fi
    fi
fi

# Step 5: Combine Models
# For Gemma3 with context > 512, use --gemma3 flag to combine 4 functions
GEMMA3_FLAG=""
SPLIT_ROTATE_FLAG=""
SKIP_DEDUP_FLAG=""
if [[ "$ARCH" == "gemma3_text"* ]] || [[ "$ARCH" == "gemma3"* ]]; then
    if [ $CONTEXT_LENGTH -gt $SLIDING_WINDOW ] && [ "$SINGLE_CACHE" != true ]; then
        GEMMA3_FLAG="--gemma3"
    fi
fi
if [ "$SPLIT_ROTATE" = true ]; then
    SPLIT_ROTATE_FLAG="--split-rotate"
    GEMMA3_FLAG=""
fi
if [ "$SKIP_ANEMLL_DEDUP" = true ]; then
    SKIP_DEDUP_FLAG="--skip-anemll-dedup"
fi

if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "5" ]; then
    if [ ! -z "$LUT_PART2" ]; then
        run_step 5 "Combining Models" "python3 \"$PROJECT_ROOT/anemll/utils/combine_models.py\" \
            --chunk $NUM_CHUNKS \
            $LUT2_PARAM \
            $SPLIT_ROTATE_FLAG \
            $GEMMA3_FLAG \
            $SKIP_DEDUP_FLAG \
            --prefix \"$PREFIX\" \
            --input \"$OUTPUT_DIR\" \
            --output \"$OUTPUT_DIR\""
    else
        run_step 5 "Combining Models" "python3 \"$PROJECT_ROOT/anemll/utils/combine_models.py\" \
            --chunk $NUM_CHUNKS \
            $SPLIT_ROTATE_FLAG \
            $GEMMA3_FLAG \
            $SKIP_DEDUP_FLAG \
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

    ARGMAX_JSON=false
    [ "$ARGMAX_IN_MODEL" = true ] && ARGMAX_JSON=true
    SPLIT_ROTATE_JSON=false
    [ "$SPLIT_ROTATE" = true ] && SPLIT_ROTATE_JSON=true
    SINGLE_CACHE_JSON=false
    [ "$SINGLE_CACHE" = true ] && SINGLE_CACHE_JSON=true
    DYNAMIC_PREFILL_JSON=false
    [ "$DYNAMIC_PREFILL_SLICE" = true ] && DYNAMIC_PREFILL_JSON=true

    SLIDING_WINDOW_JSON="null"
    if [ "$HAS_SWA" = true ] && [ -n "$SLIDING_WINDOW" ]; then
        SLIDING_WINDOW_JSON="$SLIDING_WINDOW"
    fi

    CONVERSION_INFO=$(jq -nc \
        --arg model_path "$MODEL_PATH" \
        --arg output_dir "$OUTPUT_DIR" \
        --arg lut_part1 "${LUT_PART1:-none}" \
        --arg lut_part2 "${LUT_PART2:-none}" \
        --arg lut_part3 "${LUT_PART3:-none}" \
        --arg prefix "$PREFIX" \
        --arg architecture "$ARCH" \
        --arg fp16_scale "${FP16_SCALE:-}" \
        --arg clamp "${CLAMP:-}" \
        --arg command_line "$CONVERSION_COMMAND" \
        --argjson context_length "$CONTEXT_LENGTH" \
        --argjson batch_size "$BATCH_SIZE" \
        --argjson num_chunks "$NUM_CHUNKS" \
        --argjson argmax_in_model "$ARGMAX_JSON" \
        --argjson split_rotate "$SPLIT_ROTATE_JSON" \
        --argjson sliding_window "$SLIDING_WINDOW_JSON" \
        --argjson single_cache "$SINGLE_CACHE_JSON" \
        --argjson dynamic_prefill_slice "$DYNAMIC_PREFILL_JSON" \
        --argjson vocab_size "$VOCAB_SIZE_JSON" \
        --argjson lm_head_chunk_sizes "$LM_HEAD_CHUNK_SIZES_JSON" \
        '{
            model_path: $model_path,
            output_dir: $output_dir,
            context_length: $context_length,
            batch_size: $batch_size,
            num_chunks: $num_chunks,
            lut_part1: $lut_part1,
            lut_part2: $lut_part2,
            lut_part3: $lut_part3,
            prefix: $prefix,
            architecture: $architecture,
            argmax_in_model: $argmax_in_model,
            split_rotate: $split_rotate,
            sliding_window: $sliding_window,
            fp16_scale: (if $fp16_scale == "" then null else $fp16_scale end),
            clamp: (if $clamp == "" then null else $clamp end),
            single_cache: $single_cache,
            dynamic_prefill_slice: $dynamic_prefill_slice,
            vocab_size: $vocab_size,
            lm_head_chunk_sizes: $lm_head_chunk_sizes,
            command_line: $command_line,
            monolithic: false,
            anemll_version: "0.3.5"
        }')
    export CONVERSION_INFO

    run_step 7 "Copying tokenizer files and creating meta.yaml" "
        # Copy tokenizer files if they exist
        (cp \"$MODEL_PATH/tokenizer.json\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/tokenizer_config.json\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/tokenizer.model\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/vocab.json\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/merges.txt\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/chat_template.jinja\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/added_tokens.json\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/special_tokens_map.json\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/tokenizer_config_search.json\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/generation_config.json\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/preprocessor_config.json\" \"$OUTPUT_DIR/\" || true) && \
        
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
        # Add sliding_window flag only when SWA is enabled.
        SLIDING_WINDOW_FLAG=\"\"
        if [ \"$HAS_SWA\" = true ] && [ -n \"$SLIDING_WINDOW\" ]; then
            SLIDING_WINDOW_FLAG=\"--sliding-window $SLIDING_WINDOW\"
        fi
        UPDATE_MASK_PREFILL_FLAG=\"\"
        if [[ \"$ARCH\" == \"gemma3\"* ]] && [ \"$DYNAMIC_PREFILL_SLICE\" != true ]; then
            UPDATE_MASK_PREFILL_FLAG=\"--update-mask-prefill\"
        fi
        PREFILL_DYNAMIC_SLICE_FLAG=\"\"
        if [ \"$STATIC_PREFILL_SLICE\" = true ]; then
            PREFILL_DYNAMIC_SLICE_FLAG=\"--static-prefill-slice\"
        elif [ \"$DYNAMIC_PREFILL_SLICE\" = true ]; then
            PREFILL_DYNAMIC_SLICE_FLAG=\"--dynamic-prefill-slice\"
        fi
        SINGLE_CACHE_META_FLAG=\"\"
        if [ \"$SINGLE_CACHE\" = true ]; then
            SINGLE_CACHE_META_FLAG=\"--single-cache\"
        fi

        ANEMLL_CONVERSION_INFO=\"\$CONVERSION_INFO\" python3 \"$PROJECT_ROOT/anemll/utils/generate_meta_yaml.py\" \
            \"$MODEL_NAME\" \"$CONTEXT_LENGTH\" \"$BATCH_SIZE\" \
            \"${LUT_PART1:-none}\" \"${LUT_PART2:-none}\" \"${LUT_PART3:-none}\" \
            $NUM_CHUNKS \"$PREFIX\" \"$ARCH\" \"$OUTPUT_DIR\" \$ARGMAX_META_FLAG \$SPLIT_ROTATE_META_FLAG \$SLIDING_WINDOW_FLAG \$UPDATE_MASK_PREFILL_FLAG \$PREFILL_DYNAMIC_SLICE_FLAG \$SINGLE_CACHE_META_FLAG
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
