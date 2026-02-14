#!/bin/bash
#
# Monolithic Model Conversion Script
# Converts LLM models to a single CoreML file containing embeddings + FFN + LM head
#

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default values
CONTEXT_LENGTH=512
BATCH_SIZE=64
LUT_BITS=4          # Default LUT bits for all components
LUT_EMBEDDINGS=""   # Override LUT for embeddings (empty = use LUT_BITS)
LUT_LMHEAD=""       # Override LUT for LM head (empty = use LUT_BITS)
PREFIX=""           # Auto-detect based on architecture
MODEL_PATH=""
OUTPUT_DIR=""
RESTART_STEP=1
RESTART_STEP_NUM=1
ONLY_STEP=""
SKIP_CHECK=false
FORCE_MLPROGRAM_COMPILE=false
ALLOW_MISSING_WEIGHTS=false
ARGMAX_IN_MODEL=false
ATTENTION_WINDOW=""
DYNAMIC_PREFILL_SLICE=true
STATIC_PREFILL_SLICE=false
SINGLE_CACHE=false
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
    echo ""
    echo "Monolithic Model Conversion - Creates a single CoreML model containing"
    echo "embeddings, FFN (transformer layers), and LM head in one file."
    echo ""
    echo "Options:"
    echo "  --model         Path to the model directory (required)"
    echo "  --output        Output directory for converted models (required)"
    echo "  --context       Context length (default: 512)"
    echo "  --batch         Batch size for prefill (default: 64)"
    echo "  --lut           LUT bits for FFN layers (default: 4)"
    echo "                  Format: 'bits' or 'bits,per_channel' (e.g., '4' or '4,8')"
    echo "  --lut-embeddings  Override LUT for embeddings (default: same as --lut)"
    echo "  --lut-lmhead      Override LUT for LM head (default: same as --lut)"
    echo "  --prefix        Prefix for model names (default: auto-detect)"
    echo "  --restart       Restart from specific step (1-6, 2b, 2c; default: 1)"
    echo "  --only          Run only specified step and exit (1-6, 2b, 2c)"
    echo "  --skip-check    Skip the dependency check step"
    echo "  --force-mlprogram-compile  Force ML Program when compiling .mlpackage models"
    echo "  --allow-missing-weights  Continue conversion even if some weights are missing"
    echo "  --argmax        Compute argmax inside model (outputs idx+val pairs instead of logits)"
    echo "  --dynamic-prefill-slice  Use dynamic slicing for prefill KV writes (default ON)"
    echo "  --static-prefill-slice   Disable dynamic slicing for prefill KV writes"
    echo "  --single-cache           Use unified KV cache (Gemma3 only; disables rotate)"
    echo "  --skip-anemll-dedup      Skip anemll-dedup weight dedup in combine step (default: enabled)"
    echo ""
    echo "Steps:"
    echo "  1. Convert monolithic inference model (embed + FFN + lm_head)"
    echo "  2. Convert monolithic prefill model"
    echo "  3. Combine into multi-function model"
    echo "  4. Compile to .mlmodelc"
    echo "  5. Copy tokenizer files and create meta.yaml"
    echo "  6. Test with chat.py"
    echo ""
    echo "Examples:"
    echo "  # Basic conversion with 4-bit quantization"
    echo "  $0 --model ./models/Qwen3-0.6B --output ./converted --lut 4"
    echo ""
    echo "  # With custom per_channel group size"
    echo "  $0 --model ./models/Qwen3-0.6B --output ./converted --lut 4,16"
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
        --lut)
            LUT_BITS="$2"
            shift 2
            ;;
        --lut-embeddings)
            LUT_EMBEDDINGS="$2"
            shift 2
            ;;
        --lut-lmhead)
            LUT_LMHEAD="$2"
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
        --skip-anemll-dedup)
            SKIP_ANEMLL_DEDUP=true
            shift
            ;;
        --help|-h)
            print_usage
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            ;;
    esac
done

# Normalize restart step for numeric comparisons (allow values like 2b/2c)
RESTART_STEP_NUM="${RESTART_STEP//[^0-9]/}"
if [ -z "$RESTART_STEP_NUM" ]; then
    RESTART_STEP_NUM="$RESTART_STEP"
fi

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

# Check if MODEL_PATH looks like a HuggingFace model name (e.g., "google/gemma-3-270m-it")
# HuggingFace names contain "/" but don't start with "/" or "~" or "."
if [[ "$MODEL_PATH" == *"/"* ]] && [[ ! "$MODEL_PATH" =~ ^[/~.] ]]; then
    echo "Detected HuggingFace model name: $MODEL_PATH"

    # Convert model name to cache path format (e.g., google/gemma-3-270m-it -> models--google--gemma-3-270m-it)
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
        if [ -z "$PREFIX" ]; then
            PREFIX="qwen25"
        fi
    elif [[ "$ARCH" == qwen* ]]; then
        CONVERTER="python3 -m anemll.ane_converter.qwen_converter"
        if [ -z "$PREFIX" ]; then
            PREFIX="qwen"
        fi
    elif [[ "$ARCH" == "gemma3_text"* ]] || [[ "$ARCH" == "gemma3"* ]]; then
        CONVERTER="python3 -m anemll.ane_converter.gemma3_converter"
        if [ -z "$PREFIX" ]; then
            PREFIX="gemma3"
        fi
    else
        CONVERTER="python3 -m anemll.ane_converter.llama_converter"
        if [ -z "$PREFIX" ]; then
            PREFIX="llama"
        fi
    fi

    # Detect whether this architecture actually uses sliding-window attention.
    # Qwen2.5 carries sliding_window in config but sets use_sliding_window=false.
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

echo ""
echo "=========================================="
echo "Monolithic Model Conversion"
echo "=========================================="
echo "Model path:     $MODEL_PATH"
echo "Output dir:     $OUTPUT_DIR"
echo "Architecture:   $ARCH"
echo "Prefix:         $PREFIX"
echo "Context length: $CONTEXT_LENGTH"
echo "Batch size:     $BATCH_SIZE"
echo "LUT bits:       $LUT_BITS"
if [ ! -z "$LUT_EMBEDDINGS" ]; then
    echo "LUT embeddings: $LUT_EMBEDDINGS"
fi
if [ ! -z "$LUT_LMHEAD" ]; then
    echo "LUT lm_head:    $LUT_LMHEAD"
fi
echo "Argmax in model:$ARGMAX_IN_MODEL"
echo "=========================================="
echo ""

# Step 0: Check dependencies
if [ "$SKIP_CHECK" = false ]; then
    if [ -f "$SCRIPT_DIR/check_dependencies.sh" ]; then
        "$SCRIPT_DIR/check_dependencies.sh" --model "$MODEL_PATH" --output "$OUTPUT_DIR" "$@"
        if [ $? -ne 0 ]; then
            echo "Dependency check failed. Aborting."
            exit 1
        fi
    fi
fi

# Function to run step if restart_step is less than or equal to step number
run_step() {
    local step=$1
    local description=$2
    local cmd=$3

    local step_num="${step//[^0-9]/}"
    if [ -z "$step_num" ]; then
        step_num="$step"
    fi

    if [ "$RESTART_STEP_NUM" -le "$step_num" ]; then
        # Skip if ONLY_STEP is set and doesn't match current step
        if [ ! -z "$ONLY_STEP" ] && [ "$ONLY_STEP" != "$step" ]; then
            return
        fi

        echo ""
        echo "Step $step: $description"
        echo "----------------------------------------"
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
        if [ -z "$ONLY_STEP" ]; then
            echo "Skipping step $step: $description"
        fi
    fi
}

# Prepare LUT parameters
LUT_PARAM=""
if [ ! -z "$LUT_BITS" ]; then
    LUT_PARAM="--lut $LUT_BITS"
fi
LUT_EMBED_PARAM=""
if [ ! -z "$LUT_EMBEDDINGS" ]; then
    LUT_EMBED_PARAM="--lut-embeddings $LUT_EMBEDDINGS"
fi
LUT_LMHEAD_PARAM=""
if [ ! -z "$LUT_LMHEAD" ]; then
    LUT_LMHEAD_PARAM="--lut-lmhead $LUT_LMHEAD"
fi

# Prepare argmax parameter
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

# Step 1: Convert Monolithic Inference Model
run_step 1 "Converting Monolithic Inference Model" "$CONVERTER \
    --part monolithic \
    $LUT_PARAM \
    $LUT_EMBED_PARAM \
    $LUT_LMHEAD_PARAM \
    $ARGMAX_PARAM \
    $SINGLE_CACHE_PARAM \
    --context-length $CONTEXT_LENGTH \
    --batch-size $BATCH_SIZE \
    --prefix \"$PREFIX\" \
    --model \"$MODEL_PATH\" \
    --output \"$OUTPUT_DIR\""

# Step 2: Convert Monolithic Prefill Model
run_step 2 "Converting Monolithic Prefill Model" "$CONVERTER \
    --part monolithic_prefill \
    $LUT_PARAM \
    $LUT_EMBED_PARAM \
    $LUT_LMHEAD_PARAM \
    $ARGMAX_PARAM \
    $DYNAMIC_PREFILL_SLICE_PARAM \
    $SINGLE_CACHE_PARAM \
    --context-length $CONTEXT_LENGTH \
    --batch-size $BATCH_SIZE \
    --prefix \"$PREFIX\" \
    --model \"$MODEL_PATH\" \
    --output \"$OUTPUT_DIR\""

# For context > sliding_window, also convert rotation functions (4-function model for SWA models)
if [ "$HAS_SWA" = true ] && [ "$CONTEXT_LENGTH" -gt "$SLIDING_WINDOW" ]; then
    if [ "$SINGLE_CACHE" = true ]; then
        echo "Skipping rotate conversions: --single-cache disables rotate/prefill_rotate."
    else
    echo "Context > sliding_window ($SLIDING_WINDOW): Converting rotation functions for 4-function model support..."

    # Step 2b: Convert Monolithic Inference Rotate Model
    run_step "2b" "Converting Monolithic Inference Rotate Model" "$CONVERTER \
        --part monolithic_rotate \
        $LUT_PARAM \
        $LUT_EMBED_PARAM \
        $LUT_LMHEAD_PARAM \
        $ARGMAX_PARAM \
        $SINGLE_CACHE_PARAM \
        --context-length $CONTEXT_LENGTH \
        --batch-size $BATCH_SIZE \
        --prefix \"$PREFIX\" \
        --model \"$MODEL_PATH\" \
        --output \"$OUTPUT_DIR\""

    # Step 2c: Convert Monolithic Prefill Rotate Model
    run_step "2c" "Converting Monolithic Prefill Rotate Model" "$CONVERTER \
        --part monolithic_prefill_rotate \
        $LUT_PARAM \
        $LUT_EMBED_PARAM \
        $LUT_LMHEAD_PARAM \
        $ARGMAX_PARAM \
        $DYNAMIC_PREFILL_SLICE_PARAM \
        $SINGLE_CACHE_PARAM \
        --context-length $CONTEXT_LENGTH \
        --batch-size $BATCH_SIZE \
        --prefix \"$PREFIX\" \
        --model \"$MODEL_PATH\" \
        --output \"$OUTPUT_DIR\""
    fi
fi

# Step 3: Combine into Multi-Function Model
# Includes rotation functions if context > sliding_window
ROTATE_FLAG=""
SKIP_DEDUP_FLAG=""
if [ "$HAS_SWA" = true ] && [ "$CONTEXT_LENGTH" -gt "$SLIDING_WINDOW" ]; then
    ROTATE_FLAG="--rotate"
fi
if [ "$SKIP_ANEMLL_DEDUP" = true ]; then
    SKIP_DEDUP_FLAG="--skip-anemll-dedup"
fi
run_step 3 "Combining Monolithic Models" "python3 \"$PROJECT_ROOT/anemll/utils/combine_models.py\" \
    --monolithic \
    $LUT_PARAM \
    $ROTATE_FLAG \
    $SKIP_DEDUP_FLAG \
    --prefix \"$PREFIX\" \
    --input \"$OUTPUT_DIR\" \
    --output \"$OUTPUT_DIR\""

# Step 4: Compile to .mlmodelc
FORCE_MLPROG_FLAG=""
if [ "$FORCE_MLPROGRAM_COMPILE" = true ]; then
    FORCE_MLPROG_FLAG="--force-mlprogram"
fi
run_step 4 "Compiling Monolithic Model" "python3 \"$PROJECT_ROOT/anemll/utils/compile_models.py\" \
    monolithic \
    $LUT_PARAM \
    $FORCE_MLPROG_FLAG \
    --prefix \"$PREFIX\" \
    --input \"$OUTPUT_DIR\" \
    --output \"$OUTPUT_DIR\""

# Step 5: Copy tokenizer files and create meta.yaml
if [ "$MODEL_PATH" != "$OUTPUT_DIR" ]; then
    # Detect HuggingFace cache path and extract proper model name
    if [[ "$MODEL_PATH" =~ \.cache/huggingface/hub/models--([^/]+)--([^/]+)/snapshots/ ]]; then
        HF_ORG="${BASH_REMATCH[1]}"
        HF_MODEL="${BASH_REMATCH[2]}"
        MODEL_NAME="${HF_ORG}-${HF_MODEL}"
    else
        MODEL_NAME=$(basename "$MODEL_PATH")
    fi

    ROTATION_ENABLED=false
    if [ "$HAS_SWA" = true ] && [ -n "$SLIDING_WINDOW" ] && [ "$CONTEXT_LENGTH" -gt "$SLIDING_WINDOW" ] && [ "$SINGLE_CACHE" != true ]; then
        ROTATION_ENABLED=true
    fi

    ARGMAX_JSON=false
    [ "$ARGMAX_IN_MODEL" = true ] && ARGMAX_JSON=true
    ROTATION_JSON=false
    [ "$ROTATION_ENABLED" = true ] && ROTATION_JSON=true
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
        --arg lut_bits "${LUT_BITS:-none}" \
        --arg lut_embeddings "${LUT_EMBEDDINGS:-}" \
        --arg lut_lmhead "${LUT_LMHEAD:-}" \
        --arg prefix "$PREFIX" \
        --arg architecture "$ARCH" \
        --arg fp16_scale "${FP16_SCALE:-}" \
        --arg clamp "${CLAMP:-}" \
        --arg command_line "$CONVERSION_COMMAND" \
        --argjson context_length "$CONTEXT_LENGTH" \
        --argjson batch_size "$BATCH_SIZE" \
        --argjson argmax_in_model "$ARGMAX_JSON" \
        --argjson rotate "$ROTATION_JSON" \
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
            lut_bits: $lut_bits,
            lut_embeddings: (if $lut_embeddings == "" then null else $lut_embeddings end),
            lut_lmhead: (if $lut_lmhead == "" then null else $lut_lmhead end),
            prefix: $prefix,
            architecture: $architecture,
            argmax_in_model: $argmax_in_model,
            rotate: $rotate,
            sliding_window: $sliding_window,
            fp16_scale: (if $fp16_scale == "" then null else $fp16_scale end),
            clamp: (if $clamp == "" then null else $clamp end),
            single_cache: $single_cache,
            dynamic_prefill_slice: $dynamic_prefill_slice,
            vocab_size: $vocab_size,
            lm_head_chunk_sizes: $lm_head_chunk_sizes,
            command_line: $command_line,
            monolithic: true,
            anemll_version: "0.3.5"
        }')
    export CONVERSION_INFO

    run_step 5 "Copying tokenizer files and creating meta.yaml" "
        # Copy tokenizer files if they exist
        (cp \"$MODEL_PATH/tokenizer.json\" \"$OUTPUT_DIR/\" 2>/dev/null || true) && \
        (cp \"$MODEL_PATH/tokenizer_config.json\" \"$OUTPUT_DIR/\" 2>/dev/null || true) && \
        (cp \"$MODEL_PATH/tokenizer.model\" \"$OUTPUT_DIR/\" 2>/dev/null || true) && \
        (cp \"$MODEL_PATH/vocab.json\" \"$OUTPUT_DIR/\" 2>/dev/null || true) && \
        (cp \"$MODEL_PATH/merges.txt\" \"$OUTPUT_DIR/\" 2>/dev/null || true) && \
        (cp \"$MODEL_PATH/chat_template.jinja\" \"$OUTPUT_DIR/\" 2>/dev/null || true) && \

        # Create config.json if it doesn't exist
        if [ ! -f \"$OUTPUT_DIR/config.json\" ]; then
            echo \"Creating config.json for iOS tokenizer...\" && \
            if [[ \"$ARCH\" == \"qwen2\" ]] || [[ \"$ARCH\" == *\"qwen2forcausallm\"* ]]; then
                cat > \"$OUTPUT_DIR/config.json\" <<'EOF_CONFIG'
{
  \"tokenizer_class\": \"Qwen2Tokenizer\",
  \"model_type\": \"qwen2\"
}
EOF_CONFIG
            elif [[ \"$ARCH\" == qwen* ]]; then
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

        # Create meta.yaml for monolithic model
        # Add --rotate flag if context > sliding_window (4-function model)
        ROTATE_META_FLAG=\"\"
        if [ \"$HAS_SWA\" = true ] && [ \"$CONTEXT_LENGTH\" -gt \"$SLIDING_WINDOW\" ] && [ \"$SINGLE_CACHE\" != true ]; then
            ROTATE_META_FLAG=\"--rotate\"
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
            \"${LUT_EMBEDDINGS:-${LUT_BITS:-none}}\" \"${LUT_BITS:-none}\" \"${LUT_LMHEAD:-${LUT_BITS:-none}}\" \
            1 \"$PREFIX\" \"$ARCH\" \"$OUTPUT_DIR\" \
            --monolithic $ARGMAX_PARAM $ROTATE_META_FLAG $SLIDING_WINDOW_FLAG $UPDATE_MASK_PREFILL_FLAG $PREFILL_DYNAMIC_SLICE_FLAG $SINGLE_CACHE_META_FLAG
    "
fi

# Step 6: Test with chat.py
run_step 6 "Testing with chat.py" "python3 \"$PROJECT_ROOT/tests/chat.py\" \
    --meta \"$OUTPUT_DIR/meta.yaml\" \
    --prompt \"Who are you?\" \
    --max-tokens 100"

# Print usage instructions
echo ""
echo "=========================================="
echo "Conversion completed successfully!"
echo "=========================================="
echo ""
echo "To chat with the model, use:"
echo ""
echo "python3 $PROJECT_ROOT/tests/chat.py \\"
echo "    --meta \"$OUTPUT_DIR/meta.yaml\""
echo ""
echo "Or for full conversation mode:"
echo "python3 $PROJECT_ROOT/tests/chat_full.py \\"
echo "    --meta \"$OUTPUT_DIR/meta.yaml\""
echo ""
echo "To prepare for iOS distribution:"
echo ""
echo "./anemll/utils/prepare_hf.sh --input \"$OUTPUT_DIR\" --ios"
echo ""
