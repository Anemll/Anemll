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
PREFIX=""           # Auto-detect based on architecture
MODEL_PATH=""
OUTPUT_DIR=""
RESTART_STEP=1
ONLY_STEP=""
SKIP_CHECK=false

# Default converter; may be overridden after parsing config.json
CONVERTER="python3 -m anemll.ane_converter.llama_converter"

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
    echo "  --lut           LUT bits for all components (default: 4)"
    echo "                  Format: 'bits' or 'bits,per_channel' (e.g., '4' or '4,8')"
    echo "  --prefix        Prefix for model names (default: auto-detect)"
    echo "  --restart       Restart from specific step (1-6, default: 1)"
    echo "  --only          Run only specified step and exit (1-6)"
    echo "  --skip-check    Skip the dependency check step"
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
        --help|-h)
            print_usage
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
    else
        CONVERTER="python3 -m anemll.ane_converter.llama_converter"
        if [ -z "$PREFIX" ]; then
            PREFIX="llama"
        fi
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

    if [ $RESTART_STEP -le $step ]; then
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

# Prepare LUT parameter
LUT_PARAM=""
if [ ! -z "$LUT_BITS" ]; then
    LUT_PARAM="--lut $LUT_BITS"
fi

# Step 1: Convert Monolithic Inference Model
run_step 1 "Converting Monolithic Inference Model" "$CONVERTER \
    --part monolithic \
    $LUT_PARAM \
    --context-length $CONTEXT_LENGTH \
    --batch-size $BATCH_SIZE \
    --prefix \"$PREFIX\" \
    --model \"$MODEL_PATH\" \
    --output \"$OUTPUT_DIR\""

# Step 2: Convert Monolithic Prefill Model
run_step 2 "Converting Monolithic Prefill Model" "$CONVERTER \
    --part monolithic_prefill \
    $LUT_PARAM \
    --context-length $CONTEXT_LENGTH \
    --batch-size $BATCH_SIZE \
    --prefix \"$PREFIX\" \
    --model \"$MODEL_PATH\" \
    --output \"$OUTPUT_DIR\""

# Step 3: Combine into Multi-Function Model
run_step 3 "Combining Monolithic Models" "python3 \"$PROJECT_ROOT/anemll/utils/combine_models.py\" \
    --monolithic \
    $LUT_PARAM \
    --prefix \"$PREFIX\" \
    --input \"$OUTPUT_DIR\" \
    --output \"$OUTPUT_DIR\""

# Step 4: Compile to .mlmodelc
run_step 4 "Compiling Monolithic Model" "python3 \"$PROJECT_ROOT/anemll/utils/compile_models.py\" \
    monolithic \
    $LUT_PARAM \
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

    run_step 5 "Copying tokenizer files and creating meta.yaml" "
        # Copy tokenizer files if they exist
        (cp \"$MODEL_PATH/tokenizer.json\" \"$OUTPUT_DIR/\" 2>/dev/null || true) && \
        (cp \"$MODEL_PATH/tokenizer_config.json\" \"$OUTPUT_DIR/\" 2>/dev/null || true) && \
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
        python3 \"$PROJECT_ROOT/anemll/utils/generate_meta_yaml.py\" \
            \"$MODEL_NAME\" \"$CONTEXT_LENGTH\" \"$BATCH_SIZE\" \
            \"${LUT_BITS:-none}\" \"${LUT_BITS:-none}\" \"${LUT_BITS:-none}\" \
            1 \"$PREFIX\" \"$ARCH\" \"$OUTPUT_DIR\" \
            --monolithic
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
