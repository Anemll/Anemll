#!/bin/bash

echo "=== ANEMLL DEBUG MARKER ==="
echo "RUNNING CHECK_DEPENDENCIES.SH FROM: $0"

# check_dependencies.sh
# This script checks for necessary dependencies before running convert_model.sh

# Function to display usage
usage() {
    echo "Usage: $0 [--skip-check] [--model <model_directory>] [--context <context_length>] [--batch <batch_size>] [other options for convert_model.sh]"
    exit 1
}

# Parse arguments
SKIP_CHECK=false
MODEL_DIR=""
CONVERT_ARGS=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-check)
            SKIP_CHECK=true
            shift
            ;;
        --model)
            MODEL_DIR="$2"
            CONVERT_ARGS+=" --model $2"
            shift 2
            ;;
        --context)
            CONVERT_ARGS+=" --context $2"
            shift 2
            ;;
        --batch)
            CONVERT_ARGS+=" --batch $2"
            shift 2
            ;;
        *)
            # Pass other arguments to convert_model.sh
            CONVERT_ARGS+=" $1"
            shift
            ;;
    esac
done

# Check dependencies unless --skip-check is provided
if [ "$SKIP_CHECK" = false ]; then
    echo "Checking dependencies..."
    echo "Checking if macOS version is 15 or higher..."
    macos_version=$(sw_vers -productVersion | awk -F '.' '{print $1}')
    if [ "$macos_version" -lt 15 ]; then
        echo "macOS version 15 or higher is required. Aborting. (Issue #5)"
        echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."
        exit 1
    fi

    echo "Checking if Python is installed..."
    command -v python >/dev/null 2>&1 || { echo >&2 "Python is required but it's not installed. Aborting. (Issue #1)"; echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."; exit 1; }

    echo "Checking if pip is installed..."
    command -v pip >/dev/null 2>&1 || { echo >&2 "pip is required but it's not installed. Aborting. (Issue #2)"; echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."; exit 1; }

    echo "Checking if coremltools is installed via pip..."
    if ! pip show coremltools >/dev/null 2>&1; then
        echo "coremltools is required but not installed via pip. Aborting. (Issue #3)"
        echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."
        exit 1
    fi

    # Check coremltools version
    coremltools_version=$(pip show coremltools | grep Version | awk '{print $2}')
    coremltools_major_version=$(echo "$coremltools_version" | cut -d. -f1)
    if [ "$coremltools_major_version" -lt 8 ]; then
        echo "coremltools version 8.x or higher is required. Aborting. (Issue #9)"
        echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."
        exit 1
    fi

    echo "Checking if coremlcompiler is available..."
    if ! xcrun --find coremlcompiler >/dev/null 2>&1; then
        echo "coremlcompiler is required but not found. Aborting. (Issue #4)"
        echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."
        exit 1
    fi

    echo "Displaying coremlcompiler version..."
    coremlcompiler_version=$(xcrun coremlcompiler version 2>&1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
    echo "coremlcompiler version: $coremlcompiler_version"

    echo "Checking if Python3 is installed..."
    command -v python3 >/dev/null 2>&1 || { echo >&2 "Python3 is required but it's not installed. Aborting."; echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."; exit 1; }

    # Check for model files only if MODEL_DIR is provided
    if [ ! -z "$MODEL_DIR" ]; then
        echo "Checking for model files in the provided directory: $MODEL_DIR"
        if [ ! -f "$MODEL_DIR/config.json" ]; then
            echo "config.json is required but not found in the model directory. Aborting. (Issue #5)"
            echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."
            exit 1
        fi
        if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
            echo "tokenizer.json is required but not found in the model directory. Aborting. (Issue #5)"
            echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."
            exit 1
        fi
        if [ ! -f "$MODEL_DIR/tokenizer_config.json" ]; then
            echo "tokenizer_config.json is required but not found in the model directory. Aborting. (Issue #5)"
            echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."
            exit 1
        fi

        # Check for quantization in config.json
        echo "Checking if model is quantized..."
        
        # Check for quantization_config field (used by FP8, GPTQ, AWQ, etc.)
        QUANTIZATION_CONFIG=$(jq -r '.quantization_config // empty' "$MODEL_DIR/config.json" 2>/dev/null)
        
        # Also check the old quantization field for backward compatibility
        QUANTIZATION_PRESENT=$(jq -e '.quantization | length > 0' "$MODEL_DIR/config.json" 2>/dev/null)
        
        if [ ! -z "$QUANTIZATION_CONFIG" ] || [ "$QUANTIZATION_PRESENT" = "true" ]; then
            echo ""
            echo "⚠️  ERROR: Quantized model detected!"
            echo ""
            
            # Get more details about the quantization
            if [ ! -z "$QUANTIZATION_CONFIG" ]; then
                QUANT_METHOD=$(jq -r '.quantization_config.quant_method // "unknown"' "$MODEL_DIR/config.json" 2>/dev/null)
                QUANT_FORMAT=$(jq -r '.quantization_config.fmt // .quantization_config.bits // "unknown"' "$MODEL_DIR/config.json" 2>/dev/null)
                echo "Quantization method: $QUANT_METHOD"
                echo "Format/bits: $QUANT_FORMAT"
                
                if [ "$QUANT_METHOD" = "fp8" ]; then
                    echo "This is an FP8 quantized model which cannot be properly converted."
                elif [ "$QUANT_METHOD" = "gptq" ] || [ "$QUANT_METHOD" = "awq" ]; then
                    echo "This is a $QUANT_METHOD quantized model with $QUANT_FORMAT-bit weights."
                fi
            fi
            
            echo ""
            echo "Quantized models are not supported for ANEMLL conversion. (Issue #6)"
            echo ""
            echo "Please use an unquantized (full precision) version of this model."
            echo "Look for model names without 'fp8', 'gptq', 'awq', '4bit', '8bit' etc."
            echo ""
            echo "For example:"
            echo "  ❌ qwen3_4b_fp8     -> ✅ qwen3_4B"
            echo "  ❌ llama-8b-gptq    -> ✅ llama-8b"
            echo "  ❌ model-awq-4bit   -> ✅ model"
            echo ""
            echo "Alternatively, you can convert this quantized model to unquantized FP16/BF16:"
            echo ""
            echo "  Option 1: Download the original unquantized model from Hugging Face"
            echo "  Option 2: Use the dequantization script (if available):"
            echo "           python utils/dequantize_model.py --input $MODEL_DIR --output ./dequantized_model"
            echo "  Option 3: In Python, load and save as FP16:"
            echo "           from transformers import AutoModelForCausalLM"
            echo "           model = AutoModelForCausalLM.from_pretrained('$MODEL_DIR', device_map='cpu')"
            echo "           model.save_pretrained('./unquantized_model', torch_dtype='float16')"
            echo ""
            echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."
            exit 1
        fi
        
        # Also check filename for quantization hints
        MODEL_BASENAME=$(basename "$MODEL_DIR" | tr '[:upper:]' '[:lower:]')
        if [[ "$MODEL_BASENAME" == *"fp8"* ]] || [[ "$MODEL_BASENAME" == *"gptq"* ]] || [[ "$MODEL_BASENAME" == *"awq"* ]] || [[ "$MODEL_BASENAME" == *"4bit"* ]] || [[ "$MODEL_BASENAME" == *"8bit"* ]]; then
            echo ""
            echo "⚠️  WARNING: Model directory name suggests quantization: $MODEL_BASENAME"
            echo "Please ensure you are using an unquantized model."
            echo ""
        fi

        # Check for supported architectures in config.json
        echo "Checking for supported architectures in config.json..."
        echo "DEBUG: config.json contents:"
        cat "$MODEL_DIR/config.json"

        CONFIG_ARCH=$(jq -r '.architectures[0] // empty' "$MODEL_DIR/config.json" 2>/dev/null)
        CONFIG_MODEL_TYPE=$(jq -r '.model_type // empty' "$MODEL_DIR/config.json" 2>/dev/null)

        CONFIG_ARCH_LOWER=$(echo "$CONFIG_ARCH" | tr '[:upper:]' '[:lower:]')
        CONFIG_MODEL_TYPE_LOWER=$(echo "$CONFIG_MODEL_TYPE" | tr '[:upper:]' '[:lower:]')

        echo "DEBUG: CONFIG_ARCH='$CONFIG_ARCH', CONFIG_MODEL_TYPE='$CONFIG_MODEL_TYPE', CONFIG_ARCH_LOWER='$CONFIG_ARCH_LOWER', CONFIG_MODEL_TYPE_LOWER='$CONFIG_MODEL_TYPE_LOWER'"

        if [[ -z "$CONFIG_ARCH_LOWER" && -z "$CONFIG_MODEL_TYPE_LOWER" ]]; then
            echo "Could not detect architecture or model_type in config.json. Aborting."
            exit 1
        fi

        if [[ "$CONFIG_ARCH_LOWER" != *llama* && "$CONFIG_ARCH_LOWER" != *qwen* && "$CONFIG_MODEL_TYPE_LOWER" != *llama* && "$CONFIG_MODEL_TYPE_LOWER" != *qwen* ]]; then
            echo "Unsupported architecture or model type in config.json. Supported types: llama, qwen. Aborting. (Issue #7)"
            echo "Detected architectures: $CONFIG_ARCH"
            echo "Detected model_type: $CONFIG_MODEL_TYPE"
            echo "(Lowercased: architectures='$CONFIG_ARCH_LOWER', model_type='$CONFIG_MODEL_TYPE_LOWER')"
            echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."
            exit 1
        fi
    fi

    # Add more checks as needed
    echo "All dependencies are satisfied."
fi