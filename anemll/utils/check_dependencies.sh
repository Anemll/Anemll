#!/bin/bash

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
        QUANTIZATION_PRESENT=$(jq -e '.quantization | length > 0' "$MODEL_DIR/config.json" 2>/dev/null)
        if [ "$QUANTIZATION_PRESENT" = "true" ]; then
            echo "Quantized models are not supported. Aborting. (Issue #6)"
            echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."
            exit 1
        fi

        # Check for supported architectures in config.json
        echo "Checking for supported architectures in config.json..."
        SUPPORTED_ARCHS=("llama" "qwen")
        CONFIG_ARCH=$(jq -r '.architectures[]' "$MODEL_DIR/config.json" 2>/dev/null)
        CONFIG_MODEL_TYPE=$(jq -r '.model_type' "$MODEL_DIR/config.json" 2>/dev/null)

        CONFIG_ARCH_LOWER=$(echo "$CONFIG_ARCH" | tr '[:upper:]' '[:lower:]')
        CONFIG_MODEL_TYPE_LOWER=$(echo "$CONFIG_MODEL_TYPE" | tr '[:upper:]' '[:lower:]')

        # Check if architecture contains any supported pattern
        ARCH_SUPPORTED=false
        for arch in "${SUPPORTED_ARCHS[@]}"; do
            if [[ "$CONFIG_ARCH_LOWER" == *"$arch"* ]] || [[ "$CONFIG_MODEL_TYPE_LOWER" == *"$arch"* ]]; then
                ARCH_SUPPORTED=true
                break
            fi
        done
        
        if [ "$ARCH_SUPPORTED" = false ]; then
            echo "Unsupported architecture or model type in config.json. Supported types: ${SUPPORTED_ARCHS[@]}. Aborting. (Issue #7)"
            echo "Please refer to the troubleshooting guide in docs/troubleshooting.md for more information."
            exit 1
        fi
    fi

    # Add more checks as needed
    echo "All dependencies are satisfied."
fi