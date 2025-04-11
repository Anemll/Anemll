#!/bin/bash

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

print_usage() {
    echo "Usage: $0 --input <converted_model_dir> [--output <output_dir>] [--org <huggingface_org>] [--ios]"
    echo "Options:"
    echo "  --input    Directory containing converted model files (required)"
    echo "  --output   Output directory for HF distribution (optional, defaults to input_dir/hf_dist)"
    echo "  --org      Hugging Face organization/account (optional, defaults to anemll)"
    echo "  --ios      Prepare iOS-ready version with unzipped MLMODELC files (if omitted, prepares standard distribution)"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --org)
            HF_ORG="$2"
            shift 2
            ;;
        --ios)
            PREPARE_IOS=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$INPUT_DIR" ]; then
    echo "Error: Input directory is required"
    print_usage
fi

# Convert input path to absolute path
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)" || {
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
}

# Read meta.yaml
if [ ! -f "$INPUT_DIR/meta.yaml" ]; then
    echo "Error: meta.yaml not found in input directory"
    exit 1
fi

# Validate LUT values
validate_lut_values() {
    # Check if LUT values are non-empty
    if [ -z "$LUT_FFN" ]; then
        echo "Warning: LUT value for FFN not found in meta.yaml, using default value of 8"
        LUT_FFN=8
    elif [ "$LUT_FFN" = "none" ]; then
        LUT_FFN="none"
    elif ! [[ "$LUT_FFN" =~ ^[0-9]+$ ]]; then
        echo "Warning: Invalid LUT value for FFN: $LUT_FFN, using default value of 8"
        LUT_FFN=8
    fi
    
    if [ -z "$LUT_LMHEAD" ]; then
        echo "Warning: LUT value for LM head not found in meta.yaml, using default value of 8"
        LUT_LMHEAD=8
    elif [ "$LUT_LMHEAD" = "none" ]; then
        LUT_LMHEAD="none"
    elif ! [[ "$LUT_LMHEAD" =~ ^[0-9]+$ ]]; then
        echo "Warning: Invalid LUT value for LM head: $LUT_LMHEAD, using default value of 8"
        LUT_LMHEAD=8
    fi
}

# Extract model name and parameters from meta.yaml
MODEL_NAME=$(grep "name:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f4)
MODEL_VERSION=$(grep "version:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f4)
CONTEXT_LENGTH=$(grep "context_length:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
BATCH_SIZE=$(grep "batch_size:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
MODEL_PREFIX=$(grep "model_prefix:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
NUM_CHUNKS=$(grep "num_chunks:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
LUT_FFN=$(grep "lut_ffn:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
LUT_LMHEAD=$(grep "lut_lmhead:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
LUT_EMBEDDINGS=$(grep "lut_embeddings:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6 2>/dev/null || echo "")

# Only llama is supported
MODEL_TYPE="llama"
TOKENIZER_CLASS="LlamaTokenizer"

# Validate and set default values for LUT parameters
validate_lut_values

# Set default value for LUT_EMBEDDINGS if not found in meta.yaml
if [ -z "$LUT_EMBEDDINGS" ]; then
    LUT_EMBEDDINGS=$LUT_FFN  # Default to same as FFN LUT
fi

# Construct full model name with version
FULL_MODEL_NAME="${MODEL_NAME}_${MODEL_VERSION}"

# Set output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$INPUT_DIR/hf_dist"
fi

# Set default HF organization if not specified
if [ -z "$HF_ORG" ]; then
    HF_ORG="anemll"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Preparing distribution for $FULL_MODEL_NAME..."

# Function to check if file exists in input directory
check_file_exists() {
    local file=$1
    if [ ! -d "$INPUT_DIR/$file" ]; then
        echo "Warning: $file not found in input directory"
        return 1
    fi
    return 0
}

# Function to prepare model directory (common files)
prepare_common_files() {
    local target_dir=$1
    
    # Copy configuration files
    cp "$INPUT_DIR/meta.yaml" "$target_dir/"
    cp "$INPUT_DIR/tokenizer.json" "$target_dir/"
    cp "$INPUT_DIR/tokenizer_config.json" "$target_dir/"
    
    # Always generate new config.json for tokenizer (forcing overwrite)
    echo "Generating config.json for tokenizer (overwriting any existing file)..."
    # Remove existing config.json if it exists
    if [ -f "$target_dir/config.json" ]; then
        rm "$target_dir/config.json"
    fi
    
    # Generate new config.json
    python -m anemll.ane_converter.create_config_json --output "$target_dir/config.json" --model-type "$MODEL_TYPE" --tokenizer-class "$TOKENIZER_CLASS"
    
    # Verify config.json was created
    if [ ! -f "$target_dir/config.json" ]; then
        echo "Error: Failed to generate config.json. This file is required for iOS offline tokenizer."
        echo "Please check if anemll.ane_converter.create_config_json is properly installed."
        exit 1
    fi
    
    # Copy sample code
    cp "$PROJECT_ROOT/tests/chat.py" "$target_dir/"
    cp "$PROJECT_ROOT/tests/chat_full.py" "$target_dir/"
}

# Function to compress mlmodelc directory
compress_mlmodelc() {
    local dir=$1
    local output_dir=$2
    local base=$(basename "$dir" .mlmodelc)
    
    if check_file_exists "${base}.mlmodelc"; then
        echo "[STANDARD] Compressing $base.mlmodelc..."
        (cd "$INPUT_DIR" && zip -r "$output_dir/${base}.mlmodelc.zip" "${base}.mlmodelc")
    fi
}

# Function to copy mlmodelc directory (for iOS)
copy_mlmodelc() {
    local dir=$1
    local output_dir=$2
    local base=$(basename "$dir" .mlmodelc)
    
    if check_file_exists "${base}.mlmodelc"; then
        echo "[iOS] Copying $base.mlmodelc (uncompressed)..."
        cp -r "$INPUT_DIR/${base}.mlmodelc" "$output_dir/"
    fi
}

# Function to get model filename with optional LUT
get_model_filename() {
    local prefix=$1
    local type=$2
    local lut=$3
    local chunk_info=$4
    
    if [ "$lut" = "none" ] || [ -z "$lut" ]; then
        echo "${prefix}_${type}${chunk_info}"
    else
        echo "${prefix}_${type}_lut${lut}${chunk_info}"
    fi
}

# Determine which distribution to prepare based on flags
if [ "$PREPARE_IOS" = true ]; then
    # Prepare iOS version only
    echo "============================================================"
    echo "PREPARING iOS DISTRIBUTION (with uncompressed .mlmodelc directories)"
    echo "============================================================"
    mkdir -p "$OUTPUT_DIR/ios"
    prepare_common_files "$OUTPUT_DIR/ios"
    
    # Copy all mlmodelc files uncompressed for iOS
    embeddings_file=$(get_model_filename "$MODEL_PREFIX" "embeddings" "$LUT_EMBEDDINGS")
    copy_mlmodelc "$embeddings_file" "$OUTPUT_DIR/ios"
    
    lmhead_file=$(get_model_filename "$MODEL_PREFIX" "lm_head" "$LUT_LMHEAD")
    copy_mlmodelc "$lmhead_file" "$OUTPUT_DIR/ios"
    
    for ((i=1; i<=NUM_CHUNKS; i++)); do
        chunk_num=$(printf "%02d" $i)
        chunk_info="_chunk_${chunk_num}of$(printf "%02d" $NUM_CHUNKS)"
        ffn_file=$(get_model_filename "$MODEL_PREFIX" "FFN_PF" "$LUT_FFN" "$chunk_info")
        copy_mlmodelc "$ffn_file" "$OUTPUT_DIR/ios"
    done
    
    # No need to zip iOS distribution - it should be used directly
    echo "[iOS] Distribution ready in: $OUTPUT_DIR/ios"
else
    # Prepare standard (zipped) version
    echo "============================================================"
    echo "PREPARING STANDARD DISTRIBUTION (with compressed .mlmodelc.zip files)"
    echo "============================================================"
    mkdir -p "$OUTPUT_DIR/standard"
    prepare_common_files "$OUTPUT_DIR/standard"
    
    # Compress all mlmodelc files for standard distribution
    embeddings_file=$(get_model_filename "$MODEL_PREFIX" "embeddings" "$LUT_EMBEDDINGS")
    compress_mlmodelc "$embeddings_file" "$OUTPUT_DIR/standard"
    
    lmhead_file=$(get_model_filename "$MODEL_PREFIX" "lm_head" "$LUT_LMHEAD")
    compress_mlmodelc "$lmhead_file" "$OUTPUT_DIR/standard"
    
    for ((i=1; i<=NUM_CHUNKS; i++)); do
        chunk_num=$(printf "%02d" $i)
        chunk_info="_chunk_${chunk_num}of$(printf "%02d" $NUM_CHUNKS)"
        ffn_file=$(get_model_filename "$MODEL_PREFIX" "FFN_PF" "$LUT_FFN" "$chunk_info")
        compress_mlmodelc "$ffn_file" "$OUTPUT_DIR/standard"
    done
    
    # Create standard distribution zip
    (cd "$OUTPUT_DIR" && zip -r "${FULL_MODEL_NAME}.zip" standard/)
fi

# Create README.md from template
README_TEMPLATE="$SCRIPT_DIR/readme.template"
if [ ! -f "$README_TEMPLATE" ]; then
    echo "Error: readme.template not found at $README_TEMPLATE"
    exit 1
fi

# Determine target directory for README.md
if [ "$PREPARE_IOS" = true ]; then
    TARGET_DIR="$OUTPUT_DIR/ios"
else
    TARGET_DIR="$OUTPUT_DIR/standard"
fi

# Read template and replace placeholders
sed -e "s|%NAME_OF_THE_FOLDER_WE_UPLOAD%|$FULL_MODEL_NAME|g" \
    -e "s|%PATH_TO_META_YAML%|./meta.yaml|g" \
    -e "s|%HF_ORG%|$HF_ORG|g" \
    "$README_TEMPLATE" > "$TARGET_DIR/README.md"

# Also create a copy in the main output directory for reference
cp "$TARGET_DIR/README.md" "$OUTPUT_DIR/"

echo "Distribution prepared in: $OUTPUT_DIR"
echo
echo "SUMMARY OF CREATED DISTRIBUTION:"
echo "--------------------------------"
if [ "$PREPARE_IOS" = true ]; then
    echo "iOS:      $OUTPUT_DIR/ios/ - Contains uncompressed .mlmodelc directories"
    echo "         - IMPORTANT: For iOS use these uncompressed directories directly!"
    echo "         - Generated config.json for iOS offline tokenizer"
else
    echo "STANDARD: $OUTPUT_DIR/standard/ - Contains compressed .mlmodelc.zip files"
    echo "         - Packaged as: ${FULL_MODEL_NAME}.zip"
    echo "         - Generated config.json for tokenizer"
fi
echo
echo "To upload to Hugging Face, use:"
if [ "$PREPARE_IOS" = true ]; then
    echo "huggingface-cli upload $HF_ORG/$FULL_MODEL_NAME $OUTPUT_DIR/ios"
    echo
    echo "Example:"
    echo "huggingface-cli upload $HF_ORG/$FULL_MODEL_NAME $(realpath "$OUTPUT_DIR/ios")"
else
    echo "huggingface-cli upload $HF_ORG/$FULL_MODEL_NAME $OUTPUT_DIR/standard"
    echo
    echo "Example:"
    echo "huggingface-cli upload $HF_ORG/$FULL_MODEL_NAME $(realpath "$OUTPUT_DIR/standard")"
fi 