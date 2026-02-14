#!/bin/bash

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

print_usage() {
    echo "Usage: $0 --input <converted_model_dir> [--output <output_dir>] [--org <huggingface_org>] [--name <model_name>] [--ios] [--zip]"
    echo "Options:"
    echo "  --input    Directory containing converted model files (required)"
    echo "  --output   Output directory for HF distribution (optional, defaults to input_dir/hf_dist)"
    echo "  --org      Hugging Face organization/account (optional, defaults to anemll)"
    echo "  --name     Custom model name for HuggingFace upload (optional, defaults to name from meta.yaml)"
    echo "  --ios      Prepare iOS-ready version with unzipped MLMODELC files (default)"
    echo "  --zip      Prepare standard distribution with compressed .mlmodelc.zip files"
    exit 1
}

# Default to uncompressed (iOS-style) distribution
PREPARE_IOS=true

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
        --name)
            CUSTOM_MODEL_NAME="$2"
            shift 2
            ;;
        --ios)
            PREPARE_IOS=true
            shift
            ;;
        --zip)
            PREPARE_IOS=false
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
MODEL_FORMAT=$(grep "model_type:" "$INPUT_DIR/meta.yaml" | awk '{print $2}')
NUM_CHUNKS=$(grep "num_chunks:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
LUT_FFN=$(grep "lut_ffn:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
LUT_LMHEAD=$(grep "lut_lmhead:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
LUT_EMBEDDINGS=$(grep "lut_embeddings:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6 2>/dev/null || echo "")
LUT_BITS=$(grep "lut_bits:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6 2>/dev/null || echo "")
MONOLITHIC_MODEL=$(grep "monolithic_model:" "$INPUT_DIR/meta.yaml" | awk '{print $2}')

# Check if this is a monolithic model
MODEL_TYPE_YAML=$(grep "model_type:" "$INPUT_DIR/meta.yaml" | awk '{print $2}')
MONOLITHIC_MODEL=$(grep "monolithic_model:" "$INPUT_DIR/meta.yaml" | awk '{print $2}')
IS_MONOLITHIC=false
if [ "$MODEL_TYPE_YAML" = "monolithic" ] && [ -n "$MONOLITHIC_MODEL" ]; then
    IS_MONOLITHIC=true
    echo "Detected monolithic model: $MONOLITHIC_MODEL"
fi

# Check for split-rotate mode
IS_SPLIT_ROTATE=false
SPLIT_ROTATE_YAML=$(grep "split_rotate:" "$INPUT_DIR/meta.yaml" | awk '{print $2}' 2>/dev/null || echo "")
if [ "$SPLIT_ROTATE_YAML" = "true" ]; then
    IS_SPLIT_ROTATE=true
    echo "Detected split-rotate mode (2 files per chunk)"
fi

# Detect architecture for tokenizer config
ARCH=$(grep "architecture:" "$INPUT_DIR/meta.yaml" | awk '{print $2}')
MODEL_TYPE="llama"
TOKENIZER_CLASS="LlamaTokenizer"
if [[ "$ARCH" == gemma* ]]; then
    MODEL_TYPE="gemma"
    TOKENIZER_CLASS="GemmaTokenizer"
fi

# Normalize defaults based on model type
if [ "$MODEL_FORMAT" = "monolithic" ]; then
    NUM_CHUNKS=1
    if [ -n "$LUT_BITS" ]; then
        # Use lut_bits as FFN default; per-component overrides from meta take precedence
        LUT_FFN="$LUT_BITS"
        [ -z "$LUT_LMHEAD" ] && LUT_LMHEAD="$LUT_BITS"
        [ -z "$LUT_EMBEDDINGS" ] && LUT_EMBEDDINGS="$LUT_BITS"
    fi
else
    # Validate and set default values for LUT parameters
    validate_lut_values

    # Set default value for LUT_EMBEDDINGS if not found in meta.yaml
    if [ -z "$LUT_EMBEDDINGS" ]; then
        LUT_EMBEDDINGS=$LUT_FFN  # Default to same as FFN LUT
    fi
fi

# Construct full model name with version
if [ -n "$CUSTOM_MODEL_NAME" ]; then
    FULL_MODEL_NAME="$CUSTOM_MODEL_NAME"
    echo "Using custom model name: $FULL_MODEL_NAME"
else
    FULL_MODEL_NAME="${MODEL_NAME}_${MODEL_VERSION}"
    echo "Using model name from meta.yaml: $FULL_MODEL_NAME"
fi

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
    if [ -f "$INPUT_DIR/tokenizer.model" ]; then
        cp "$INPUT_DIR/tokenizer.model" "$target_dir/"
    fi

    # Copy additional tokenizer/support files if they exist
    for extra_file in \
        added_tokens.json \
        special_tokens_map.json \
        tokenizer_config_search.json \
        generation_config.json \
        preprocessor_config.json
    do
        if [ -f "$INPUT_DIR/$extra_file" ]; then
            cp "$INPUT_DIR/$extra_file" "$target_dir/"
            echo "Copied $extra_file"
        fi
    done
    
    # Copy tokenizer vocabulary files if they exist
    if [ -f "$INPUT_DIR/vocab.json" ]; then
        cp "$INPUT_DIR/vocab.json" "$target_dir/"
    else
        echo "Warning: vocab.json not found - may be required for tokenization"
    fi

    if [ -f "$INPUT_DIR/merges.txt" ]; then
        cp "$INPUT_DIR/merges.txt" "$target_dir/"
    else
        echo "Warning: merges.txt not found - may be required for BPE tokenization"
    fi

    # Copy chat template if it exists
    if [ -f "$INPUT_DIR/chat_template.jinja" ]; then
        cp "$INPUT_DIR/chat_template.jinja" "$target_dir/"
        echo "Copied chat_template.jinja"
    fi
    
    # Always generate new config.json for tokenizer (forcing overwrite)
    echo "Generating config.json for tokenizer (overwriting any existing file)..."
    # Remove existing config.json if it exists
    if [ -f "$target_dir/config.json" ]; then
        rm "$target_dir/config.json"
    fi
    
    # Generate new config.json
    python3 -m anemll.ane_converter.create_config_json --output "$target_dir/config.json" --model-type "$MODEL_TYPE" --tokenizer-class "$TOKENIZER_CLASS"
    
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

# Function to get model filename with fallback for embeddings LUT naming
get_model_filename_with_fallback() {
    local prefix=$1
    local type=$2
    local lut=$3
    local chunk_info=$4
    local input_dir=$5
    
    # Get the primary filename
    local primary_name=$(get_model_filename "$prefix" "$type" "$lut" "$chunk_info")
    
    # For embeddings with LUT, check if primary name exists, otherwise fallback to no-LUT version
    if [ "$type" = "embeddings" ] && [ "$lut" != "none" ] && [ -n "$lut" ]; then
        if [ ! -d "$input_dir/${primary_name}.mlmodelc" ]; then
            echo "Warning: ${primary_name}.mlmodelc not found, falling back to ${prefix}_${type}${chunk_info}.mlmodelc" >&2
            echo "${prefix}_${type}${chunk_info}"
            return
        fi
    fi
    
    echo "$primary_name"
}

# Determine which distribution to prepare based on flags
if [ "$PREPARE_IOS" = true ]; then
    # Prepare iOS version only
    echo "============================================================"
    echo "PREPARING iOS DISTRIBUTION (with uncompressed .mlmodelc directories)"
    echo "============================================================"
    mkdir -p "$OUTPUT_DIR/ios"
    prepare_common_files "$OUTPUT_DIR/ios"

    if [ "$IS_MONOLITHIC" = true ]; then
        # Copy monolithic model for iOS
        echo "[iOS] Copying monolithic model: $MONOLITHIC_MODEL (uncompressed)..."
        if [ -d "$INPUT_DIR/$MONOLITHIC_MODEL" ]; then
            cp -r "$INPUT_DIR/$MONOLITHIC_MODEL" "$OUTPUT_DIR/ios/"
            echo "[iOS] Monolithic model copied successfully"
        else
            echo "Error: Monolithic model not found: $INPUT_DIR/$MONOLITHIC_MODEL"
            exit 1
        fi
    else
        # Copy all mlmodelc files uncompressed for iOS (standard chunked model)
        embeddings_file=$(get_model_filename_with_fallback "$MODEL_PREFIX" "embeddings" "$LUT_EMBEDDINGS" "" "$INPUT_DIR")
        copy_mlmodelc "$embeddings_file" "$OUTPUT_DIR/ios"

        lmhead_file=$(get_model_filename "$MODEL_PREFIX" "lm_head" "$LUT_LMHEAD")
        copy_mlmodelc "$lmhead_file" "$OUTPUT_DIR/ios"


        for ((i=1; i<=NUM_CHUNKS; i++)); do
            chunk_num=$(printf "%02d" $i)
            chunk_info="_chunk_${chunk_num}of$(printf "%02d" $NUM_CHUNKS)"
            ffn_file=$(get_model_filename "$MODEL_PREFIX" "FFN_PF" "$LUT_FFN" "$chunk_info")
            copy_mlmodelc "$ffn_file" "$OUTPUT_DIR/ios"
            # Copy rotate file if in split-rotate mode
            if [ "$IS_SPLIT_ROTATE" = true ]; then
                rot_file="${ffn_file}_rot"
                copy_mlmodelc "$rot_file" "$OUTPUT_DIR/ios"
            fi
        done
    fi


    # No need to zip iOS distribution - it should be used directly
    echo "[iOS] Distribution ready in: $OUTPUT_DIR/ios"
else
    # Prepare standard (zipped) version
    echo "============================================================"
    echo "PREPARING STANDARD DISTRIBUTION (with compressed .mlmodelc.zip files)"
    echo "============================================================"
    mkdir -p "$OUTPUT_DIR/standard"
    prepare_common_files "$OUTPUT_DIR/standard"

    if [ "$IS_MONOLITHIC" = true ]; then
        # Compress monolithic model for standard distribution
        echo "[STANDARD] Compressing monolithic model: $MONOLITHIC_MODEL..."
        if [ -d "$INPUT_DIR/$MONOLITHIC_MODEL" ]; then
            (cd "$INPUT_DIR" && zip -r "$OUTPUT_DIR/standard/${MONOLITHIC_MODEL}.zip" "$MONOLITHIC_MODEL")
            echo "[STANDARD] Monolithic model compressed successfully"
        else
            echo "Error: Monolithic model not found: $INPUT_DIR/$MONOLITHIC_MODEL"
            exit 1
        fi
    else
        # Compress all mlmodelc files for standard distribution (standard chunked model)
        embeddings_file=$(get_model_filename_with_fallback "$MODEL_PREFIX" "embeddings" "$LUT_EMBEDDINGS" "" "$INPUT_DIR")
        compress_mlmodelc "$embeddings_file" "$OUTPUT_DIR/standard"

        lmhead_file=$(get_model_filename "$MODEL_PREFIX" "lm_head" "$LUT_LMHEAD")
        compress_mlmodelc "$lmhead_file" "$OUTPUT_DIR/standard"


        for ((i=1; i<=NUM_CHUNKS; i++)); do
            chunk_num=$(printf "%02d" $i)
            chunk_info="_chunk_${chunk_num}of$(printf "%02d" $NUM_CHUNKS)"
            ffn_file=$(get_model_filename "$MODEL_PREFIX" "FFN_PF" "$LUT_FFN" "$chunk_info")
            compress_mlmodelc "$ffn_file" "$OUTPUT_DIR/standard"
            # Compress rotate file if in split-rotate mode
            if [ "$IS_SPLIT_ROTATE" = true ]; then
                rot_file="${ffn_file}_rot"
                compress_mlmodelc "$rot_file" "$OUTPUT_DIR/standard"
            fi
        done
    fi


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

# Determine LUT bits for display (use LUT_BITS for monolithic, LUT_FFN otherwise)
if [ "$MODEL_FORMAT" = "monolithic" ]; then
    DISPLAY_LUT_BITS="${LUT_BITS:-none}"
else
    DISPLAY_LUT_BITS="${LUT_FFN:-none}"
fi

# Read template and replace placeholders
sed -e "s|%NAME_OF_THE_FOLDER_WE_UPLOAD%|$FULL_MODEL_NAME|g" \
    -e "s|%PATH_TO_META_YAML%|./meta.yaml|g" \
    -e "s|%HF_ORG%|$HF_ORG|g" \
    -e "s|%CONTEXT_LENGTH%|$CONTEXT_LENGTH|g" \
    -e "s|%BATCH_SIZE%|$BATCH_SIZE|g" \
    -e "s|%NUM_CHUNKS%|$NUM_CHUNKS|g" \
    -e "s|%LUT_BITS%|$DISPLAY_LUT_BITS|g" \
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
    echo "hf upload $HF_ORG/$FULL_MODEL_NAME $OUTPUT_DIR/ios"
    echo
    echo "Example:"
    echo "hf upload $HF_ORG/$FULL_MODEL_NAME $(realpath "$OUTPUT_DIR/ios")"
else
    echo "hf upload $HF_ORG/$FULL_MODEL_NAME $OUTPUT_DIR/standard"
    echo
    echo "Example:"
    echo "hf upload $HF_ORG/$FULL_MODEL_NAME $(realpath "$OUTPUT_DIR/standard")"
fi 
