#!/bin/bash

# Default values
DEFAULT_PROMPT="who are you"
DEFAULT_ITERATIONS=5
DEFAULT_MODEL_PATH="~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/Models/llama_3_2_1b_iosv2_0/meta.yaml"
DEFAULT_IMPLEMENTATION="swift"

# Token checking mode flag
CHECK_TOKENS=false

# Temporary directory for token decoding
DECODE_DIR=$(mktemp -d)
DECODE_SCRIPT="$DECODE_DIR/decode_token.swift"

# Function to create token decoder script
create_token_decoder() {
    cat > "$DECODE_SCRIPT" << 'EOF'
import Foundation
import ArgumentParser
import AnemllCore

@main
struct TokenDecoder: ParsableCommand {
    @Option(name: .shortAndLong, help: "Path to tokenizer model file")
    var tokenizerPath: String
    
    @Option(name: .shortAndLong, help: "Token ID to decode")
    var tokenId: Int
    
    @Flag(name: .long, help: "Include special tokens in output")
    var includeSpecial = true
    
    mutating func run() async throws {
        let tokenizer = try await Tokenizer(modelPath: tokenizerPath, template: "deephermes")
        let decoded = tokenizer.decode(tokens: [tokenId], skipSpecialTokens: !includeSpecial)
        print(decoded)
    }
}
EOF
}

# Function to decode a token
decode_token() {
    local token_id=$1
    local tokenizer_path=$2
    
    # Build and run the decoder
    cd "$DECODE_DIR" || return
    swift build -c release > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        local decoded=$(swift run -c release TokenDecoder -t "$tokenizer_path" -i "$token_id" --include-special 2>/dev/null)
        echo "$decoded"
    else
        echo "Failed to build token decoder"
    fi
}

# Get tokenizer path from meta.yaml
get_tokenizer_path() {
    local meta_path=$1
    local dir=$(dirname "$meta_path")
    local tokenizer_file=$(grep "tokenizerModel:" "$meta_path" | cut -d':' -f2 | tr -d ' "')
    
    # Check if it's an absolute path or relative
    if [[ "$tokenizer_file" == /* ]]; then
        echo "$tokenizer_file"
    else
        echo "$dir/$tokenizer_file"
    fi
}

# Check if token consistency check is requested
if [[ "$1" == "@test_anemllcli.sh" ]]; then
    CHECK_TOKENS=true
    # Remove the first argument so the rest can be processed normally
    shift
fi

# Parse command line arguments
prompt="${1:-$DEFAULT_PROMPT}"
iterations="${2:-$DEFAULT_ITERATIONS}"

# Check if third argument is one of the implementations
if [[ "$3" == "swift" || "$3" == "python" ]]; then
    # Third argument is implementation, use default model path
    model_path="$DEFAULT_MODEL_PATH"
    implementation="$3"
else
    # Third argument is model path (or empty), fourth is implementation (or empty)
    model_path="${3:-$DEFAULT_MODEL_PATH}"
    implementation="${4:-$DEFAULT_IMPLEMENTATION}"
fi

# Expand tilde in model path
model_path="${model_path/#\~/$HOME}"

# Set default behavior to interrupt on divergence
INTERRUPT_ON_DIVERGENCE=true

# Set path to Python script
PYTHON_SCRIPT="../tests/chat.py"

# Function to show usage
usage() {
    echo "Usage: $0 [@test_anemllcli.sh] [prompt] [iterations] [model_path|implementation] [implementation]"
    echo ""
    echo "Arguments:"
    echo "  @test_anemllcli.sh  Optional: Check token ID consistency across runs"
    echo "  prompt              The prompt to send to the model (default: \"$DEFAULT_PROMPT\")"
    echo "  iterations          Number of times to run the test (default: $DEFAULT_ITERATIONS)"
    echo "  model_path          Path to model meta.yaml (default: $DEFAULT_MODEL_PATH)"
    echo "  implementation      Implementation to use: 'swift' or 'python' (default: swift)"
    echo ""
    echo "Flexible arguments:"
    echo "  You can specify implementation as the third argument to use the default model path:"
    echo "  $0 \"your prompt\" 5 python"
    echo ""
    echo "Environment variables:"
    echo "  NO_INTERRUPT    Set to 1 to continue testing after divergence is detected"
    echo ""
    echo "Example:"
    echo "  $0 \"tell me a joke\" 3                       # Swift with default model path"
    echo "  $0 \"tell me a joke\" 5 python               # Python with default model path" 
    echo "  $0 \"tell me a joke\" 5 ~/path/to/meta.yaml  # Swift with custom model path"
    echo "  $0 \"tell me a joke\" 5 ~/path/to/meta.yaml python  # Python with custom model path"
    echo "  $0 @test_anemllcli.sh \"tell me a joke\" 3    # Test token ID consistency"
    exit 1
}

# Show help if requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
fi

# Check if NO_INTERRUPT is set
if [[ "$NO_INTERRUPT" == "1" ]]; then
    INTERRUPT_ON_DIVERGENCE=false
fi

# Validate iterations is a number
if ! [[ "$iterations" =~ ^[0-9]+$ ]]; then
    echo "Error: iterations must be a number"
    usage
fi

# Validate implementation is either swift or python
if [[ "$implementation" != "swift" && "$implementation" != "python" ]]; then
    echo "Error: implementation must be either 'swift' or 'python'"
    usage
fi

# If checking tokens, force Swift implementation
if [[ "$CHECK_TOKENS" == true && "$implementation" != "swift" ]]; then
    echo "⚠️ Token consistency check requires Swift implementation. Switching to Swift."
    implementation="swift"
fi

# Check if model file exists
if [ ! -f "$model_path" ]; then
    echo "❌ Error: Model file not found at: $model_path"
    exit 1
fi

# Check if Python script exists when using python implementation
if [[ "$implementation" == "python" && ! -f "$PYTHON_SCRIPT" ]]; then
    echo "❌ Error: Python script not found at: $PYTHON_SCRIPT"
    exit 1
fi

echo "╔════════════════════════════════════════╗"
echo "║         ANEMLLCLI Testing Script        ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Running $iterations tests with prompt: \"$prompt\""
echo "Using model at: $model_path"
echo "Implementation: $implementation"
echo "Check token consistency: $([ "$CHECK_TOKENS" == true ] && echo "YES" || echo "NO")"
echo "Interrupt on divergence: $([ "$INTERRUPT_ON_DIVERGENCE" == true ] && echo "YES" || echo "NO")"
echo ""

# Create temp directory for outputs
TEMP_DIR=$(mktemp -d)
BASE_OUTPUT_FILE="$TEMP_DIR/response_base.txt"
BASE_OUTPUT_LOG="$TEMP_DIR/output_log_base.txt"
BASE_INPUT_TOKENS=""
BASE_OUTPUT_TOKENS=""
DIVERGENCE_DETECTED=false
TOKEN_DIVERGENCE_DETECTED=false
INTERRUPTED=false
COMPLETED_ITERATIONS=1

# Create token decoder script if token checking is enabled
if [[ "$CHECK_TOKENS" == true ]]; then
    create_token_decoder
    echo "Token decoder created for text representation"
    echo ""
fi

# Function to run command based on implementation
run_command() {
    local output_file=$1
    local log_file=$2
    
    if [[ "$implementation" == "swift" ]]; then
        if [[ "$CHECK_TOKENS" == true ]]; then
            # Run with debug level 1 to get token IDs
            swift run -c release anemllcli --meta "$model_path" --prompt "$prompt" --save "$output_file" --debug-level 2 > "$log_file" 2>&1
        else
            swift run -c release anemllcli --meta "$model_path" --prompt "$prompt" --save "$output_file" > "$log_file" 2>&1
        fi
    else
        python "$PYTHON_SCRIPT" --meta "$model_path" --prompt "$prompt" --save "$output_file" > "$log_file" 2>&1
    fi
    return $?
}

# Function to extract token IDs from log
extract_tokens() {
    local log_file=$1
    local input_tokens=$(grep "INPUT_TOKENS:" "$log_file" | cut -d' ' -f2)
    local output_tokens=$(grep "OUTPUT_TOKENS:" "$log_file" | cut -d' ' -f2)
    local stop_reason=$(grep "STOP_REASON:" "$log_file" | cut -d' ' -f2)
    local stop_token=$(grep "STOP_TOKEN:" "$log_file" | cut -d' ' -f2)
    local stop_token_text=$(grep "STOP_TOKEN_TEXT:" "$log_file" | cut -d' ' -f2-)
    
    echo "$input_tokens|$output_tokens|$stop_reason|$stop_token|$stop_token_text"
}

# Function to find first divergent token
find_first_divergence() {
    local tokens1=$1
    local tokens2=$2
    
    # Convert comma-separated tokens to arrays
    IFS=',' read -ra ARRAY1 <<< "$tokens1"
    IFS=',' read -ra ARRAY2 <<< "$tokens2"
    
    # Find the position of first difference
    local min_len=$((${#ARRAY1[@]} < ${#ARRAY2[@]} ? ${#ARRAY1[@]} : ${#ARRAY2[@]}))
    local first_diff_idx=-1
    
    for ((i=0; i<min_len; i++)); do
        if [[ "${ARRAY1[$i]}" != "${ARRAY2[$i]}" ]]; then
            first_diff_idx=$i
            break
        fi
    done
    
    # If no difference found in common length, check if one is shorter
    if [[ $first_diff_idx -eq -1 && ${#ARRAY1[@]} != ${#ARRAY2[@]} ]]; then
        first_diff_idx=$min_len
    fi
    
    # Return the index and both tokens
    if [[ $first_diff_idx -ne -1 ]]; then
        local token1="${ARRAY1[$first_diff_idx]:-EOA}"  # End of Array if index is beyond
        local token2="${ARRAY2[$first_diff_idx]:-EOA}"
        echo "$first_diff_idx:$token1:$token2"
    else
        echo "-1:N/A:N/A"  # No difference found
    fi
}

# Run the first iteration
echo "📝 Running test iteration 1 of $iterations..."
run_command "$BASE_OUTPUT_FILE" "$BASE_OUTPUT_LOG"
if [ $? -ne 0 ]; then
    echo "❌ Error running test. Check if the model path is correct."
    echo "📋 Error details from log file:"
    echo "-----------------------------------"
    cat "$BASE_OUTPUT_LOG" | tail -n 20
    echo "-----------------------------------"
    echo "📁 Model path: $model_path"
    # Check if file exists and show permissions
    if [ -f "$model_path" ]; then
        echo "✅ Model file exists"
        ls -l "$model_path"
    else
        echo "❌ Model file does not exist"
    fi
    exit 1
fi

echo "✅ Base response saved to $BASE_OUTPUT_FILE"

# Extract token IDs if checking tokens
if [[ "$CHECK_TOKENS" == true ]]; then
    TOKENS=$(extract_tokens "$BASE_OUTPUT_LOG")
    if [[ -n "$TOKENS" ]]; then
        BASE_INPUT_TOKENS=$(echo "$TOKENS" | cut -d'|' -f1)
        BASE_OUTPUT_TOKENS=$(echo "$TOKENS" | cut -d'|' -f2)
        BASE_STOP_REASON=$(echo "$TOKENS" | cut -d'|' -f3)
        BASE_STOP_TOKEN=$(echo "$TOKENS" | cut -d'|' -f4)
        BASE_STOP_TOKEN_TEXT=$(echo "$TOKENS" | cut -d'|' -f5)
        echo "✅ Base token IDs extracted"
        echo "   Stop reason: $BASE_STOP_REASON"
        if [[ -n "$BASE_STOP_TOKEN" ]]; then
            echo "   Stop token: $BASE_STOP_TOKEN ($BASE_STOP_TOKEN_TEXT)"
        fi
    else
        echo "⚠️ Could not extract token IDs. Make sure debug level is enabled."
    fi
fi
echo ""

# Run remaining iterations and compare
for ((i=2; i<=iterations; i++)); do
    CURRENT_OUTPUT_FILE="$TEMP_DIR/response_$i.txt"
    CURRENT_OUTPUT_LOG="$TEMP_DIR/output_log_$i.txt"
    
    echo "📝 Running test iteration $i of $iterations..."
    run_command "$CURRENT_OUTPUT_FILE" "$CURRENT_OUTPUT_LOG"
    
    # Compare with base response
    echo "🔍 Comparing with base response..."
    DIFF_OUTPUT=$(diff -u "$BASE_OUTPUT_FILE" "$CURRENT_OUTPUT_FILE")
    DIFF_STATUS=$?
    
    COMPLETED_ITERATIONS=$i
    
    # Check for text divergence
    if [ $DIFF_STATUS -eq 0 ]; then
        echo "✅ No text divergence detected in iteration $i"
    else
        echo "⚠️  TEXT DIVERGENCE DETECTED in iteration $i"
        DIVERGENCE_DETECTED=true
        
        # Create a diff file
        DIFF_FILE="$TEMP_DIR/diff_$i.txt"
        echo "$DIFF_OUTPUT" > "$DIFF_FILE"
        echo "   Diff saved to $DIFF_FILE"
        
        # Show a brief summary of differences
        DIFFERENT_LINES=$(echo "$DIFF_OUTPUT" | grep -E "^[\+\-]" | wc -l)
        echo "   $DIFFERENT_LINES lines differ between base response and iteration $i"
    fi
    
    # Check token consistency if enabled
    if [[ "$CHECK_TOKENS" == true && -n "$BASE_INPUT_TOKENS" ]]; then
        TOKENS=$(extract_tokens "$CURRENT_OUTPUT_LOG")
        CURRENT_INPUT_TOKENS=$(echo "$TOKENS" | cut -d'|' -f1)
        CURRENT_OUTPUT_TOKENS=$(echo "$TOKENS" | cut -d'|' -f2)
        CURRENT_STOP_REASON=$(echo "$TOKENS" | cut -d'|' -f3)
        CURRENT_STOP_TOKEN=$(echo "$TOKENS" | cut -d'|' -f4)
        CURRENT_STOP_TOKEN_TEXT=$(echo "$TOKENS" | cut -d'|' -f5)
        
        # Compare stop reason/token
        if [[ "$BASE_STOP_REASON" == "$CURRENT_STOP_REASON" ]]; then
            echo "✅ Stop reason matches: $CURRENT_STOP_REASON"
        else
            echo "⚠️  STOP REASON DIVERGENCE in iteration $i"
            echo "   Base stop reason: $BASE_STOP_REASON"
            echo "   Current stop reason: $CURRENT_STOP_REASON"
            TOKEN_DIVERGENCE_DETECTED=true
        fi
        
        if [[ -n "$BASE_STOP_TOKEN" && -n "$CURRENT_STOP_TOKEN" ]]; then
            if [[ "$BASE_STOP_TOKEN" == "$CURRENT_STOP_TOKEN" ]]; then
                echo "✅ Stop token matches: $CURRENT_STOP_TOKEN ($CURRENT_STOP_TOKEN_TEXT)"
            else
                echo "⚠️  STOP TOKEN DIVERGENCE in iteration $i"
                echo "   Base stop token: $BASE_STOP_TOKEN ($BASE_STOP_TOKEN_TEXT)"
                echo "   Current stop token: $CURRENT_STOP_TOKEN ($CURRENT_STOP_TOKEN_TEXT)"
                TOKEN_DIVERGENCE_DETECTED=true
            fi
        fi
        
        # Compare input tokens
        if [[ "$BASE_INPUT_TOKENS" == "$CURRENT_INPUT_TOKENS" ]]; then
            echo "✅ Input tokens match in iteration $i"
        else
            echo "⚠️  INPUT TOKEN DIVERGENCE in iteration $i"
            TOKEN_DIVERGENCE_DETECTED=true
            # Find first divergent token
            DIVERGENCE_INFO=$(find_first_divergence "$BASE_INPUT_TOKENS" "$CURRENT_INPUT_TOKENS")
            DIVERGENCE_IDX=$(echo "$DIVERGENCE_INFO" | cut -d':' -f1)
            BASE_TOKEN=$(echo "$DIVERGENCE_INFO" | cut -d':' -f2)
            CURRENT_TOKEN=$(echo "$DIVERGENCE_INFO" | cut -d':' -f3)
            
            echo "   First divergence at position $DIVERGENCE_IDX:"
            echo "   Base token ID: $BASE_TOKEN"
            echo "   Current token ID: $CURRENT_TOKEN"
            
            # Decode tokens if decoder exists
            if [[ -f "$DECODE_SCRIPT" && "$BASE_TOKEN" != "EOA" && "$CURRENT_TOKEN" != "EOA" ]]; then
                TOKENIZER_PATH=$(get_tokenizer_path "$model_path")
                if [[ -n "$TOKENIZER_PATH" && -f "$TOKENIZER_PATH" ]]; then
                    BASE_TOKEN_TEXT=$(decode_token "$BASE_TOKEN" "$TOKENIZER_PATH")
                    CURRENT_TOKEN_TEXT=$(decode_token "$CURRENT_TOKEN" "$TOKENIZER_PATH")
                    echo "   Base token text: '$BASE_TOKEN_TEXT'"
                    echo "   Current token text: '$CURRENT_TOKEN_TEXT'"
                fi
            fi
            
            # Save token diff
            echo "BASE INPUT TOKENS: $BASE_INPUT_TOKENS" > "$TEMP_DIR/token_diff_input_$i.txt"
            echo "ITER INPUT TOKENS: $CURRENT_INPUT_TOKENS" >> "$TEMP_DIR/token_diff_input_$i.txt"
            echo "FIRST DIVERGENCE AT: $DIVERGENCE_IDX (Base: $BASE_TOKEN, Current: $CURRENT_TOKEN)" >> "$TEMP_DIR/token_diff_input_$i.txt"

            # Also record stop reason/token in diff file
            echo "STOP INFORMATION:" >> "$TEMP_DIR/token_diff_input_$i.txt"
            echo "BASE STOP REASON: $BASE_STOP_REASON" >> "$TEMP_DIR/token_diff_input_$i.txt"
            echo "ITER STOP REASON: $CURRENT_STOP_REASON" >> "$TEMP_DIR/token_diff_input_$i.txt"
            if [[ -n "$BASE_STOP_TOKEN" && -n "$CURRENT_STOP_TOKEN" ]]; then
                echo "BASE STOP TOKEN: $BASE_STOP_TOKEN ($BASE_STOP_TOKEN_TEXT)" >> "$TEMP_DIR/token_diff_input_$i.txt"
                echo "ITER STOP TOKEN: $CURRENT_STOP_TOKEN ($CURRENT_STOP_TOKEN_TEXT)" >> "$TEMP_DIR/token_diff_input_$i.txt"
            fi
        fi
        
        # Compare output tokens
        if [[ "$BASE_OUTPUT_TOKENS" == "$CURRENT_OUTPUT_TOKENS" ]]; then
            echo "✅ Output tokens match in iteration $i"
        else
            echo "⚠️  OUTPUT TOKEN DIVERGENCE in iteration $i"
            TOKEN_DIVERGENCE_DETECTED=true
            # Find first divergent token
            DIVERGENCE_INFO=$(find_first_divergence "$BASE_OUTPUT_TOKENS" "$CURRENT_OUTPUT_TOKENS")
            DIVERGENCE_IDX=$(echo "$DIVERGENCE_INFO" | cut -d':' -f1)
            BASE_TOKEN=$(echo "$DIVERGENCE_INFO" | cut -d':' -f2)
            CURRENT_TOKEN=$(echo "$DIVERGENCE_INFO" | cut -d':' -f3)
            
            echo "   First divergence at position $DIVERGENCE_IDX:"
            echo "   Base token ID: $BASE_TOKEN"
            echo "   Current token ID: $CURRENT_TOKEN"
            
            # Decode tokens if decoder exists
            if [[ -f "$DECODE_SCRIPT" && "$BASE_TOKEN" != "EOA" && "$CURRENT_TOKEN" != "EOA" ]]; then
                TOKENIZER_PATH=$(get_tokenizer_path "$model_path")
                if [[ -n "$TOKENIZER_PATH" && -f "$TOKENIZER_PATH" ]]; then
                    BASE_TOKEN_TEXT=$(decode_token "$BASE_TOKEN" "$TOKENIZER_PATH")
                    CURRENT_TOKEN_TEXT=$(decode_token "$CURRENT_TOKEN" "$TOKENIZER_PATH")
                    echo "   Base token text: '$BASE_TOKEN_TEXT'"
                    echo "   Current token text: '$CURRENT_TOKEN_TEXT'"
                fi
            fi
            
            # Save token diff
            echo "BASE OUTPUT TOKENS: $BASE_OUTPUT_TOKENS" > "$TEMP_DIR/token_diff_output_$i.txt"
            echo "ITER OUTPUT TOKENS: $CURRENT_OUTPUT_TOKENS" >> "$TEMP_DIR/token_diff_output_$i.txt"
            echo "FIRST DIVERGENCE AT: $DIVERGENCE_IDX (Base: $BASE_TOKEN, Current: $CURRENT_TOKEN)" >> "$TEMP_DIR/token_diff_output_$i.txt"

            # Also record stop reason/token in diff file
            echo "STOP INFORMATION:" >> "$TEMP_DIR/token_diff_output_$i.txt"
            echo "BASE STOP REASON: $BASE_STOP_REASON" >> "$TEMP_DIR/token_diff_output_$i.txt"
            echo "ITER STOP REASON: $CURRENT_STOP_REASON" >> "$TEMP_DIR/token_diff_output_$i.txt"
            if [[ -n "$BASE_STOP_TOKEN" && -n "$CURRENT_STOP_TOKEN" ]]; then
                echo "BASE STOP TOKEN: $BASE_STOP_TOKEN ($BASE_STOP_TOKEN_TEXT)" >> "$TEMP_DIR/token_diff_output_$i.txt"
                echo "ITER STOP TOKEN: $CURRENT_STOP_TOKEN ($CURRENT_STOP_TOKEN_TEXT)" >> "$TEMP_DIR/token_diff_output_$i.txt"
            fi
        fi
    fi
    
    # Interrupt if flag is set and divergence detected
    SHOULD_INTERRUPT=false
    if [[ "$DIVERGENCE_DETECTED" == true || "$TOKEN_DIVERGENCE_DETECTED" == true ]]; then
        SHOULD_INTERRUPT=true
    fi
    
    if [[ "$INTERRUPT_ON_DIVERGENCE" == true && "$SHOULD_INTERRUPT" == true ]]; then
        echo ""
        echo "⛔ Test interrupted after detecting divergence in iteration $i"
        INTERRUPTED=true
        break
    fi
    echo ""
done

# Summary
echo "╔════════════════════════════════════════╗"
echo "║               Test Summary              ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Completed iterations: $COMPLETED_ITERATIONS of $iterations"
if [ "$INTERRUPTED" = true ]; then
    echo "⛔ Test was interrupted after divergence detected"
fi
echo "Prompt used: \"$prompt\""

# Display results based on what was tested
if [[ "$CHECK_TOKENS" == true ]]; then
    INPUT_TOKEN_MATCH="${BASE_INPUT_TOKENS:+true}"
    OUTPUT_TOKEN_MATCH="${BASE_OUTPUT_TOKENS:+true}"
    
    # More detailed summary with specific details about which tokens diverged
    echo "Token consistency check results:"
    if [[ "$TOKEN_DIVERGENCE_DETECTED" == true ]]; then
        # Check which type of token divergence was detected
        INPUT_DIVERGED=false
        OUTPUT_DIVERGED=false
        
        # Check if we have input token diff files
        if ls "$TEMP_DIR"/token_diff_input_*.txt &> /dev/null; then
            INPUT_DIVERGED=true
            echo "  ⚠️  INPUT TOKENS diverged across iterations"
        else
            echo "  ✅ INPUT TOKENS consistent across all iterations"
        fi
        
        # Check if we have output token diff files
        if ls "$TEMP_DIR"/token_diff_output_*.txt &> /dev/null; then
            OUTPUT_DIVERGED=true
            echo "  ⚠️  OUTPUT TOKENS diverged across iterations"
        else
            echo "  ✅ OUTPUT TOKENS consistent across all iterations"
        fi
    else
        echo "  ✅ All tokens (input and output) consistent across iterations"
    fi
    
    # Overall result summary
    if [[ "$DIVERGENCE_DETECTED" == true && "$TOKEN_DIVERGENCE_DETECTED" == true ]]; then
        echo "⚠️  RESULT: Both text output and token IDs diverged"
    elif [[ "$DIVERGENCE_DETECTED" == true ]]; then
        echo "⚠️  RESULT: Text output diverged but token IDs were consistent"
    elif [[ "$TOKEN_DIVERGENCE_DETECTED" == true ]]; then
        if [[ "$INPUT_DIVERGED" == true && "$OUTPUT_DIVERGED" == true ]]; then
            echo "⚠️  RESULT: Both input and output token IDs diverged (text was consistent)"
        elif [[ "$INPUT_DIVERGED" == true ]]; then
            echo "⚠️  RESULT: Input token IDs diverged (output tokens and text were consistent)"
        else
            echo "⚠️  RESULT: Output token IDs diverged (input tokens and text were consistent)"
        fi
    else
        echo "✅ RESULT: No divergence detected in text or tokens across all iterations"
    fi
else
    if [[ "$DIVERGENCE_DETECTED" == true ]]; then
        echo "⚠️  RESULT: Text divergence detected in at least one iteration"
    else
        echo "✅ RESULT: No divergence detected across all iterations"
    fi
fi

echo "Output files saved in $TEMP_DIR"
echo ""
if [[ "$DIVERGENCE_DETECTED" == true || "$TOKEN_DIVERGENCE_DETECTED" == true ]]; then
    echo "To examine differences:"
    echo "  cat $TEMP_DIR/diff_*.txt        # For text differences"
    if [[ "$CHECK_TOKENS" == true ]]; then
        echo "  cat $TEMP_DIR/token_diff_*.txt  # For token differences"
    fi
    echo ""
else
    echo "All responses were consistent!"
    # Optional cleanup if no divergence
    read -p "Clean up temporary files? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TEMP_DIR"
        echo "Temporary files deleted."
    else
        echo "Files saved in $TEMP_DIR"
    fi
fi

# Clean up token decoder
if [[ "$CHECK_TOKENS" == true && -d "$DECODE_DIR" ]]; then
    rm -rf "$DECODE_DIR"
fi
