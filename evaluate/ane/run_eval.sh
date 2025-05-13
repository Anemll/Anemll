#!/bin/bash
# run_eval.sh - ANE model evaluation with lm-evaluation-harness
# Uses the abstracted ANE_Model class for better state handling

# Set default values
MODEL_PATH="/Users/anemll/Models/ANE/anemll-Llama-3.2-1B-FP16-b64-ctx1024"
TASKS="hellaswag"
NUM_SHOTS=0
BATCH_SIZE=1  # Always use batch_size=1 for ANE models
OUTPUT_DIR="results"
LIMIT=""
MAX_TOKENS=""
CHAT_TEMPLATE=""
SEED=123
DEBUG=""
PERPLEXITY_TEXT=""
SKIP=""
CHUNK_SIZE=""
SAFETY_MARGIN=100  # Default safety margin

print_header() {
    echo "=================== ANE MODEL EVALUATION ==================="
    echo "Model path:   ${MODEL_PATH}"
    echo "Tasks:        ${TASKS}"
    echo "Num shots:    ${NUM_SHOTS}"
    echo "Batch size:   ${BATCH_SIZE} (for strict serial execution)"
    echo "Output dir:   ${OUTPUT_DIR}"
    if [ -n "$DEBUG" ]; then
        echo "Debug mode:   ENABLED"
    fi
    if [ -n "$LIMIT" ]; then
        echo "Limit:        ${LIMIT} samples"
    fi
    if [ -n "$SKIP" ]; then
        echo "Skip:         ${SKIP} samples"
    fi
    if [ -n "$CHUNK_SIZE" ]; then
        echo "Chunk size:   ${CHUNK_SIZE} tokens"
    fi
    if [ -n "$SAFETY_MARGIN" ]; then
        echo "Safety margin: ${SAFETY_MARGIN} tokens"
    fi
    echo "=========================================================="
}

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --shots)
            NUM_SHOTS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --skip)
            SKIP="--skip $2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --chat-template)
            CHAT_TEMPLATE="--chat-template $2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="--max_tokens $2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --perplexity)
            # If no argument is provided, use wikitext as default
            if [[ "$2" == --* ]] || [[ -z "$2" ]]; then
                PERPLEXITY_TEXT="wikitext"
            shift
            else
                PERPLEXITY_TEXT="$2"
                shift 2
            fi
            ;;
        --chunk-size)
            CHUNK_SIZE="--chunk-size $2"
            shift 2
            ;;
        --safety-margin)
            SAFETY_MARGIN="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run evaluation based on task selection
if [ -z "$PERPLEXITY_TEXT" ]; then
    # Non-perplexity tasks - standard benchmarks
    echo "Running LM evaluation with ANE_Model on ${TASKS}..."
    python anelm_harness.py \
        --model ${MODEL_PATH} \
        --tasks ${TASKS} \
        --num-shots ${NUM_SHOTS} \
        --batch-size ${BATCH_SIZE} \
        ${CHAT_TEMPLATE} \
        ${LIMIT} \
        ${MAX_TOKENS} \
        ${DEBUG} \
        ${SKIP} \
        ${CHUNK_SIZE} \
        --safety-margin ${SAFETY_MARGIN} \
        --seed ${SEED} \
        --output-dir ${OUTPUT_DIR} \
        --output-path ${OUTPUT_DIR}/results.json
else
    # Perplexity evaluation with appropriate handling
    echo "Running perplexity evaluation..."
    
    # Check if perplexity_text is "wikitext" to use the dataset
    if [ "$PERPLEXITY_TEXT" = "wikitext" ]; then
        echo "Using wikitext dataset for perplexity"
        
        # Use the harness for wikitext perplexity
        echo "Running with harness perplexity evaluation on wikitext..."
        python anelm_harness.py \
            --model ${MODEL_PATH} \
            --tasks wikitext \
            --batch-size ${BATCH_SIZE} \
            ${DEBUG} \
            ${CHUNK_SIZE} \
            --safety-margin ${SAFETY_MARGIN} \
            --seed ${SEED} \
            --perplexity \
            --output-dir ${OUTPUT_DIR} \
            --output-path ${OUTPUT_DIR}/perplexity_results.json
            
    elif [ -f "$PERPLEXITY_TEXT" ]; then
        # Custom text file perplexity calculation
        echo "Using custom text file: ${PERPLEXITY_TEXT}"
        
        # Create a custom perplexity script - Part 1: Imports and setup
        TEMP_SCRIPT="/tmp/calc_perplexity_$$.py"
        cat > "${TEMP_SCRIPT}" << 'EOF'
#!/usr/bin/env python3
# Temporary script for perplexity calculation

import os
import sys
import numpy as np
import math
import torch
from tqdm import tqdm

# Import ANE_Model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ane_model import ANE_Model
except ImportError:
    print("Failed to import ANE_Model from standard paths")
    # Try alternative paths
    alt_paths = [
        os.path.dirname(os.path.abspath("${0}")),  # Script directory
        "${PWD}",  # Current working directory
        "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/evaluate/ane",  # Absolute path
        os.path.abspath(".")  # Current directory
    ]
    
    imported = False
    for path in alt_paths:
        try:
            print(f"Trying path: {path}")
            sys.path.append(path)
            from ane_model import ANE_Model
            print(f"Successfully imported ANE_Model from {path}")
            imported = True
            break
        except ImportError:
            continue
    
    if not imported:
        print("Error: Could not import ANE_Model from any path")
        sys.exit(1)

# Try to import tokenizer
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers package not found. Please install it with:")
    print("pip install transformers")
    sys.exit(1)
EOF

        # Part 2: Main perplexity calculation function
        cat >> "${TEMP_SCRIPT}" << 'EOF'

def calculate_perplexity(model_path, text_file, debug=False, chunk_size=None, safety_margin=100):
    """Calculate perplexity for text in a file using sliding window approach."""
    # Initialize ANE model
    print(f"Loading model from {model_path}")
    model = ANE_Model(model_path)
    model.debug = 1 if debug else 0
    
    # Get context length from model metadata
    ctx_length = model.metadata.get('context_length', 2048)
    safe_length = ctx_length - safety_margin
    print(f"Model context length: {ctx_length}")
    print(f"Safe length with margin: {safe_length}")
    
    # Use provided chunk size or a default based on safe_length
    if chunk_size is None:
        chunk_size = min(safe_length, 512)  # Default to 512 or less if safe_length is smaller
    chunk_size = min(chunk_size, safe_length)  # Ensure chunk_size doesn't exceed safe_length
    print(f"Using chunk size: {chunk_size}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer is None:
            raise ValueError("Tokenizer not found")
    except Exception as e:
        print(f"Error loading tokenizer from {model_path}: {str(e)}")
        print("Falling back to Llama tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Read text file
    print(f"Reading text from {text_file}")
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(text_file, 'r', encoding='latin-1') as f:
            text = f.read()
            
    # Tokenize the text
    tokens = tokenizer.encode(text)
    print(f"Text tokenized to {len(tokens)} tokens")
    
    # Split into manageable chunks
    chunks = []
    for i in range(0, len(tokens) - 1, chunk_size // 2):  # 50% overlap between chunks
        end = min(i + chunk_size, len(tokens))
        if end - i > 1:  # Only use chunks with at least 2 tokens
            chunks.append(tokens[i:end])
    
    print(f"Split into {len(chunks)} chunks for processing")
    
    # Calculate log-likelihood for each chunk
    total_log_likelihood = 0.0
    total_tokens = 0
    
    # Create progress bar
    pbar = tqdm(total=len(chunks), desc="Perplexity", unit="chunk")
    
    for chunk_idx, chunk in enumerate(chunks):
        if len(chunk) <= 1:
            pbar.update(1)
            continue
            
        # Split into input and target
        inputs, targets = chunk[:-1], chunk[1:]
        
        try:
            # Reset model state for this chunk
            model.reset_state()
            
            # Prefill with input tokens
            input_tensor = torch.tensor([inputs], dtype=torch.int32)
            _ = model.prefill(input_tensor)
            
            # Score each target token
            chunk_log_likelihood = 0.0
            processed_tokens = 0
            
            for i, target in enumerate(targets):
                # Get current token
                current_token = inputs[i] if i < len(inputs) else targets[i-1]
                
                try:
                    # Get log probabilities for current token
                    token_tensor = torch.tensor([[current_token]], dtype=torch.int32)
                    log_probs = model.compute_logprobs(token_tensor)
                    
                    if log_probs is None:
                        if debug:
                            print(f"Warning: No log probabilities for token at position {i}")
                        continue
                    
                    # Score the target token
                    token_score = log_probs[target].item()
                    chunk_log_likelihood += token_score
                    processed_tokens += 1
                    
                    # Update state if not the last token
                    if i < len(targets) - 1:
                        _ = model.predict(token_tensor)
                        
                except Exception as e:
                    if debug:
                        print(f"Error processing token {i} in chunk {chunk_idx}: {str(e)}")
                    # Continue with next token
                    continue
            
            # Update totals
            if processed_tokens > 0:
                total_log_likelihood += chunk_log_likelihood
                total_tokens += processed_tokens
                
                # Update progress bar
                current_ppl = math.exp(-chunk_log_likelihood / processed_tokens) if processed_tokens > 0 else 0
                pbar.set_postfix(ppl=f"{current_ppl:.2f}")
                
        except Exception as e:
            print(f"\nError processing chunk {chunk_idx}: {str(e)}")
            if debug:
                import traceback
                traceback.print_exc()
        
        pbar.update(1)
    
    pbar.close()
    
    # Calculate perplexity
    if total_tokens > 0:
        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = math.exp(-avg_log_likelihood)
        print(f"\nTotal tokens scored: {total_tokens}")
        print(f"Average log likelihood: {avg_log_likelihood:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
        return perplexity
    else:
        print("Error: No tokens were successfully processed")
        return float('inf')
EOF

        # Part 3: Main function
        cat >> "${TEMP_SCRIPT}" << EOF
if __name__ == "__main__":
    # Get script directory from first argument if provided
    if len(sys.argv) > 1:
        script_dir = sys.argv[1]
        sys.path.append(script_dir)
        print(f"Added script directory to path: {script_dir}")
    
    model_path = "${MODEL_PATH}"
    text_file = "${PERPLEXITY_TEXT}"
    debug = ${DEBUG:+True} ${DEBUG:-False}
    chunk_size = ${CHUNK_SIZE:+${CHUNK_SIZE##*--chunk-size }} ${CHUNK_SIZE:-None}
    safety_margin = ${SAFETY_MARGIN}
    
    perplexity = calculate_perplexity(model_path, text_file, debug, chunk_size, safety_margin)
    
    # Save result to file
    output_dir = "${OUTPUT_DIR}"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "perplexity_results.txt"), "w") as f:
        f.write(f"Perplexity: {perplexity:.4f}\n")
        f.write(f"Text file: {text_file}\n")
        f.write(f"Model: {model_path}\n")
EOF

        # Make the script executable
        chmod +x "${TEMP_SCRIPT}"
        
        # Run the perplexity script
        echo "Running custom perplexity calculation..."
        SCRIPT_DIR="$(pwd)"
        python3 "${TEMP_SCRIPT}" "${SCRIPT_DIR}"
        
        # Clean up
        rm "${TEMP_SCRIPT}"
    else
        # Default sample text for quick testing
        echo "Using default sample text for perplexity testing"
        
        # Create a simple perplexity script with default text
        TEMP_SCRIPT="/tmp/ppl_default_$$.py"
        cat > "${TEMP_SCRIPT}" << 'EOF'
#!/usr/bin/env python3
# Quick perplexity test with default text

import os
import sys
import math
import torch

# Get the absolute path to the ANE modules
current_script_dir = os.path.dirname(os.path.abspath("${0}"))
sys.path.append(current_script_dir)

# Import ANE_Model - first try with absolute path
try:
    script_dir = "${PWD}"
    sys.path.append(script_dir)
    from ane_model import ANE_Model
except ImportError:
    print(f"Failed to import ANE_Model from {script_dir}")
    print("Trying alternate paths...")
    
    # Try alternative paths
    alt_paths = [
        "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/evaluate/ane",
        os.path.dirname(os.path.abspath(__file__)),
        os.path.abspath(".")
    ]
    
    imported = False
    for path in alt_paths:
        try:
            print(f"Trying path: {path}")
            sys.path.append(path)
            from ane_model import ANE_Model
            print(f"Successfully imported ANE_Model from {path}")
            imported = True
            break
        except ImportError:
            continue
    
    if not imported:
        print("Error: Could not import ANE_Model from any path")
        sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers package not found")
    sys.exit(1)

# Sample text for testing
SAMPLE_TEXT = """
The quick brown fox jumps over the lazy dog. This pangram contains all the letters of the English alphabet.
Machine learning models process text by converting words into numerical representations called embeddings.
These embeddings capture semantic relationships between words, allowing models to understand language.
The Apple Neural Engine accelerates machine learning tasks on Apple devices, enabling efficient inference.
"""

def main():
    model_path = sys.argv[1]
    debug = len(sys.argv) > 2 and sys.argv[2] == "True"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "results"
    script_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.dirname(os.path.abspath(__file__))
    
    # Add script directory to path for importing modules
    sys.path.append(script_dir)
    print(f"Added script directory to path: {script_dir}")
    
    # Initialize model
    print(f"Loading model from {model_path}")
    model = ANE_Model(model_path)
    model.debug = 1 if debug else 0
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        print("Falling back to default tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Tokenize sample text
    tokens = tokenizer.encode(SAMPLE_TEXT)
    print(f"Sample text tokenized to {len(tokens)} tokens")
    
    # Process the tokens
    inputs, targets = tokens[:-1], tokens[1:]
    
    # Reset model state
    model.reset_state()
    
    # Prefill with input tokens
    input_tensor = torch.tensor([inputs], dtype=torch.int32)
    _ = model.prefill(input_tensor)
    
    # Score each target token
    total_log_likelihood = 0.0
    total_tokens = 0
    
    for i, target in enumerate(targets):
        # Get current token
        current_token = inputs[i]
        
        # Get log probabilities
        token_tensor = torch.tensor([[current_token]], dtype=torch.int32)
        log_probs = model.compute_logprobs(token_tensor)
        
        if log_probs is None:
            print(f"Warning: No log probabilities for token at position {i}")
            continue
        
        # Score target token
        try:
            token_score = log_probs[target].item()
            total_log_likelihood += token_score
            total_tokens += 1
            
            if debug and i < 5:
                print(f"Token {i}: {current_token} → {target}, score: {token_score:.4f}")
            
            # Update model state
            if i < len(targets) - 1:
                _ = model.predict(token_tensor)
                
        except Exception as e:
            print(f"Error scoring token {target}: {str(e)}")
    
    # Calculate perplexity
    if total_tokens > 0:
        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = math.exp(-avg_log_likelihood)
        print(f"\nTotal tokens scored: {total_tokens}")
        print(f"Average log likelihood: {avg_log_likelihood:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
        
        # Save result
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "perplexity_results.txt"), "w") as f:
            f.write(f"Sample text perplexity: {perplexity:.4f}\n")
    else:
        print("Error: No tokens were successfully processed")

if __name__ == "__main__":
    main()
EOF

        # Make the script executable
        chmod +x "${TEMP_SCRIPT}"
        
        # Run the default perplexity script
        echo "Running default sample text perplexity test..."
        SCRIPT_DIR="$(pwd)"
        python3 "${TEMP_SCRIPT}" "${MODEL_PATH}" "${DEBUG:+True}${DEBUG:-False}" "${OUTPUT_DIR}" "${SCRIPT_DIR}"
        
        # Clean up
        rm "${TEMP_SCRIPT}"
    fi
fi

echo "Evaluation complete! Results saved to ${OUTPUT_DIR}"
exit 0 