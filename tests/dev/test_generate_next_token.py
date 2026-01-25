#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat import (
    run_prefill,
    generate_next_token,
    create_unified_state,
    initialize_causal_mask,
    initialize_tokenizer,
    parse_model_path,
    load_models
)
import torch
from transformers import AutoTokenizer

def test_generate_next_token():
    model_dir = "/tmp/test-qwen25-0.5b-base"
    
    # Load models
    models = parse_model_path(model_dir)
    embed_model, ffn_models, lmhead_model, metadata = load_models(model_dir, models)
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer(model_dir, eval_mode=True)
    
    # Create state and causal mask
    state = create_unified_state(ffn_models, metadata['context_length'], eval_mode=True)
    causal_mask = initialize_causal_mask(metadata['context_length'], eval_mode=True)
    
    # Read the exact prompt
    with open('/tmp/boolq_exact_prompt.txt', 'r') as f:
        prompt_text = f.read().strip()
    
    # Tokenize exactly like chat.py does
    input_ids = tokenizer(
        prompt_text,
        return_tensors='pt',
        add_special_tokens=True
    ).input_ids.to(torch.int32)
    
    print(f"Chat.py style tokenization: {input_ids.shape[1]} tokens")
    print(f"Last 5 tokens: {input_ids[0, -5:].tolist()}")
    
    # Run prefill exactly like chat.py does
    context_pos = input_ids.size(1)
    
    _ = run_prefill(
        embed_model,
        ffn_models,
        input_ids,
        context_pos,
        metadata['context_length'],
        metadata.get('batch_size', 64),
        state,
        causal_mask
    )
    
    print(f"Ran prefill with context_pos={context_pos}")
    
    # Generate next token exactly like chat.py does
    next_token = generate_next_token(
        embed_model,
        ffn_models,
        lmhead_model,
        input_ids,
        context_pos,  # This is the position where chat.py generates
        metadata['context_length'],
        metadata,
        state,
        causal_mask
    )
    
    print(f"Chat.py style prediction: token {next_token}")
    print(f"Decoded: '{tokenizer.decode([next_token])}'")

if __name__ == "__main__":
    test_generate_next_token()