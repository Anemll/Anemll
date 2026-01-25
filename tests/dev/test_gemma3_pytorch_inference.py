#!/usr/bin/env python3
"""Test Gemma3 inference in pure PyTorch using the ANEMLL model implementation.

This test verifies that the ANEMLL PyTorch model (gemma3_model.py) produces
correct outputs when compared to HuggingFace reference.
"""

import sys
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from anemll.models.gemma3_model import (
    Gemma3ForCausalLM,
    Gemma3Config,
    MODEL_DTYPE,
)
from anemll.ane_converter.gemma3_converter import Gemma3Converter


def load_anemll_model(model_path, context_length=64):
    """Load ANEMLL model with weights from HuggingFace."""
    config = Gemma3Config.from_json(f'{model_path}/config.json')
    config.context_length = context_length
    config.state_length = context_length

    model = Gemma3ForCausalLM(config)
    converter = Gemma3Converter(model, context_length=context_length, batch_size=context_length)
    converter.load_weights_from_hf(model_path)
    model.eval()
    return model


def generate_single_token(model, hidden_states, position, causal_mask=None):
    """Generate a single token using the ANEMLL model's process_layers."""
    seq_len = hidden_states.shape[1]

    # Create position_ids for the current position
    position_ids = torch.arange(seq_len).unsqueeze(0)
    current_pos = torch.tensor([position])

    # Process through all layers
    for i, layer in enumerate(model.model.layers):
        # Get layer-specific RoPE
        if hasattr(model.model.config, "layer_types") and model.model.config.layer_types[i] == "full_attention":
            rotary_emb = model.model.rotary_emb_global
        else:
            rotary_emb = model.model.rotary_emb_local
        layer.self_attn.rotary_emb = rotary_emb

        # Forward through layer
        hidden_states = layer(hidden_states, causal_mask, position_ids, current_pos)

    # Apply final norm
    hidden_states = model.model.norm(hidden_states)
    return hidden_states


def get_logits(model, hidden_states):
    """Get logits from hidden states using split LM head."""
    # Take only the last token
    hidden_for_lm = hidden_states[:, -1:, :]  # [1, 1, hidden_size]
    hidden_for_lm = hidden_for_lm.permute(0, 2, 1).unsqueeze(2)  # [1, hidden_size, 1, 1]

    # Apply all 16 lm_head splits
    all_logits = []
    for j in range(1, 17):
        lm_head = getattr(model, f'lm_head16_{j}')
        part_logits = lm_head(hidden_for_lm).squeeze(2).transpose(1, 2)  # [1, 1, vocab/16]
        all_logits.append(part_logits)

    logits = torch.cat(all_logits, dim=-1).squeeze(1)  # [1, vocab]
    return logits


def main():
    parser = argparse.ArgumentParser(description='Test Gemma3 PyTorch inference')
    parser.add_argument('--model', type=str,
                        default='/Users/anemll/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3',
                        help='Path to HuggingFace model')
    parser.add_argument('--prompt', type=str, default='What is the capital of France?',
                        help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=20, help='Max tokens to generate')
    parser.add_argument('--context-length', type=int, default=64, help='Context length')
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load ANEMLL model
    anemll_model = load_anemll_model(args.model, args.context_length)

    # Load HF model for comparison (bfloat16)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True
    )
    hf_model.eval()

    # Prepare input
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

    print(f"\nPrompt: {args.prompt}")
    print(f"Tokenized input: {input_ids.shape}")
    print(f"Tokens: {input_ids[0].tolist()[:10]}...")

    # === ANEMLL Inference ===
    print("\n=== ANEMLL PyTorch Inference ===")
    with torch.no_grad():
        # Get embeddings
        embed = anemll_model.model.embed_tokens(input_ids)
        embed_scaled = embed * anemll_model.model.embedding_scale
        hidden_states = embed_scaled.to(MODEL_DTYPE)

        seq_len = hidden_states.shape[1]

        # Create causal mask
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).to(MODEL_DTYPE)

        # Generate tokens
        generated_tokens = []
        current_hidden = hidden_states

        for step in range(args.max_tokens):
            # Process through model
            output_hidden = generate_single_token(
                anemll_model, current_hidden, position=0, causal_mask=causal_mask
            )

            # Get logits and sample
            logits = get_logits(anemll_model, output_hidden)
            next_token = logits.argmax(dim=-1).item()
            generated_tokens.append(next_token)

            print(f"Step {step}: token={next_token} -> '{tokenizer.decode([next_token])}'")

            # Check for EOS
            if next_token in [1, 106]:  # Gemma3 EOS tokens
                break

            # For next iteration, append new token to input
            new_token = torch.tensor([[next_token]], dtype=torch.long)
            new_embed = anemll_model.model.embed_tokens(new_token)
            new_embed_scaled = new_embed * anemll_model.model.embedding_scale
            current_hidden = torch.cat([current_hidden, new_embed_scaled.to(MODEL_DTYPE)], dim=1)

            # Expand causal mask
            new_seq_len = current_hidden.shape[1]
            causal_mask = torch.triu(torch.full((new_seq_len, new_seq_len), float('-inf')), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).to(MODEL_DTYPE)

    anemll_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"\nANEMLL Output: {anemll_output}")

    # === HuggingFace Reference ===
    print("\n=== HuggingFace Reference (bfloat16) ===")
    with torch.no_grad():
        hf_output_ids = hf_model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        hf_generated = hf_output_ids[0, input_ids.shape[1]:]
        hf_output = tokenizer.decode(hf_generated, skip_special_tokens=True)
        print(f"HF Output: {hf_output}")

    print("\n=== Comparison ===")
    print(f"ANEMLL: {anemll_output}")
    print(f"HF:     {hf_output}")
    match = anemll_output.strip() == hf_output.strip()
    print(f"Match: {match}")


if __name__ == "__main__":
    main()
