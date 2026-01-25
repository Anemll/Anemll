#!/usr/bin/env python3
"""Compare PyTorch FFNWrapper/PrefillWrapper outputs with expected results.

This test verifies that the PyTorch wrappers (same code used for CoreML tracing)
produce correct outputs for Gemma3 generation.
"""

import sys
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

import os
import torch
import numpy as np
from transformers import AutoTokenizer

from anemll.models.gemma3_model import (
    Gemma3ForCausalLM,
    Gemma3Config,
    MODEL_DTYPE,
    TEST_DEVICE,
)
from anemll.ane_converter.gemma3_converter import Gemma3Converter


def load_model(model_path, context_length=64):
    """Load ANEMLL model."""
    config = Gemma3Config.from_json(f'{model_path}/config.json')
    config.context_length = context_length
    config.state_length = context_length

    model = Gemma3ForCausalLM(config)
    converter = Gemma3Converter(model, context_length=context_length, batch_size=context_length)
    converter.load_weights_from_hf(model_path)
    model.eval()

    return model, converter


def create_ffn_wrapper(model, start_layer=0, end_layer=None):
    """Create the same FFNWrapper used in convert_part_2."""
    class FFNWrapper(torch.nn.Module):
        def __init__(self, model, start_layer, end_layer):
            super().__init__()
            self.model = model
            self.start_layer = start_layer
            self.end_layer = end_layer

        def forward(self, hidden_states, position_ids, causal_mask, current_pos):
            out = self.model.model.process_layers(
                hidden_states,
                position_ids,
                causal_mask,
                current_pos,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                IN_PREFILL=False,
            )
            if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                out = self.model.model.norm(out)
            return out

    wrapper = FFNWrapper(model, start_layer, end_layer)
    wrapper.eval()
    return wrapper


def create_prefill_wrapper(model, start_layer=0, end_layer=None):
    """Create the same PrefillWrapper used in convert_part_2_prefill."""
    class PrefillWrapper(torch.nn.Module):
        def __init__(self, model, start_layer, end_layer):
            super().__init__()
            self.model = model
            self.start_layer = start_layer
            self.end_layer = end_layer

        def forward(self, hidden_states, position_ids, causal_mask, current_pos):
            out = self.model.model.process_layers(
                hidden_states,
                position_ids,
                causal_mask,
                current_pos,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                IN_PREFILL=True,
            )
            if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                return out[:, 0:1, :]
            return out

    wrapper = PrefillWrapper(model, start_layer, end_layer)
    wrapper.eval()
    return wrapper


def get_logits(model, hidden_states):
    """Get logits from hidden states using split LM head."""
    hidden_for_lm = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

    all_logits = []
    for j in range(1, 17):
        lm_head = getattr(model, f'lm_head16_{j}')
        part_logits = lm_head(hidden_for_lm).squeeze(2).transpose(1, 2)
        all_logits.append(part_logits)

    logits = torch.cat(all_logits, dim=-1).squeeze(1)
    return logits


def test_full_generation():
    """Test full generation sequence using PyTorch wrappers."""
    model_path = '/Users/anemll/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3'
    context_length = 64

    print("Loading model...")
    model, converter = load_model(model_path, context_length)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Prepare input
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    seq_len = input_ids.shape[1]

    print(f"Input sequence length: {seq_len}")
    print(f"Tokens: {input_ids[0].tolist()}")

    # Create wrappers
    prefill_wrapper = create_prefill_wrapper(model)
    ffn_wrapper = create_ffn_wrapper(model)

    with torch.no_grad():
        # Reset KV cache
        model.model.kv_cache_0.zero_()

        # Get embeddings
        embed = model.model.embed_tokens(input_ids)
        hidden_states = embed * model.model.embedding_scale
        hidden_states = hidden_states.to(MODEL_DTYPE)

        print(f"\nInput hidden states: shape={hidden_states.shape}")
        print(f"  mean={hidden_states.float().mean():.4f}, std={hidden_states.float().std():.4f}")

        # Prefill
        position_ids = torch.arange(seq_len, dtype=torch.int32)
        causal_mask = torch.zeros((1, 1, seq_len, context_length), dtype=MODEL_DTYPE)
        for i in range(seq_len):
            for j in range(i + 1, context_length):
                causal_mask[0, 0, i, j] = float('-inf')
        current_pos = torch.tensor([0], dtype=torch.int32)

        print("\n=== Running Prefill ===")
        prefill_out = prefill_wrapper(hidden_states, position_ids, causal_mask, current_pos)
        print(f"Prefill done. Output shape: {prefill_out.shape}")
        print(f"  mean={prefill_out.float().mean():.4f}, std={prefill_out.float().std():.4f}")

        # Check KV cache
        kv_cache = model.model.kv_cache_0
        print(f"\nKV cache after prefill:")
        print(f"  shape={kv_cache.shape}")
        print(f"  non-zero elements: {(kv_cache != 0).sum().item()}")
        # Should have seq_len * 18 layers * 2 (K+V) * 1 head * 256 dims = seq_len * 9216 non-zero
        expected_nonzero = seq_len * 18 * 2 * 1 * 256
        print(f"  expected non-zero: {expected_nonzero}")

        # Generate tokens
        print("\n=== Generating Tokens ===")
        generated_tokens = []
        current_gen_pos = seq_len

        for step in range(20):
            if step == 0:
                # First token: need to get logits from the last prefilled position
                # We need to run model at position seq_len-1 with the KV cache filled
                # But the KV cache is already filled from prefill!

                # The hidden state we need is for position seq_len-1 (the last input token)
                # After prefill, the KV cache has entries for positions 0 to seq_len-1
                # To get the next token, we run inference at position seq_len-1
                # (using the last input token's embedding)

                last_embed = model.model.embed_tokens(input_ids[:, -1:])
                current_hidden = last_embed * model.model.embedding_scale
                current_hidden = current_hidden.to(MODEL_DTYPE)

                pos = torch.tensor([seq_len - 1], dtype=torch.int32)

                # Mask: attend to all previous positions (0 to seq_len-1)
                mask = torch.zeros((1, 1, 1, context_length), dtype=MODEL_DTYPE)
                for j in range(seq_len, context_length):
                    mask[0, 0, 0, j] = float('-inf')

                current = torch.tensor([seq_len - 1], dtype=torch.int32)
            else:
                # Subsequent tokens: use the previously generated token
                prev_token = torch.tensor([[generated_tokens[-1]]], dtype=torch.long)
                new_embed = model.model.embed_tokens(prev_token)
                current_hidden = new_embed * model.model.embedding_scale
                current_hidden = current_hidden.to(MODEL_DTYPE)

                pos = torch.tensor([current_gen_pos - 1], dtype=torch.int32)

                # Mask: attend to all previous positions (0 to current_gen_pos-1)
                mask = torch.zeros((1, 1, 1, context_length), dtype=MODEL_DTYPE)
                for j in range(current_gen_pos, context_length):
                    mask[0, 0, 0, j] = float('-inf')

                current = torch.tensor([current_gen_pos - 1], dtype=torch.int32)

            # Run FFN wrapper
            output = ffn_wrapper(current_hidden, pos, mask, current)

            # Get logits and predict token
            logits = get_logits(model, output)
            next_token = logits.argmax(dim=-1).item()
            generated_tokens.append(next_token)

            print(f"Step {step}: pos={pos.item()}, token={next_token} -> '{tokenizer.decode([next_token])}'")

            # Check for EOS
            if next_token in [1, 106]:
                break

            current_gen_pos += 1

        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"\nGenerated: {output_text}")


if __name__ == "__main__":
    print("=" * 70)
    print("PYTORCH WRAPPER GENERATION TEST")
    print("=" * 70)
    test_full_generation()
