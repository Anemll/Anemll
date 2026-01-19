#!/usr/bin/env python3
"""PyTorch inference test using QwenModel from qwen_model.py with AQ1 quantization.

This test uses QwenModel directly from anemll/models/qwen_model.py -
the exact same model class used for CoreML/ANE conversion.

Usage:
    python tests/dev/test_qwen_aq1_pytorch.py ~/Downloads/snapped_step1800.pt --prompt "History of England and UK"
    python tests/dev/test_qwen_aq1_pytorch.py ~/Downloads/snapped_step1800.pt --prompt "What is AI?" --no-think
    python tests/dev/test_qwen_aq1_pytorch.py ~/Downloads/snapped_step1800.pt -v --max-tokens 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import QwenModel and QwenConfig directly from qwen_model.py
from anemll.models.qwen_model import QwenModel, QwenConfig, MODEL_DTYPE

# Import weight loading from anemll_quant.py
from anemll.models.anemll_quant import load_baked_weights_for_ane


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_tokenizer(model_id: str):
    """Load tokenizer from HuggingFace (only tokenizer, not model)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def create_causal_mask(seq_len: int, state_length: int, dtype=MODEL_DTYPE, current_pos: int = 0):
    """Create a causal attention mask."""
    mask = torch.zeros((1, 1, seq_len, state_length), dtype=dtype)
    for i in range(seq_len):
        actual_pos = current_pos + i
        mask[0, 0, i, actual_pos + 1:] = float('-inf')
    return mask


def detect_model_config(checkpoint_path: str) -> dict:
    """Detect model configuration from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = {}

    if 'model.embed_tokens.weight' in state_dict:
        vocab_size, hidden_size = state_dict['model.embed_tokens.weight'].shape
        config['vocab_size'] = vocab_size
        config['hidden_size'] = hidden_size

    layer_indices = set()
    for key in state_dict.keys():
        if 'model.layers.' in key:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_indices.add(int(parts[i + 1]))
    if layer_indices:
        config['num_hidden_layers'] = max(layer_indices) + 1

    for key in state_dict.keys():
        if 'gate_proj' in key and (key.endswith('.weight') or key.endswith('._Q')):
            config['intermediate_size'] = state_dict[key].shape[0]
            break

    for key in state_dict.keys():
        if 'q_proj' in key and (key.endswith('.weight') or key.endswith('._Q')):
            q_size = state_dict[key].shape[0]
            if q_size % 128 == 0:
                config['head_dim'] = 128
                config['num_attention_heads'] = q_size // 128
            elif q_size % 64 == 0:
                config['head_dim'] = 64
                config['num_attention_heads'] = q_size // 64
            break

    for key in state_dict.keys():
        if 'k_proj' in key and (key.endswith('.weight') or key.endswith('._Q')):
            kv_size = state_dict[key].shape[0]
            head_dim = config.get('head_dim', 128)
            config['num_key_value_heads'] = kv_size // head_dim
            break

    return config


# =============================================================================
# MAIN MODEL WRAPPER (uses QwenModel directly)
# =============================================================================

class QwenInferenceModel(nn.Module):
    """Wrapper that uses QwenModel from qwen_model.py + LMHead for inference."""

    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config

        # Use QwenModel directly from qwen_model.py
        self.model = QwenModel(config)

        # LM head with 16-way split (matching conversion structure)
        # Create the split heads directly on this model (not wrapped in LMHead)
        # This matches what load_baked_weights_for_ane expects
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_splits = 16

        split_size = vocab_size // num_splits
        remainder = vocab_size % num_splits
        self._lm_head_split_sizes = []

        for i in range(num_splits):
            size = split_size + (1 if i < remainder else 0)
            self._lm_head_split_sizes.append(size)
            setattr(self, f'lm_head16_{i+1}',
                   nn.Conv2d(hidden_size, size, kernel_size=1, bias=False))

    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: int,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        """Forward pass returning logits."""
        # Use QwenModel.forward() to get hidden states
        hidden_states = self.model(
            input_ids=input_ids,
            causal_mask=causal_mask,
            position_ids=position_ids,
            current_pos=current_pos,
            IN_PREFILL=IN_PREFILL,
        )

        # Apply split LM heads to get logits
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Reshape for Conv2d: [batch, hidden_size, 1, seq_len]
        x = hidden_states.transpose(1, 2).unsqueeze(2)

        # Apply each split head and concatenate
        logits_parts = []
        for i in range(16):
            head = getattr(self, f'lm_head16_{i+1}')
            part = head(x)  # [batch, split_vocab, 1, seq_len]
            logits_parts.append(part)

        # Concatenate along vocab dimension
        logits = torch.cat(logits_parts, dim=1)  # [batch, vocab_size, 1, seq_len]

        # Reshape back: [batch, seq_len, vocab_size]
        logits = logits.squeeze(2).transpose(1, 2)

        return logits


def load_qwen_with_aq1(
    checkpoint_path: str,
    context_length: int = 512,
    state_length: int = 512,
    model_id: str = None,
    verbose: bool = False,
) -> tuple:
    """Load QwenModel from qwen_model.py with AQ1 weights.

    Uses:
    - QwenModel from qwen_model.py (the exact class used for conversion)
    - load_baked_weights_for_ane() from anemll_quant.py
    """
    if verbose:
        print(f"Detecting config from checkpoint: {checkpoint_path}")

    detected_config = detect_model_config(checkpoint_path)
    if verbose:
        print(f"  Detected: {detected_config}")

    # Create QwenConfig
    config = QwenConfig(
        vocab_size=detected_config.get('vocab_size', 151936),
        hidden_size=detected_config.get('hidden_size', 1024),
        num_hidden_layers=detected_config.get('num_hidden_layers', 28),
        num_attention_heads=detected_config.get('num_attention_heads', 16),
        num_key_value_heads=detected_config.get('num_key_value_heads', 8),
        head_dim=detected_config.get('head_dim', 128),
        intermediate_size=detected_config.get('intermediate_size', 3072),
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        context_length=context_length,
        state_length=state_length,
    )

    if verbose:
        print(f"\nCreating QwenModel (from qwen_model.py):")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_layers: {config.num_hidden_layers}")

    # Create inference model (QwenModel + LMHead)
    model = QwenInferenceModel(config)

    # Load weights using ANEMLL API
    if verbose:
        print(f"\nLoading weights using load_baked_weights_for_ane():")

    # Load all weights (transformer layers + lm_head16_* splits)
    success = load_baked_weights_for_ane(
        model=model,
        checkpoint_path=checkpoint_path,
        verbose=verbose,
    )

    if not success:
        print("Warning: load_baked_weights_for_ane returned False")

    # Load tokenizer
    if model_id is None:
        vocab_size = config.vocab_size
        if vocab_size == 151936:
            model_id = 'Qwen/Qwen3-0.6B'
        else:
            model_id = 'Qwen/Qwen3-0.6B'
        if verbose:
            print(f"\n  Inferred tokenizer: {model_id}")

    tokenizer = load_tokenizer(model_id)

    return model, tokenizer, config


@torch.no_grad()
def generate(
    model: QwenInferenceModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    no_think: bool = False,
    verbose: bool = False,
    stream: bool = True,
) -> str:
    """Generate text using QwenModel."""

    if no_think:
        messages = [{"role": "user", "content": f"/no_think {prompt}"}]
    else:
        messages = [{"role": "user", "content": prompt}]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]

    batch_size, seq_len = input_ids.shape
    state_length = model.config.state_length

    # Reset KV cache
    if hasattr(model.model, 'kv_cache_0'):
        model.model.kv_cache_0.zero_()

    model.eval()

    # Prefill phase
    if verbose:
        print(f"Prefilling {seq_len} tokens...")

    t_prefill = time.time()
    position_ids = torch.arange(seq_len)
    causal_mask = create_causal_mask(seq_len, state_length)

    logits = model(
        input_ids=input_ids,
        causal_mask=causal_mask,
        position_ids=position_ids,
        current_pos=0,
        IN_PREFILL=True,
    )

    prefill_time = time.time() - t_prefill

    # Get first token
    next_logits = logits[:, -1, :]

    if temperature > 0:
        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = next_logits.argmax(dim=-1, keepdim=True)

    generated_tokens = [next_token.item()]

    if verbose:
        print(f"Prefill: {prefill_time:.3f}s")

    if stream:
        token_str = tokenizer.decode([next_token.item()], skip_special_tokens=True)
        print(token_str, end="", flush=True)

    # Decode phase
    current_pos = seq_len
    t_decode = time.time()

    for _ in range(max_new_tokens - 1):
        if current_pos >= state_length:
            if verbose:
                print(f"\nReached state_length limit ({state_length})")
            break

        if next_token.item() in [151643, 151644, 151645]:
            break

        single_mask = create_causal_mask(1, state_length, current_pos=current_pos)

        logits = model(
            input_ids=next_token,
            causal_mask=single_mask,
            position_ids=torch.tensor([current_pos]),
            current_pos=current_pos,
            IN_PREFILL=False,
        )

        next_logits = logits[:, -1, :]

        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        generated_tokens.append(next_token.item())
        current_pos += 1

        if stream:
            token_str = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(token_str, end="", flush=True)

    decode_time = time.time() - t_decode
    tokens_per_sec = len(generated_tokens) / decode_time if decode_time > 0 else 0

    if stream:
        print()

    if verbose:
        print(f"Decode: {len(generated_tokens)} tokens in {decode_time:.3f}s ({tokens_per_sec:.1f} tok/s)")

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    response = response.replace('<|im_end|>', '').strip()

    return response


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch inference using QwenModel from qwen_model.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('checkpoint', type=str, help='Path to AQ1 checkpoint')
    parser.add_argument('--prompt', '-p', type=str, default='What is the capital of France?')
    parser.add_argument('--model-id', '-m', type=str, default=None)
    parser.add_argument('--max-tokens', '-n', type=int, default=100)
    parser.add_argument('--temperature', '-t', type=float, default=0.0)
    parser.add_argument('--context-length', type=int, default=512)
    parser.add_argument('--state-length', type=int, default=512)
    parser.add_argument('--no-think', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-stream', action='store_true')

    args = parser.parse_args()

    checkpoint_path = os.path.expanduser(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("=" * 60)
    print("ANEMLL Inference Test")
    print("=" * 60)
    print(f"Model: QwenModel (from anemll/models/qwen_model.py)")
    print(f"Loader: load_baked_weights_for_ane (from anemll_quant.py)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prompt: {args.prompt}")
    if args.no_think:
        print(f"Mode: No-think")

    print("\nLoading model...")
    t0 = time.time()
    model, tokenizer, config = load_qwen_with_aq1(
        checkpoint_path=checkpoint_path,
        context_length=args.context_length,
        state_length=args.state_length,
        model_id=args.model_id,
        verbose=args.verbose,
    )
    print(f"Model loaded in {time.time() - t0:.2f}s")

    print("\n" + "-" * 60)
    print("Response:")
    print("-" * 60)

    response = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        no_think=args.no_think,
        verbose=args.verbose,
        stream=not args.no_stream,
    )

    if args.no_stream:
        print(response)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
