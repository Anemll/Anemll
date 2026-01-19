#!/usr/bin/env python3
"""Compare PyTorch (QwenModel) vs CoreML inference token-by-token.

This test compares:
1. Token-by-token generation (argmax)
2. Logits values
3. Probability distributions

Usage:
    python tests/dev/test_qwen_aq1_compare.py \
        ~/Downloads/snapped_step1800.pt \
        /Users/anemll/Models/ANE/q4_r32_lut_ka \
        --prompt "What is AI?"

    python tests/dev/test_qwen_aq1_compare.py \
        ~/Downloads/snapped_step1800.pt \
        /Users/anemll/Models/ANE/q4_r32_lut_ka \
        --prompt "Hello" --max-tokens 20 --no-think

    # With custom context/state length
    python tests/dev/test_qwen_aq1_compare.py \
        ~/Downloads/snapped_step1800.pt \
        /Users/anemll/Models/ANE/q4_r32_lut_ka \
        --prompt "History of England and UK" --no-think \
        --context-length 1024 --state-length 1024 --max-tokens 2000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from anemll.models.qwen_model import QwenModel, QwenConfig, MODEL_DTYPE
from anemll.models.anemll_quant import load_baked_weights_for_ane


# =============================================================================
# PYTORCH MODEL (from test_qwen_aq1_pytorch.py)
# =============================================================================

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


class QwenInferenceModel(nn.Module):
    """Wrapper that uses QwenModel from qwen_model.py + LMHead for inference."""

    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.model = QwenModel(config)

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
        hidden_states = self.model(
            input_ids=input_ids,
            causal_mask=causal_mask,
            position_ids=position_ids,
            current_pos=current_pos,
            IN_PREFILL=IN_PREFILL,
        )

        batch_size, seq_len, hidden_size = hidden_states.shape
        x = hidden_states.transpose(1, 2).unsqueeze(2)

        logits_parts = []
        for i in range(16):
            head = getattr(self, f'lm_head16_{i+1}')
            part = head(x)
            logits_parts.append(part)

        logits = torch.cat(logits_parts, dim=1)
        logits = logits.squeeze(2).transpose(1, 2)

        return logits


def create_causal_mask(seq_len: int, state_length: int, dtype=MODEL_DTYPE, current_pos: int = 0):
    """Create a causal attention mask."""
    mask = torch.zeros((1, 1, seq_len, state_length), dtype=dtype)
    for i in range(seq_len):
        actual_pos = current_pos + i
        mask[0, 0, i, actual_pos + 1:] = float('-inf')
    return mask


def load_pytorch_model(checkpoint_path: str, context_length: int, state_length: int = None, verbose: bool = False):
    """Load PyTorch QwenModel with AQ1 weights."""
    if state_length is None:
        state_length = context_length

    if verbose:
        print(f"Loading PyTorch model from: {checkpoint_path}")
        print(f"  Context length: {context_length}, State length: {state_length}")

    detected_config = detect_model_config(checkpoint_path)
    if verbose:
        print(f"  Detected config: {detected_config}")

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

    model = QwenInferenceModel(config)
    load_baked_weights_for_ane(model=model, checkpoint_path=checkpoint_path, verbose=verbose)
    model.eval()

    return model, config


# =============================================================================
# COREML MODEL LOADING
# =============================================================================

def load_coreml_model(path, function_name=None):
    """Load a CoreML model."""
    path = Path(path)
    compute_unit = ct.ComputeUnit.CPU_AND_NE

    if path.suffix == '.mlmodelc':
        if function_name:
            return ct.models.CompiledMLModel(str(path), compute_unit, function_name=function_name)
        else:
            return ct.models.CompiledMLModel(str(path), compute_unit)
    else:
        if function_name:
            return ct.models.MLModel(str(path), function_name=function_name)
        else:
            return ct.models.MLModel(str(path))


def load_coreml_models(model_dir: str, verbose: bool = False):
    """Load CoreML models from directory."""
    model_dir = Path(model_dir)

    # Load meta.yaml
    meta_path = model_dir / 'meta.yaml'
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = yaml.safe_load(f)
        params = meta['model_info']['parameters']
        context_length = int(params['context_length'])
        batch_size = int(params['batch_size'])
        split_lm_head = int(params.get('split_lm_head', 16))

        embed_name = params.get('embeddings', 'qwen_embeddings.mlmodelc')
        lmhead_name = params.get('lm_head', 'qwen_lm_head_lut6.mlmodelc')
        ffn_name = params.get('ffn', 'qwen_FFN_PF_lut4_chunk_01of01.mlmodelc')
    else:
        context_length = 1024
        batch_size = 64
        split_lm_head = 16
        embed_name = 'qwen_embeddings.mlmodelc'
        lmhead_name = 'qwen_lm_head_lut6.mlmodelc'
        ffn_name = 'qwen_FFN_PF_lut4_chunk_01of01.mlmodelc'

    if verbose:
        print(f"Loading CoreML models from: {model_dir}")
        print(f"  Context length: {context_length}")
        print(f"  Batch size: {batch_size}")
        print(f"  Split LM head: {split_lm_head}")

    # Load models
    embed_path = model_dir / embed_name
    lmhead_path = model_dir / lmhead_name
    ffn_path = model_dir / ffn_name

    if verbose:
        print(f"  Loading embeddings: {embed_path.name}")
    embed_model = load_coreml_model(embed_path)

    if verbose:
        print(f"  Loading LM head: {lmhead_path.name}")
    lmhead_model = load_coreml_model(lmhead_path)

    if verbose:
        print(f"  Loading FFN (infer): {ffn_path.name}")
    ffn_infer = load_coreml_model(ffn_path, function_name='infer')

    if verbose:
        print(f"  Loading FFN (prefill): {ffn_path.name}")
    ffn_prefill = load_coreml_model(ffn_path, function_name='prefill')

    metadata = {
        'context_length': context_length,
        'batch_size': batch_size,
        'split_lm_head': split_lm_head,
    }

    return embed_model, ffn_infer, ffn_prefill, lmhead_model, metadata


# =============================================================================
# TOKENIZER
# =============================================================================

def load_tokenizer(model_dir: str):
    """Load tokenizer from model directory or HuggingFace."""
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    tokenizer_json = model_dir / 'tokenizer.json'

    if tokenizer_json.exists():
        return AutoTokenizer.from_pretrained(str(model_dir), use_fast=False, trust_remote_code=True)
    else:
        return AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

@torch.no_grad()
def pytorch_forward_single(model, input_ids, position_ids, causal_mask, current_pos, prefill=False):
    """Run single forward pass on PyTorch model, return logits."""
    logits = model(
        input_ids=input_ids,
        causal_mask=causal_mask,
        position_ids=position_ids,
        current_pos=current_pos,
        IN_PREFILL=prefill,
    )
    return logits


def coreml_prefill(embed_model, ffn_prefill, input_ids, context_length, batch_size, state):
    """Run prefill on CoreML model."""
    seq_len = input_ids.shape[1]

    # Create causal mask for prefill
    causal_mask = np.full((1, 1, batch_size, context_length), -np.inf, dtype=np.float16)
    for i in range(batch_size):
        causal_mask[0, 0, i, :i+1] = 0

    batch_pos = 0
    while batch_pos < seq_len:
        batch_end = min(batch_pos + batch_size, seq_len)
        current_batch_size = batch_end - batch_pos

        # Get current batch
        batch_input = input_ids[:, batch_pos:batch_end]

        # Pad to full batch size
        if current_batch_size < batch_size:
            batch_input = np.pad(batch_input, ((0, 0), (0, batch_size - current_batch_size)), mode='constant')

        # Position IDs
        position_ids = np.arange(batch_pos, batch_pos + batch_size, dtype=np.int32)

        # Run embeddings
        hidden_states = embed_model.predict({'input_ids': batch_input.astype(np.int32)})['hidden_states']

        # Update causal mask for this batch position
        batch_causal_mask = causal_mask[:, :, :, :]

        # Run FFN prefill
        inputs = {
            'hidden_states': hidden_states.astype(np.float16),
            'position_ids': position_ids,
            'causal_mask': batch_causal_mask.astype(np.float16),
            'current_pos': np.array([batch_pos], dtype=np.int32),
        }
        ffn_prefill.predict(inputs, state)

        batch_pos = batch_end


def coreml_forward_single(embed_model, ffn_infer, lmhead_model, input_ids, pos, context_length, state, split_lm_head):
    """Run single token forward pass on CoreML model, return logits."""
    # Run embeddings
    hidden_states = embed_model.predict({'input_ids': input_ids.astype(np.int32)})['hidden_states']

    # Create masks
    update_mask = np.zeros((1, 1, context_length, 1), dtype=np.float16)
    update_mask[0, 0, pos, 0] = 1.0
    position_ids = np.array([pos], dtype=np.int32)

    # Causal mask for single token
    causal_mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
    causal_mask[0, 0, 0, pos+1:] = -np.inf

    # Run FFN
    inputs = {
        'hidden_states': hidden_states.astype(np.float16),
        'update_mask': update_mask,
        'position_ids': position_ids,
        'causal_mask': causal_mask,
        'current_pos': position_ids,
    }
    output = ffn_infer.predict(inputs, state)
    hidden_out = output['output_hidden_states']

    # Run LM head
    lm_output = lmhead_model.predict({'hidden_states': hidden_out.astype(np.float16)})

    # Combine logits
    logits_parts = []
    for i in range(1, split_lm_head + 1):
        key = f'logits{i}'
        if key in lm_output:
            logits_parts.append(lm_output[key])
    logits = np.concatenate(logits_parts, axis=-1)  # [1, 1, vocab_size]

    return logits


# =============================================================================
# COMPARISON
# =============================================================================

def compare_logits(pytorch_logits, coreml_logits, tokenizer, top_k=10, verbose=True):
    """Compare logits between PyTorch and CoreML."""
    # Get last token logits
    pt_logits = pytorch_logits[0, -1, :].numpy()  # [vocab_size]
    cm_logits = coreml_logits[0, 0, :]  # [vocab_size]

    # Compute softmax probabilities (scaled by 100 for readability)
    def softmax(x):
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum()

    pt_probs = softmax(pt_logits.astype(np.float64)) * 100  # Percentage
    cm_probs = softmax(cm_logits.astype(np.float64)) * 100  # Percentage

    # Top-k tokens
    pt_top_k_idx = np.argsort(pt_logits)[::-1][:top_k]
    cm_top_k_idx = np.argsort(cm_logits)[::-1][:top_k]

    # Argmax tokens
    pt_argmax = pt_top_k_idx[0]
    cm_argmax = cm_top_k_idx[0]

    match = pt_argmax == cm_argmax

    if verbose:
        print(f"\n  PyTorch argmax: {pt_argmax} ('{tokenizer.decode([pt_argmax])}') prob={pt_probs[pt_argmax]:.2f}%")
        print(f"  CoreML argmax:  {cm_argmax} ('{tokenizer.decode([cm_argmax])}') prob={cm_probs[cm_argmax]:.2f}%")
        print(f"  Match: {'YES' if match else 'NO'}")

        # Logits statistics
        print(f"\n  PyTorch logits: min={pt_logits.min():.4f}, max={pt_logits.max():.4f}, mean={pt_logits.mean():.4f}")
        print(f"  CoreML logits:  min={cm_logits.min():.4f}, max={cm_logits.max():.4f}, mean={cm_logits.mean():.4f}")

        # Correlation
        corr = np.corrcoef(pt_logits, cm_logits)[0, 1]
        print(f"  Correlation: {corr:.6f}")

        # Top-k comparison with probabilities
        print(f"\n  Top-{top_k} comparison (prob = softmax %):")
        print(f"  {'Rank':<5} {'PyTorch':<22} {'CoreML':<22} {'Match'}")
        print(f"  {'-'*5} {'-'*22} {'-'*22} {'-'*5}")
        for i in range(top_k):
            pt_tok = pt_top_k_idx[i]
            cm_tok = cm_top_k_idx[i]
            pt_str = tokenizer.decode([pt_tok]).replace('\n', '\\n')[:8]
            cm_str = tokenizer.decode([cm_tok]).replace('\n', '\\n')[:8]
            pt_p = pt_probs[pt_tok]
            cm_p = cm_probs[cm_tok]
            m = 'YES' if pt_tok == cm_tok else 'NO'
            print(f"  {i+1:<5} {pt_tok:<6} '{pt_str:<8}' {pt_p:>5.2f}%  {cm_tok:<6} '{cm_str:<8}' {cm_p:>5.2f}%  {m}")

    return {
        'match': match,
        'pt_argmax': pt_argmax,
        'cm_argmax': cm_argmax,
        'correlation': np.corrcoef(pt_logits, cm_logits)[0, 1],
        'pt_logits': pt_logits,
        'cm_logits': cm_logits,
    }


def run_comparison(
    pytorch_model,
    pytorch_config,
    coreml_models,
    coreml_metadata,
    tokenizer,
    prompt: str,
    max_tokens: int = 20,
    no_think: bool = False,
    stop_on_mismatch: bool = False,
    verbose: bool = True,
):
    """Run token-by-token comparison."""
    embed_model, ffn_infer, ffn_prefill, lmhead_model = coreml_models
    context_length = coreml_metadata['context_length']
    batch_size = coreml_metadata['batch_size']
    split_lm_head = coreml_metadata['split_lm_head']

    # Format prompt
    if no_think:
        messages = [{"role": "user", "content": f"/no_think {prompt}"}]
    else:
        messages = [{"role": "user", "content": prompt}]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    input_ids_np = input_ids.numpy().astype(np.int32)

    seq_len = input_ids.shape[1]
    print(f"\nPrompt tokens: {seq_len}")
    print(f"Prompt: {prompt}")

    # Reset PyTorch KV cache
    if hasattr(pytorch_model.model, 'kv_cache_0'):
        pytorch_model.model.kv_cache_0.zero_()

    # Create CoreML state
    coreml_state = ffn_prefill.make_state()

    # === PREFILL ===
    print("\n" + "="*60)
    print("PREFILL PHASE")
    print("="*60)

    # PyTorch prefill
    t0 = time.time()
    position_ids = torch.arange(seq_len)
    causal_mask = create_causal_mask(seq_len, pytorch_config.state_length)
    pt_logits = pytorch_forward_single(
        pytorch_model, input_ids, position_ids, causal_mask, current_pos=0, prefill=True
    )
    pt_prefill_time = time.time() - t0

    # CoreML prefill
    t0 = time.time()
    coreml_prefill(embed_model, ffn_prefill, input_ids_np, context_length, batch_size, coreml_state)
    cm_prefill_time = time.time() - t0

    # Get first token logits from CoreML (need to run inference for last position)
    # Actually for prefill comparison, let's compare the last prefill position
    last_token = input_ids_np[:, -1:]  # [1, 1]
    cm_logits = coreml_forward_single(
        embed_model, ffn_infer, lmhead_model,
        last_token, seq_len - 1, context_length, coreml_state, split_lm_head
    )

    print(f"\nPrefill timing:")
    print(f"  PyTorch: {pt_prefill_time*1000:.1f}ms ({seq_len/pt_prefill_time:.1f} tok/s)")
    print(f"  CoreML:  {cm_prefill_time*1000:.1f}ms ({seq_len/cm_prefill_time:.1f} tok/s)")

    print("\nComparing prefill output (last position):")
    prefill_result = compare_logits(pt_logits, cm_logits, tokenizer, verbose=verbose)

    # === DECODE ===
    print("\n" + "="*60)
    print("DECODE PHASE (token-by-token)")
    print("="*60)

    # Get first generated token
    if prefill_result['match']:
        next_token = prefill_result['pt_argmax']
    else:
        # Use PyTorch result
        next_token = prefill_result['pt_argmax']

    pt_tokens = [next_token]
    cm_tokens = [prefill_result['cm_argmax']]

    current_pos = seq_len
    results = []

    for step in range(max_tokens - 1):
        if current_pos >= context_length - 1:
            print(f"\nReached context length limit ({context_length})")
            break

        # Check EOS
        if next_token in [151643, 151644, 151645]:
            print(f"\nEOS token generated at step {step}")
            break

        print(f"\n--- Step {step + 1} (pos={current_pos}) ---")
        print(f"  Input token: {next_token} ('{tokenizer.decode([next_token])}')")

        # PyTorch decode
        single_input = torch.tensor([[next_token]], dtype=torch.long)
        single_mask = create_causal_mask(1, pytorch_config.state_length, current_pos=current_pos)
        pt_logits = pytorch_forward_single(
            pytorch_model, single_input, torch.tensor([current_pos]),
            single_mask, current_pos=current_pos, prefill=False
        )

        # CoreML decode
        cm_input = np.array([[next_token]], dtype=np.int32)
        cm_logits = coreml_forward_single(
            embed_model, ffn_infer, lmhead_model,
            cm_input, current_pos, context_length, coreml_state, split_lm_head
        )

        # Compare
        result = compare_logits(pt_logits, cm_logits, tokenizer, verbose=verbose)
        results.append(result)

        # Check for mismatch and stop if requested
        if stop_on_mismatch and not result['match']:
            print(f"\n*** STOPPING: Token mismatch at step {step + 1} ***")
            print(f"  PyTorch: {result['pt_argmax']} ('{tokenizer.decode([result['pt_argmax']])}')")
            print(f"  CoreML:  {result['cm_argmax']} ('{tokenizer.decode([result['cm_argmax']])}')")
            pt_tokens.append(result['pt_argmax'])
            cm_tokens.append(result['cm_argmax'])
            break

        # Next token (use PyTorch)
        next_token = result['pt_argmax']
        pt_tokens.append(next_token)
        cm_tokens.append(result['cm_argmax'])
        current_pos += 1

    # === SUMMARY ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    matches = sum(1 for r in results if r['match'])
    total = len(results)
    correlations = [r['correlation'] for r in results]

    print(f"\nToken matches: {matches}/{total} ({100*matches/total:.1f}%)")
    print(f"Avg correlation: {np.mean(correlations):.6f}")
    print(f"Min correlation: {np.min(correlations):.6f}")
    print(f"Max correlation: {np.max(correlations):.6f}")

    print(f"\nPyTorch tokens: {pt_tokens}")
    print(f"CoreML tokens:  {cm_tokens}")

    print(f"\nPyTorch output: {tokenizer.decode(pt_tokens)}")
    print(f"CoreML output:  {tokenizer.decode(cm_tokens)}")

    return {
        'matches': matches,
        'total': total,
        'match_rate': matches / total if total > 0 else 0,
        'avg_correlation': np.mean(correlations),
        'pt_tokens': pt_tokens,
        'cm_tokens': cm_tokens,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare PyTorch (QwenModel) vs CoreML inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('checkpoint', type=str, help='Path to AQ1 checkpoint for PyTorch')
    parser.add_argument('coreml_dir', type=str, help='Path to CoreML model directory')
    parser.add_argument('--prompt', '-p', type=str, default='What is AI?')
    parser.add_argument('--max-tokens', '-n', type=int, default=20)
    parser.add_argument('--context-length', type=int, default=None,
                        help='Context length (default: from CoreML meta.yaml)')
    parser.add_argument('--state-length', type=int, default=None,
                        help='State length for KV cache (default: same as context-length)')
    parser.add_argument('--no-think', action='store_true')
    parser.add_argument('--stop', action='store_true',
                        help='Stop when predicted tokens mismatch between PyTorch and CoreML')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    checkpoint_path = os.path.expanduser(args.checkpoint)
    coreml_dir = os.path.expanduser(args.coreml_dir)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not os.path.exists(coreml_dir):
        print(f"Error: CoreML directory not found: {coreml_dir}")
        sys.exit(1)

    print("="*60)
    print("PyTorch vs CoreML Comparison")
    print("="*60)
    print(f"PyTorch checkpoint: {checkpoint_path}")
    print(f"CoreML directory: {coreml_dir}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")

    # Load CoreML models first to get default context_length
    print("\n--- Loading CoreML models ---")
    embed_model, ffn_infer, ffn_prefill, lmhead_model, coreml_metadata = load_coreml_models(
        coreml_dir, verbose=True
    )

    # Override context_length if specified
    if args.context_length is not None:
        context_length = args.context_length
        coreml_metadata['context_length'] = context_length
        print(f"  Overriding context_length: {context_length}")
    else:
        context_length = coreml_metadata['context_length']

    # State length defaults to context_length
    state_length = args.state_length if args.state_length is not None else context_length

    # Load PyTorch model with matching context/state length
    print("\n--- Loading PyTorch model ---")
    pytorch_model, pytorch_config = load_pytorch_model(
        checkpoint_path, context_length, state_length=state_length, verbose=True
    )

    # Load tokenizer
    print("\n--- Loading tokenizer ---")
    tokenizer = load_tokenizer(coreml_dir)
    print(f"  Vocab size: {len(tokenizer)}")

    # Run comparison
    result = run_comparison(
        pytorch_model=pytorch_model,
        pytorch_config=pytorch_config,
        coreml_models=(embed_model, ffn_infer, ffn_prefill, lmhead_model),
        coreml_metadata=coreml_metadata,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        no_think=args.no_think,
        stop_on_mismatch=args.stop,
        verbose=True,
    )

    print("\n" + "="*60)
    if result['match_rate'] >= 0.9:
        print("SUCCESS: Models produce very similar outputs!")
    elif result['match_rate'] >= 0.5:
        print("PARTIAL: Models produce somewhat similar outputs")
    else:
        print("DIVERGENT: Models produce different outputs")
    print("="*60)


if __name__ == '__main__':
    main()
