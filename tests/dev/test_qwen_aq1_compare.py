#!/usr/bin/env python3
"""Compare PyTorch (QwenModel) vs CoreML inference token-by-token.

This test compares:
1. Token-by-token generation (argmax)
2. Logits values and probabilities
3. Stability metrics: KL divergence, entropy, top1_margin
4. Repetition detection (4-gram)

Driver modes:
- pt: PyTorch drives token selection (good for parity testing)
- coreml: CoreML drives (realistic ANE behavior, catches instability)
- teacher: Use provided token sequence (teacher forcing)

Usage:
    # Basic comparison (PyTorch drives)
    python tests/dev/test_qwen_aq1_compare.py \
        ~/Downloads/snapped_step1800.pt \
        /Users/anemll/Models/ANE/q4_r32_lut_ka \
        --prompt "What is AI?"

    # CoreML-driven (realistic ANE behavior)
    python tests/dev/test_qwen_aq1_compare.py \
        ~/Downloads/snapped_step1800.pt \
        /Users/anemll/Models/ANE/q4_r32_lut_ka \
        --prompt "Hello" --max-tokens 100 --driver coreml

    # With custom context/state length
    python tests/dev/test_qwen_aq1_compare.py \
        ~/Downloads/snapped_step1800.pt \
        /Users/anemll/Models/ANE/q4_r32_lut_ka \
        --prompt "History of England and UK" --no-think \
        --context-length 1024 --state-length 1024 --max-tokens 2000

    # Stop on first mismatch
    python tests/dev/test_qwen_aq1_compare.py \
        ~/Downloads/snapped_step1800.pt \
        /Users/anemll/Models/ANE/q4_r32_lut_ka \
        --prompt "List all US presidents" --driver coreml --stop

    # Use HuggingFace model as reference (isolates quantization effects)
    python tests/dev/test_qwen_aq1_compare.py \
        --hf-reference Qwen/Qwen3-0.6B \
        /Users/anemll/Models/ANE/q4_r32_lut_ka \
        --prompt "What is AI?" --driver coreml --no-think
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
# HUGGINGFACE MODEL SUPPORT
# =============================================================================

class HFQwenInferenceModel(nn.Module):
    """Wrapper that uses HuggingFace Qwen3 model for inference (reference baseline)."""

    def __init__(self, hf_model, context_length: int, state_length: int):
        super().__init__()
        self.hf_model = hf_model
        self.context_length = context_length
        self.state_length = state_length
        self.config = hf_model.config

        # Create a fake model attribute for KV cache compatibility
        self.model = self

        # Initialize simple KV cache storage for single-token decode
        self._kv_cache = None
        self._cache_pos = 0

    def reset_cache(self):
        """Reset KV cache state."""
        self._kv_cache = None
        self._cache_pos = 0

    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor = None,  # Unused, HF handles internally
        position_ids: torch.LongTensor = None,
        current_pos: int = 0,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        """Forward pass returning logits."""
        with torch.no_grad():
            if IN_PREFILL:
                # Prefill: process full sequence
                outputs = self.hf_model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                )
                self._kv_cache = outputs.past_key_values
                self._cache_pos = input_ids.shape[1]
                return outputs.logits
            else:
                # Decode: single token with KV cache
                outputs = self.hf_model(
                    input_ids=input_ids,
                    past_key_values=self._kv_cache,
                    use_cache=True,
                    return_dict=True,
                )
                self._kv_cache = outputs.past_key_values
                self._cache_pos += 1
                return outputs.logits


def load_hf_pytorch_model(model_id: str, context_length: int, state_length: int = None, verbose: bool = False):
    """Load HuggingFace Qwen3 model as reference baseline.

    Args:
        model_id: HuggingFace model ID (e.g., 'Qwen/Qwen3-0.6B')
        context_length: Maximum context length
        state_length: State length for KV cache (default: same as context_length)

    Returns:
        model: HFQwenInferenceModel wrapper
        config: Model config object with required attributes
    """
    from transformers import AutoModelForCausalLM

    if state_length is None:
        state_length = context_length

    if verbose:
        print(f"Loading HuggingFace model: {model_id}")
        print(f"  Context length: {context_length}, State length: {state_length}")

    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use fp32 for reference accuracy
        trust_remote_code=True,
    )
    hf_model.eval()

    if verbose:
        print(f"  Loaded config: hidden_size={hf_model.config.hidden_size}, "
              f"num_layers={hf_model.config.num_hidden_layers}, "
              f"vocab_size={hf_model.config.vocab_size}")

    # Wrap in inference model
    model = HFQwenInferenceModel(hf_model, context_length, state_length)

    # Create a config-like object for compatibility
    class HFConfig:
        def __init__(self, hf_config, context_length, state_length):
            self.vocab_size = hf_config.vocab_size
            self.hidden_size = hf_config.hidden_size
            self.num_hidden_layers = hf_config.num_hidden_layers
            self.num_attention_heads = hf_config.num_attention_heads
            self.num_key_value_heads = getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads)
            self.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
            self.intermediate_size = hf_config.intermediate_size
            self.context_length = context_length
            self.state_length = state_length

    config = HFConfig(hf_model.config, context_length, state_length)
    return model, config


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


def tokenwise_prompt_feed_teacher(
    pytorch_model,
    pytorch_config,
    embed_model,
    ffn_infer,
    lmhead_model,
    input_ids_np: np.ndarray,        # shape [1, seq_len]
    context_length: int,
    split_lm_head: int,
    coreml_state,
    probe_metrics: bool = True,
):
    """
    Feed prompt one token at a time to BOTH PyTorch and CoreML (teacher forcing).
    Uses single-token path only (no batch prefill).

    This bypasses coreml_prefill() and exercises only ffn_infer.predict(...),
    giving a clean "prompt divergence profile" per token.

    Returns:
        pt_logits_last: PyTorch logits for last prompt token [1, 1, vocab_size]
        cm_logits_last: CoreML logits for last prompt token [1, 1, vocab_size]
        prompt_metrics: List of stability metrics dicts per position (if probe_metrics=True)
        first_prompt_mismatch_pos: Position of first argmax mismatch, or None
    """
    seq_len = input_ids_np.shape[1]

    pt_logits_last = None
    cm_logits_last = None
    prompt_metrics = []
    first_prompt_mismatch_pos = None

    # Reset PyTorch KV cache
    if hasattr(pytorch_model, 'reset_cache'):
        # HF model wrapper
        pytorch_model.reset_cache()
    elif hasattr(pytorch_model.model, 'kv_cache_0'):
        # Baked model
        pytorch_model.model.kv_cache_0.zero_()

    for pos in range(seq_len):
        tok_id = int(input_ids_np[0, pos])

        # --- PyTorch single-token step ---
        pt_in = torch.tensor([[tok_id]], dtype=torch.long)
        pt_mask = create_causal_mask(1, pytorch_config.state_length, current_pos=pos)
        pt_logits_last = pytorch_forward_single(
            pytorch_model,
            pt_in,
            torch.tensor([pos]),
            pt_mask,
            current_pos=pos,
            prefill=False,   # IMPORTANT: single-token stepping only
        )

        # --- CoreML single-token step ---
        cm_in = np.array([[tok_id]], dtype=np.int32)
        cm_logits_last = coreml_forward_single(
            embed_model,
            ffn_infer,
            lmhead_model,
            cm_in,
            pos,
            context_length,
            coreml_state,
            split_lm_head,
        )

        if probe_metrics:
            pt_vec = pt_logits_last[0, -1, :].detach().cpu().numpy()
            cm_vec = cm_logits_last[0, 0, :]
            m = compute_stability_metrics(pt_vec, cm_vec)
            prompt_metrics.append(m)

            if first_prompt_mismatch_pos is None:
                if int(np.argmax(pt_vec)) != int(np.argmax(cm_vec)):
                    first_prompt_mismatch_pos = pos

    return pt_logits_last, cm_logits_last, prompt_metrics, first_prompt_mismatch_pos


# =============================================================================
# STABILITY METRICS
# =============================================================================

def compute_stability_metrics(pt_logits, cm_logits):
    """Compute stability metrics for ANE analysis.

    Returns dict with:
    - kl_divergence: KL(pt || cm)
    - pt_entropy, cm_entropy: Distribution entropy
    - pt_top1_margin, cm_top1_margin: logit[top1] - logit[top2]
    - pt_max_logit, cm_max_logit: Maximum logit values
    - pt_std_logit, cm_std_logit: Logit standard deviation
    - correlation: Pearson correlation
    """
    eps = 1e-10

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + eps)

    def entropy(probs):
        probs = np.clip(probs, eps, 1.0)
        return -np.sum(probs * np.log(probs))

    def kl_divergence(p, q):
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        return np.sum(p * np.log(p / q))

    pt_probs = softmax(pt_logits.astype(np.float64))
    cm_probs = softmax(cm_logits.astype(np.float64))

    # Top-1 margin (difference between top 2 logits)
    pt_sorted = np.sort(pt_logits)[::-1]
    cm_sorted = np.sort(cm_logits)[::-1]
    pt_top1_margin = pt_sorted[0] - pt_sorted[1] if len(pt_sorted) > 1 else 0
    cm_top1_margin = cm_sorted[0] - cm_sorted[1] if len(cm_sorted) > 1 else 0

    return {
        'kl_divergence': kl_divergence(pt_probs, cm_probs),
        'pt_entropy': entropy(pt_probs),
        'cm_entropy': entropy(cm_probs),
        'pt_top1_margin': pt_top1_margin,
        'cm_top1_margin': cm_top1_margin,
        'pt_max_logit': float(pt_logits.max()),
        'cm_max_logit': float(cm_logits.max()),
        'pt_std_logit': float(pt_logits.std()),
        'cm_std_logit': float(cm_logits.std()),
        'correlation': np.corrcoef(pt_logits, cm_logits)[0, 1],
    }


def detect_instability(metrics_history, window=10):
    """Detect instability patterns from metrics history.

    Returns dict with flags:
    - entropy_collapse: entropy below threshold for consecutive steps
    - margin_explosion: top1_margin growing rapidly
    - logit_explosion: max_logit exceeding threshold
    """
    if len(metrics_history) < 2:
        return {'entropy_collapse': False, 'margin_explosion': False, 'logit_explosion': False}

    recent = metrics_history[-window:] if len(metrics_history) >= window else metrics_history

    # Entropy collapse: CoreML entropy < 0.5 for 3+ consecutive steps
    cm_entropies = [m['cm_entropy'] for m in recent]
    entropy_collapse = sum(1 for e in cm_entropies[-3:] if e < 0.5) >= 3 if len(cm_entropies) >= 3 else False

    # Margin explosion: top1_margin > 20 (very peaky distribution)
    cm_margins = [m['cm_top1_margin'] for m in recent]
    margin_explosion = any(m > 20 for m in cm_margins[-3:]) if cm_margins else False

    # Logit explosion: max_logit > 50
    cm_max = [m['cm_max_logit'] for m in recent]
    logit_explosion = any(m > 50 for m in cm_max[-3:]) if cm_max else False

    return {
        'entropy_collapse': entropy_collapse,
        'margin_explosion': margin_explosion,
        'logit_explosion': logit_explosion,
    }


def compute_repetition_score(tokens, n=4, window=50):
    """Compute n-gram repetition score over last `window` tokens."""
    if len(tokens) < n + 1:
        return 0.0

    recent = tokens[-window:] if len(tokens) >= window else tokens

    # Count n-grams
    ngrams = []
    for i in range(len(recent) - n + 1):
        ngrams.append(tuple(recent[i:i+n]))

    if not ngrams:
        return 0.0

    unique_ratio = len(set(ngrams)) / len(ngrams)
    repetition_score = 1.0 - unique_ratio  # Higher = more repetition
    return repetition_score


# =============================================================================
# COMPARISON
# =============================================================================

def compare_logits(pytorch_logits, coreml_logits, tokenizer, top_k=10, verbose=True):
    """Compare logits between PyTorch and CoreML with stability metrics."""
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

    # Compute stability metrics
    stability = compute_stability_metrics(pt_logits, cm_logits)

    if verbose:
        print(f"\n  PyTorch argmax: {pt_argmax} ('{tokenizer.decode([pt_argmax])}') prob={pt_probs[pt_argmax]:.2f}%")
        print(f"  CoreML argmax:  {cm_argmax} ('{tokenizer.decode([cm_argmax])}') prob={cm_probs[cm_argmax]:.2f}%")
        print(f"  Match: {'YES' if match else 'NO'}")

        # Stability metrics
        print(f"\n  Stability: KL={stability['kl_divergence']:.4f} | "
              f"Entropy PT={stability['pt_entropy']:.2f} CM={stability['cm_entropy']:.2f} | "
              f"Margin PT={stability['pt_top1_margin']:.2f} CM={stability['cm_top1_margin']:.2f}")

        # Logits statistics
        print(f"  Logits: PT max={stability['pt_max_logit']:.2f} std={stability['pt_std_logit']:.2f} | "
              f"CM max={stability['cm_max_logit']:.2f} std={stability['cm_std_logit']:.2f} | "
              f"Corr={stability['correlation']:.4f}")

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
        'correlation': stability['correlation'],
        'pt_logits': pt_logits,
        'cm_logits': cm_logits,
        'stability': stability,
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
    driver: str = 'pt',  # 'pt', 'coreml', or 'teacher'
    prefill_mode: str = 'batch',  # 'batch' or 'token'
    verbose: bool = True,
):
    """Run token-by-token comparison with stability tracking.

    Args:
        driver: Which model drives next token selection:
            - 'pt': PyTorch argmax (default, good for parity testing)
            - 'coreml': CoreML argmax (realistic ANE behavior, catches instability)
            - 'teacher': Use provided token sequence (teacher forcing)
        prefill_mode: How to process the prompt:
            - 'batch': Use batch prefill (default, faster)
            - 'token': Single-token stepping (better for dataset generation, exercises infer path)
    """
    embed_model, ffn_infer, ffn_prefill, lmhead_model = coreml_models
    context_length = coreml_metadata['context_length']
    batch_size = coreml_metadata['batch_size']
    split_lm_head = coreml_metadata['split_lm_head']

    # Format prompt
    # Use enable_thinking=False to match chat.py behavior (pre-fills <think></think>)
    messages = [{"role": "user", "content": prompt}]
    template_kwargs = {
        "tokenize": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    if no_think:
        template_kwargs["enable_thinking"] = False

    input_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
    input_ids_np = input_ids.numpy().astype(np.int32)

    seq_len = input_ids.shape[1]
    print(f"\nPrompt tokens: {seq_len}")
    print(f"Prompt: {prompt}")
    print(f"Driver mode: {driver.upper()}")
    print(f"Prefill mode: {prefill_mode.upper()}")

    # Debug: show full tokenized prompt
    if verbose:
        decoded_prompt = tokenizer.decode(input_ids[0])
        print(f"\n--- Full tokenized prompt ---")
        print(repr(decoded_prompt))
        print(f"--- Token IDs: {input_ids[0].tolist()[:20]}{'...' if seq_len > 20 else ''}")

    # Reset PyTorch KV cache
    if hasattr(pytorch_model, 'reset_cache'):
        # HF model wrapper
        pytorch_model.reset_cache()
    elif hasattr(pytorch_model.model, 'kv_cache_0'):
        # Baked model
        pytorch_model.model.kv_cache_0.zero_()

    # Create CoreML state (prefer infer state for token mode)
    try:
        coreml_state = ffn_infer.make_state()
    except Exception:
        coreml_state = ffn_prefill.make_state()

    # Track prompt-phase metrics (only populated in token mode)
    prompt_metrics = []
    prompt_first_mismatch = None

    if prefill_mode == 'token':
        # === TOKENWISE PREFILL (teacher forcing) ===
        print("\n" + "="*60)
        print("PROMPT FEED (tokenwise teacher forcing)")
        print("="*60)

        t0 = time.time()
        pt_logits, cm_logits, prompt_metrics, prompt_first_mismatch = tokenwise_prompt_feed_teacher(
            pytorch_model=pytorch_model,
            pytorch_config=pytorch_config,
            embed_model=embed_model,
            ffn_infer=ffn_infer,
            lmhead_model=lmhead_model,
            input_ids_np=input_ids_np,
            context_length=context_length,
            split_lm_head=split_lm_head,
            coreml_state=coreml_state,
            probe_metrics=True,
        )
        prompt_feed_time = time.time() - t0

        print(f"\nPrompt feed timing (tokenwise): {prompt_feed_time*1000:.1f}ms ({seq_len/prompt_feed_time:.1f} tok/s)")

        if prompt_first_mismatch is not None:
            print(f"First prompt-token mismatch at pos={prompt_first_mismatch}")

        if prompt_metrics:
            kls = [m['kl_divergence'] for m in prompt_metrics]
            print(f"Prompt feed KL: avg={float(np.mean(kls)):.4f} max={float(np.max(kls)):.4f}")

        print("\nComparing logits after last prompt token:")
        prefill_result = compare_logits(pt_logits, cm_logits, tokenizer, verbose=verbose)

    else:
        # === BATCH PREFILL (original behavior) ===
        print("\n" + "="*60)
        print("PREFILL PHASE (batch)")
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

        # Get first token logits from CoreML
        last_token = input_ids_np[:, -1:]
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
    print(f"DECODE PHASE (driver={driver.upper()})")
    print("="*60)

    # Select first token based on driver
    if driver == 'coreml':
        next_token = prefill_result['cm_argmax']
    else:  # 'pt' or 'teacher'
        next_token = prefill_result['pt_argmax']

    pt_tokens = [prefill_result['pt_argmax']]
    cm_tokens = [prefill_result['cm_argmax']]
    driver_tokens = [next_token]  # Actual tokens fed to both models

    current_pos = seq_len
    results = []
    metrics_history = []
    first_mismatch_step = None

    for step in range(max_tokens - 1):
        if current_pos >= context_length - 1:
            print(f"\nReached context length limit ({context_length})")
            break

        # Check EOS (for driver token)
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

        # CoreML decode (same input token for fair comparison)
        cm_input = np.array([[next_token]], dtype=np.int32)
        cm_logits = coreml_forward_single(
            embed_model, ffn_infer, lmhead_model,
            cm_input, current_pos, context_length, coreml_state, split_lm_head
        )

        # Compare and get stability metrics
        result = compare_logits(pt_logits, cm_logits, tokenizer, verbose=verbose)
        results.append(result)

        # Track stability metrics
        if 'stability' in result:
            metrics_history.append(result['stability'])

            # Check for instability
            instability = detect_instability(metrics_history)
            if any(instability.values()):
                flags = [k for k, v in instability.items() if v]
                print(f"  *** INSTABILITY DETECTED: {', '.join(flags)} ***")

        # Track first mismatch
        if not result['match'] and first_mismatch_step is None:
            first_mismatch_step = step + 1
            print(f"  *** FIRST MISMATCH at step {first_mismatch_step} ***")

        # Check for stop on mismatch
        if stop_on_mismatch and not result['match']:
            print(f"\n*** STOPPING: Token mismatch at step {step + 1} ***")
            print(f"  PyTorch: {result['pt_argmax']} ('{tokenizer.decode([result['pt_argmax']])}')")
            print(f"  CoreML:  {result['cm_argmax']} ('{tokenizer.decode([result['cm_argmax']])}')")
            pt_tokens.append(result['pt_argmax'])
            cm_tokens.append(result['cm_argmax'])
            break

        # Record predictions
        pt_tokens.append(result['pt_argmax'])
        cm_tokens.append(result['cm_argmax'])

        # Select next token based on driver mode
        if driver == 'coreml':
            next_token = result['cm_argmax']
        else:  # 'pt' or 'teacher'
            next_token = result['pt_argmax']

        driver_tokens.append(next_token)
        current_pos += 1

        # Track repetition
        rep_score = compute_repetition_score(driver_tokens)
        if rep_score > 0.3:
            print(f"  *** HIGH REPETITION: {rep_score:.2f} ***")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    matches = sum(1 for r in results if r['match'])
    total = len(results)
    correlations = [r['correlation'] for r in results]

    print(f"\nDriver mode: {driver.upper()}")
    print(f"Token matches: {matches}/{total} ({100*matches/total:.1f}%)" if total > 0 else "No tokens generated")
    if first_mismatch_step:
        print(f"First mismatch at step: {first_mismatch_step}")

    if correlations:
        print(f"\nCorrelation stats:")
        print(f"  Avg: {np.mean(correlations):.6f}")
        print(f"  Min: {np.min(correlations):.6f}")
        print(f"  Max: {np.max(correlations):.6f}")

    # Prompt-phase stability summary (token mode only)
    if prompt_metrics:
        prompt_kl_values = [m['kl_divergence'] for m in prompt_metrics]
        prompt_cm_entropies = [m['cm_entropy'] for m in prompt_metrics]

        print(f"\nPrompt-phase stability (tokenwise):")
        print(f"  KL divergence:  avg={np.mean(prompt_kl_values):.4f} max={np.max(prompt_kl_values):.4f}")
        print(f"  CM entropy:     avg={np.mean(prompt_cm_entropies):.2f} min={np.min(prompt_cm_entropies):.2f}")
        if prompt_first_mismatch is not None:
            print(f"  First mismatch: pos={prompt_first_mismatch}")

    # Decode-phase stability summary
    if metrics_history:
        kl_values = [m['kl_divergence'] for m in metrics_history]
        cm_entropies = [m['cm_entropy'] for m in metrics_history]
        cm_margins = [m['cm_top1_margin'] for m in metrics_history]

        print(f"\nDecode-phase stability:")
        print(f"  KL divergence:  avg={np.mean(kl_values):.4f} max={np.max(kl_values):.4f}")
        print(f"  CM entropy:     avg={np.mean(cm_entropies):.2f} min={np.min(cm_entropies):.2f}")
        print(f"  CM top1_margin: avg={np.mean(cm_margins):.2f} max={np.max(cm_margins):.2f}")

    # Repetition analysis
    pt_rep = compute_repetition_score(pt_tokens)
    cm_rep = compute_repetition_score(cm_tokens)
    print(f"\nRepetition scores (4-gram):")
    print(f"  PyTorch: {pt_rep:.3f}")
    print(f"  CoreML:  {cm_rep:.3f}")

    print(f"\nPyTorch tokens: {pt_tokens[:20]}{'...' if len(pt_tokens) > 20 else ''}")
    print(f"CoreML tokens:  {cm_tokens[:20]}{'...' if len(cm_tokens) > 20 else ''}")
    if driver == 'coreml':
        print(f"Driver tokens:  {driver_tokens[:20]}{'...' if len(driver_tokens) > 20 else ''}")

    # Decode outputs with clear visual separation
    pt_output = tokenizer.decode(pt_tokens)
    cm_output = tokenizer.decode(cm_tokens)

    print("\n" + "="*60)
    print("PYTORCH OUTPUT")
    print("="*60)
    print(pt_output)
    print("\n" + "-"*60)
    print("COREML OUTPUT")
    print("-"*60)
    print(cm_output)
    print("-"*60)

    return {
        'matches': matches,
        'total': total,
        'match_rate': matches / total if total > 0 else 0,
        'avg_correlation': np.mean(correlations) if correlations else 0,
        'first_mismatch_step': first_mismatch_step,
        'pt_tokens': pt_tokens,
        'cm_tokens': cm_tokens,
        'driver_tokens': driver_tokens,
        'metrics_history': metrics_history,
        'pt_repetition': pt_rep,
        'cm_repetition': cm_rep,
        # Token-mode specific (empty for batch mode)
        'prompt_metrics': prompt_metrics,
        'prompt_first_mismatch': prompt_first_mismatch,
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
    parser.add_argument('checkpoint', type=str, nargs='?', default=None,
                        help='Path to AQ1 checkpoint for PyTorch (required unless --hf-reference is used)')
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
    parser.add_argument('--driver', type=str, default='pt', choices=['pt', 'coreml', 'teacher'],
                        help='Driver mode: pt (PyTorch drives), coreml (CoreML drives, realistic ANE), teacher')
    parser.add_argument('--prefill-mode', type=str, default='batch', choices=['batch', 'token'],
                        help='Prefill mode: batch (default, faster) or token (single-token stepping, better for dataset generation)')
    parser.add_argument('--hf-reference', type=str, default=None,
                        help='Use HuggingFace model as reference instead of baked checkpoint (e.g., Qwen/Qwen3-0.6B)')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    # Validate arguments
    use_hf_reference = args.hf_reference is not None
    if not use_hf_reference and args.checkpoint is None:
        parser.error("checkpoint is required unless --hf-reference is specified")

    coreml_dir = os.path.expanduser(args.coreml_dir)

    if not use_hf_reference:
        checkpoint_path = os.path.expanduser(args.checkpoint)
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

    if not os.path.exists(coreml_dir):
        print(f"Error: CoreML directory not found: {coreml_dir}")
        sys.exit(1)

    print("="*60)
    if use_hf_reference:
        print("HuggingFace vs CoreML Comparison")
    else:
        print("PyTorch vs CoreML Comparison")
    print("="*60)
    if use_hf_reference:
        print(f"HuggingFace model: {args.hf_reference}")
    else:
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

    # Load reference model (HF or baked checkpoint)
    if use_hf_reference:
        print(f"\n--- Loading HuggingFace reference model ---")
        pytorch_model, pytorch_config = load_hf_pytorch_model(
            args.hf_reference, context_length, state_length=state_length, verbose=True
        )
    else:
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
        driver=args.driver,
        prefill_mode=args.prefill_mode,
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
