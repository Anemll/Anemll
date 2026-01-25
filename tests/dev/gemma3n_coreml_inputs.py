#!/usr/bin/env python3
"""Shared helpers for Gemma3n CoreML chunk inputs."""

from __future__ import annotations

import numpy as np


def create_position_mask(current_pos: int, ctx_len: int) -> np.ndarray:
    """Create causal mask row for a given position."""
    mask = np.full((1, 1, ctx_len, ctx_len), float("-inf"), dtype=np.float16)
    mask[:, :, 0, : current_pos + 1] = 0.0
    return mask


def create_position_one_hot(current_pos: int, ctx_len: int) -> np.ndarray:
    """Return [1,1,ctx_len,1] one-hot tensor for KV cache updates."""
    one_hot = np.zeros((1, 1, ctx_len, 1), dtype=np.float16)
    one_hot[0, 0, current_pos, 0] = 1.0
    return one_hot


def create_rotary_embeddings(
    current_pos: int,
    head_dim: int,
    theta_local: float = 10_000.0,
    theta_global: float = 1_000_000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (cos_local, sin_local, cos_global, sin_global) for current position."""

    def compute_rotary(theta: float) -> tuple[np.ndarray, np.ndarray]:
        inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        freqs = current_pos * inv_freq
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos = np.cos(emb).astype(np.float16).reshape(1, 1, -1)
        sin = np.sin(emb).astype(np.float16).reshape(1, 1, -1)
        return cos, sin

    cos_local, sin_local = compute_rotary(theta_local)
    cos_global, sin_global = compute_rotary(theta_global)
    return cos_local, sin_local, cos_global, sin_global


__all__ = [
    "create_position_mask",
    "create_position_one_hot",
    "create_rotary_embeddings",
]
