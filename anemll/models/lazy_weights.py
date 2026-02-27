"""Reusable lazy safetensors loading helpers for conversion workflows.

This module keeps weight IO scoped to the keys needed by the current conversion
part/chunk. It is model-agnostic and can be reused by Qwen/Gemma/Llama loaders.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import safetensors
import torch

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class WeightLoadSpec:
    """Declarative key filter for lazy safetensors loading."""

    include_prefixes: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()
    include_exact: frozenset[str] = field(default_factory=frozenset)
    exclude_exact: frozenset[str] = field(default_factory=frozenset)
    layer_range: tuple[int, int] | None = None
    strip_prefix: str | None = None
    description: str = "weights"

    def matches(self, key: str) -> bool:
        if key in self.exclude_exact:
            return False
        if self.exclude_prefixes and any(key.startswith(p) for p in self.exclude_prefixes):
            return False

        has_include_filters = bool(self.include_exact) or bool(self.include_prefixes)
        if has_include_filters:
            included = key in self.include_exact or any(
                key.startswith(p) for p in self.include_prefixes
            )
        else:
            included = True
        if not included:
            return False

        if self.layer_range is None:
            return True

        layer_idx = extract_layer_index(key)
        if layer_idx is None:
            return True

        start, end = self.layer_range
        return start <= layer_idx < end


@dataclass(frozen=True)
class WeightLoadStats:
    description: str
    matched_keys: int
    total_keys: int
    files_touched: int
    total_files: int
    loaded_bytes: int
    elapsed_sec: float

    @property
    def loaded_mb(self) -> float:
        return self.loaded_bytes / (1024 * 1024)

    def summary(self) -> str:
        return (
            f"{self.description}: matched {self.matched_keys}/{self.total_keys} tensors, "
            f"files {self.files_touched}/{self.total_files}, "
            f"loaded {self.loaded_mb:.1f} MB in {self.elapsed_sec:.2f}s"
        )


def extract_layer_index(key: str) -> int | None:
    """Return decoder layer index from keys like 'model.layers.12.self_attn...'."""
    m = _LAYER_RE.search(key)
    if not m:
        return None
    return int(m.group(1))


def compute_chunk_layer_range(total_layers: int, total_chunks: int, chunk_idx: int) -> Tuple[int, int]:
    """Remainder-aware chunk split used by converters."""
    if total_chunks <= 0:
        raise ValueError("total_chunks must be > 0")
    if chunk_idx < 0 or chunk_idx >= total_chunks:
        raise ValueError(f"chunk_idx out of range: {chunk_idx} not in [0, {total_chunks})")

    base, rem = divmod(total_layers, total_chunks)
    start = chunk_idx * base + min(chunk_idx, rem)
    end = start + base + (1 if chunk_idx < rem else 0)
    return start, end


class LazySafeTensorLoader:
    """Index safetensors files once and fetch only selected tensors on demand."""

    def __init__(self, model_path: str) -> None:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)

        self.model_path = model_path
        self._paths = sorted(
            os.path.join(model_path, f)
            for f in os.listdir(model_path)
            if f.endswith(".safetensors")
        )
        if not self._paths:
            raise FileNotFoundError(f"No .safetensors found in {model_path}")

        self._keys_by_file: dict[str, tuple[str, ...]] = {}
        self._key_to_file: dict[str, str] = {}

    def _ensure_index(self) -> None:
        if self._key_to_file:
            return

        for path in self._paths:
            with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
                keys = tuple(handle.keys())
            self._keys_by_file[path] = keys
            for key in keys:
                self._key_to_file[key] = path

    def load_state_dict(self, spec: WeightLoadSpec) -> tuple[Dict[str, torch.Tensor], WeightLoadStats]:
        """Load tensors that match `spec` and optionally strip a key prefix."""
        self._ensure_index()

        start = time.perf_counter()
        selected_keys = [k for k in self._key_to_file if spec.matches(k)]

        keys_by_path: dict[str, list[str]] = {}
        for key in selected_keys:
            path = self._key_to_file[key]
            keys_by_path.setdefault(path, []).append(key)

        loaded_bytes = 0
        out: Dict[str, torch.Tensor] = {}
        for path in self._paths:
            keys = keys_by_path.get(path)
            if not keys:
                continue
            with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
                for key in keys:
                    out_key = key
                    if spec.strip_prefix and out_key.startswith(spec.strip_prefix):
                        out_key = out_key[len(spec.strip_prefix) :]
                    tensor = handle.get_tensor(key)
                    loaded_bytes += int(tensor.numel() * tensor.element_size())
                    out[out_key] = tensor

        elapsed = time.perf_counter() - start
        stats = WeightLoadStats(
            description=spec.description,
            matched_keys=len(selected_keys),
            total_keys=len(self._key_to_file),
            files_touched=len(keys_by_path),
            total_files=len(self._paths),
            loaded_bytes=loaded_bytes,
            elapsed_sec=elapsed,
        )
        return out, stats

    def available_keys(self) -> Iterable[str]:
        self._ensure_index()
        return self._key_to_file.keys()
