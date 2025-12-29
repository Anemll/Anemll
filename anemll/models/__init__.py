"""ANEMLL Models Package

This package contains model implementations optimized for Apple Neural Engine.
"""

# ANEMLL Quantization components
from .anemll_quant import (
    AnemllConv2d,
    AnemllQuantConfig,
    load_anemll_weights,
    load_anemll_checkpoint_full,
    convert_conv2d_to_anemll,
    compute_lut_indices,
    reconstruct_from_lut,
    DEFAULT_MLP_SCALE_RANK,
    DEFAULT_ATTN_SCALE_RANK,
)

# Qwen models
from .qwen_model import QwenConfig, QwenForCausalLM, QwenModel
from .qwenAQ1_model import (
    QwenConfig as QwenAQ1Config,
    QwenForCausalLM as QwenAQ1ForCausalLM,
    QwenModel as QwenAQ1Model,
)

__all__ = [
    # Quantization
    "AnemllConv2d",
    "AnemllQuantConfig",
    "load_anemll_weights",
    "load_anemll_checkpoint_full",
    "convert_conv2d_to_anemll",
    "compute_lut_indices",
    "reconstruct_from_lut",
    "DEFAULT_MLP_SCALE_RANK",
    "DEFAULT_ATTN_SCALE_RANK",
    # Qwen
    "QwenConfig",
    "QwenForCausalLM",
    "QwenModel",
    # Qwen AQ1 (ANEMLL-QUANT-1)
    "QwenAQ1Config",
    "QwenAQ1ForCausalLM",
    "QwenAQ1Model",
]
