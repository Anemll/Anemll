#!/usr/bin/env python3
"""
Fixed RMSNorm implementation for Qwen3 model
This shows the correct implementation matching HuggingFace transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QwenRMSNorm(nn.Module):
    """Standard RMSNorm implementation matching HuggingFace Qwen3."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class QwenHeadNorm(nn.Module):
    """Per-head RMSNorm for query and key projections - Standard RMSNorm matching HuggingFace."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Test comparison between old and new implementations
def test_rmsnorm_comparison():
    """Test the difference between ANEMLL's doubled tensor trick and standard RMSNorm."""
    
    # Create test input
    batch_size, seq_len, hidden_size = 1, 10, 512
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    
    # Old ANEMLL implementation (doubled tensor trick)
    class OldQwenRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            x = hidden_states
            # Doubled tensor trick
            doubled = torch.cat([x, -x], dim=-1)
            hidden_size = hidden_states.shape[-1]
            normed = F.layer_norm(
                doubled,
                normalized_shape=(2 * hidden_size,),
                weight=None,
                bias=None,
                eps=float(self.variance_epsilon)
            )
            normed = normed[..., : hidden_size]
            return (normed * self.weight
                           .to(normed.dtype, copy=False)
                           .to(normed.device, copy=False))
    
    # Initialize both normalization layers
    old_norm = OldQwenRMSNorm(hidden_size)
    new_norm = QwenRMSNorm(hidden_size)
    
    # Use same weights for fair comparison
    new_norm.weight.data = old_norm.weight.data.clone()
    
    # Forward pass
    with torch.no_grad():
        old_output = old_norm(x)
        new_output = new_norm(x)
    
    # Compare outputs
    diff = (old_output - new_output).abs()
    print(f"Max difference: {diff.max().item():.6f}")
    print(f"Mean difference: {diff.mean().item():.6f}")
    print(f"Output shapes - Old: {old_output.shape}, New: {new_output.shape}")
    
    # Show sample values
    print(f"Old output sample: {old_output[0, 0, :5].tolist()}")
    print(f"New output sample: {new_output[0, 0, :5].tolist()}")
    
    return diff.max().item() > 0.01  # Return True if significant difference


if __name__ == "__main__":
    print("Testing RMSNorm implementation differences...")
    significant_diff = test_rmsnorm_comparison()
    print(f"Significant difference detected: {significant_diff}")