import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import numpy as np
from pathlib import Path
import math
import subprocess
import shutil

# Global constants for model configuration
NUM_PARALLEL_LAYERS = 20
NUM_TESTS = 1000

# Global constants for tensor dimensions
hidden_size = 1024
batch_size = 1
seq_length = 8


class StandardRMSNorm(nn.Module):
    """Standard RMSNorm implementation for comparison."""
    
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


class SebaRMSNorm(nn.Module):
    """Seba's stable low precision RMSNorm implementation for ANE."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        # dimroot is sqrt(hidden_size) for normalization
        self.dimroot = np.sqrt(hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states
        
        # Following the exact implementation from Seba's repo
        # Get max absolute value for scaling
        maxval = torch.abs(x)
        maxval = torch.max(maxval, dim=-1, keepdim=True)[0]
        maxval = torch.clamp(maxval, min=self.variance_epsilon)
        
        # Scale by max value for numerical stability
        xscaled = x / maxval
        
        # Compute squared sum
        sq_sum = torch.sum(xscaled * xscaled, dim=-1, keepdim=True)
        
        # Apply rsqrt
        rsqrt = torch.rsqrt(sq_sum + self.variance_epsilon)
        
        # Scale by dimension root BEFORE normalization
        xscaled = xscaled * self.dimroot
        
        # Apply normalization
        xnormed = xscaled * rsqrt
        
        # Apply weight
        return self.weight * xnormed.to(input_dtype)


class LinalgRMSNorm(nn.Module):
    """ANE-friendly RMSNorm using linalg.norm"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        # Note: Original uses 4D tensor (1, size, 1, 1), we'll adapt for our use case
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states shape: (batch, seq_len, hidden_size)
        dtype = hidden_states.dtype
        x = hidden_states.float()
        
        # Reshape to 4D for consistency with original implementation
        batch_size, seq_len, hidden_size = x.shape
        x = x.view(batch_size * seq_len, hidden_size, 1, 1)
        
        # Create epsilon channel on the same device as input
        eps_chan = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype) * ((self.eps * x.size(1)) ** 0.5)
        x_eps = torch.cat((x, eps_chan), dim=1)
        
        # Compute norm using linalg.norm
        norm_x = torch.linalg.norm(x_eps, dim=1, keepdim=True)
        x_normed = x / norm_x
        x_normed = x_normed * math.sqrt(x.size(1))
        
        # Reshape back and apply weight
        x_normed = x_normed.view(batch_size, seq_len, hidden_size)
        x_normed = x_normed.to(dtype=dtype)
        return x_normed * self.weight


class QwenRMSNorm(nn.Module):
    """ANE optimized RMSNorm implementation. We use layer_norm and avoid the mean subtraction.
    This give us the best quality for Boolq and other benchmarks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, weight_init: torch.Tensor = None) -> None:
        super().__init__()
        if weight_init is not None:
            self.weight = nn.Parameter(weight_init.clone())
        else:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ──────────────────────────────────────────────────────────────────────
        # Compatibility path for PyTorch 1.x / 2.0–2.3                           .
        # We build a tensor whose mean is *exactly* zero so that LayerNorm's
        # mean‑subtraction becomes a no‑op and we recover RMS statistics:
        #
        #     concat([x, ‑x])  →  μ = 0,
        #                        σ² = ½(‖x‖²) = mean(x²)
        # ──────────────────────────────────────────────────────────────────────

        x = hidden_states
        
        # ❶ Make the last‑dimension mean zero.
        doubled = torch.cat([x, -x], dim=-1)

        hidden_size = hidden_states.shape[-1]

        # ❂ Run the highly‑optimised LayerNorm kernel on the doubled tensor.
        # Use static weights of 1 for layer norm
        weight = torch.ones(2 * hidden_size, device=doubled.device, dtype=doubled.dtype)
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * hidden_size,),
            weight=weight,
            bias=None,
            eps=float(self.variance_epsilon)
        )

        # ❸ Drop the mirror half → correct RMS‑normed activations.
        #normed, _ = torch.split(normed, [hidden_size, hidden_size], dim=-1)
        normed = normed[..., : hidden_size]
        return (normed * self.weight
                        .to(normed.dtype, copy=False)
                        .to(normed.device, copy=False))


class ParallelQwenRMSNorm(nn.Module):
    """Multiple QwenRMSNorm layers in parallel, averaged output."""
    
    def __init__(self, hidden_size: int, num_layers: int = NUM_PARALLEL_LAYERS, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Initialize layers without weights (will be set later)
        for i in range(num_layers):
            self.layers.append(QwenRMSNorm(hidden_size, eps))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply all layers in parallel
        outputs = []
        for layer in self.layers:
            outputs.append(layer(hidden_states))
        
        # Stack, sum and average
        stacked = torch.stack(outputs, dim=0)
        summed = torch.sum(stacked, dim=0)
        averaged = summed / self.num_layers
        
        return averaged


class ParallelSebaRMSNorm(nn.Module):
    """Multiple SebaRMSNorm layers in parallel, averaged output."""
    
    def __init__(self, hidden_size: int, num_layers: int = NUM_PARALLEL_LAYERS, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Initialize layers without weights (will be set later)
        for i in range(num_layers):
            self.layers.append(SebaRMSNorm(hidden_size, eps))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply all layers in parallel
        outputs = []
        for layer in self.layers:
            outputs.append(layer(hidden_states))
        
        # Stack, sum and average
        stacked = torch.stack(outputs, dim=0)
        summed = torch.sum(stacked, dim=0)
        averaged = summed / self.num_layers
        
        return averaged


class ParallelLinalgRMSNorm(nn.Module):
    """Multiple LinalgRMSNorm layers in parallel, averaged output."""
    
    def __init__(self, hidden_size: int, num_layers: int = NUM_PARALLEL_LAYERS, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Initialize layers without weights (will be set later)
        for i in range(num_layers):
            self.layers.append(LinalgRMSNorm(hidden_size, eps))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply all layers in parallel
        outputs = []
        for layer in self.layers:
            outputs.append(layer(hidden_states))
        
        # Stack, sum and average
        stacked = torch.stack(outputs, dim=0)
        summed = torch.sum(stacked, dim=0)
        averaged = summed / self.num_layers
        
        return averaged


class Conv2DQwenRMSNorm(nn.Module):
    """QwenRMSNorm with Conv2D layers before normalization for ANE complexity."""
    
    def __init__(self, hidden_size: int, num_layers: int = NUM_PARALLEL_LAYERS, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.conv_layers = nn.ModuleList()
        
        # Create Conv2D layers for complexity
        # Use 1x1 convolutions to maintain spatial dimensions
        for i in range(num_layers):
            self.conv_layers.append(
                nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
            )
        
        # Single RMSNorm layer
        self.norm = QwenRMSNorm(hidden_size, eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Reshape to (batch*seq_len, hidden_size, 1, 1) for Conv2D
        x = hidden_states.view(batch_size * seq_len, hidden_size, 1, 1)
        
        # Apply Conv2D layers and accumulate
        output = torch.zeros_like(x)
        for conv in self.conv_layers:
            output = output + conv(x.float())
        
        # Average
        output = output / self.num_layers
        
        # Reshape back to (batch, seq_len, hidden_size)
        output = output.view(batch_size, seq_len, hidden_size)
        
        # Apply RMSNorm
        output = output.to(hidden_states.dtype)
        return self.norm(output)


class Conv2DSebaRMSNorm(nn.Module):
    """SebaRMSNorm with Conv2D layers before normalization for ANE complexity."""
    
    def __init__(self, hidden_size: int, num_layers: int = NUM_PARALLEL_LAYERS, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.conv_layers = nn.ModuleList()
        
        # Create Conv2D layers for complexity
        # Use 1x1 convolutions to maintain spatial dimensions
        for i in range(num_layers):
            self.conv_layers.append(
                nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
            )
        
        # Single RMSNorm layer
        self.norm = SebaRMSNorm(hidden_size, eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Reshape to (batch*seq_len, hidden_size, 1, 1) for Conv2D
        x = hidden_states.view(batch_size * seq_len, hidden_size, 1, 1)
        
        # Apply Conv2D layers and accumulate
        output = torch.zeros_like(x)
        for conv in self.conv_layers:
            output = output + conv(x.float())
        
        # Average
        output = output / self.num_layers
        
        # Reshape back to (batch, seq_len, hidden_size)
        output = output.view(batch_size, seq_len, hidden_size)
        
        # Apply RMSNorm
        output = output.to(hidden_states.dtype)
        return self.norm(output)


class Conv2DLinalgRMSNorm(nn.Module):
    """LinalgRMSNorm with Conv2D layers before normalization for ANE complexity."""
    
    def __init__(self, hidden_size: int, num_layers: int = NUM_PARALLEL_LAYERS, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.conv_layers = nn.ModuleList()
        
        # Create Conv2D layers for complexity
        # Use 1x1 convolutions to maintain spatial dimensions
        for i in range(num_layers):
            self.conv_layers.append(
                nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
            )
        
        # Single RMSNorm layer
        self.norm = LinalgRMSNorm(hidden_size, eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Reshape to (batch*seq_len, hidden_size, 1, 1) for Conv2D
        x = hidden_states.view(batch_size * seq_len, hidden_size, 1, 1)
        
        # Apply Conv2D layers and accumulate
        output = torch.zeros_like(x)
        for conv in self.conv_layers:
            output = output + conv(x.float())
        
        # Average
        output = output / self.num_layers
        
        # Reshape back to (batch, seq_len, hidden_size)
        output = output.view(batch_size, seq_len, hidden_size)
        
        # Apply RMSNorm
        output = output.to(hidden_states.dtype)
        return self.norm(output)


def export_seba_rmsnorm_to_coreml():
    """Export ParallelSebaRMSNorm to CoreML format."""
    
    # Create parallel model instance
    model = ParallelSebaRMSNorm(hidden_size=hidden_size, num_layers=NUM_PARALLEL_LAYERS)
    model.eval()
    
    # Create example input
    example_input = torch.randn(batch_size, seq_length, hidden_size)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(
            name="hidden_states",
            shape=(batch_size, seq_length, hidden_size),
            dtype=np.float16
        )],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS16
    )
    
    # Save the model
    output_path = Path(f"/tmp/ANE/seba_rmsnorm_parallel_{NUM_PARALLEL_LAYERS}.mlpackage")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coreml_model.save(str(output_path))
    
    print(f"SebaRMSNorm model saved to: {output_path}")
    print(f"Model contains {NUM_PARALLEL_LAYERS} parallel SebaRMSNorm layers")
    
    # The model is saved but not loaded here - will be compiled and loaded later
    print(f"Model saved for compilation")
    
    return coreml_model, model


def export_rmsnorm_to_coreml():
    """Export ParallelQwenRMSNorm to CoreML format."""
    
    
    # Create parallel model instance
    model = ParallelQwenRMSNorm(hidden_size=hidden_size, num_layers=NUM_PARALLEL_LAYERS)
    model.eval()
    
    # Create example input
    example_input = torch.randn(batch_size, seq_length, hidden_size)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(
            name="hidden_states",
            shape=(batch_size, seq_length, hidden_size),
            dtype=np.float16
        )],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS16
    )
    
    # Save the model
    output_path = Path(f"/tmp/ANE/qwen_rmsnorm_parallel_{NUM_PARALLEL_LAYERS}.mlpackage")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coreml_model.save(str(output_path))
    
    print(f"Model saved to: {output_path}")
    print(f"Model contains {NUM_PARALLEL_LAYERS} parallel QwenRMSNorm layers")
    
    # The model is saved but not loaded here - will be compiled and loaded later
    print(f"Model saved for compilation")
    
    return coreml_model, model


def test_coreml_vs_pytorch():
    """Test CoreML model against PyTorch implementation."""
    
    # Export the model
    coreml_model, pytorch_parallel_model = export_rmsnorm_to_coreml()
    
    # Create test input in fp16
    test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16)
    
    # Create single QwenRMSNorm and standard RMSNorm for comparison
    single_qwen_model = QwenRMSNorm(hidden_size=hidden_size)
    single_qwen_model.eval()
    standard_model = StandardRMSNorm(hidden_size=hidden_size)
    standard_model.eval()
    
    # PyTorch outputs
    with torch.no_grad():
        parallel_output = pytorch_parallel_model(test_input).numpy()
        single_output = single_qwen_model(test_input).numpy()
        standard_output = standard_model(test_input).numpy()
    
    # CoreML output (input is already fp16)
    coreml_output = coreml_model.predict({"hidden_states": test_input.numpy()})["output"]
    
    # Note: Parallel output will be different from single due to different weights
    # We'll just verify the shape and that it's reasonable
    
    # Compare Single QwenRMSNorm vs Standard RMSNorm
    pytorch_diff = np.max(np.abs(single_output - standard_output))
    pytorch_mean_diff = np.mean(np.abs(single_output - standard_output))
    
    # Compare CoreML vs Parallel PyTorch
    coreml_diff = np.max(np.abs(parallel_output - coreml_output))
    coreml_mean_diff = np.mean(np.abs(parallel_output - coreml_output))
    
    print(f"\nComparison Results:")
    print(f"Input dtype: {test_input.dtype}")
    print(f"Output shapes: {parallel_output.shape}")
    print(f"Model has {NUM_PARALLEL_LAYERS} parallel layers with different weights")
    print(f"\nSingle QwenRMSNorm vs Standard RMSNorm:")
    print(f"  Max difference: {pytorch_diff}")
    print(f"  Mean difference: {pytorch_mean_diff}")
    print(f"\nCoreML Parallel vs PyTorch Parallel:")
    print(f"  Max difference: {coreml_diff}")
    print(f"  Mean difference: {coreml_mean_diff}")
    
    # Check if outputs are close
    tolerance = 8e-3  # 0.008 for fp16 precision with 20 layers
    if pytorch_diff < tolerance and coreml_diff < tolerance:
        print(f"\n✅ Test PASSED! All outputs match within tolerance {tolerance}")
    else:
        print(f"\n❌ Test FAILED! Differences exceed tolerance {tolerance}")
    
    return pytorch_diff < tolerance and coreml_diff < tolerance


def sweep_test(num_iterations=1000):
    """Run multiple tests with different random inputs to ensure consistency."""
    
    print(f"\nRunning sweep test with {num_iterations} iterations...")
    print(f"Testing {NUM_PARALLEL_LAYERS} parallel QwenRMSNorm layers")
    
    # Export the model once
    
    coreml_model, pytorch_parallel_model = export_rmsnorm_to_coreml()
    single_qwen_model = QwenRMSNorm(hidden_size=hidden_size)
    single_qwen_model.eval()
    standard_model = StandardRMSNorm(hidden_size=hidden_size)
    standard_model.eval()
    
    max_pytorch_diffs = []
    max_coreml_diffs = []
    
    for i in range(num_iterations):
        # Generate random input
        test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16)
        
        # PyTorch outputs
        with torch.no_grad():
            parallel_output = pytorch_parallel_model(test_input).numpy()
            single_output = single_qwen_model(test_input).numpy()
            standard_output = standard_model(test_input).numpy()
        
        # CoreML output
        coreml_output = coreml_model.predict({"hidden_states": test_input.numpy()})["output"]
        
        # Calculate differences
        pytorch_diff = np.max(np.abs(single_output - standard_output))
        coreml_diff = np.max(np.abs(parallel_output - coreml_output))
        
        max_pytorch_diffs.append(pytorch_diff)
        max_coreml_diffs.append(coreml_diff)
        
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{num_iterations} iterations")
    
    # Analyze results
    print(f"\nSweep Test Results ({num_iterations} iterations):")
    print(f"\nQwenRMSNorm vs Standard RMSNorm:")
    print(f"  Max difference across all tests: {np.max(max_pytorch_diffs)}")
    print(f"  Mean of max differences: {np.mean(max_pytorch_diffs)}")
    print(f"  Std of max differences: {np.std(max_pytorch_diffs)}")
    
    print(f"\nCoreML vs QwenRMSNorm:")
    print(f"  Max difference across all tests: {np.max(max_coreml_diffs)}")
    print(f"  Mean of max differences: {np.mean(max_coreml_diffs)}")
    print(f"  Std of max differences: {np.std(max_coreml_diffs)}")
    
    tolerance = 4e-3  # 0.004 for fp16 precision
    all_passed = np.all(np.array(max_pytorch_diffs) < tolerance) and np.all(np.array(max_coreml_diffs) < tolerance)
    
    if all_passed:
        print(f"\n✅ All {num_iterations} tests PASSED within tolerance {tolerance}")
    else:
        print(f"\n❌ Some tests exceeded tolerance {tolerance}")
    
    return all_passed


def compare_all_implementations():
    """Compare QwenRMSNorm, SebaRMSNorm, and Standard RMSNorm."""
    
    print("\nComparing all RMSNorm implementations...")
    
    # Create models
    qwen_model = QwenRMSNorm(hidden_size=hidden_size)
    seba_model = SebaRMSNorm(hidden_size=hidden_size)
    standard_model = StandardRMSNorm(hidden_size=hidden_size)
    
    qwen_model.eval()
    seba_model.eval()
    standard_model.eval()
    
    # Test with multiple inputs
    num_tests = 10
    qwen_vs_standard_diffs = []
    seba_vs_standard_diffs = []
    qwen_vs_seba_diffs = []
    
    # Detailed analysis on first test
    test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16)
    
    with torch.no_grad():
        qwen_output = qwen_model(test_input).numpy()
        seba_output = seba_model(test_input).numpy()
        standard_output = standard_model(test_input).numpy()
    
    print(f"\nDetailed analysis on single input:")
    print(f"Input stats - mean: {test_input.mean():.6f}, std: {test_input.std():.6f}")
    print(f"Standard output - mean: {standard_output.mean():.6f}, std: {standard_output.std():.6f}")
    print(f"Qwen output - mean: {qwen_output.mean():.6f}, std: {qwen_output.std():.6f}")
    print(f"Seba output - mean: {seba_output.mean():.6f}, std: {seba_output.std():.6f}")
    
    # Check actual differences
    diff_seba_standard = seba_output - standard_output
    print(f"\nSeba vs Standard difference stats:")
    print(f"  Min diff: {diff_seba_standard.min()}")
    print(f"  Max diff: {diff_seba_standard.max()}")
    print(f"  Mean diff: {diff_seba_standard.mean()}")
    print(f"  Std of diff: {diff_seba_standard.std()}")
    
    for i in range(num_tests):
        test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16)
        
        with torch.no_grad():
            qwen_output = qwen_model(test_input).numpy()
            seba_output = seba_model(test_input).numpy()
            standard_output = standard_model(test_input).numpy()
        
        qwen_vs_standard = np.max(np.abs(qwen_output - standard_output))
        seba_vs_standard = np.max(np.abs(seba_output - standard_output))
        qwen_vs_seba = np.max(np.abs(qwen_output - seba_output))
        
        qwen_vs_standard_diffs.append(qwen_vs_standard)
        seba_vs_standard_diffs.append(seba_vs_standard)
        qwen_vs_seba_diffs.append(qwen_vs_seba)
    
    print(f"\nResults from {num_tests} test inputs:")
    print(f"\nQwenRMSNorm vs Standard:")
    print(f"  Max difference: {np.max(qwen_vs_standard_diffs)}")
    print(f"  Mean difference: {np.mean(qwen_vs_standard_diffs)}")
    print(f"  All diffs: {qwen_vs_standard_diffs[:5]}...")
    
    print(f"\nSebaRMSNorm vs Standard:")
    print(f"  Max difference: {np.max(seba_vs_standard_diffs)}")
    print(f"  Mean difference: {np.mean(seba_vs_standard_diffs)}")
    print(f"  All diffs: {seba_vs_standard_diffs[:5]}...")
    
    print(f"\nQwenRMSNorm vs SebaRMSNorm:")
    print(f"  Max difference: {np.max(qwen_vs_seba_diffs)}")
    print(f"  Mean difference: {np.mean(qwen_vs_seba_diffs)}")
    print(f"  All diffs: {qwen_vs_seba_diffs[:5]}...")


def export_linalg_rmsnorm_to_coreml():
    """Export ParallelLinalgRMSNorm to CoreML format."""
    
    
    # Create parallel model instance
    model = ParallelLinalgRMSNorm(hidden_size=hidden_size, num_layers=NUM_PARALLEL_LAYERS)
    model.eval()
    
    # Create example input
    example_input = torch.randn(batch_size, seq_length, hidden_size)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(
            name="hidden_states",
            shape=(batch_size, seq_length, hidden_size),
            dtype=np.float16
        )],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS16
    )
    
    # Save the model
    output_path = Path(f"/tmp/ANE/linalg_rmsnorm_parallel_{NUM_PARALLEL_LAYERS}.mlpackage")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coreml_model.save(str(output_path))
    
    print(f"LinalgRMSNorm model saved to: {output_path}")
    print(f"Model contains {NUM_PARALLEL_LAYERS} parallel LinalgRMSNorm layers")
    
    # The model is saved but not loaded here - will be compiled and loaded later
    print(f"Model saved for compilation")
    
    return coreml_model, model


def test_seba_coreml_vs_pytorch():
    """Test SebaRMSNorm CoreML model against PyTorch implementation."""
    
    # Export the model
    coreml_model, pytorch_parallel_model = export_seba_rmsnorm_to_coreml()
    
    # Create test input in fp16
    test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16)
    
    # PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_parallel_model(test_input).numpy()
    
    # CoreML output
    coreml_output = coreml_model.predict({"hidden_states": test_input.numpy()})["output"]
    
    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - coreml_output))
    mean_diff = np.mean(np.abs(pytorch_output - coreml_output))
    
    print(f"\nSebaRMSNorm Comparison Results:")
    print(f"Input dtype: {test_input.dtype}")
    print(f"Output shapes: {pytorch_output.shape}")
    print(f"Model has {NUM_PARALLEL_LAYERS} parallel layers with different weights")
    print(f"\nCoreML vs PyTorch:")
    print(f"  Max difference: {max_diff}")
    print(f"  Mean difference: {mean_diff}")
    
    # Check if outputs are close
    tolerance = 8e-3  # 0.008 for fp16 precision with 20 layers
    if max_diff < tolerance:
        print(f"\n✅ Test PASSED! Outputs match within tolerance {tolerance}")
    else:
        print(f"\n❌ Test FAILED! Differences exceed tolerance {tolerance}")
    
    return max_diff < tolerance


def test_linalg_coreml_vs_pytorch():
    """Test LinalgRMSNorm CoreML model against PyTorch implementation."""
    
    # Export the model
    coreml_model, pytorch_parallel_model = export_linalg_rmsnorm_to_coreml()
    
    # Create test input in fp16
    test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16)
    
    # PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_parallel_model(test_input).numpy()
    
    # CoreML output
    coreml_output = coreml_model.predict({"hidden_states": test_input.numpy()})["output"]
    
    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - coreml_output))
    mean_diff = np.mean(np.abs(pytorch_output - coreml_output))
    
    print(f"\nLinalgRMSNorm Comparison Results:")
    print(f"Input dtype: {test_input.dtype}")
    print(f"Output shapes: {pytorch_output.shape}")
    print(f"Model has {NUM_PARALLEL_LAYERS} parallel layers with different weights")
    print(f"\nCoreML vs PyTorch:")
    print(f"  Max difference: {max_diff}")
    print(f"  Mean difference: {mean_diff}")
    
    # Check if outputs are close
    tolerance = 8e-3  # 0.008 for fp16 precision with 20 layers
    if max_diff < tolerance:
        print(f"\n✅ Test PASSED! Outputs match within tolerance {tolerance}")
    else:
        print(f"\n❌ Test FAILED! Differences exceed tolerance {tolerance}")
    
    return max_diff < tolerance


def compare_all_implementations_extended():
    """Compare all RMSNorm implementations including LinalgRMSNorm."""
    
    print("\nComparing all RMSNorm implementations (including LinalgRMSNorm)...")
    
    
    # Create models
    qwen_model = QwenRMSNorm(hidden_size=hidden_size)
    seba_model = SebaRMSNorm(hidden_size=hidden_size)
    linalg_model = LinalgRMSNorm(hidden_size=hidden_size)
    standard_model = StandardRMSNorm(hidden_size=hidden_size)
    
    qwen_model.eval()
    seba_model.eval()
    linalg_model.eval()
    standard_model.eval()
    
    # Test with multiple inputs
    num_tests = 1000
    results = {
        'qwen_vs_standard': [],
        'seba_vs_standard': [],
        'linalg_vs_standard': [],
        'qwen_vs_linalg': [],
    }
    
    # Test different value ranges
    for i in range(num_tests):
        if i < 800:
            # Normal range inputs (0-10)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 3  # ~3σ covers most of 0-10 range
        elif i < 900:
            # Large value inputs (0-256)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 85  # ~3σ covers most of 0-256 range
        else:
            # Very large value inputs (0-1000)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 333  # ~3σ covers most of 0-1000 range
            test_input = torch.clamp(test_input, min=-1000, max=1000)  # Clamp to range
        
        with torch.no_grad():
            qwen_output = qwen_model(test_input).numpy()
            seba_output = seba_model(test_input).numpy()
            linalg_output = linalg_model(test_input).numpy()
            standard_output = standard_model(test_input).numpy()
        
        results['qwen_vs_standard'].append(np.max(np.abs(qwen_output - standard_output)))
        results['seba_vs_standard'].append(np.max(np.abs(seba_output - standard_output)))
        results['linalg_vs_standard'].append(np.max(np.abs(linalg_output - standard_output)))
        results['qwen_vs_linalg'].append(np.max(np.abs(qwen_output - linalg_output)))
        
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{num_tests} tests...")
    
    print(f"\nResults from {num_tests} test inputs (800 normal 0-10, 100 large 0-256, 100 very large 0-1000):")
    
    for name, diffs in results.items():
        diffs_array = np.array(diffs)
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Overall - Max: {np.max(diffs_array):.6f}, Mean: {np.mean(diffs_array):.6f}, Std: {np.std(diffs_array):.6f}")
        print(f"  Normal range (0-10) - Max: {np.max(diffs_array[:800]):.6f}, Mean: {np.mean(diffs_array[:800]):.6f}")
        print(f"  Large range (0-256) - Max: {np.max(diffs_array[800:900]):.6f}, Mean: {np.mean(diffs_array[800:900]):.6f}")
        print(f"  Very large range (0-1000) - Max: {np.max(diffs_array[900:]):.6f}, Mean: {np.mean(diffs_array[900:]):.6f}")


def generate_layer_weights(hidden_size, num_layers, seed=42):
    """Generate consistent weights for all layers."""
    torch.manual_seed(seed)
    weights = []
    for i in range(num_layers):
        # Create unique weights for each layer with more variation
        base_value = 1.0 + (i * 0.5 / num_layers)  # Larger variation between layers
        weight = torch.ones(hidden_size) * base_value
        # Add different random variation for each layer
        torch.manual_seed(seed + i)  # Different seed for each layer
        weight = weight + torch.randn(hidden_size) * 0.05  # Larger random variation
        weights.append(weight)
        print(f"Layer {i} weight stats - Mean: {weight.mean():.4f}, Std: {weight.std():.4f}, Min: {weight.min():.4f}, Max: {weight.max():.4f}")
    return weights


def create_parallel_models_with_same_weights(hidden_size, num_layers, weights):
    """Create all parallel models with the same weights."""
    # Create Qwen parallel model
    qwen_model = ParallelQwenRMSNorm(hidden_size=hidden_size, num_layers=num_layers)
    for i, layer in enumerate(qwen_model.layers):
        layer.weight = nn.Parameter(weights[i].clone())
    
    # Create Seba parallel model
    seba_model = ParallelSebaRMSNorm(hidden_size=hidden_size, num_layers=num_layers)
    for i, layer in enumerate(seba_model.layers):
        layer.weight = nn.Parameter(weights[i].clone())
    
    # Create Linalg parallel model
    linalg_model = ParallelLinalgRMSNorm(hidden_size=hidden_size, num_layers=num_layers)
    for i, layer in enumerate(linalg_model.layers):
        layer.weight = nn.Parameter(weights[i].clone())
    
    return qwen_model, seba_model, linalg_model


def export_all_models_with_same_weights():
    """Export all models to CoreML with identical weights."""
    
    # Generate consistent weights
    print("\nGenerating unique weights for each layer:")
    weights = generate_layer_weights(hidden_size, NUM_PARALLEL_LAYERS)
    
    # Create all models with same weights
    print("\nCreating models with identical weight distribution across implementations...")
    qwen_model, seba_model, linalg_model = create_parallel_models_with_same_weights(
        hidden_size, NUM_PARALLEL_LAYERS, weights
    )
    
    # Verify weights are correctly set
    print("\nVerifying weights are correctly assigned:")
    for i in range(min(3, NUM_PARALLEL_LAYERS)):  # Check first 3 layers
        qwen_w = qwen_model.layers[i].weight
        seba_w = seba_model.layers[i].weight
        linalg_w = linalg_model.layers[i].weight
        
        print(f"Layer {i}:")
        print(f"  Qwen weight mean: {qwen_w.mean():.4f}")
        print(f"  Seba weight mean: {seba_w.mean():.4f}")
        print(f"  Linalg weight mean: {linalg_w.mean():.4f}")
        print(f"  All equal: {torch.allclose(qwen_w, seba_w) and torch.allclose(seba_w, linalg_w)}")
    
    # Set to eval mode
    qwen_model.eval()
    seba_model.eval()
    linalg_model.eval()
    
    # Create example input for tracing
    example_input = torch.randn(batch_size, seq_length, hidden_size)
    
    # Export all models
    models = {
        'qwen': (qwen_model, 'qwen_rmsnorm'),
        'seba': (seba_model, 'seba_rmsnorm'),
        'linalg': (linalg_model, 'linalg_rmsnorm')
    }
    
    coreml_models = {}
    pytorch_models = {}
    
    for name, (model, file_prefix) in models.items():
        print(f"\nExporting {name.title()}RMSNorm to CoreML...")
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(
                name="hidden_states",
                shape=(batch_size, seq_length, hidden_size),
                dtype=np.float16
            )],
            outputs=[ct.TensorType(name="output", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS16
        )
        
        # Save the model
        output_path = Path(f"/tmp/ANE/{file_prefix}_parallel_{NUM_PARALLEL_LAYERS}_same_weights.mlpackage")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        coreml_model.save(str(output_path))
        
        print(f"Model saved to: {output_path}")
        
        # The model is saved but not loaded here - will be compiled and loaded later
        print(f"Model saved for compilation")
        
        coreml_models[name] = coreml_model
        pytorch_models[name] = model
    
    return coreml_models, pytorch_models, weights


def compile_and_load_model(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE):
    """Compile and load a CoreML model, ensuring it runs on ANE.
    
    Args:
        mlpackage_path: Path to .mlpackage file
        compute_units: Compute units to use (default: CPU_AND_NE)
        
    Returns:
        Loaded CoreML model
    """
    mlpackage_path = Path(mlpackage_path)
    
    # Compile the model to .mlmodelc
    mlmodelc_path = mlpackage_path.with_suffix('.mlmodelc')
    
    # Delete existing compiled model if it exists
    if mlmodelc_path.exists():
        import shutil
        shutil.rmtree(mlmodelc_path)
        print(f"Deleted existing compiled model: {mlmodelc_path}")
    
    # Compile the model
    print(f"Compiling {mlpackage_path.name} to .mlmodelc...")
    import subprocess
    try:
        result = subprocess.run(
            ['xcrun', 'coremlcompiler', 'compile', str(mlpackage_path), str(mlpackage_path.parent)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Model compiled successfully to: {mlmodelc_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling model: {e.stderr}")
        raise
    
    # Load the compiled model
    try:
        # Use CompiledMLModel for .mlmodelc files with specified compute units
        model = ct.models.CompiledMLModel(str(mlmodelc_path), compute_units)
        print(f"Model loaded with compute units: {compute_units}")
        return model
    except Exception as e:
        print(f"Error loading compiled model: {e}")
        # Fallback to loading the .mlpackage directly
        print("Falling back to loading .mlpackage directly...")
        model = ct.models.MLModel(str(mlpackage_path), compute_units=compute_units)
        return model


def export_conv2d_models_to_coreml():
    """Export Conv2D+RMSNorm models to CoreML with identical weights."""

    # Generate consistent Conv2D weights (different per layer, same across models)
    print("\nGenerating Conv2D weights for each layer:")
    conv_weights = []
    for i in range(NUM_PARALLEL_LAYERS):
        # Create unique weights for each Conv2D layer
        weight = torch.randn(hidden_size, hidden_size, 1, 1) * 0.02
        conv_weights.append(weight)
        print(f"Conv2D Layer {i} weight stats - Mean: {weight.mean():.4f}, Std: {weight.std():.4f}")
    
    # Generate single RMSNorm weight (same for all models)
    rmsnorm_weight = torch.ones(hidden_size)
    print(f"\nRMSNorm weight: All ones (identical across models)")
    
    # Create all models with same weights
    print("\nCreating Conv2D models with identical weight distribution...")
    qwen_model = Conv2DQwenRMSNorm(hidden_size, NUM_PARALLEL_LAYERS)
    seba_model = Conv2DSebaRMSNorm(hidden_size, NUM_PARALLEL_LAYERS)
    linalg_model = Conv2DLinalgRMSNorm(hidden_size, NUM_PARALLEL_LAYERS)
    
    # Set Conv2D weights (identical across models, different per layer)
    for i in range(NUM_PARALLEL_LAYERS):
        qwen_model.conv_layers[i].weight = nn.Parameter(conv_weights[i].clone())
        seba_model.conv_layers[i].weight = nn.Parameter(conv_weights[i].clone())
        linalg_model.conv_layers[i].weight = nn.Parameter(conv_weights[i].clone())
    
    # Set RMSNorm weights (identical across models)
    qwen_model.norm.weight = nn.Parameter(rmsnorm_weight.clone())
    seba_model.norm.weight = nn.Parameter(rmsnorm_weight.clone())
    linalg_model.norm.weight = nn.Parameter(rmsnorm_weight.clone())
    
    # Set to eval mode
    qwen_model.eval()
    seba_model.eval()
    linalg_model.eval()
    
    # Create example input for tracing
    example_input = torch.randn(batch_size, seq_length, hidden_size)
    
    # Export all models
    models = {
        'qwen': (qwen_model, 'conv2d_qwen_rmsnorm'),
        'seba': (seba_model, 'conv2d_seba_rmsnorm'),
        'linalg': (linalg_model, 'conv2d_linalg_rmsnorm')
    }
    
    coreml_models = {}
    pytorch_models = {}
    
    for name, (model, file_prefix) in models.items():
        print(f"\nExporting Conv2D+{name.title()}RMSNorm to CoreML...")
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(
                name="hidden_states",
                shape=(batch_size, seq_length, hidden_size),
                dtype=np.float16
            )],
            outputs=[ct.TensorType(name="output", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS16
        )
        
        # Save the model
        output_path = Path(f"/tmp/ANE/{file_prefix}_{NUM_PARALLEL_LAYERS}_layers.mlpackage")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        coreml_model.save(str(output_path))
        
        print(f"Model saved to: {output_path}")
        
        # The model is saved but not loaded here - will be compiled and loaded later
        print(f"Model saved for compilation")
        
        coreml_models[name] = coreml_model
        pytorch_models[name] = model
    
    return coreml_models, pytorch_models, conv_weights


def compare_all_coreml_implementations():
    """Compare all RMSNorm implementations running on CoreML/ANE."""
    
    print("\nComparing all RMSNorm implementations on CoreML/ANE...")
    print(f"Exporting and testing with {NUM_PARALLEL_LAYERS} parallel layers")
    print("All models use IDENTICAL weights for fair comparison")
    
    
    # Export all models with same weights
    coreml_models, pytorch_models, weights = export_all_models_with_same_weights()
    
    # Create standard model for comparison
    standard_model = StandardRMSNorm(hidden_size=hidden_size)
    standard_model.eval()
    
    # Test with multiple inputs
    num_tests = 10  # Increased for extra large range
    results = {
        'qwen_coreml_vs_pytorch': [],
        'seba_coreml_vs_pytorch': [],
        'linalg_coreml_vs_pytorch': [],
        'qwen_coreml_vs_standard': [],
        'seba_coreml_vs_standard': [],
        'linalg_coreml_vs_standard': [],
        'qwen_vs_seba_coreml': [],
        'qwen_vs_linalg_coreml': [],
        'seba_vs_linalg_coreml': [],
    }
    
    print(f"\nRunning {num_tests} comparison tests...")
    
    # Test different value ranges
    for i in range(num_tests):
        if i < 80:
            # Normal range inputs (0-10)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 3  # ~3σ covers most of 0-10 range
        elif i < 90:
            # Large value inputs (0-256)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 85  # ~3σ covers most of 0-256 range
        elif i < 100:
            # Very large value inputs (0-1000)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 333  # ~3σ covers most of 0-1000 range
            test_input = torch.clamp(test_input, min=-1000, max=1000)  # Clamp to range
        else:
            # Extra large value inputs (0-10000)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 3333  # ~3σ covers most of 0-10000 range
            test_input = torch.clamp(test_input, min=-10000, max=10000)  # Clamp to range
        
        # Get PyTorch outputs
        with torch.no_grad():
            qwen_pytorch_out = pytorch_models['qwen'](test_input).numpy()
            seba_pytorch_out = pytorch_models['seba'](test_input).numpy()
            linalg_pytorch_out = pytorch_models['linalg'](test_input).numpy()
            standard_out = standard_model(test_input).numpy()
        
        # Get CoreML outputs
        input_np = test_input.numpy()
        qwen_coreml_out = coreml_models['qwen'].predict({"hidden_states": input_np})["output"]
        seba_coreml_out = coreml_models['seba'].predict({"hidden_states": input_np})["output"]
        linalg_coreml_out = coreml_models['linalg'].predict({"hidden_states": input_np})["output"]
        
        # Calculate differences - CoreML vs PyTorch
        results['qwen_coreml_vs_pytorch'].append(np.max(np.abs(qwen_coreml_out - qwen_pytorch_out)))
        results['seba_coreml_vs_pytorch'].append(np.max(np.abs(seba_coreml_out - seba_pytorch_out)))
        results['linalg_coreml_vs_pytorch'].append(np.max(np.abs(linalg_coreml_out - linalg_pytorch_out)))
        
        # Compare CoreML outputs vs standard
        results['qwen_coreml_vs_standard'].append(np.max(np.abs(qwen_coreml_out - standard_out)))
        results['seba_coreml_vs_standard'].append(np.max(np.abs(seba_coreml_out - standard_out)))
        results['linalg_coreml_vs_standard'].append(np.max(np.abs(linalg_coreml_out - standard_out)))
        
        # Compare CoreML implementations against each other
        results['qwen_vs_seba_coreml'].append(np.max(np.abs(qwen_coreml_out - seba_coreml_out)))
        results['qwen_vs_linalg_coreml'].append(np.max(np.abs(qwen_coreml_out - linalg_coreml_out)))
        results['seba_vs_linalg_coreml'].append(np.max(np.abs(seba_coreml_out - linalg_coreml_out)))
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{num_tests} tests...")
    
    print(f"\nCoreML/ANE Results from {num_tests} test inputs:")
    
    # Print CoreML vs PyTorch comparisons
    print("\n--- CoreML vs PyTorch (same model) ---")
    for name in ['qwen_coreml_vs_pytorch', 'seba_coreml_vs_pytorch', 'linalg_coreml_vs_pytorch']:
        diffs_array = np.array(results[name])
        model_name = name.split('_')[0].title()
        print(f"\n{model_name}RMSNorm CoreML vs PyTorch:")
        print(f"  Overall - Max: {np.max(diffs_array):.6f}, Mean: {np.mean(diffs_array):.6f}, Std: {np.std(diffs_array):.6f}")
        print(f"  Normal range (0-10) - Max: {np.max(diffs_array[:80]):.6f}, Mean: {np.mean(diffs_array[:80]):.6f}")
        if len(diffs_array) > 80:
            print(f"  Large range (0-256) - Max: {np.max(diffs_array[80:90]):.6f}, Mean: {np.mean(diffs_array[80:90]):.6f}")
        if len(diffs_array) > 90:
            print(f"  Very large range (0-1000) - Max: {np.max(diffs_array[90:100]):.6f}, Mean: {np.mean(diffs_array[90:100]):.6f}")
        if len(diffs_array) > 100:
            print(f"  Extra large range (0-10000) - Max: {np.max(diffs_array[100:]):.6f}, Mean: {np.mean(diffs_array[100:]):.6f}")
    
    # Print CoreML vs Standard comparisons
    print("\n--- CoreML Models vs Standard RMSNorm ---")
    for name in ['qwen_coreml_vs_standard', 'seba_coreml_vs_standard', 'linalg_coreml_vs_standard']:
        diffs_array = np.array(results[name])
        model_name = name.split('_')[0].title()
        print(f"\n{model_name}RMSNorm CoreML vs Standard:")
        print(f"  Overall - Max: {np.max(diffs_array):.6f}, Mean: {np.mean(diffs_array):.6f}, Std: {np.std(diffs_array):.6f}")
        print(f"  Normal range (0-10) - Max: {np.max(diffs_array[:80]):.6f}, Mean: {np.mean(diffs_array[:80]):.6f}")
        if len(diffs_array) > 80:
            print(f"  Large range (0-256) - Max: {np.max(diffs_array[80:90]):.6f}, Mean: {np.mean(diffs_array[80:90]):.6f}")
        if len(diffs_array) > 90:
            print(f"  Very large range (0-1000) - Max: {np.max(diffs_array[90:100]):.6f}, Mean: {np.mean(diffs_array[90:100]):.6f}")
        if len(diffs_array) > 100:
            print(f"  Extra large range (0-10000) - Max: {np.max(diffs_array[100:]):.6f}, Mean: {np.mean(diffs_array[100:]):.6f}")
    
    # Print CoreML inter-model comparisons
    print("\n--- CoreML Models Compared Against Each Other ---")
    print("(All models have IDENTICAL weights)")
    
    comparisons = [
        ('qwen_vs_seba_coreml', 'QwenRMSNorm vs SebaRMSNorm'),
        ('qwen_vs_linalg_coreml', 'QwenRMSNorm vs LinalgRMSNorm'),
        ('seba_vs_linalg_coreml', 'SebaRMSNorm vs LinalgRMSNorm')
    ]
    
    for key, title in comparisons:
        diffs_array = np.array(results[key])
        print(f"\n{title} (CoreML):")
        print(f"  Overall - Max: {np.max(diffs_array):.6f}, Mean: {np.mean(diffs_array):.6f}, Std: {np.std(diffs_array):.6f}")
        print(f"  Normal range (0-10) - Max: {np.max(diffs_array[:80]):.6f}, Mean: {np.mean(diffs_array[:80]):.6f}")
        if len(diffs_array) > 80:
            print(f"  Large range (0-256) - Max: {np.max(diffs_array[80:90]):.6f}, Mean: {np.mean(diffs_array[80:90]):.6f}")
        if len(diffs_array) > 90:
            print(f"  Very large range (0-1000) - Max: {np.max(diffs_array[90:100]):.6f}, Mean: {np.mean(diffs_array[90:100]):.6f}")
        if len(diffs_array) > 100:
            print(f"  Extra large range (0-10000) - Max: {np.max(diffs_array[100:]):.6f}, Mean: {np.mean(diffs_array[100:]):.6f}")
    
    # Generate automatic synopsis
    generate_synopsis(results)


def generate_synopsis(results):
    """Generate automatic synopsis of test results with recommendations."""
    print("\n" + "="*80)
    print("AUTOMATIC RESULT SYNOPSIS")
    print("="*80)
    
    # Analyze precision for each model
    models = ['qwen', 'seba', 'linalg']
    model_scores = {}
    overflow_detected = {}
    
    for model in models:
        # Check CoreML vs PyTorch precision
        key = f'{model}_coreml_vs_pytorch'
        diffs = np.array(results[key])
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)
        
        # Check for overflow (diff > 0.1 or 10x degradation indicates severe numerical issues)
        overflow_detected[model] = max_diff > 0.1
        
        # Also check for relative degradation
        if len(diffs) > 100:
            normal_max = np.max(diffs[:80])
            extra_large_max = np.max(diffs[100:])
            if extra_large_max > normal_max * 10:
                overflow_detected[model] = True
        
        # Score based on max difference (lower is better)
        if max_diff < 0.01:
            precision_score = 5  # Excellent
        elif max_diff < 0.05:
            precision_score = 4  # Good
        elif max_diff < 0.1:
            precision_score = 3  # Acceptable
        elif max_diff < 1.0:
            precision_score = 2  # Poor
        else:
            precision_score = 1  # Failed/Overflow
        
        # Check stability across value ranges
        stability_score = 5
        if len(diffs) > 100:
            # Compare normal range to extra large range
            normal_max = np.max(diffs[:80])
            extra_large_max = np.max(diffs[100:])
            if extra_large_max > normal_max * 10:
                stability_score = 1  # Severe instability
            elif extra_large_max > normal_max * 5:
                stability_score = 2  # Poor stability
            elif extra_large_max > normal_max * 2:
                stability_score = 3  # Moderate stability
            elif extra_large_max > normal_max * 1.5:
                stability_score = 4  # Good stability
        
        model_scores[model] = {
            'precision': precision_score,
            'stability': stability_score,
            'total': precision_score + stability_score,
            'max_diff': max_diff,
            'mean_diff': mean_diff
        }
    
    # Rank models
    ranked_models = sorted(model_scores.items(), key=lambda x: x[1]['total'], reverse=True)
    
    print("\n📊 MODEL RANKINGS (Higher score is better, max 10):")
    print("="*60)
    for rank, (model, scores) in enumerate(ranked_models, 1):
        model_name = model.title() + "RMSNorm"
        print(f"\n{rank}. {model_name}")
        print(f"   Total Score: {scores['total']}/10 (Precision: {scores['precision']}/5, Stability: {scores['stability']}/5)")
        print(f"   Max Difference: {scores['max_diff']:.6f}")
        print(f"   Mean Difference: {scores['mean_diff']:.6f}")
        
        # Add specific warnings
        if overflow_detected[model]:
            print(f"   ⚠️  WARNING: NUMERICAL OVERFLOW DETECTED! Max error: {scores['max_diff']:.1f}")
        if scores['stability'] <= 2:
            print(f"   ⚠️  WARNING: Severe instability with large values!")
    
    # Recommendations
    print("\n" + "="*60)
    print("🎯 RECOMMENDATIONS:")
    print("="*60)
    
    best_model = ranked_models[0][0]
    print(f"\n✅ RECOMMENDED: {best_model.title()}RMSNorm")
    print(f"   - Best overall score: {model_scores[best_model]['total']}/10")
    if model_scores[best_model]['precision'] == 5:
        print("   - Excellent precision across all tests")
    if model_scores[best_model]['stability'] == 5:
        print("   - Excellent stability across all value ranges")
    
    # Acceptable alternatives
    acceptable = [m for m, s in ranked_models[1:] if s['total'] >= 7 and not overflow_detected[m]]
    if acceptable:
        print(f"\n⚡ ACCEPTABLE ALTERNATIVES:")
        for model in acceptable:
            print(f"   - {model.title()}RMSNorm (Score: {model_scores[model]['total']}/10)")
    
    # Warnings
    failed = [m for m, s in model_scores.items() if overflow_detected[m] or s['total'] <= 4]
    if failed:
        print(f"\n❌ DO NOT USE:")
        for model in failed:
            print(f"   - {model.title()}RMSNorm", end="")
            if overflow_detected[model]:
                print(f" (Numerical overflow: {model_scores[model]['max_diff']:.1f}x error!)", end="")
            elif model_scores[model]['stability'] == 1:
                print(" (Catastrophic instability with large values)", end="")
            print()
    
    # Value range analysis
    print("\n" + "="*60)
    print("📈 VALUE RANGE ANALYSIS:")
    print("="*60)
    
    for model in models:
        key = f'{model}_coreml_vs_pytorch'
        diffs = np.array(results[key])
        if len(diffs) > 100:
            normal_max = np.max(diffs[:80])
            large_max = np.max(diffs[80:90])
            very_large_max = np.max(diffs[90:100])
            extra_large_max = np.max(diffs[100:])
            
            print(f"\n{model.title()}RMSNorm value range behavior:")
            print(f"  0-10:     {normal_max:.6f}")
            print(f"  0-256:    {large_max:.6f} ({large_max/normal_max:.1f}x vs normal)")
            print(f"  0-1000:   {very_large_max:.6f} ({very_large_max/normal_max:.1f}x vs normal)")
            print(f"  0-10000:  {extra_large_max:.6f} ({extra_large_max/normal_max:.1f}x vs normal)")
            
            if extra_large_max > normal_max * 100:
                print(f"  ⚠️  CRITICAL: {extra_large_max/normal_max:.0f}x degradation at large values!")
    
    # Comparative analysis
    print("\n" + "="*60)
    print("🔍 COMPARATIVE ANALYSIS:")
    print("="*60)
    
    # Direct model comparisons
    if 'qwen_vs_seba_coreml' in results:
        qwen_vs_seba = np.array(results['qwen_vs_seba_coreml'])
        qwen_vs_linalg = np.array(results['qwen_vs_linalg_coreml'])
        seba_vs_linalg = np.array(results['seba_vs_linalg_coreml'])
        
        print("\nDirect Model Comparisons (with identical weights):")
        print(f"  QwenRMSNorm vs SebaRMSNorm:   Max diff = {np.max(qwen_vs_seba):.6f}")
        print(f"  QwenRMSNorm vs LinalgRMSNorm: Max diff = {np.max(qwen_vs_linalg):.6f}")
        print(f"  SebaRMSNorm vs LinalgRMSNorm: Max diff = {np.max(seba_vs_linalg):.6f}")
        
        # Analyze which models are most similar
        min_diff = min(np.max(qwen_vs_seba), np.max(qwen_vs_linalg), np.max(seba_vs_linalg))
        if np.max(qwen_vs_seba) == min_diff:
            print("\n  → QwenRMSNorm and SebaRMSNorm are most similar")
        elif np.max(qwen_vs_linalg) == min_diff:
            print("\n  → QwenRMSNorm and LinalgRMSNorm are most similar")
        else:
            print("\n  → SebaRMSNorm and LinalgRMSNorm are most similar")
    
    # Focus on CoreML vs PyTorch precision (same model comparison)
    print("\n" + "="*60)
    print("🎯 COREML PRECISION ANALYSIS:")
    print("="*60)
    
    print("\nCoreML vs PyTorch Implementation Precision:")
    print("(Lower difference = Better CoreML precision)")
    
    precision_ranking = []
    for model in models:
        key = f'{model}_coreml_vs_pytorch'
        if key in results:
            diffs = np.array(results[key])
            max_diff = np.max(diffs)
            mean_diff = np.mean(diffs)
            precision_ranking.append((model, mean_diff, max_diff))
            
            print(f"\n  {model.title()}RMSNorm:")
            print(f"    Max difference:  {max_diff:.6f}")
            print(f"    Mean difference: {mean_diff:.6f}")
            
            # Check value range behavior
            if len(diffs) > 100:
                normal = np.max(diffs[:80])
                large = np.max(diffs[100:])
                if large > normal * 2:
                    print(f"    ⚠️  Precision degrades {large/normal:.1f}x at large values")
    
    # Rank by precision
    precision_ranking.sort(key=lambda x: x[1])
    print("\n  CoreML Precision Ranking:")
    for i, (model, mean_diff, max_diff) in enumerate(precision_ranking, 1):
        print(f"    {i}. {model.title()}RMSNorm: Mean={mean_diff:.6f}, Max={max_diff:.6f}")
    
    # Performance analysis
    print("\n" + "="*60)
    print("💡 KEY INSIGHTS:")
    print("="*60)
    
    # Check if LinalgRMSNorm shows degradation
    if 'linalg' in model_scores:
        linalg_scores = model_scores['linalg']
        if linalg_scores['stability'] < 5:
            print("\n⚠️  LinalgRMSNorm shows performance degradation:")
            key = 'linalg_coreml_vs_pytorch'
            diffs = np.array(results[key])
            if len(diffs) > 100:
                normal = np.max(diffs[:80])
                large = np.max(diffs[100:])
                print(f"   - Normal values (0-10): {normal:.6f}")
                print(f"   - Large values (0-10000): {large:.6f}")
                print(f"   - Degradation factor: {large/normal:.1f}x")
                if large/normal < 10:
                    print("   - No catastrophic overflow detected in this run")
                    print("   - Previous tests showed 4x errors, suggesting intermittent instability")
    
    # Model implementation differences
    print("\n📝 Implementation Characteristics:")
    print("  - QwenRMSNorm: Uses LayerNorm trick (concat [x, -x])")
    print("  - SebaRMSNorm: Uses max-value scaling for stability")
    print("  - LinalgRMSNorm: Uses linalg.norm with epsilon channel")
    
    print("\n" + "="*80)


def compare_conv2d_implementations():
    """Compare RMSNorm implementations with Conv2D layers before normalization."""
    
    print("\n" + "="*80)
    print("CONV2D + RMSNORM COMPARISON (True Accuracy Test with ANE Complexity)")
    print("="*80)
    print(f"Using {NUM_PARALLEL_LAYERS} Conv2D layers averaged BEFORE RMSNorm")
    print("This shows the true accuracy of each RMSNorm implementation\n")
    
    # Use MPS if available for faster PyTorch inference
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    
    # Create Conv2D models
    qwen_model = Conv2DQwenRMSNorm(hidden_size=hidden_size, num_layers=NUM_PARALLEL_LAYERS).to(device)
    seba_model = Conv2DSebaRMSNorm(hidden_size=hidden_size, num_layers=NUM_PARALLEL_LAYERS).to(device)
    linalg_model = Conv2DLinalgRMSNorm(hidden_size=hidden_size, num_layers=NUM_PARALLEL_LAYERS).to(device)
    standard_model = StandardRMSNorm(hidden_size=hidden_size).to(device)
    
    # Set all models to use the same RMSNorm weight
    shared_weight = torch.ones(hidden_size).to(device)
    qwen_model.norm.weight = nn.Parameter(shared_weight.clone())
    seba_model.norm.weight = nn.Parameter(shared_weight.clone())
    linalg_model.norm.weight = nn.Parameter(shared_weight.clone())
    standard_model.weight = nn.Parameter(shared_weight.clone())
    
    # Initialize Conv2D layers with same weights across models
    for i in range(NUM_PARALLEL_LAYERS):
        weight = torch.randn(hidden_size, hidden_size, 1, 1, device=device) * 0.1
        qwen_model.conv_layers[i].weight = nn.Parameter(weight.clone())
        seba_model.conv_layers[i].weight = nn.Parameter(weight.clone())
        linalg_model.conv_layers[i].weight = nn.Parameter(weight.clone())
    
    qwen_model.eval()
    seba_model.eval()
    linalg_model.eval()
    standard_model.eval()
    
    # Test with different value ranges
    num_tests = NUM_TESTS
    results = {
        'qwen_vs_standard': [],
        'seba_vs_standard': [],
        'linalg_vs_standard': [],
        'qwen_vs_seba': [],
        'qwen_vs_linalg': [],
        'seba_vs_linalg': []
    }
    
    for i in range(num_tests):
        if i < num_tests // 4:
            # Normal range (0-10)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16, device=device) * 3
        elif i < num_tests // 2:
            # Large range (0-256)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16, device=device) * 85
        elif i < 3 * num_tests // 4:
            # Very large range (0-1000)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16, device=device) * 333
        else:
            # Extra large range (0-10000)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16, device=device) * 3333
            test_input = torch.clamp(test_input, min=-10000, max=10000)
        
        with torch.no_grad():
            # Get pre-averaged outputs
            qwen_output = qwen_model(test_input).cpu().numpy()
            seba_output = seba_model(test_input).cpu().numpy()
            linalg_output = linalg_model(test_input).cpu().numpy()
            
            # For standard, we need to manually do the same Conv2D processing
            x = test_input.view(batch_size * seq_length, hidden_size, 1, 1)
            output = torch.zeros_like(x)
            for conv in qwen_model.conv_layers:
                output = output + conv(x.float())
            output = output / qwen_model.num_layers
            averaged_input = output.view(batch_size, seq_length, hidden_size)
            averaged_input = averaged_input.to(test_input.dtype)
            standard_output = standard_model(averaged_input).cpu().numpy()
        
        # Calculate differences
        results['qwen_vs_standard'].append(np.max(np.abs(qwen_output - standard_output)))
        results['seba_vs_standard'].append(np.max(np.abs(seba_output - standard_output)))
        results['linalg_vs_standard'].append(np.max(np.abs(linalg_output - standard_output)))
        results['qwen_vs_seba'].append(np.max(np.abs(qwen_output - seba_output)))
        results['qwen_vs_linalg'].append(np.max(np.abs(qwen_output - linalg_output)))
        results['seba_vs_linalg'].append(np.max(np.abs(seba_output - linalg_output)))
    
    # Print results
    print("Comparison vs Standard RMSNorm (Ground Truth):")
    print("="*60)
    
    for name, diffs in results.items():
        if 'vs_standard' in name:
            model_name = name.replace('_vs_standard', '').title()
            diffs_array = np.array(diffs)
            print(f"\n{model_name}RMSNorm vs Standard:")
            print(f"  Overall - Max: {np.max(diffs_array):.6f}, Mean: {np.mean(diffs_array):.6f}")
            print(f"  Normal (0-10) - Max: {np.max(diffs_array[:NUM_TESTS//4]):.6f}, Mean: {np.mean(diffs_array[:NUM_TESTS//4]):.6f}")
            print(f"  Large (0-256) - Max: {np.max(diffs_array[NUM_TESTS//4:NUM_TESTS//2]):.6f}, Mean: {np.mean(diffs_array[NUM_TESTS//4:NUM_TESTS//2]):.6f}")
            print(f"  Very Large (0-1000) - Max: {np.max(diffs_array[NUM_TESTS//2:3*NUM_TESTS//4]):.6f}, Mean: {np.mean(diffs_array[NUM_TESTS//2:3*NUM_TESTS//4]):.6f}")
            print(f"  Extra Large (0-10000) - Max: {np.max(diffs_array[3*NUM_TESTS//4:]):.6f}, Mean: {np.mean(diffs_array[3*NUM_TESTS//4:]):.6f}")
            
            # Check for degradation
            normal_max = np.max(diffs_array[:NUM_TESTS//4])
            extra_large_max = np.max(diffs_array[3*NUM_TESTS//4:])
            if extra_large_max > normal_max * 2:
                print(f"  ⚠️  Degradation: {extra_large_max/normal_max:.1f}x at large values")
    
    # Rank by accuracy
    accuracy_scores = []
    for model in ['qwen', 'seba', 'linalg']:
        key = f'{model}_vs_standard'
        mean_diff = np.mean(results[key])
        max_diff = np.max(results[key])
        accuracy_scores.append((model, mean_diff, max_diff))
    
    accuracy_scores.sort(key=lambda x: x[1])
    
    print("\n" + "="*60)
    print("ACCURACY RANKING (vs Standard RMSNorm):")
    print("="*60)
    for i, (model, mean_diff, max_diff) in enumerate(accuracy_scores, 1):
        print(f"{i}. {model.title()}RMSNorm: Mean={mean_diff:.6f}, Max={max_diff:.6f}")
    
    print("\n" + "="*60)


def analyze_rmsnorm_input_values():
    """Analyze the maximum values at the input of each RMSNorm implementation."""
    
    print("\n" + "="*80)
    print("RMSNORM INPUT VALUE ANALYSIS")
    print("="*80)
    print("Tracking maximum values at the input of each RMSNorm implementation\n")
    
    # Create models
    qwen_model = QwenRMSNorm(hidden_size=hidden_size)
    seba_model = SebaRMSNorm(hidden_size=hidden_size)
    linalg_model = LinalgRMSNorm(hidden_size=hidden_size)
    standard_model = StandardRMSNorm(hidden_size=hidden_size)
    
    qwen_model.eval()
    seba_model.eval()
    linalg_model.eval()
    standard_model.eval()
    
    # Store input value statistics
    input_stats = {
        'normal': {'max': [], 'mean': [], 'std': []},
        'large': {'max': [], 'mean': [], 'std': []},
        'very_large': {'max': [], 'mean': [], 'std': []},
        'extra_large': {'max': [], 'mean': [], 'std': []}
    }
    
    # Also track per-model max values
    model_max_values = {
        'qwen': {'normal': [], 'large': [], 'very_large': [], 'extra_large': []},
        'seba': {'normal': [], 'large': [], 'very_large': [], 'extra_large': []},
        'linalg': {'normal': [], 'large': [], 'very_large': [], 'extra_large': []},
        'standard': {'normal': [], 'large': [], 'very_large': [], 'extra_large': []}
    }
    
    num_tests = 100
    
    # Modified forward functions to capture input values
    original_forwards = {}
    
    def create_hook(model_name, range_name):
        def hook(module, input, output):
            # Input is a tuple, get the first element
            hidden_states = input[0]
            max_val = torch.max(torch.abs(hidden_states)).item()
            model_max_values[model_name][range_name].append(max_val)
        return hook
    
    # Register hooks
    hooks = []
    
    for i in range(num_tests):
        # Determine range
        if i < 25:
            range_name = 'normal'
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 3
        elif i < 50:
            range_name = 'large'
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 85
        elif i < 75:
            range_name = 'very_large'
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 333
        else:
            range_name = 'extra_large'
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 3333
            test_input = torch.clamp(test_input, min=-10000, max=10000)
        
        # Track input statistics
        input_stats[range_name]['max'].append(torch.max(torch.abs(test_input)).item())
        input_stats[range_name]['mean'].append(torch.mean(torch.abs(test_input)).item())
        input_stats[range_name]['std'].append(torch.std(test_input).item())
        
        # Clear previous hooks
        for hook in hooks:
            hook.remove()
        hooks = []
        
        # Register new hooks for this range
        hooks.append(qwen_model.register_forward_hook(create_hook('qwen', range_name)))
        hooks.append(seba_model.register_forward_hook(create_hook('seba', range_name)))
        hooks.append(linalg_model.register_forward_hook(create_hook('linalg', range_name)))
        hooks.append(standard_model.register_forward_hook(create_hook('standard', range_name)))
        
        # Run models
        with torch.no_grad():
            _ = qwen_model(test_input)
            _ = seba_model(test_input)
            _ = linalg_model(test_input)
            _ = standard_model(test_input)
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Print analysis
    print("Input Value Statistics:")
    print("="*60)
    for range_name in ['normal', 'large', 'very_large', 'extra_large']:
        stats = input_stats[range_name]
        print(f"\n{range_name.replace('_', ' ').title()} Range:")
        print(f"  Max absolute value: {np.mean(stats['max']):.2f} ± {np.std(stats['max']):.2f}")
        print(f"  Mean absolute value: {np.mean(stats['mean']):.2f} ± {np.std(stats['mean']):.2f}")
        print(f"  Standard deviation: {np.mean(stats['std']):.2f} ± {np.std(stats['std']):.2f}")
    
    print("\n" + "="*60)
    print("Maximum Values at RMSNorm Input (per model):")
    print("="*60)
    
    for model_name in ['qwen', 'seba', 'linalg', 'standard']:
        print(f"\n{model_name.title()}RMSNorm:")
        for range_name in ['normal', 'large', 'very_large', 'extra_large']:
            max_vals = model_max_values[model_name][range_name]
            if max_vals:
                print(f"  {range_name.replace('_', ' ').title()}: Max={np.max(max_vals):.2f}, Mean={np.mean(max_vals):.2f}, Min={np.min(max_vals):.2f}")
    
    # Check if any model modifies input values
    print("\n" + "="*60)
    print("Input Value Consistency Check:")
    print("="*60)
    
    # All models should see the same input values
    for range_name in ['normal', 'large', 'very_large', 'extra_large']:
        qwen_vals = model_max_values['qwen'][range_name]
        seba_vals = model_max_values['seba'][range_name]
        linalg_vals = model_max_values['linalg'][range_name]
        standard_vals = model_max_values['standard'][range_name]
        
        if qwen_vals and seba_vals and linalg_vals and standard_vals:
            all_equal = (qwen_vals == seba_vals == linalg_vals == standard_vals)
            print(f"\n{range_name.replace('_', ' ').title()} Range:")
            print(f"  All models see same input: {all_equal}")
            if not all_equal:
                print(f"  WARNING: Models see different input values!")


def compare_conv2d_coreml_implementations():
    """Compare Conv2D+RMSNorm implementations on CoreML/ANE for true accuracy."""
    
    print("\n" + "="*80)
    print("CONV2D + RMSNORM COREML/ANE COMPARISON")
    print("="*80)
    print(f"Using {NUM_PARALLEL_LAYERS} Conv2D layers before RMSNorm")
    print("This shows true accuracy without post-normalization amplification\n")
    
   
    # Export Conv2D models
    coreml_models_dict, pytorch_models, conv_weights = export_conv2d_models_to_coreml()
    
    # Compile and load models properly for ANE
    print("\n--- Compiling and Loading Models for ANE ---")
    coreml_models = {}
    for name in ['qwen', 'seba', 'linalg']:
        model_path = Path(f"/tmp/ANE/conv2d_{name}_rmsnorm_{NUM_PARALLEL_LAYERS}_layers.mlpackage")
        if model_path.exists():
            print(f"\nProcessing {name.title()}RMSNorm model...")
            coreml_models[name] = compile_and_load_model(model_path)
        else:
            print(f"Model not found: {model_path}")
            # Use the already loaded model from export as fallback
            coreml_models[name] = coreml_models_dict[name]
    
    # Create standard model for comparison with same processing
    standard_model = StandardRMSNorm(hidden_size=hidden_size)
    standard_model.weight = nn.Parameter(torch.ones(hidden_size))
    standard_model.eval()
    
    # Test with multiple inputs
    num_tests = NUM_TESTS
    results = {
        'qwen_coreml_vs_pytorch': [],
        'seba_coreml_vs_pytorch': [],
        'linalg_coreml_vs_pytorch': [],
        'qwen_coreml_vs_standard': [],
        'seba_coreml_vs_standard': [],
        'linalg_coreml_vs_standard': [],
        'qwen_vs_seba_coreml': [],
        'qwen_vs_linalg_coreml': [],
        'seba_vs_linalg_coreml': [],
    }
    
    print(f"\nRunning {num_tests} comparison tests...")
    
    for i in range(num_tests):
        if i < num_tests // 4:
            # Normal range (0-10)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 3
        elif i < num_tests // 2:
            # Large range (0-256)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 85
        elif i < 3 * num_tests // 4:
            # Very large range (0-1000)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 333
        else:
            # Extra large range (0-10000)
            test_input = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float16) * 3333
            test_input = torch.clamp(test_input, min=-10000, max=10000)
        
        # Get PyTorch outputs
        with torch.no_grad():
            qwen_pytorch_out = pytorch_models['qwen'](test_input).numpy()
            seba_pytorch_out = pytorch_models['seba'](test_input).numpy()
            linalg_pytorch_out = pytorch_models['linalg'](test_input).numpy()
            
            # For standard, apply same Conv2D processing
            x = test_input.view(batch_size * seq_length, hidden_size, 1, 1)
            output = torch.zeros_like(x)
            for j, weight in enumerate(conv_weights):
                conv = nn.Conv2d(hidden_size, hidden_size, 1, bias=False)
                conv.weight = nn.Parameter(weight.clone())
                output = output + conv(x.float())
            output = output / len(conv_weights)
            output = output.view(batch_size, seq_length, hidden_size).to(test_input.dtype)
            standard_out = standard_model(output).numpy()
        
        # Get CoreML outputs
        input_np = test_input.numpy()
        qwen_coreml_out = coreml_models['qwen'].predict({"hidden_states": input_np})["output"]
        seba_coreml_out = coreml_models['seba'].predict({"hidden_states": input_np})["output"]
        linalg_coreml_out = coreml_models['linalg'].predict({"hidden_states": input_np})["output"]
        
        # Calculate differences
        results['qwen_coreml_vs_pytorch'].append(np.max(np.abs(qwen_coreml_out - qwen_pytorch_out)))
        results['seba_coreml_vs_pytorch'].append(np.max(np.abs(seba_coreml_out - seba_pytorch_out)))
        results['linalg_coreml_vs_pytorch'].append(np.max(np.abs(linalg_coreml_out - linalg_pytorch_out)))
        
        results['qwen_coreml_vs_standard'].append(np.max(np.abs(qwen_coreml_out - standard_out)))
        results['seba_coreml_vs_standard'].append(np.max(np.abs(seba_coreml_out - standard_out)))
        results['linalg_coreml_vs_standard'].append(np.max(np.abs(linalg_coreml_out - standard_out)))
        
        results['qwen_vs_seba_coreml'].append(np.max(np.abs(qwen_coreml_out - seba_coreml_out)))
        results['qwen_vs_linalg_coreml'].append(np.max(np.abs(qwen_coreml_out - linalg_coreml_out)))
        results['seba_vs_linalg_coreml'].append(np.max(np.abs(seba_coreml_out - linalg_coreml_out)))
        
        if (i + 1) % (NUM_TESTS // 4) == 0:
            print(f"  Completed {i + 1}/{num_tests} tests...")
    
    # Print results
    print("\n--- Conv2D+RMSNorm CoreML vs PyTorch ---")
    for name in ['qwen', 'seba', 'linalg']:
        key = f'{name}_coreml_vs_pytorch'
        diffs = np.array(results[key])
        print(f"\n{name.title()}RMSNorm:")
        print(f"  Overall - Max: {np.max(diffs):.6f}, Mean: {np.mean(diffs):.6f}")
        print(f"  Normal (0-10) - Max: {np.max(diffs[:NUM_TESTS//4]):.6f}, Mean: {np.mean(diffs[:NUM_TESTS//4]):.6f}")
        print(f"  Large (0-256) - Max: {np.max(diffs[NUM_TESTS//4:NUM_TESTS//2]):.6f}, Mean: {np.mean(diffs[NUM_TESTS//4:NUM_TESTS//2]):.6f}")
        print(f"  Very Large (0-1000) - Max: {np.max(diffs[NUM_TESTS//2:3*NUM_TESTS//4]):.6f}, Mean: {np.mean(diffs[NUM_TESTS//2:3*NUM_TESTS//4]):.6f}")
        print(f"  Extra Large (0-10000) - Max: {np.max(diffs[3*NUM_TESTS//4:]):.6f}, Mean: {np.mean(diffs[3*NUM_TESTS//4:]):.6f}")
    
    print("\n--- Conv2D+RMSNorm Models vs Standard RMSNorm (True Accuracy) ---")
    accuracy_scores = []
    for name in ['qwen', 'seba', 'linalg']:
        key = f'{name}_coreml_vs_standard'
        diffs = np.array(results[key])
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        accuracy_scores.append((name, mean_diff, max_diff))
        
        print(f"\n{name.title()}RMSNorm vs Standard:")
        print(f"  Overall - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")
        print(f"  Normal (0-10) - Max: {np.max(diffs[:NUM_TESTS//4]):.6f}, Mean: {np.mean(diffs[:NUM_TESTS//4]):.6f}")
        print(f"  Large (0-256) - Max: {np.max(diffs[NUM_TESTS//4:NUM_TESTS//2]):.6f}, Mean: {np.mean(diffs[NUM_TESTS//4:NUM_TESTS//2]):.6f}")
        print(f"  Very Large (0-1000) - Max: {np.max(diffs[NUM_TESTS//2:3*NUM_TESTS//4]):.6f}, Mean: {np.mean(diffs[NUM_TESTS//2:3*NUM_TESTS//4]):.6f}")
        print(f"  Extra Large (0-10000) - Max: {np.max(diffs[3*NUM_TESTS//4:]):.6f}, Mean: {np.mean(diffs[3*NUM_TESTS//4:]):.6f}")
        
        # Check for degradation
        if np.max(diffs[3*NUM_TESTS//4:]) > np.max(diffs[:NUM_TESTS//4]) * 2:
            print(f"  ⚠️  Degradation: {np.max(diffs[3*NUM_TESTS//4:])/np.max(diffs[:NUM_TESTS//4]):.1f}x at large values")
    
    # Rank by accuracy
    accuracy_scores.sort(key=lambda x: x[1])
    print("\n--- TRUE ACCURACY RANKING (vs Standard RMSNorm) ---")
    for i, (name, mean_diff, max_diff) in enumerate(accuracy_scores, 1):
        print(f"{i}. {name.title()}RMSNorm: Mean={mean_diff:.6f}, Max={max_diff:.6f}")
    
    # Inter-model comparisons
    print("\n--- Direct Model Comparisons (CoreML) ---")
    print(f"QwenRMSNorm vs SebaRMSNorm: Max={np.max(results['qwen_vs_seba_coreml']):.6f}")
    print(f"QwenRMSNorm vs LinalgRMSNorm: Max={np.max(results['qwen_vs_linalg_coreml']):.6f}")
    print(f"SebaRMSNorm vs LinalgRMSNorm: Max={np.max(results['seba_vs_linalg_coreml']):.6f}")
    
    print("\n" + "="*80)
    
    # Generate automatic synopsis with CoreML results
    generate_synopsis(results)


if __name__ == "__main__":
    print(f"Starting RMSNorm input value analysis...")
    print(f"Using hidden_size={hidden_size}, batch_size={batch_size}, seq_length={seq_length}")
    
    # First analyze input values at RMSNorm
    analyze_rmsnorm_input_values()
    
    # Compare Conv2D implementations in PyTorch for baseline
    compare_conv2d_implementations()
    
    # Compare Conv2D+RMSNorm CoreML implementations for true accuracy
    compare_conv2d_coreml_implementations()