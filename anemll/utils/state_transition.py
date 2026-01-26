"""
State Transition Utility for KV Cache

This module provides utilities for transitioning KV cache states between models
with different context sizes. This is useful when you want to:
- Start inference with a smaller, faster model (e.g., 256 context)
- Transition to a larger model (e.g., 512 context) after initial tokens
- Preserve the accumulated KV cache state during the transition

It also supports COMPACTION (larger to smaller):
- Prefill with a larger state model (e.g., 512 context) for batch efficiency
- Compact to a smaller state model (e.g., 256 context) for token-by-token prediction
- This works when current_position <= target_seq_length

State Shape: [num_layers, num_kv_heads, state_length, head_dim]
Example Expansion: Float16 [36, 1, 256, 256] -> [36, 1, 512, 256]
Example Compaction: Float16 [36, 1, 512, 256] -> [36, 1, 256, 256] (if pos <= 256)

Usage (Expansion - small to large):
    from anemll.utils.state_transition import transition_kv_state

    # Read state from smaller model (numpy array from CoreML)
    small_state = state1.read_state(name="kv_cache")  # Shape: [36, 1, 256, 256]

    # Transition to larger state
    large_state = transition_kv_state(
        source_state=small_state,
        target_seq_length=512,
        current_position=256  # How many tokens have been processed
    )

    # Write to larger model's state
    state2.write_state(name="kv_cache", value=large_state)

Usage (Compaction - large to small):
    from anemll.utils.state_transition import compact_kv_state

    # Prefill with 512 state model (faster batch processing)
    large_state = state_512.read_state(name="kv_cache")  # Shape: [36, 1, 512, 256]

    # After prefill, compact to 256 state for prediction (more efficient)
    small_state = compact_kv_state(
        source_state=large_state,
        target_seq_length=256,
        current_position=200  # Must be <= 256 for compaction to work
    )

    # Write to smaller model's state for token-by-token generation
    state_256.write_state(name="kv_cache", value=small_state)
"""

import numpy as np
from typing import Union, Optional, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def transition_kv_state(
    source_state: Union[np.ndarray, "torch.Tensor"],
    target_seq_length: int,
    current_position: Optional[int] = None,
    pad_value: float = 0.0,
    dtype: Optional[np.dtype] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Transition KV cache state from a smaller context model to a larger context model.

    This function takes a KV cache state tensor and expands it to a larger sequence
    length, padding the additional positions with the specified value (default: 0.0).

    Args:
        source_state: The source KV cache state tensor.
            Shape: [num_layers, num_kv_heads, source_seq_length, head_dim]
            Can be numpy array or PyTorch tensor.
        target_seq_length: The target sequence length to expand to.
            Must be >= source sequence length.
        current_position: Optional. The number of tokens that have been processed
            in the source state. If provided, only positions 0:current_position
            contain valid data; the rest will be zeroed. If None, assumes the
            entire source state is valid (i.e., current_position = source_seq_length).
        pad_value: The value to use for padding new positions. Default: 0.0
        dtype: Optional dtype for the output. If None, uses source dtype.

    Returns:
        The expanded state tensor with shape:
        [num_layers, num_kv_heads, target_seq_length, head_dim]
        Returns same type as input (numpy or torch).

    Raises:
        ValueError: If target_seq_length < source_seq_length
        ValueError: If current_position > source_seq_length
        ValueError: If source_state doesn't have 4 dimensions

    Example:
        >>> import numpy as np
        >>> # Create a sample state: 36 layers, 1 head, 256 seq, 256 dim
        >>> small_state = np.random.randn(36, 1, 256, 256).astype(np.float16)
        >>>
        >>> # Transition to 512 sequence length
        >>> large_state = transition_kv_state(
        ...     source_state=small_state,
        ...     target_seq_length=512,
        ...     current_position=200  # Only first 200 tokens are valid
        ... )
        >>> print(large_state.shape)  # (36, 1, 512, 256)
    """
    # Validate input shape
    if len(source_state.shape) != 4:
        raise ValueError(
            f"Expected 4D state tensor [num_layers, num_kv_heads, seq_length, head_dim], "
            f"got shape {source_state.shape}"
        )

    num_layers, num_kv_heads, source_seq_length, head_dim = source_state.shape

    # Validate target size
    if target_seq_length < source_seq_length:
        raise ValueError(
            f"Target sequence length ({target_seq_length}) must be >= "
            f"source sequence length ({source_seq_length})"
        )

    # Handle current_position
    if current_position is None:
        current_position = source_seq_length
    elif current_position > source_seq_length:
        raise ValueError(
            f"current_position ({current_position}) cannot exceed "
            f"source sequence length ({source_seq_length})"
        )

    # Check if no transition needed
    if target_seq_length == source_seq_length and current_position == source_seq_length:
        return source_state

    # Determine if we're working with PyTorch or NumPy
    is_torch = HAS_TORCH and isinstance(source_state, torch.Tensor)

    if is_torch:
        return _transition_torch(
            source_state, target_seq_length, current_position, pad_value, dtype
        )
    else:
        return _transition_numpy(
            source_state, target_seq_length, current_position, pad_value, dtype
        )


def compact_kv_state(
    source_state: Union[np.ndarray, "torch.Tensor"],
    target_seq_length: int,
    current_position: int,
    dtype: Optional[np.dtype] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Compact KV cache state from a larger context model to a smaller context model.

    This is useful when you:
    1. Use a larger model for prefill (better batch efficiency)
    2. Switch to a smaller model for token-by-token prediction (more efficient)

    The key constraint is that current_position must fit in the target state,
    i.e., current_position <= target_seq_length.

    Args:
        source_state: The source KV cache state tensor from larger model.
            Shape: [num_layers, num_kv_heads, source_seq_length, head_dim]
            Can be numpy array or PyTorch tensor.
        target_seq_length: The target sequence length to compact to.
            Can be smaller than source sequence length.
        current_position: The number of tokens that have been processed.
            MUST be <= target_seq_length for compaction to succeed.
        dtype: Optional dtype for the output. If None, uses source dtype.

    Returns:
        The compacted state tensor with shape:
        [num_layers, num_kv_heads, target_seq_length, head_dim]
        Returns same type as input (numpy or torch).

    Raises:
        ValueError: If current_position > target_seq_length (won't fit)
        ValueError: If source_state doesn't have 4 dimensions

    Example:
        >>> import numpy as np
        >>> # Prefill with 512 state model
        >>> large_state = np.random.randn(36, 1, 512, 256).astype(np.float16)
        >>> current_pos = 200  # Processed 200 tokens during prefill
        >>>
        >>> # Compact to 256 state for prediction
        >>> small_state = compact_kv_state(
        ...     source_state=large_state,
        ...     target_seq_length=256,
        ...     current_position=current_pos  # 200 <= 256, so this works
        ... )
        >>> print(small_state.shape)  # (36, 1, 256, 256)

    Typical Use Case:
        # 1. Load 512-state model for prefill (faster batch processing)
        model_512 = load_model(state_size=512)
        state_512 = model_512.make_state()

        # 2. Run prefill with prompt (e.g., 200 tokens)
        for batch in prompt_batches:
            model_512.predict(batch, state=state_512)
        current_pos = 200

        # 3. Load 256-state model for prediction (more efficient per token)
        model_256 = load_model(state_size=256)
        state_256 = model_256.make_state()

        # 4. Compact state from 512 to 256
        large_kv = state_512.read_state(name="kv_cache")
        small_kv = compact_kv_state(large_kv, target_seq_length=256, current_position=200)
        state_256.write_state(name="kv_cache", value=small_kv)

        # 5. Continue with 256 model for generation
        for _ in range(max_tokens):
            model_256.predict(next_token, state=state_256)
    """
    # Validate input shape
    if len(source_state.shape) != 4:
        raise ValueError(
            f"Expected 4D state tensor [num_layers, num_kv_heads, seq_length, head_dim], "
            f"got shape {source_state.shape}"
        )

    num_layers, num_kv_heads, source_seq_length, head_dim = source_state.shape

    # Key validation: current_position must fit in target
    if current_position > target_seq_length:
        raise ValueError(
            f"Cannot compact: current_position ({current_position}) exceeds "
            f"target_seq_length ({target_seq_length}). "
            f"The processed tokens won't fit in the smaller state."
        )

    if current_position > source_seq_length:
        raise ValueError(
            f"current_position ({current_position}) cannot exceed "
            f"source sequence length ({source_seq_length})"
        )

    # Determine if we're working with PyTorch or NumPy
    is_torch = HAS_TORCH and isinstance(source_state, torch.Tensor)

    if is_torch:
        return _compact_torch(source_state, target_seq_length, current_position, dtype)
    else:
        return _compact_numpy(source_state, target_seq_length, current_position, dtype)


def _compact_numpy(
    source_state: np.ndarray,
    target_seq_length: int,
    current_position: int,
    dtype: Optional[np.dtype],
) -> np.ndarray:
    """NumPy implementation of state compaction."""
    num_layers, num_kv_heads, source_seq_length, head_dim = source_state.shape

    # Determine output dtype
    out_dtype = dtype if dtype is not None else source_state.dtype

    # Create target state filled with zeros
    target_state = np.zeros(
        (num_layers, num_kv_heads, target_seq_length, head_dim),
        dtype=out_dtype
    )

    # Copy valid positions from source to target
    if current_position > 0:
        target_state[:, :, :current_position, :] = source_state[:, :, :current_position, :].astype(out_dtype)

    return target_state


def _compact_torch(
    source_state: "torch.Tensor",
    target_seq_length: int,
    current_position: int,
    dtype: Optional["torch.dtype"],
) -> "torch.Tensor":
    """PyTorch implementation of state compaction."""
    num_layers, num_kv_heads, source_seq_length, head_dim = source_state.shape

    # Determine output dtype
    out_dtype = dtype if dtype is not None else source_state.dtype

    # Create target state filled with zeros
    target_state = torch.zeros(
        (num_layers, num_kv_heads, target_seq_length, head_dim),
        dtype=out_dtype,
        device=source_state.device
    )

    # Copy valid positions from source to target
    if current_position > 0:
        target_state[:, :, :current_position, :] = source_state[:, :, :current_position, :].to(out_dtype)

    return target_state


def _transition_numpy(
    source_state: np.ndarray,
    target_seq_length: int,
    current_position: int,
    pad_value: float,
    dtype: Optional[np.dtype],
) -> np.ndarray:
    """NumPy implementation of state transition."""
    num_layers, num_kv_heads, source_seq_length, head_dim = source_state.shape

    # Determine output dtype
    out_dtype = dtype if dtype is not None else source_state.dtype

    # Create target state filled with pad value
    target_state = np.full(
        (num_layers, num_kv_heads, target_seq_length, head_dim),
        fill_value=pad_value,
        dtype=out_dtype
    )

    # Copy valid positions from source to target
    if current_position > 0:
        target_state[:, :, :current_position, :] = source_state[:, :, :current_position, :].astype(out_dtype)

    return target_state


def _transition_torch(
    source_state: "torch.Tensor",
    target_seq_length: int,
    current_position: int,
    pad_value: float,
    dtype: Optional["torch.dtype"],
) -> "torch.Tensor":
    """PyTorch implementation of state transition."""
    num_layers, num_kv_heads, source_seq_length, head_dim = source_state.shape

    # Determine output dtype
    out_dtype = dtype if dtype is not None else source_state.dtype

    # Create target state filled with pad value
    target_state = torch.full(
        (num_layers, num_kv_heads, target_seq_length, head_dim),
        fill_value=pad_value,
        dtype=out_dtype,
        device=source_state.device
    )

    # Copy valid positions from source to target
    if current_position > 0:
        target_state[:, :, :current_position, :] = source_state[:, :, :current_position, :].to(out_dtype)

    return target_state


def get_transition_info(
    source_seq_length: int,
    target_seq_length: int,
    current_position: int,
) -> dict:
    """
    Get information about a state transition.

    Useful for debugging and understanding the transition.

    Args:
        source_seq_length: The source state sequence length
        target_seq_length: The target state sequence length
        current_position: Number of tokens processed in source

    Returns:
        Dictionary with transition information
    """
    return {
        "source_seq_length": source_seq_length,
        "target_seq_length": target_seq_length,
        "current_position": current_position,
        "valid_tokens": current_position,
        "padded_positions_source": source_seq_length - current_position,
        "padded_positions_target": target_seq_length - current_position,
        "expansion_factor": target_seq_length / source_seq_length,
        "additional_capacity": target_seq_length - source_seq_length,
    }


def validate_state_shapes(
    source_state_shape: Tuple[int, ...],
    target_state_shape: Tuple[int, ...],
) -> bool:
    """
    Validate that two state shapes are compatible for transition.

    States are compatible if they have the same:
    - Number of layers
    - Number of KV heads
    - Head dimension

    And the target sequence length >= source sequence length.

    Args:
        source_state_shape: Shape of source state (num_layers, num_kv_heads, seq_len, head_dim)
        target_state_shape: Shape of target state

    Returns:
        True if shapes are compatible for transition

    Raises:
        ValueError: If shapes are incompatible, with explanation
    """
    if len(source_state_shape) != 4:
        raise ValueError(f"Source state must be 4D, got {len(source_state_shape)}D")
    if len(target_state_shape) != 4:
        raise ValueError(f"Target state must be 4D, got {len(target_state_shape)}D")

    src_layers, src_heads, src_seq, src_dim = source_state_shape
    tgt_layers, tgt_heads, tgt_seq, tgt_dim = target_state_shape

    if src_layers != tgt_layers:
        raise ValueError(
            f"Layer count mismatch: source has {src_layers}, target has {tgt_layers}"
        )

    if src_heads != tgt_heads:
        raise ValueError(
            f"KV head count mismatch: source has {src_heads}, target has {tgt_heads}"
        )

    if src_dim != tgt_dim:
        raise ValueError(
            f"Head dimension mismatch: source has {src_dim}, target has {tgt_dim}"
        )

    if tgt_seq < src_seq:
        raise ValueError(
            f"Target sequence length ({tgt_seq}) must be >= source ({src_seq})"
        )

    return True


class StateTransitionManager:
    """
    Manager for handling state transitions between models of different context sizes.

    This class provides a higher-level interface for managing state transitions,
    particularly useful when working with CoreML models that have the read_state
    and write_state APIs.

    Example:
        >>> manager = StateTransitionManager()
        >>>
        >>> # Register models with their state configurations
        >>> manager.register_model("model_256", state_seq_length=256)
        >>> manager.register_model("model_512", state_seq_length=512)
        >>>
        >>> # Perform transition
        >>> large_state = manager.transition(
        ...     source_state=small_state,
        ...     source_model="model_256",
        ...     target_model="model_512",
        ...     current_position=200
        ... )
    """

    def __init__(self):
        self.models = {}

    def register_model(
        self,
        model_name: str,
        state_seq_length: int,
        num_layers: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        """
        Register a model configuration for state management.

        Args:
            model_name: Unique identifier for the model
            state_seq_length: The sequence length of this model's state
            num_layers: Optional number of layers (for validation)
            num_kv_heads: Optional number of KV heads (for validation)
            head_dim: Optional head dimension (for validation)
        """
        self.models[model_name] = {
            "state_seq_length": state_seq_length,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }

    def transition(
        self,
        source_state: Union[np.ndarray, "torch.Tensor"],
        source_model: str,
        target_model: str,
        current_position: int,
        pad_value: float = 0.0,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Transition state from one model to another.

        Args:
            source_state: The source KV cache state
            source_model: Name of the source model (must be registered)
            target_model: Name of the target model (must be registered)
            current_position: Number of tokens processed
            pad_value: Value for padding

        Returns:
            The transitioned state tensor
        """
        if source_model not in self.models:
            raise ValueError(f"Source model '{source_model}' not registered")
        if target_model not in self.models:
            raise ValueError(f"Target model '{target_model}' not registered")

        source_config = self.models[source_model]
        target_config = self.models[target_model]

        # Validate current position
        if current_position > source_config["state_seq_length"]:
            raise ValueError(
                f"current_position ({current_position}) exceeds source model's "
                f"state length ({source_config['state_seq_length']})"
            )

        return transition_kv_state(
            source_state=source_state,
            target_seq_length=target_config["state_seq_length"],
            current_position=current_position,
            pad_value=pad_value,
        )

    def get_transition_path(
        self,
        start_model: str,
        end_model: str,
    ) -> list:
        """
        Get the list of models in order from start to end by sequence length.

        Useful for planning multi-stage transitions.

        Args:
            start_model: Starting model name
            end_model: Ending model name

        Returns:
            List of model names in order of increasing sequence length
        """
        if start_model not in self.models:
            raise ValueError(f"Start model '{start_model}' not registered")
        if end_model not in self.models:
            raise ValueError(f"End model '{end_model}' not registered")

        start_len = self.models[start_model]["state_seq_length"]
        end_len = self.models[end_model]["state_seq_length"]

        if start_len > end_len:
            raise ValueError(
                f"Start model seq length ({start_len}) > end model ({end_len}). "
                "Can only transition to larger states."
            )

        # Sort models by sequence length and filter to range
        sorted_models = sorted(
            self.models.items(),
            key=lambda x: x[1]["state_seq_length"]
        )

        path = []
        for name, config in sorted_models:
            seq_len = config["state_seq_length"]
            if start_len <= seq_len <= end_len:
                path.append(name)

        return path


# Convenience functions for CoreML state objects
def transition_coreml_state(
    source_mlstate,
    target_mlstate,
    state_name: str,
    current_position: int,
    pad_value: float = 0.0,
):
    """
    Convenience function to transition state between CoreML MLState objects.

    This function reads from the source state, performs the transition,
    and writes to the target state.

    Args:
        source_mlstate: Source CoreML MLState object (from mlmodel.make_state())
        target_mlstate: Target CoreML MLState object
        state_name: Name of the state to transition (e.g., "kv_cache")
        current_position: Number of tokens that have been processed
        pad_value: Value for padding

    Example:
        >>> # Assuming you have two CoreML models with different state sizes
        >>> state_256 = model_256.make_state()
        >>> state_512 = model_512.make_state()
        >>>
        >>> # Run predictions on the smaller model
        >>> for i in range(200):
        ...     model_256.predict(input_dict, state=state_256)
        >>>
        >>> # Transition state to larger model
        >>> transition_coreml_state(
        ...     source_mlstate=state_256,
        ...     target_mlstate=state_512,
        ...     state_name="kv_cache",
        ...     current_position=200
        ... )
        >>>
        >>> # Continue with larger model
        >>> model_512.predict(input_dict, state=state_512)
    """
    # Read source state
    source_state = source_mlstate.read_state(name=state_name)

    # Get target shape to determine target sequence length
    # We need to infer target_seq_length from the target state
    # Read current target state to get its shape
    target_current = target_mlstate.read_state(name=state_name)
    target_seq_length = target_current.shape[2]

    # Perform transition
    transitioned_state = transition_kv_state(
        source_state=source_state,
        target_seq_length=target_seq_length,
        current_position=current_position,
        pad_value=pad_value,
    )

    # Write to target state
    target_mlstate.write_state(name=state_name, value=transitioned_state)


def compact_coreml_state(
    source_mlstate,
    target_mlstate,
    state_name: str,
    current_position: int,
):
    """
    Convenience function to compact state from larger CoreML model to smaller.

    This is useful when:
    1. You prefill with a larger state model (e.g., 512) for batch efficiency
    2. You want to switch to a smaller state model (e.g., 256) for prediction

    The current_position must fit within the target model's state size.

    Args:
        source_mlstate: Source CoreML MLState object (larger model)
        target_mlstate: Target CoreML MLState object (smaller model)
        state_name: Name of the state to compact (e.g., "kv_cache")
        current_position: Number of tokens processed (must be <= target size)

    Example:
        >>> # Prefill with 512 state model (faster batch processing)
        >>> state_512 = model_512.make_state()
        >>> for batch in prompt_batches:
        ...     model_512.predict(batch, state=state_512)
        >>> current_pos = 200  # Processed 200 tokens
        >>>
        >>> # Compact to 256 state for prediction
        >>> state_256 = model_256.make_state()
        >>> compact_coreml_state(
        ...     source_mlstate=state_512,
        ...     target_mlstate=state_256,
        ...     state_name="kv_cache",
        ...     current_position=200  # 200 <= 256, fits!
        ... )
        >>>
        >>> # Continue with smaller model for generation
        >>> model_256.predict(next_token, state=state_256)
    """
    # Read source state
    source_state = source_mlstate.read_state(name=state_name)

    # Get target shape to determine target sequence length
    target_current = target_mlstate.read_state(name=state_name)
    target_seq_length = target_current.shape[2]

    # Perform compaction
    compacted_state = compact_kv_state(
        source_state=source_state,
        target_seq_length=target_seq_length,
        current_position=current_position,
    )

    # Write to target state
    target_mlstate.write_state(name=state_name, value=compacted_state)


if __name__ == "__main__":
    # Demo/test code
    import numpy as np

    print("=== State Transition Utility Demo ===\n")

    # Create sample state (simulating 36 layers, 1 head, 256 seq, 256 dim)
    print("Creating sample source state: [36, 1, 256, 256]")
    source = np.random.randn(36, 1, 256, 256).astype(np.float16)

    # Simulate that we've processed 200 tokens
    current_pos = 200
    print(f"Simulating {current_pos} tokens processed")

    # Transition to 512
    print("\nTransitioning to target sequence length: 512")
    target = transition_kv_state(
        source_state=source,
        target_seq_length=512,
        current_position=current_pos
    )

    print(f"Source shape: {source.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target dtype: {target.dtype}")

    # Verify data integrity
    print("\n=== Verification ===")
    print(f"Valid positions (0:{current_pos}) match: {np.allclose(source[:, :, :current_pos, :], target[:, :, :current_pos, :])}")
    print(f"Padded positions ({current_pos}:512) are zeros: {np.allclose(target[:, :, current_pos:, :], 0.0)}")

    # Show transition info
    print("\n=== Transition Info ===")
    info = get_transition_info(256, 512, current_pos)
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Demo PyTorch if available
    if HAS_TORCH:
        print("\n=== PyTorch Demo ===")
        source_torch = torch.from_numpy(source)
        target_torch = transition_kv_state(
            source_state=source_torch,
            target_seq_length=512,
            current_position=current_pos
        )
        print(f"PyTorch target shape: {target_torch.shape}")
        print(f"PyTorch target dtype: {target_torch.dtype}")
        print(f"PyTorch target device: {target_torch.device}")

    print("\n=== Demo Complete ===")
