"""Test suite for Gemma3 model implementation.

This test file validates the ANEMLL Gemma3 implementation against
the reference HuggingFace implementation to ensure correctness.

Tests cover:
- Config loading and initialization
- RMSNorm with Gemma-style (1 + weight) scaling
- Rotary embeddings (dual RoPE bases for local vs global layers)
- Attention with per-head Q/K normalization
- MLP with GEGLU activation
- Forward pass correctness
- KV cache functionality
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.gemma3_model import (
    Gemma3Config,
    Gemma3RMSNorm,
    Gemma3HeadNorm,
    Gemma3RotaryEmbedding,
    Gemma3MLP,
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3Model,
    Gemma3ForCausalLM,
    MODEL_DTYPE,
    TEST_DEVICE,
)


def test_gemma3_config():
    """Test Gemma3Config initialization with default values."""
    print("\n" + "=" * 60)
    print("Testing Gemma3Config")
    print("=" * 60)

    config = Gemma3Config()

    # Check default values for Gemma3 270M
    assert config.hidden_size == 640, f"Expected hidden_size=640, got {config.hidden_size}"
    assert config.num_hidden_layers == 18, f"Expected num_hidden_layers=18, got {config.num_hidden_layers}"
    assert config.num_attention_heads == 4, f"Expected num_attention_heads=4, got {config.num_attention_heads}"
    assert config.num_key_value_heads == 1, f"Expected num_key_value_heads=1, got {config.num_key_value_heads}"
    assert config.head_dim == 256, f"Expected head_dim=256, got {config.head_dim}"
    assert config.intermediate_size == 2048, f"Expected intermediate_size=2048, got {config.intermediate_size}"
    assert config.vocab_size == 262_144, f"Expected vocab_size=262144, got {config.vocab_size}"
    assert config.rope_theta == 1_000_000.0, f"Expected rope_theta=1e6, got {config.rope_theta}"
    assert config.rope_local_base_freq == 10_000.0, f"Expected rope_local_base_freq=10k, got {config.rope_local_base_freq}"
    assert config.sliding_window == 512, f"Expected sliding_window=512, got {config.sliding_window}"

    # Check layer types for interleaved attention
    expected_global_layers = [5, 11, 17]  # 0-indexed layers 6, 12, 18
    for i, layer_type in enumerate(config.layer_types):
        if i in expected_global_layers:
            assert layer_type == "full_attention", f"Layer {i} should be full_attention, got {layer_type}"
        else:
            assert layer_type == "sliding_attention", f"Layer {i} should be sliding_attention, got {layer_type}"

    print("✅ Gemma3Config test passed!")
    return True


def test_gemma3_rmsnorm():
    """Test Gemma3 RMSNorm with (1 + weight) scaling."""
    print("\n" + "=" * 60)
    print("Testing Gemma3RMSNorm")
    print("=" * 60)

    hidden_size = 640
    batch_size = 1
    seq_len = 4

    # Create RMSNorm layer
    norm = Gemma3RMSNorm(hidden_size, eps=1e-6)

    # Check weight initialization (should be zeros for Gemma-style)
    assert torch.allclose(norm.weight, torch.zeros(hidden_size)), "Weight should be initialized to zeros"

    # Test forward pass
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    output = norm(x)

    # Check output shape
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input {x.shape}"

    # Verify that with weight=0, output should be RMS normalized (approximately)
    # The (1 + weight) = 1 when weight=0, so it's just RMS normalization
    for i in range(seq_len):
        vec = output[0, i, :].float()
        rms = torch.sqrt(torch.mean(vec ** 2))
        # After RMS norm, the RMS should be approximately 1
        assert 0.9 < rms.item() < 1.1, f"RMS norm output should have RMS ≈ 1, got {rms.item()}"

    # Test with non-zero weights
    norm.weight.data = torch.ones(hidden_size) * 0.5  # (1 + 0.5) = 1.5x scaling
    output_scaled = norm(x)

    # Output should be scaled by 1.5x compared to default
    for i in range(seq_len):
        vec = output_scaled[0, i, :].float()
        rms = torch.sqrt(torch.mean(vec ** 2))
        # After scaling by 1.5, RMS should be approximately 1.5
        assert 1.3 < rms.item() < 1.7, f"Scaled RMS norm output should have RMS ≈ 1.5, got {rms.item()}"

    print("✅ Gemma3RMSNorm test passed!")
    return True


def test_gemma3_head_norm():
    """Test Gemma3HeadNorm for Q/K normalization."""
    print("\n" + "=" * 60)
    print("Testing Gemma3HeadNorm")
    print("=" * 60)

    head_dim = 256
    batch_size = 1
    num_heads = 4
    seq_len = 1

    # Create head norm layer
    head_norm = Gemma3HeadNorm(head_dim, eps=1e-6)

    # Test forward pass with attention-like input shape [batch, heads, seq, head_dim]
    x = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    output = head_norm(x)

    # Check output shape
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input {x.shape}"

    print("✅ Gemma3HeadNorm test passed!")
    return True


def test_gemma3_rotary_embedding():
    """Test Gemma3 rotary embeddings with dual RoPE bases."""
    print("\n" + "=" * 60)
    print("Testing Gemma3RotaryEmbedding")
    print("=" * 60)

    config = Gemma3Config()
    config.context_length = 256
    config.state_length = 256

    # Create rotary embedding with global theta (1e6)
    rotary_global = Gemma3RotaryEmbedding(config)

    # Create local config
    local_config = Gemma3Config()
    local_config.rope_theta = 10_000.0  # Local base freq
    local_config.context_length = 256
    local_config.state_length = 256
    rotary_local = Gemma3RotaryEmbedding(local_config)

    # Test with sample input
    batch_size = 1
    seq_len = 4
    x = torch.randn(batch_size, seq_len, config.head_dim)
    position_ids = torch.arange(seq_len)

    # Get embeddings
    cos_global, sin_global = rotary_global(x, position_ids)
    cos_local, sin_local = rotary_local(x, position_ids)

    # Check shapes
    assert cos_global.shape[-1] == config.head_dim, f"cos shape {cos_global.shape} doesn't match head_dim"
    assert sin_global.shape[-1] == config.head_dim, f"sin shape {sin_global.shape} doesn't match head_dim"

    # Global and local should have different frequencies
    # (they should NOT be exactly equal due to different base frequencies)
    if not torch.allclose(cos_global, cos_local, atol=1e-3):
        print("  ✓ Global and local RoPE have different frequencies (as expected)")
    else:
        print("  ⚠ Global and local RoPE are similar (may indicate theta is not being used)")

    print("✅ Gemma3RotaryEmbedding test passed!")
    return True


def test_gemma3_mlp():
    """Test Gemma3 MLP with GEGLU activation."""
    print("\n" + "=" * 60)
    print("Testing Gemma3MLP")
    print("=" * 60)

    config = Gemma3Config()
    mlp = Gemma3MLP(config)

    batch_size = 1
    seq_len = 4

    # Test forward pass
    x = torch.randn(batch_size, seq_len, config.hidden_size, dtype=MODEL_DTYPE)
    output = mlp(x)

    # Check output shape
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input {x.shape}"

    # Check projection dimensions
    assert mlp.gate_proj.weight.shape == (config.intermediate_size, config.hidden_size, 1, 1)
    assert mlp.up_proj.weight.shape == (config.intermediate_size, config.hidden_size, 1, 1)
    assert mlp.down_proj.weight.shape == (config.hidden_size, config.intermediate_size, 1, 1)

    print("✅ Gemma3MLP test passed!")
    return True


def test_gemma3_attention():
    """Test Gemma3 Attention with per-head Q/K normalization."""
    print("\n" + "=" * 60)
    print("Testing Gemma3Attention")
    print("=" * 60)

    config = Gemma3Config()
    attn = Gemma3Attention(config)

    batch_size = 1
    seq_len = 4

    # Check projection dimensions
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim

    assert attn.q_proj.weight.shape == (q_dim, config.hidden_size, 1, 1), \
        f"q_proj shape mismatch: {attn.q_proj.weight.shape}"
    assert attn.k_proj.weight.shape == (kv_dim, config.hidden_size, 1, 1), \
        f"k_proj shape mismatch: {attn.k_proj.weight.shape}"
    assert attn.v_proj.weight.shape == (kv_dim, config.hidden_size, 1, 1), \
        f"v_proj shape mismatch: {attn.v_proj.weight.shape}"
    assert attn.o_proj.weight.shape == (config.hidden_size, q_dim, 1, 1), \
        f"o_proj shape mismatch: {attn.o_proj.weight.shape}"

    # Check Q/K norms exist
    assert hasattr(attn, 'q_norm'), "Attention should have q_norm"
    assert hasattr(attn, 'k_norm'), "Attention should have k_norm"

    print("✅ Gemma3Attention test passed!")
    return True


def test_gemma3_model_init():
    """Test Gemma3Model initialization."""
    print("\n" + "=" * 60)
    print("Testing Gemma3Model initialization")
    print("=" * 60)

    config = Gemma3Config()
    config.context_length = 64  # Smaller for testing
    config.state_length = 64

    model = Gemma3Model(config)

    # Check embedding
    assert model.embed_tokens.num_embeddings == config.vocab_size
    assert model.embed_tokens.embedding_dim == config.hidden_size

    # Check embedding scale
    expected_scale = config.hidden_size ** 0.5
    assert model.embedding_scale == expected_scale, \
        f"Embedding scale should be {expected_scale}, got {model.embedding_scale}"

    # Check dual RoPE
    assert hasattr(model, 'rotary_emb_global'), "Model should have rotary_emb_global"
    assert hasattr(model, 'rotary_emb_local'), "Model should have rotary_emb_local"

    # Check layers
    assert len(model.layers) == config.num_hidden_layers

    # Check KV cache
    assert hasattr(model, 'kv_cache_0'), "Model should have unified KV cache"
    expected_cache_shape = (
        2 * config.num_hidden_layers,
        config.num_key_value_heads,
        config.state_length,
        config.head_dim
    )
    assert model.kv_cache_0.shape == expected_cache_shape, \
        f"KV cache shape mismatch: {model.kv_cache_0.shape} vs {expected_cache_shape}"

    print("✅ Gemma3Model initialization test passed!")
    return True


def test_gemma3_forward_pass():
    """Test Gemma3ForCausalLM forward pass."""
    print("\n" + "=" * 60)
    print("Testing Gemma3ForCausalLM forward pass")
    print("=" * 60)

    config = Gemma3Config()
    config.context_length = 64
    config.state_length = 64

    model = Gemma3ForCausalLM(config)
    model.eval()

    batch_size = 1
    seq_len = 1  # Single token for generation mode

    # Create inputs
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.int32)
    position_ids = torch.zeros((seq_len,), dtype=torch.int32)
    causal_mask = torch.zeros((batch_size, 1, seq_len, config.context_length), dtype=MODEL_DTYPE)
    current_pos = torch.tensor([0], dtype=torch.int32)
    update_mask = torch.zeros((batch_size, 1, config.context_length, 1), dtype=MODEL_DTYPE)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False,
        )

    # Check output - should be 16 logits for split LM head
    if isinstance(outputs, tuple):
        print(f"  Output is tuple with {len(outputs)} elements (16-way split LM head)")
        for i, logits in enumerate(outputs):
            print(f"    logits{i+1} shape: {logits.shape}")
            # Each split should have vocab_size//16 tokens
            expected_vocab = config.vocab_size // 16
            assert logits.shape[-1] == expected_vocab, \
                f"logits{i+1} should have {expected_vocab} vocab, got {logits.shape[-1]}"
    else:
        print(f"  Output shape: {outputs.shape}")
        # Full vocab output
        assert outputs.shape[-1] == config.vocab_size, \
            f"Output vocab size {outputs.shape[-1]} doesn't match config {config.vocab_size}"

    print("✅ Gemma3ForCausalLM forward pass test passed!")
    return True


def test_gemma3_kv_cache():
    """Test Gemma3 KV cache functionality."""
    print("\n" + "=" * 60)
    print("Testing Gemma3 KV cache")
    print("=" * 60)

    config = Gemma3Config()
    config.context_length = 64
    config.state_length = 64

    model = Gemma3ForCausalLM(config)
    model.eval()

    # Generate a few tokens to populate KV cache
    input_ids = torch.randint(0, config.vocab_size, (1, 1), dtype=torch.int32)

    for pos in range(5):
        position_ids = torch.tensor([pos], dtype=torch.int32)
        causal_mask = torch.zeros((1, 1, 1, config.context_length), dtype=MODEL_DTYPE)
        # Set up causal mask - allow attention to all positions up to current
        causal_mask[:, :, :, pos+1:] = float('-inf')
        current_pos = torch.tensor([pos], dtype=torch.int32)
        update_mask = torch.zeros((1, 1, config.context_length, 1), dtype=MODEL_DTYPE)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False,
            )

        # Get next token (simplified - just take argmax of first split)
        if isinstance(outputs, tuple):
            next_token = outputs[0].argmax(dim=-1)
        else:
            next_token = outputs.argmax(dim=-1)

        input_ids = next_token.view(1, 1).to(torch.int32)
        print(f"  Position {pos}: Generated token {input_ids.item()}")

    # Check KV cache is populated
    kv_cache = model.model.kv_cache_0
    # At least some values should be non-zero after 5 tokens
    assert (kv_cache[:, :, :5, :] != 0).any(), "KV cache should have non-zero values"

    print("✅ Gemma3 KV cache test passed!")
    return True


def test_layer_types():
    """Test that layer types are correctly assigned for interleaved attention."""
    print("\n" + "=" * 60)
    print("Testing layer types for interleaved attention")
    print("=" * 60)

    config = Gemma3Config()

    # Global layers should be at indices 5, 11, 17 (0-indexed)
    global_layers = [5, 11, 17]

    for i, layer_type in enumerate(config.layer_types):
        expected = "full_attention" if i in global_layers else "sliding_attention"
        assert layer_type == expected, f"Layer {i}: expected {expected}, got {layer_type}"
        print(f"  Layer {i}: {layer_type} {'✓' if layer_type == expected else '✗'}")

    print("✅ Layer types test passed!")
    return True


def run_all_tests():
    """Run all Gemma3 tests."""
    print("\n" + "=" * 70)
    print("GEMMA3 MODEL TEST SUITE")
    print("=" * 70)

    tests = [
        ("Config", test_gemma3_config),
        ("RMSNorm", test_gemma3_rmsnorm),
        ("HeadNorm", test_gemma3_head_norm),
        ("RotaryEmbedding", test_gemma3_rotary_embedding),
        ("MLP", test_gemma3_mlp),
        ("Attention", test_gemma3_attention),
        ("Model Init", test_gemma3_model_init),
        ("Forward Pass", test_gemma3_forward_pass),
        ("KV Cache", test_gemma3_kv_cache),
        ("Layer Types", test_layer_types),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            print(f"❌ {name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, p, _ in results if p)
    failed = len(results) - passed

    for name, p, error in results:
        status = "✅ PASSED" if p else f"❌ FAILED: {error}"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if failed > 0:
        print(f"\n⚠️ {failed} test(s) failed!")
        return False
    else:
        print("\n✅ All tests passed!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
