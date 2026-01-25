"""Test ANEMLL Gemma3 implementation against HuggingFace reference.

This test loads the actual Gemma3-270M model weights and compares
the ANEMLL implementation output against the HuggingFace implementation.
"""

import os
import sys
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL_PATH = os.path.expanduser("~/models/Models/gemma-3-270m-it")


def test_config_match():
    """Test that ANEMLL config matches HuggingFace config."""
    print("\n" + "=" * 60)
    print("Testing Config Match")
    print("=" * 60)

    import json
    from anemll.models.gemma3_model import Gemma3Config

    # Load HF config
    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        hf_config = json.load(f)

    # Create ANEMLL config
    anemll_config = Gemma3Config()

    # Compare key values
    checks = [
        ("hidden_size", hf_config["hidden_size"], anemll_config.hidden_size),
        ("num_hidden_layers", hf_config["num_hidden_layers"], anemll_config.num_hidden_layers),
        ("num_attention_heads", hf_config["num_attention_heads"], anemll_config.num_attention_heads),
        ("num_key_value_heads", hf_config["num_key_value_heads"], anemll_config.num_key_value_heads),
        ("head_dim", hf_config["head_dim"], anemll_config.head_dim),
        ("intermediate_size", hf_config["intermediate_size"], anemll_config.intermediate_size),
        ("vocab_size", hf_config["vocab_size"], anemll_config.vocab_size),
        ("rope_theta", hf_config["rope_theta"], anemll_config.rope_theta),
        ("rope_local_base_freq", hf_config["rope_local_base_freq"], anemll_config.rope_local_base_freq),
        ("sliding_window", hf_config["sliding_window"], anemll_config.sliding_window),
        ("rms_norm_eps", hf_config["rms_norm_eps"], anemll_config.rms_norm_eps),
    ]

    all_passed = True
    for name, hf_val, anemll_val in checks:
        match = hf_val == anemll_val
        status = "✅" if match else "❌"
        print(f"  {status} {name}: HF={hf_val}, ANEMLL={anemll_val}")
        if not match:
            all_passed = False

    # Check layer types
    print(f"\n  Layer types comparison:")
    for i, (hf_type, anemll_type) in enumerate(zip(hf_config["layer_types"], anemll_config.layer_types)):
        match = hf_type == anemll_type
        if not match:
            print(f"    ❌ Layer {i}: HF={hf_type}, ANEMLL={anemll_type}")
            all_passed = False

    if all_passed:
        print("  ✅ All layer types match")

    return all_passed


def test_weight_loading():
    """Test that weights can be loaded correctly."""
    print("\n" + "=" * 60)
    print("Testing Weight Loading")
    print("=" * 60)

    from anemll.models.gemma3_model import Gemma3Config, Gemma3ForCausalLM

    config = Gemma3Config()
    config.context_length = 64
    config.state_length = 64

    model = Gemma3ForCausalLM(config)

    print(f"  Loading weights from {MODEL_PATH}...")
    success = model.load_pretrained_weights(MODEL_PATH)

    if success:
        print("  ✅ Weights loaded successfully")
    else:
        print("  ❌ Weight loading failed")

    return success


def test_embedding_layer():
    """Test embedding layer output against HuggingFace."""
    print("\n" + "=" * 60)
    print("Testing Embedding Layer")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from anemll.models.gemma3_model import Gemma3Config, Gemma3ForCausalLM

    # Load HF model
    print("  Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # Use float32 for comparison
        device_map="cpu"
    )
    hf_model.eval()

    # Load ANEMLL model
    print("  Loading ANEMLL model...")
    config = Gemma3Config()
    config.context_length = 64
    config.state_length = 64
    anemll_model = Gemma3ForCausalLM(config)
    anemll_model.load_pretrained_weights(MODEL_PATH)
    anemll_model.eval()

    # Test input
    input_ids = torch.tensor([[2, 1234, 5678, 9012]], dtype=torch.long)  # BOS + some tokens

    # Get HF embeddings (Gemma3TextScaledWordEmbedding already scales internally)
    with torch.no_grad():
        hf_embeds_scaled = hf_model.model.embed_tokens(input_ids)

    # Get ANEMLL embeddings (we scale manually)
    with torch.no_grad():
        anemll_embeds = anemll_model.model.embed_tokens(input_ids)
        anemll_embeds_scaled = anemll_embeds * anemll_model.model.embedding_scale

    # Compare
    hf_embeds_scaled = hf_embeds_scaled.float()
    anemll_embeds_scaled = anemll_embeds_scaled.float()

    diff = torch.abs(hf_embeds_scaled - anemll_embeds_scaled).max().item()
    mean_diff = torch.abs(hf_embeds_scaled - anemll_embeds_scaled).mean().item()

    print(f"  Embedding shape: HF={hf_embeds_scaled.shape}, ANEMLL={anemll_embeds_scaled.shape}")
    print(f"  Max diff: {diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    # Allow for float16 precision differences
    passed = diff < 0.01
    if passed:
        print("  ✅ Embeddings match")
    else:
        print("  ❌ Embeddings don't match")

    return passed


def test_rmsnorm_layer():
    """Test RMSNorm against HuggingFace."""
    print("\n" + "=" * 60)
    print("Testing RMSNorm Layer")
    print("=" * 60)

    from transformers import AutoModelForCausalLM
    from anemll.models.gemma3_model import Gemma3Config, Gemma3ForCausalLM, Gemma3RMSNorm

    # Load HF model to get weights
    print("  Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    # Get HF layer norm weight from first layer
    hf_norm_weight = hf_model.model.layers[0].input_layernorm.weight.clone()

    # Load ANEMLL model
    print("  Loading ANEMLL model...")
    config = Gemma3Config()
    config.context_length = 64
    config.state_length = 64
    anemll_model = Gemma3ForCausalLM(config)
    anemll_model.load_pretrained_weights(MODEL_PATH)

    # Get ANEMLL layer norm weight
    anemll_norm_weight = anemll_model.model.layers[0].input_layernorm.weight.clone()

    # Test input
    x = torch.randn(1, 4, config.hidden_size, dtype=torch.float32)

    # HF RMSNorm forward (Gemma uses (1 + weight) scaling)
    with torch.no_grad():
        hf_norm = hf_model.model.layers[0].input_layernorm
        hf_out = hf_norm(x)

    # ANEMLL RMSNorm forward
    with torch.no_grad():
        anemll_norm = anemll_model.model.layers[0].input_layernorm
        anemll_out = anemll_norm(x.to(torch.float32))

    hf_out = hf_out.float()
    anemll_out = anemll_out.float()

    diff = torch.abs(hf_out - anemll_out).max().item()
    mean_diff = torch.abs(hf_out - anemll_out).mean().item()

    print(f"  HF norm weight (first 5): {hf_norm_weight[:5].tolist()}")
    print(f"  ANEMLL norm weight (first 5): {anemll_norm_weight[:5].tolist()}")
    print(f"  Max diff: {diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    # Allow some tolerance for numerical differences
    passed = diff < 0.01
    if passed:
        print("  ✅ RMSNorm outputs match")
    else:
        print("  ❌ RMSNorm outputs don't match")

    return passed


def test_single_token_forward():
    """Test full forward pass for single token."""
    print("\n" + "=" * 60)
    print("Testing Single Token Forward Pass")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # NOTE: Gemma3's architecture produces large intermediate values (~100k)
    # that overflow float16 (max ~65504). We test using a manual float32 forward pass.
    # For ANE deployment, this needs to be addressed with mixed precision or scaling.

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load HF model
    print("  Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    hf_model.eval()

    # Load ANEMLL model (will create in float16 but we'll run in float32)
    print("  Loading ANEMLL model...")
    from anemll.models.gemma3_model import Gemma3Config, Gemma3ForCausalLM

    config = Gemma3Config()
    config.context_length = 64
    config.state_length = 64
    anemll_model = Gemma3ForCausalLM(config, disable_kv_cache=True)
    anemll_model.load_pretrained_weights(MODEL_PATH)
    anemll_model.eval()

    # Test input - simple prompt
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    print(f"  Input: '{prompt}'")
    print(f"  Token IDs: {input_ids.tolist()}")

    # HF forward
    with torch.no_grad():
        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits

    # ANEMLL forward - do it manually layer by layer in float32
    print("  Running ANEMLL forward (manual float32)...")
    with torch.no_grad():
        seq_len = input_ids.shape[1]

        # Embeddings (in float32)
        hidden_states = anemll_model.model.embed_tokens(input_ids).float()
        hidden_states = hidden_states * anemll_model.model.embedding_scale

        # Process each layer
        position_ids = torch.arange(seq_len, dtype=torch.int32)
        causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
        for i in range(seq_len):
            causal_mask[:, :, i, i+1:] = float('-inf')

        # Convert model to float32 for testing
        anemll_model = anemll_model.float()

        for layer_idx, layer in enumerate(anemll_model.model.layers):
            rotary_emb = anemll_model.model.get_rotary_embedding_prefill(position_ids, layer_idx)

            # Input layernorm
            normed = layer.input_layernorm(hidden_states)

            # Get Q, K, V
            hs = normed.permute(0, 2, 1).unsqueeze(2)
            query_states = layer.self_attn.q_proj(hs).view(1, layer.self_attn.num_heads, layer.self_attn.head_dim, seq_len).permute(0, 1, 3, 2)
            key_states = layer.self_attn.k_proj(hs).view(1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim, seq_len).permute(0, 1, 3, 2)
            value_states = layer.self_attn.v_proj(hs).view(1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim, seq_len).permute(0, 1, 3, 2)

            # Q/K norms
            query_states = layer.self_attn.q_norm(query_states)
            key_states = layer.self_attn.k_norm(key_states)

            # Apply rotary embeddings
            cos, sin = rotary_emb
            cos = cos.permute(0, 2, 1, 3).float()
            sin = sin.permute(0, 2, 1, 3).float()
            from anemll.models.gemma3_model import apply_rotary_pos_emb_prefill
            query_states, key_states = apply_rotary_pos_emb_prefill(query_states, key_states, cos, sin)

            # Repeat KV for GQA
            n_rep = layer.self_attn.num_heads // layer.self_attn.num_kv_heads
            if n_rep > 1:
                key_states = key_states.repeat(1, n_rep, 1, 1)
                value_states = value_states.repeat(1, n_rep, 1, 1)

            # Attention
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * layer.self_attn.scale
            attn_weights = attn_weights + causal_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)

            # Output projection
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(1, seq_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
            attn_output = attn_output.squeeze(2).permute(0, 2, 1)

            # Post attention norm + residual
            attn_output = layer.post_attention_layernorm(attn_output)
            hidden_states = hidden_states + attn_output

            # MLP
            residual = hidden_states
            hidden_states = layer.pre_feedforward_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = layer.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states

        # Final norm
        hidden_states = anemll_model.model.norm(hidden_states)

        # LM head
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2)
        logits_parts = []
        for i in range(1, 17):
            head = getattr(anemll_model, f"lm_head16_{i}")
            logits_parts.append(head(hs).squeeze(2).transpose(1, 2))
        anemll_logits = torch.cat(logits_parts, dim=-1)

    hf_logits = hf_logits.float()
    anemll_logits = anemll_logits.float()

    # Compare last position logits
    hf_last = hf_logits[0, -1, :]
    anemll_last = anemll_logits[0, -1, :]

    diff = torch.abs(hf_last - anemll_last).max().item()
    mean_diff = torch.abs(hf_last - anemll_last).mean().item()

    # Get top-5 predictions
    hf_top5 = hf_last.topk(5)
    anemll_top5 = anemll_last.topk(5)

    print(f"\n  HF top-5 tokens: {[tokenizer.decode([t]) for t in hf_top5.indices.tolist()]}")
    print(f"  HF top-5 IDs: {hf_top5.indices.tolist()}")
    print(f"  ANEMLL top-5 tokens: {[tokenizer.decode([t]) for t in anemll_top5.indices.tolist()]}")
    print(f"  ANEMLL top-5 IDs: {anemll_top5.indices.tolist()}")
    print(f"\n  Max logit diff: {diff:.4f}")
    print(f"  Mean logit diff: {mean_diff:.4f}")

    # Check if top prediction matches
    top_match = hf_top5.indices[0] == anemll_top5.indices[0]

    if top_match:
        print("  ✅ Top prediction matches")
    else:
        print("  ❌ Top prediction doesn't match")

    # Note about float16
    print("\n  ⚠️  NOTE: Gemma3 was trained in bfloat16 and produces large intermediate values.")
    print("     FP16 clamping is applied after residual connections for ANE compatibility.")

    return top_match


def run_all_tests():
    """Run all comparison tests."""
    print("\n" + "=" * 70)
    print("GEMMA3 vs HUGGINGFACE COMPARISON TESTS")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please download the model first.")
        return False

    tests = [
        ("Config Match", test_config_match),
        ("Weight Loading", test_weight_loading),
        ("Embedding Layer", test_embedding_layer),
        ("RMSNorm Layer", test_rmsnorm_layer),
        ("Single Token Forward", test_single_token_forward),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            print(f"❌ {name} test FAILED with exception: {e}")
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
        status = "✅ PASSED" if p else f"❌ FAILED: {error}" if error else "❌ FAILED"
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
