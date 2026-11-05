"""Offline regressions for HY-MT2 architecture, loading and state behavior."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from anemll.models.hunyuan_model import (
    HunyuanAttention,
    HunyuanConfig,
    HunyuanForCausalLM,
    HunyuanRMSNorm,
    HunyuanRotaryEmbedding,
)


def tiny_config(**overrides):
    options = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "attention_head_dim": 8,
        "num_hidden_layers": 2,
        "vocab_size": 67,
        "context_length": 16,
        "state_length": 16,
        "rope_scaling": {"type": "dynamic", "alpha": 1000.0},
    }
    options.update(overrides)
    return HunyuanConfig(**options)


def causal_mask(start, length, context=16):
    queries = torch.arange(start, start + length).unsqueeze(1)
    keys = torch.arange(context).unsqueeze(0)
    return torch.where(keys <= queries, 0.0, -10000.0).half()[None, None]


def run_batch(model, tokens, start=0, prefill=True):
    length = tokens.shape[1]
    return model(
        tokens,
        torch.zeros(1, 1, 16, 1).half(),
        torch.arange(start, start + length),
        causal_mask(start, length),
        torch.tensor([start]),
        IN_PREFILL=prefill,
    )


def checkpoint_state(model):
    state = {}
    for name, weight in model.state_dict().items():
        if (
            not name.startswith("model.")
            or "kv_cache_" in name
            or "rotary_emb.inv_freq" in name
        ):
            continue
        name = name.replace(".q_norm.", ".query_layernorm.").replace(
            ".k_norm.", ".key_layernorm."
        )
        if weight.ndim == 4:
            weight = weight[:, :, 0, 0]
        state[name] = weight.contiguous().clone()
    return state


def test_head_dim_and_nested_alpha_ntk():
    config = tiny_config(
        attention_head_dim=12,
        head_dim=8,
        rope_scaling=None,
        rope_parameters={"type": "dynamic", "alpha": 1000.0, "rope_theta": 10000.0},
    )
    assert config.head_dim == 12
    rotary = HunyuanRotaryEmbedding(config)
    base = 10000.0 * 1000.0 ** (12 / 10)
    expected = 1 / base ** (torch.arange(0, 12, 2).float() / 12)
    torch.testing.assert_close(rotary.inv_freq, expected)


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_cla": True},
        {"use_qk_norm": False},
        {"attention_bias": True},
        {"hidden_act": "gelu"},
        {"model_type": "hunyuan_moe"},
        {"state_length": 15},
        {"rope_scaling": {"type": "dynamic"}},
    ],
)
def test_reject_unsupported_checkpoint(overrides):
    with pytest.raises(ValueError):
        tiny_config(**overrides)


def test_norm_matches_rms_reference_with_nonzero_mean():
    norm = HunyuanRMSNorm(8)
    values = torch.randn(1, 3, 8) + 2
    expected = values * torch.rsqrt(
        values.square().mean(-1, keepdim=True) + norm.variance_epsilon
    )
    torch.testing.assert_close(norm(values), expected, atol=1e-6, rtol=1e-6)


def test_rope_precedes_nonuniform_qk_norm():
    attention = HunyuanAttention(tiny_config())
    with torch.no_grad():
        attention.q_norm.weight.copy_(torch.linspace(0.3, 1.7, 8))
        attention.k_norm.weight.copy_(torch.linspace(1.6, 0.4, 8))
    values = torch.randn(1, 1, 32).half()
    rotary = attention.rotary_emb(values, torch.tensor([7]))
    rotary = tuple(value.unsqueeze(1) for value in rotary)
    query, key, _ = attention.get_new_kv_cache(values, torch.tensor([7]), rotary)
    from anemll.models.hunyuan_model import apply_rotary_pos_emb_single

    projected = values.permute(0, 2, 1).unsqueeze(2)
    q = attention.q_proj(projected).view(1, 4, 1, 8)
    k = attention.k_proj(projected).view(1, 2, 1, 8)
    q_rotated, k_rotated = apply_rotary_pos_emb_single(q, k, *rotary)
    torch.testing.assert_close(query, attention.q_norm(q_rotated))
    torch.testing.assert_close(key, attention.k_norm(k_rotated))
    wrong, _ = apply_rotary_pos_emb_single(
        attention.q_norm(q), attention.k_norm(k), *rotary
    )
    assert not torch.allclose(query, wrong, atol=1e-3)


@pytest.mark.parametrize("state_pad", [0, 1])
def test_prefill_batches_decode_and_padding_are_equivalent(state_pad):
    torch.manual_seed(41)
    baseline = HunyuanForCausalLM(tiny_config()).eval().half()
    model = HunyuanForCausalLM(tiny_config(state_length=16 + state_pad)).eval().half()
    weights = {
        name: value
        for name, value in baseline.state_dict().items()
        if name != "model.kv_cache_0"
    }
    model.load_state_dict(weights, strict=False)
    if state_pad:
        model.model.kv_cache_0[:, :, 16:, :].fill_(17)
    tokens = torch.tensor([[2, 6, 8, 3, 5, 10]])
    with torch.no_grad():
        expected = run_batch(baseline, tokens)
        actual = torch.cat(
            [run_batch(model, tokens[:, :3]), run_batch(model, tokens[:, 3:], start=3)],
            dim=1,
        )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        model.model.kv_cache_0[:, :, :16],
        baseline.model.kv_cache_0,
        atol=2e-3,
        rtol=2e-3,
    )
    if state_pad:
        assert torch.all(model.model.kv_cache_0[:, :, 16:] == 17)
    model.model.kv_cache_0[:, :, :16].zero_()
    with torch.no_grad():
        decoded = torch.cat(
            [
                run_batch(
                    model, tokens[:, index : index + 1], start=index, prefill=False
                )
                for index in range(6)
            ],
            dim=1,
        )
    torch.testing.assert_close(decoded, expected, atol=3e-3, rtol=3e-3)
    if state_pad:
        assert torch.all(model.model.kv_cache_0[:, :, 16:] == 17)


def test_nonzero_prefill_survives_tracing():
    model = HunyuanForCausalLM(tiny_config(state_length=17)).eval().half()
    model.config.prefill_batch_size = 2

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, tokens, positions, mask, start):
            return self.model(
                tokens,
                torch.zeros(1, 1, 16, 1).half(),
                positions,
                mask,
                start,
                IN_PREFILL=True,
            )

    wrapper = Wrapper()
    inputs = (
        torch.tensor([[2, 4]]),
        torch.tensor([0, 1]),
        causal_mask(0, 2),
        torch.tensor([0]),
    )
    traced = torch.jit.trace(wrapper, inputs, check_trace=False)
    for module in (model, traced.model):
        module.model.kv_cache_0.zero_()
        module.model.kv_cache_0[:, :, 16:].fill_(17)
    with torch.no_grad():
        actual = traced(
            torch.tensor([[2, 4]]),
            torch.tensor([5, 6]),
            causal_mask(5, 2),
            torch.tensor([5]),
        )
        traced_cache = traced.model.model.kv_cache_0.clone()
        model.model.kv_cache_0[:, :, :16].zero_()
        expected = run_batch(model, torch.tensor([[2, 4]]), start=5)
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(traced_cache[:, :, :5]) == 0
    assert torch.count_nonzero(traced_cache[:, :, 5:7]) > 0
    assert torch.all(traced_cache[:, :, 16:] == 17)


@pytest.mark.parametrize("tied", [False, True])
def test_hf_checkpoint_mapping_and_split_head(tmp_path, tied):
    model = HunyuanForCausalLM(tiny_config(tie_word_embeddings=tied)).eval().half()
    state = checkpoint_state(model)
    head = state["model.embed_tokens.weight"] if tied else torch.randn(67, 32).half()
    if not tied:
        state["lm_head.weight"] = head
    save_file(state, str(tmp_path / "model.safetensors"))
    loaded = HunyuanForCausalLM(tiny_config(tie_word_embeddings=tied)).eval().half()
    assert loaded.load_pretrained_weights(str(tmp_path))
    joined = torch.cat(
        [
            getattr(loaded, f"lm_head16_{index+1}").weight[:, :, 0, 0]
            for index in range(16)
        ]
    )
    torch.testing.assert_close(joined, head)
    assert [getattr(loaded, f"lm_head16_{i+1}").out_channels for i in range(16)][
        :4
    ] == [5, 5, 5, 4]
    torch.testing.assert_close(
        loaded.model.layers[0].self_attn.q_norm.weight,
        state["model.layers.0.self_attn.query_layernorm.weight"],
    )


def test_missing_checkpoint_weight_fails(tmp_path):
    model = HunyuanForCausalLM(tiny_config()).half()
    state = checkpoint_state(model)
    del state["model.layers.0.self_attn.q_proj.weight"]
    save_file(state, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="q_proj.weight"):
        model.load_pretrained_weights(str(tmp_path))


def test_tiny_huggingface_logits_parity(tmp_path):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "HunYuanDenseV1ForCausalLM"):
        pytest.skip("Installed transformers does not include HunYuanDenseV1")
    # A fixed alpha-NTK base is also expressible as default RoPE with a
    # rescaled theta, which keeps this test independent of HF config drift.
    theta = 10000.0 * 1000.0 ** (8 / 6)
    config = transformers.HunYuanDenseV1Config(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=67,
        tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": theta},
    )
    torch.manual_seed(42)
    reference = transformers.HunYuanDenseV1ForCausalLM(config).eval().half()
    save_file(
        {
            name: value.clone().contiguous()
            for name, value in reference.state_dict().items()
        },
        str(tmp_path / "model.safetensors"),
    )
    model = HunyuanForCausalLM(tiny_config()).eval().half()
    model.load_pretrained_weights(str(tmp_path))
    tokens = torch.tensor([[1, 3, 5, 6, 2, 4]])
    with torch.no_grad():
        expected = reference(tokens, use_cache=False).logits
        actual = run_batch(model, tokens)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))


@pytest.mark.parametrize("prefill", [True, False])
def test_chunked_layer_ranges_preserve_states_and_padding(prefill):
    torch.manual_seed(15)
    model = HunyuanForCausalLM(tiny_config(state_length=17)).eval().half()
    tokens = torch.tensor([[2, 3]]) if prefill else torch.tensor([[2]])
    start = 3
    positions = torch.arange(start, start + tokens.shape[1])
    mask = causal_mask(start, tokens.shape[1])
    current = torch.tensor([start])
    hidden = model.model.embed_tokens(tokens)
    rotary = (
        model.model.get_rotary_embedding_prefill(positions)
        if prefill
        else model.model.get_rotary_embeddings_s(current)
    )
    model.model.kv_cache_0[:, :, 16:].fill_(99)
    with torch.no_grad():
        expected = model.model.process_layers(
            hidden, positions, mask, current, rotary, IN_PREFILL=prefill
        )
        expected_cache = model.model.kv_cache_0.clone()
        model.model.kv_cache_0[:, :, :16].zero_()
        partial = model.model.process_layers(
            hidden,
            positions,
            mask,
            current,
            rotary,
            start_layer=0,
            end_layer=1,
            IN_PREFILL=prefill,
        )
        actual = model.model.process_layers(
            partial,
            positions,
            mask,
            current,
            rotary,
            start_layer=1,
            end_layer=2,
            IN_PREFILL=prefill,
        )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.model.kv_cache_0, expected_cache)
    assert torch.all(model.model.kv_cache_0[:, :, 16:] == 99)
