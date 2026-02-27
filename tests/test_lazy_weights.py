import pytest
import torch
from safetensors.torch import save_file

from anemll.models.lazy_weights import LazySafeTensorLoader, WeightLoadSpec
from anemll.models.qwen_model import QwenConfig, QwenForCausalLM


def _write_ckpt(tmp_path):
    tensors = {
        "model.embed_tokens.weight": torch.zeros((4, 8), dtype=torch.float16),
        "model.layers.0.self_attn.q_proj.weight": torch.ones((8, 8), dtype=torch.float16),
        "model.layers.1.self_attn.q_proj.weight": torch.full((8, 8), 2.0, dtype=torch.float16),
        "lm_head.weight": torch.full((4, 8), 3.0, dtype=torch.float16),
    }
    ckpt = tmp_path / "model.safetensors"
    save_file(tensors, str(ckpt))
    return ckpt


def test_include_exact_does_not_fall_back_to_all_keys(tmp_path):
    _write_ckpt(tmp_path)
    loader = LazySafeTensorLoader(str(tmp_path))

    spec = WeightLoadSpec(
        include_exact=frozenset({"lm_head.weight"}),
        description="exact-only",
    )
    state, stats = loader.load_state_dict(spec)

    assert list(state.keys()) == ["lm_head.weight"]
    assert stats.matched_keys == 1


def test_layer_range_filters_layer_tensors(tmp_path):
    _write_ckpt(tmp_path)
    loader = LazySafeTensorLoader(str(tmp_path))

    spec = WeightLoadSpec(
        include_prefixes=("model.layers.",),
        layer_range=(1, 2),
        strip_prefix="model.",
        description="layer-range",
    )
    state, stats = loader.load_state_dict(spec)

    assert list(state.keys()) == ["layers.1.self_attn.q_proj.weight"]
    assert stats.matched_keys == 1


def _tiny_qwen_config():
    return QwenConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        context_length=8,
        state_length=8,
    )


def test_chunk_local_model_initialization_skips_unused_modules():
    cfg = _tiny_qwen_config()
    model = QwenForCausalLM(
        cfg,
        build_embeddings=False,
        build_layers=True,
        build_norm=False,
        layer_range=(2, 4),
        build_kv_cache=True,
        build_lm_head=False,
    )

    assert model.model.embed_tokens is None
    assert model.model.norm is None
    assert model.model.active_layer_start == 2
    assert model.model.active_layer_end == 4
    assert len(model.model.layers) == 2
    assert not hasattr(model, "lm_head16_1")
    assert model.model.kv_cache_0.shape[0] == 2 * cfg.num_hidden_layers


def test_chunk_local_process_layers_rejects_outside_active_range():
    cfg = _tiny_qwen_config()
    model = QwenForCausalLM(
        cfg,
        build_embeddings=False,
        build_layers=True,
        build_norm=False,
        layer_range=(2, 4),
        build_kv_cache=True,
        build_lm_head=False,
    )

    hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16)
    position_ids = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, 1, cfg.state_length), dtype=torch.float16)
    current_pos = torch.zeros((1,), dtype=torch.int32)
    rotary = model.model.get_rotary_embeddings_s(current_pos)

    with pytest.raises(ValueError):
        model.model.process_layers(
            hidden_states,
            position_ids,
            causal_mask,
            current_pos,
            rotary,
            start_layer=0,
            end_layer=2,
            IN_PREFILL=False,
        )
