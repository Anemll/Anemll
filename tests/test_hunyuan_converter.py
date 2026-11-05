"""Core ML state contracts and offline tiny-model export regressions."""

from __future__ import annotations

import platform
import sys
from unittest.mock import patch

import pytest

ct = pytest.importorskip("coremltools")

from anemll.ane_converter.hunyuan_converter import HunyuanConverter, parse_args
from anemll.models.hunyuan_model import HunyuanConfig, HunyuanForCausalLM


def tiny_model(state_pad=0):
    return (
        HunyuanForCausalLM(
            HunyuanConfig(
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=4,
                num_key_value_heads=2,
                attention_head_dim=16,
                num_hidden_layers=2,
                vocab_size=67,
                context_length=16,
                state_length=16 + state_pad,
            )
        )
        .eval()
        .half()
    )


@pytest.mark.parametrize("state_pad", [0, 1])
def test_state_shape_uses_explicit_head_dim_and_padding(state_pad):
    model = tiny_model(state_pad)
    converter = HunyuanConverter(model, context_length=16, batch_size=2, lut_bits=None)
    assert converter.head_dim == 16
    states = converter.GetTransformerStates(model)
    assert states[0].wrapped_type.shape.shape == (4, 2, 16 + state_pad, 16)


@pytest.mark.parametrize("value", ["0", "1"])
def test_state_pad_cli(value):
    with patch.object(sys, "argv", ["hunyuan_converter", "--state-pad", value]):
        assert parse_args().state_pad == int(value)


def test_state_pad_defaults_to_zero():
    with patch.object(sys, "argv", ["hunyuan_converter"]):
        assert parse_args().state_pad == 0


@pytest.mark.skipif(
    platform.system() != "Darwin", reason="Core ML runtime requires macOS"
)
@pytest.mark.parametrize(
    "part", ["1", "2", "2_prefill", "3", "all", "monolithic", "monolithic_prefill"]
)
def test_tiny_export(part):
    model = tiny_model(state_pad=1)
    converter = HunyuanConverter(model, context_length=16, batch_size=2, lut_bits=None)
    converted = converter.convert(part=part)
    models = converted if isinstance(converted, list) else [converted]
    for mlmodel in models:
        spec = mlmodel.get_spec()
        assert spec.WhichOneof("Type") == "mlProgram"
        assert (
            mlmodel.user_defined_metadata["com.anemll.position_mask_prefill"] == "true"
        )
        assert mlmodel.user_defined_metadata["com.anemll.state_pad"] == "1"
        if part in ("2", "2_prefill", "monolithic", "monolithic_prefill"):
            state = spec.description.state[0]
            assert list(state.type.stateType.arrayType.shape) == [4, 2, 17, 16]


@pytest.mark.skipif(
    platform.system() != "Darwin", reason="Core ML runtime requires macOS"
)
def test_tiny_lut4_export():
    model = tiny_model(state_pad=1)
    converter = HunyuanConverter(
        model, context_length=16, batch_size=2, lut_bits=4, per_channel=8
    )
    mlmodel = converter.convert(part="2")
    operations = mlmodel._mil_program.functions["main"].operations
    assert any(op.op_type.startswith("constexpr_lut") for op in operations)


def test_quantization_failure_is_not_silenced(monkeypatch):
    from anemll.ane_converter import hunyuan_converter

    converter = HunyuanConverter(
        tiny_model(), context_length=16, batch_size=2, lut_bits=4
    )
    converter.converted_model = object()

    def fail(*args, **kwargs):
        raise RuntimeError("quantization failed")

    monkeypatch.setattr(hunyuan_converter.cto.coreml, "palettize_weights", fail)
    with pytest.raises(RuntimeError, match="quantization failed"):
        converter.postprocess()


@pytest.mark.skipif(
    platform.system() != "Darwin", reason="Core ML runtime requires macOS"
)
@pytest.mark.parametrize("part", ["2", "2_prefill"])
def test_chunked_exports_use_identical_padded_state(part):
    converter = HunyuanConverter(
        tiny_model(state_pad=1),
        context_length=16,
        batch_size=2,
        lut_bits=None,
        num_chunks=2,
    )
    models = converter.convert(part=part)
    assert len(models) == 2
    states = [mlmodel.get_spec().description.state[0] for mlmodel in models]
    assert states[0].name == states[1].name
    assert all(
        list(state.type.stateType.arrayType.shape) == [4, 2, 17, 16] for state in states
    )
