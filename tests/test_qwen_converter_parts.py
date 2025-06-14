import pytest
import coremltools as ct
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from anemll.ane_converter.qwen_converter import QwenConverter
from anemll.models.qwen_model import QwenConfig, QwenForCausalLM, TEST_DEVICE

pytestmark = pytest.mark.skip(
    reason="Core ML runtime unavailable in Codex VM; test is compile-only."
)


def make_model():
    config = QwenConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=8,
        vocab_size=100,
    )
    model = QwenForCausalLM(config, enable_coreml=True).to(TEST_DEVICE)
    model.eval()
    return model


def test_convert_part_1():
    model = make_model()
    converter = QwenConverter(model, lut_bits=None)
    mlmodel = converter.convert(part="1")
    assert isinstance(mlmodel, ct.models.MLModel)


def test_convert_part_2():
    model = make_model()
    converter = QwenConverter(model, lut_bits=4)
    mlmodel = converter.convert(part="2")
    if isinstance(mlmodel, list):
        mlmodel = mlmodel[0]
    assert isinstance(mlmodel, ct.models.MLModel)


def test_convert_part_2_prefill():
    model = make_model()
    converter = QwenConverter(model, lut_bits=4)
    mlmodel = converter.convert(part="2_prefill")
    if isinstance(mlmodel, list):
        mlmodel = mlmodel[0]
    assert isinstance(mlmodel, ct.models.MLModel)


def test_convert_part_3():
    model = make_model()
    converter = QwenConverter(model, lut_bits=6)
    mlmodel = converter.convert(part="3")
    assert isinstance(mlmodel, ct.models.MLModel)
