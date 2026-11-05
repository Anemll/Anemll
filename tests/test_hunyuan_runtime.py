"""Execute combined prefill/decode functions against one shared Core ML state."""

from __future__ import annotations

import platform

import numpy as np
import pytest
import torch

ct = pytest.importorskip("coremltools")

from anemll.ane_converter.hunyuan_converter import HunyuanConverter
from anemll.models.hunyuan_model import HunyuanConfig, HunyuanForCausalLM


@pytest.mark.skipif(
    platform.system() != "Darwin", reason="Core ML runtime requires macOS"
)
def test_shared_state_prefill_decode_and_padding(tmp_path):
    torch.manual_seed(21)
    model = (
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
                state_length=17,
            )
        )
        .eval()
        .half()
    )
    converter = HunyuanConverter(model, context_length=16, batch_size=2, lut_bits=None)
    for part, name in [("monolithic", "infer"), ("monolithic_prefill", "prefill")]:
        converter.convert(part).save(str(tmp_path / (name + ".mlpackage")))

    descriptor = ct.utils.MultiFunctionDescriptor()
    descriptor.add_function(str(tmp_path / "infer.mlpackage"), "main", "infer")
    descriptor.add_function(str(tmp_path / "prefill.mlpackage"), "main", "prefill")
    descriptor.default_function_name = "infer"
    combined = str(tmp_path / "combined.mlpackage")
    ct.utils.save_multifunction(descriptor, combined)
    infer = ct.models.MLModel(
        combined, function_name="infer", compute_units=ct.ComputeUnit.CPU_ONLY
    )
    prefill = ct.models.MLModel(
        combined, function_name="prefill", compute_units=ct.ComputeUnit.CPU_ONLY
    )
    state = infer.make_state()
    state_name = infer.get_spec().description.functions[0].state[0].name
    cache = state.read_state(state_name).copy()
    cache[:, :, 16, :] = 99
    # The Python MLState bridge accepts float32 arrays for write_state, including
    # when the underlying model state uses float16 storage.
    state.write_state(state_name, cache.astype(np.float32))
    model.model.kv_cache_0.zero_()
    model.model.kv_cache_0[:, :, 16, :] = 99
    tokens = np.array([[2, 6, 9, 4, 8]], dtype=np.int32)

    for start, length, compiled, is_prefill in [
        (0, 2, prefill, True),
        (2, 2, prefill, True),
        (4, 1, infer, False),
    ]:
        positions = np.arange(start, start + length, dtype=np.int32)
        mask = np.where(np.arange(16)[None, :] <= positions[:, None], 0.0, -10000.0)
        mask = mask.astype(np.float16)[None, None]
        inputs = {
            "input_ids": tokens[:, start : start + length],
            "position_ids": positions,
            "causal_mask": mask,
            "current_pos": np.array([start], dtype=np.int32),
        }
        output = compiled.predict(inputs, state=state)
        actual = np.concatenate(
            [output["logits" + str(i)] for i in range(1, 17)], axis=-1
        )
        with torch.no_grad():
            expected = model(
                torch.from_numpy(inputs["input_ids"]),
                None,
                torch.from_numpy(positions),
                torch.from_numpy(mask),
                torch.tensor([start]),
                IN_PREFILL=is_prefill,
            ).numpy()
        np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0.05)
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
        assert np.all(state.read_state(state_name)[:, :, 16, :] == 99)
