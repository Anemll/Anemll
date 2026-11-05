"""Verify runtime prefill capabilities and visible masks without loading a model."""

from types import SimpleNamespace

import pytest

from tests.chat import _prefill_options, initialize_causal_mask, load_metadata


@pytest.mark.parametrize("allow_dynamic_slice", [False, True])
@pytest.mark.parametrize("state_pad", [0, 1])
def test_position_mask_prefill_uses_batches_and_visible_context(
    allow_dynamic_slice, state_pad
):
    model = SimpleNamespace(
        user_defined_metadata={
            "com.anemll.context_length": "16",
            "com.anemll.state_length": str(16 + state_pad),
            "com.anemll.position_mask_prefill": "true",
        }
    )
    metadata = load_metadata(model, SimpleNamespace(eval=True))
    single_token, use_update_mask, mask_len = _prefill_options(
        metadata, allow_dynamic_slice
    )
    assert single_token is False
    assert use_update_mask is False
    assert metadata["state_length"] == 16 + state_pad
    assert mask_len == 16
    assert initialize_causal_mask(mask_len, eval_mode=True).shape[-1] == 16


@pytest.mark.parametrize(
    "flags,allow_dynamic_slice,expected",
    [
        ({}, False, (True, False, 17)),
        ({"update_mask_prefill": True}, False, (False, True, 17)),
        ({"prefill_dynamic_slice": True}, False, (True, False, 17)),
        ({"prefill_dynamic_slice": True}, True, (False, False, 17)),
    ],
)
def test_existing_prefill_capability_selection(flags, allow_dynamic_slice, expected):
    metadata = {"context_length": 16, "state_length": 17, **flags}
    assert _prefill_options(metadata, allow_dynamic_slice) == expected


def test_yaml_prefill_capability_overrides_model_metadata():
    model = SimpleNamespace(user_defined_metadata={"com.anemll.context_length": "16"})
    metadata = load_metadata(
        model, SimpleNamespace(eval=True, position_mask_prefill=True)
    )
    assert _prefill_options(metadata) == (False, False, 16)
