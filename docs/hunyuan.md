# HY-MT2 / Hunyuan Dense V1

ANEMLL recognizes Hugging Face `model_type: "hunyuan_v1_dense"` and
`architectures: ["HunYuanDenseV1ForCausalLM"]`. The initial target checkpoint is
[`tencent/Hy-MT2-1.8B`](https://huggingface.co/tencent/Hy-MT2-1.8B).
This contribution relates to [issue #47](https://github.com/Anemll/Anemll/issues/47),
which requests HY-MT1.5; HY-MT1.5 checkpoints have not been validated here.

## Convert

Download the original Hugging Face checkpoint, including its `.safetensors`,
`config.json`, `tokenizer.json` and `tokenizer_config.json`. A prequantized
MLX or GGUF checkpoint is not an input to this converter.

```bash
# Split embeddings, transformer chunks, and LM head.
./anemll/utils/convert_model.sh \
  --model /path/to/Hy-MT2-1.8B --output ./converted/hy-mt2 \
  --context 1024 --batch 64 --chunk 2 --lut2 4 --lut3 6

# Combined embeddings + transformer + LM head, with infer and prefill functions.
./anemll/utils/convert_monolith.sh \
  --model /path/to/Hy-MT2-1.8B --output ./converted/hy-mt2-monolithic \
  --context 1024 --batch 64 --lut 4 --lut-lmhead 6
```

Both scripts select `anemll.ane_converter.hunyuan_converter` and use the default
filename prefix `hunyuan`. `--prefix` overrides it. Exported tokenizer config
retains `hunyuan_v1_dense` and `PreTrainedTokenizerFast`; `meta.yaml` records the
16 LM-head splits, including the unequal split sizes for the 120,818-token
vocabulary.

The converter can also be invoked directly:

```bash
python -m anemll.ane_converter.hunyuan_converter \
  --model /path/to/Hy-MT2-1.8B --output ./converted/hy-mt2 \
  --part monolithic --context-length 1024 --batch-size 64 --lut 4
```

Parts `1`, `2`, `2_prefill`, `3`, `monolithic`, and `monolithic_prefill` follow
the existing converter naming and combination pipeline. Decode and prefill must
use the same context, padding and chunk configuration to share an `MLState`.

## State+1 compatibility option

`--state-pad 0` is the default. Set `--state-pad 1` explicitly when exporting
for a device affected by the full-extent KV-state write compiler failure:

```bash
./anemll/utils/convert_monolith.sh \
  --model /path/to/Hy-MT2-1.8B --output ./converted/hy-mt2-stateplus1 \
  --context 1024 --batch 64 --lut 4 --state-pad 1
```

The direct Python converter accepts the same `--state-pad 0|1` option. It adds
one storage row to each K/V cache, making `state_length = context_length + 1`.
Attention reads and cache writes use only `[0:context_length]`; the padding row
is never read or written. Input masks and visible context length stay unchanged.
Both functions export the same padded state shape, and `meta.yaml` records
`state_pad` and `state_length`.

The original integration reproduced an ANE load failure on macOS 27.0
(26A428), M4 Max, with a full-extent state write, and the additional row allowed
the model to load. A separate M1 machine was reported to work without padding;
its OS build was not recorded. See the associated
[Apple Developer Forums report](https://developer.apple.com/forums/thread/846497).
This is an explicit device compatibility option; the exporter does not select
it from the host OS or chip.

## Architecture and runtime contract

- Use `attention_head_dim` from the checkpoint; it need not be
  `hidden_size / num_attention_heads`.
- Apply alpha-NTK RoPE from `rope_scaling` (or newer `rope_parameters`), then
  query/key RMS normalization. The loader maps Hugging Face
  `query_layernorm` / `key_layernorm` weights to their ANEMLL modules.
- Preserve the checkpoint's RMS normalization rather than subtracting the
  feature mean; Hunyuan Q/K normalization and numerical reference tests require
  its original variance-only definition.
- Support tied input embeddings and LM-head weights and grouped-query attention.
  Projections use 1x1 convolutions, and the LM head is split into 16 outputs.
- Use fixed context/cache geometry. Decode performs position-masked cache
  updates; prefill supports successive fixed-size batches at `current_pos`.
  The existing `--dynamic-prefill-slice` and `--static-prefill-slice` flags are
  accepted for pipeline compatibility; Hunyuan uses its position-mask updates.
  Metadata advertises `position_mask_prefill: true`, which enables batched prefill
  without a caller-supplied update mask. Runtime causal masks use the visible
  context length even when the KV state includes an extra storage row.
- Clear the shared state between requests. Do not exceed the exported context;
  the original checkpoint's long-context limit does not enlarge the compiled
  cache.

The initial implementation targets the dense, SiLU, bias-free, RoPE and Q/K
normalization configuration used by HY-MT2-1.8B. CLA (`use_cla: true`), MoE,
attention biases and other activation functions are outside this port.
Use the checkpoint's official translation prompt and tokenizer template when
trying translations.

## Offline checks

```bash
python -m pytest -q tests/test_hunyuan*.py
```

Tests use tiny random checkpoints and exercise architecture numerics, checkpoint
loading, cached prefill/decode, state padding, Core ML conversion and LUT4
quantization, and shell routing/metadata. On macOS, the runtime check combines
prefill and decode functions and executes them against one shared `MLState`
using `CPU_ONLY`. Shell tests require `jq`; they replace only the heavyweight
converter call and execute the actual metadata-generation scripts.
