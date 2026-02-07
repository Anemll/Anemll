# ANEMLL Swift CLI

Reference implementation of ANEMLL inference engine in Swift.

## Documentation

For detailed documentation about usage, installation, and features, please see:
[Swift CLI Documentation](../docs/swift_cli.md)

## Quick Start

```bash
# Build the CLI
swift build -c release

# Run with a model
swift run -c release anemllcli --meta <path_to_model>/meta.yaml

# Get help
swift run -c release anemllcli --help
```

Example running model from `anemll.anemll-chat.demo` container:

```bash
swift run -c release anemllcli \
  --meta ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/Models/llama_3_2_1b_iosv2_0/meta.yaml \
  --prompt "List US Presidents"
```

Example with saved output (for automated tests):

```bash
swift run -c release anemllcli \
  --meta ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/Models/llama_3_2_1b_iosv2_0/meta.yaml \
  --prompt "who are you" \
  --save /tmp/chat.txt
```

## Divergence Debug Run

Use these flags when reproducing state/KV divergence:

```bash
swift run -c release anemllcli \
  --meta <path_to_model>/meta.yaml \
  --prompt "what is apple neural engine?" \
  --temperature 0 \
  --debug-level 1 \
  --debug-single-token-prefill \
  --debug-disable-io-backings \
  --debug-repeat-infer-count 2 \
  --debug-compare-kv-state-every-token true \
  --debug-predict-read-delay-ms 0
```

`--debug-predict-read-delay-ms` accepts fractional milliseconds. Useful sweep points:
`0`, `0.3`, `0.5`, `1`, `2`, `3`, `5`, `8`, `10`.

Optional noise reduction:

```bash
--debug-repeat-only-divergence
```

The same divergence flags are available in `anemllcli_adv`.

## License

This project is part of the ANEMLL framework. See the LICENSE file for details. 
