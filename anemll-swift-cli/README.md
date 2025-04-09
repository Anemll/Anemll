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

# Example running model from anemll.anemll-chat.demo container
swift run -c release anemllcli --meta ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/Models/llama_3_2_1b_iosv2_0/meta.yaml --prompt "List US Presidents"

# Example running model from anemll.anemll-chat.demo container and save output to file ( for automated test)
swift run -c release anemllcli --meta ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/Models/llama_3_2_1b_iosv2_0/meta.yaml --prompt "who are you" --save /tmp/chat.txt

## License

This project is part of the ANEMLL framework. See the LICENSE file for details. 