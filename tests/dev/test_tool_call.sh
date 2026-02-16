#!/bin/bash
# Test Nanbeige4.1-3B tool calling on ANE
# Usage: bash tests/dev/test_tool_call.sh [meta.yaml path]

META="${1:-/Volumes/Models/ANE/nanbeige41_3b_lut64_2048/meta.yaml}"
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

PROMPT='<|im_start|>system
You are a helpful assistant with access to tools.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What is the weather in San Francisco?<|im_end|>
<|im_start|>assistant
'

echo "Using meta: $META"
echo "---"

python3 "$SCRIPT_DIR/tests/chat.py" \
  --meta "$META" \
  --no-template \
  --max-tokens 200 \
  --prompt "$PROMPT"
