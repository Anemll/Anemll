#!/usr/bin/env python3
"""
Test Nanbeige4.1-3B tool calling with live URL fetching on ANE.

The model decides which tool to call, we execute it, feed the result back,
and the model summarizes the answer.

Usage:
    python tests/dev/test_tool_call_live.py --meta /Volumes/Models/ANE/nanbeige41_3b_lut64_1024/meta.yaml
    python tests/dev/test_tool_call_live.py --meta /Volumes/Models/ANE/nanbeige41_3b_lut64_2048/meta.yaml \
        --question "What is the weather in Tokyo?"
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error


# ── Project root ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CHAT_PY = os.path.join(PROJECT_ROOT, "tests", "chat.py")


# ── Tool definitions (what the model sees) ──────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco'"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

TOOLS_JSON = "\n".join(json.dumps(t) for t in TOOLS)


# ── Tool implementations (actual URL calls) ─────────────────────────────────

def get_weather(location: str) -> str:
    """Fetch weather via open-meteo (free, no API key, reliable)."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.request.quote(location)}&count=1"
        print(f"  [Tool] GET {geo_url}")
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo = json.loads(resp.read())
        if not geo.get("results"):
            return json.dumps({"error": f"Location '{location}' not found"})
        r = geo["results"][0]
        lat, lon = r["latitude"], r["longitude"]
        wx_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
            f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
        )
        print(f"  [Tool] GET {wx_url}")
        with urllib.request.urlopen(wx_url, timeout=10) as resp:
            wx = json.loads(resp.read())
        cur = wx["current"]
        # Map WMO weather codes to descriptions
        wmo = {0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
               45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle",
               55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
               71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
               95: "Thunderstorm"}
        code = cur.get("weather_code", -1)
        condition = wmo.get(code, f"code {code}")
        return json.dumps({
            "location": location,
            "temp_f": cur["temperature_2m"],
            "humidity": cur["relative_humidity_2m"],
            "condition": condition,
            "wind_mph": cur["wind_speed_10m"],
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_DISPATCH = {
    "get_weather": lambda args: get_weather(args["location"]),
}


# ── Build prompts (raw ChatML, no tokenizer needed) ─────────────────────────

def build_tool_prompt(user_question: str) -> str:
    """Build ChatML prompt with tool definitions."""
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant with access to tools.\n\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n" + TOOLS_JSON + "\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call><|im_end|>\n"
        "<|im_start|>user\n" + user_question + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_followup_prompt(user_question: str, tool_name: str,
                          tool_args_json: str, tool_result: str) -> str:
    """Build prompt with tool response for the model to summarize."""
    tool_call_block = (
        "<tool_call>\n"
        f'{{"name": "{tool_name}", "arguments": {tool_args_json}}}\n'
        "</tool_call>"
    )
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant with access to tools.\n\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n" + TOOLS_JSON + "\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call><|im_end|>\n"
        "<|im_start|>user\n" + user_question + "<|im_end|>\n"
        "<|im_start|>assistant\n" + tool_call_block + "<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n" + tool_result + "\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n</think>\n\n"
    )


# ── Run inference via chat.py subprocess ─────────────────────────────────────

def run_inference(meta_path: str, prompt: str, max_tokens: int = 150,
                  verbose: bool = False) -> str:
    """Run chat.py with --eval --nw --no-template and capture clean output."""
    cmd = [
        sys.executable, CHAT_PY,
        "--meta", meta_path,
        "--no-template",
        "--eval", "--nw",
        "--max-tokens", str(max_tokens),
        "--prompt", prompt,
    ]
    if verbose:
        print(f"  [Running: chat.py --eval --nw --max-tokens {max_tokens}]")
    else:
        print(f"  [Running chat.py with {max_tokens} max tokens...]")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
        print(f"  [STDERR] {stderr_tail}")
    return result.stdout.strip()


# ── Parse tool call from model output ────────────────────────────────────────

def parse_tool_call(text: str):
    """Extract tool name and arguments from model output."""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not match:
        return None, None
    try:
        call = json.loads(match.group(1))
        return call.get("name"), call.get("arguments", {})
    except json.JSONDecodeError:
        return None, None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test tool calling with live URL fetches on ANE")
    parser.add_argument("--meta", required=True, help="Path to meta.yaml")
    parser.add_argument("--question", default="What is the current weather in San Francisco?",
                        help="Question to ask the model")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per generation")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full chat.py output (model loading, prefill, token streaming)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Nanbeige4.1-3B Tool Call Test (ANE)")
    print(f"{'='*60}")
    print(f"  Meta:     {args.meta}")
    print(f"  Question: {args.question}")
    print(f"  Verbose:  {args.verbose}")
    print()

    # ── Step 1: First inference — model picks a tool ──
    print("[Step 1] Building tool-call prompt...")
    prompt = build_tool_prompt(args.question)
    print(f"  Prompt length: ~{len(prompt.split())} words")
    print()

    print("[Step 2] Running first inference (model decides tool call)...")
    output = run_inference(args.meta, prompt, args.max_tokens, args.verbose)
    print(f"\n  Model output:")
    print(f"  {'─'*50}")
    for line in output.split('\n'):
        print(f"  {line}")
    print(f"  {'─'*50}\n")

    # ── Step 2: Parse tool call ──
    tool_name, tool_args = parse_tool_call(output)
    if not tool_name:
        print("[!] No <tool_call> found. Model may have answered directly.")
        return

    print(f"[Step 3] Parsed: {tool_name}({json.dumps(tool_args)})")

    # ── Step 3: Execute tool (live URL) ──
    if tool_name not in TOOL_DISPATCH:
        print(f"  [!] Unknown tool: {tool_name}")
        return

    print(f"\n[Step 4] Executing '{tool_name}' (live HTTP call)...")
    tool_result = TOOL_DISPATCH[tool_name](tool_args)
    print(f"  Result: {tool_result}\n")

    # ── Step 4: Feed result back, get final answer ──
    print("[Step 5] Running second inference (model summarizes result)...")
    followup = build_followup_prompt(
        args.question, tool_name,
        json.dumps(tool_args), tool_result
    )
    final_output = run_inference(args.meta, followup, args.max_tokens, args.verbose)

    # Strip thinking for clean display
    clean = re.sub(r"<think>.*?</think>", "", final_output, flags=re.DOTALL).strip()
    print(f"\n  Final answer:")
    print(f"  {'─'*50}")
    for line in clean.split('\n'):
        print(f"  {line}")
    print(f"  {'─'*50}")
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
