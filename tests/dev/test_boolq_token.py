#!/usr/bin/env python3

from transformers import AutoTokenizer

# Test what tokens the model should predict
tokenizer = AutoTokenizer.from_pretrained('/tmp/test-qwen25-0.5b-base', use_fast=False)

# Check token encodings
print("Token encodings:")
print(f"' no' -> {tokenizer.encode(' no', add_special_tokens=False)}")
print(f"' yes' -> {tokenizer.encode(' yes', add_special_tokens=False)}")
print(f"' ' -> {tokenizer.encode(' ', add_special_tokens=False)}")
print(f"'no' -> {tokenizer.encode('no', add_special_tokens=False)}")
print(f"'yes' -> {tokenizer.encode('yes', add_special_tokens=False)}")

# Test decoding
print("\nToken decodings:")
print(f"902 -> '{tokenizer.decode([902])}'")
print(f"9834 -> '{tokenizer.decode([9834])}'") 
print(f"220 -> '{tokenizer.decode([220])}'")

# Test what comes after "Answer:"
prompt = "Question: does ethanol take more energy make that produces?\nAnswer:"
print(f"\nPrompt: {prompt}")
print("Expected continuations in BoolQ:")
print("1. 'Answer: no' (direct answer)")
print("2. 'Answer: yes' (direct answer)")
print("Current prediction: 'Answer: ' (space, then presumably 'no' or 'yes')")