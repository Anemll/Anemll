#!/usr/bin/env python3

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/tmp/test-qwen25-0.5b-base', use_fast=False)

with open('/tmp/boolq_exact_prompt.txt', 'r') as f:
    prompt = f.read().strip()

# Test both tokenization methods
chat_tokens = tokenizer(prompt, return_tensors='pt', add_special_tokens=True).input_ids
harness_tokens = tokenizer.encode(prompt, add_special_tokens=True)

print(f'Chat method: {chat_tokens.shape[1]} tokens')
print(f'Harness method: {len(harness_tokens)} tokens')
print(f'Difference: {chat_tokens.shape[1] - len(harness_tokens)} tokens')

if chat_tokens.shape[1] != len(harness_tokens):
    print('\nToken count mismatch!')
    chat_list = chat_tokens[0].tolist()
    print(f'Chat first 5: {chat_list[:5]}')
    print(f'Chat last 5: {chat_list[-5:]}')
    print(f'Harness first 5: {harness_tokens[:5]}')
    print(f'Harness last 5: {harness_tokens[-5:]}')
else:
    print('Token counts match!')