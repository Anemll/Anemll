#!/usr/bin/env python3
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)

print('Expected tokens:')
for token_id in [2585, 1558, 432, 975, 1939]:
    try:
        text = tokenizer.decode([token_id])
        print(f'  {token_id}: "{text}"')
    except:
        print(f'  {token_id}: <decode_error>')

print('\nActual tokens:')        
for token_id in [1532, 451]:
    try:
        text = tokenizer.decode([token_id])
        print(f'  {token_id}: "{text}"')
    except:
        print(f'  {token_id}: <decode_error>')

print('\nPrompt tokens:')
for token_id in [3838, 374, 8162, 60477, 8200, 30]:
    try:
        text = tokenizer.decode([token_id])
        print(f'  {token_id}: "{text}"')
    except:
        print(f'  {token_id}: <decode_error>') 