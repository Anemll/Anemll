#!/usr/bin/env python3
"""
Generate meta.yaml with correct LUT values based on actual file existence
"""

import sys
import os

def parse_lut_value(lut_str):
    """Parse LUT string to extract bits and per_channel values.

    Args:
        lut_str: String like "6" or "6,4" or "none"

    Returns:
        tuple: (bits, per_channel, original_string)
    """
    if lut_str == 'none' or not lut_str:
        return 'none', None, 'none'

    if ',' in lut_str:
        parts = lut_str.split(',')
        bits = parts[0]
        per_channel = int(parts[1]) if len(parts) > 1 else None
        return bits, per_channel, lut_str
    else:
        return lut_str, None, lut_str

def check_file_exists(output_dir, base_name, lut_value):
    """Check if file exists and return actual name and LUT value"""
    if lut_value == 'none':
        return base_name, 'none', None

    # Check if LUT version exists (only use bits for filename)
    lut_name = f'{base_name}_lut{lut_value}'
    if os.path.exists(os.path.join(output_dir, f'{lut_name}.mlmodelc')):
        return lut_name, lut_value, None
    else:
        # Fallback to non-LUT version
        print(f"Warning: {lut_name}.mlmodelc not found, using {base_name}.mlmodelc instead")
        return base_name, 'none', None

def main():
    if len(sys.argv) != 11:
        print("Usage: python3 generate_meta_yaml.py <model_name> <context> <batch> <lut_emb> <lut_ffn> <lut_lmh> <num_chunks> <prefix> <arch> <output_dir>")
        sys.exit(1)

    MODEL_NAME = sys.argv[1]
    CONTEXT = sys.argv[2]
    BATCH = sys.argv[3]
    LUT_EMB = sys.argv[4]
    LUT_FFN = sys.argv[5]
    LUT_LMH = sys.argv[6]
    NUM_CHUNKS = sys.argv[7]
    PREFIX = sys.argv[8]
    ARCH = sys.argv[9]
    OUTPUT_DIR = sys.argv[10]

    # Parse LUT values to extract bits and per_channel
    lut_emb_bits, lut_emb_per_channel, _ = parse_lut_value(LUT_EMB)
    lut_ffn_bits, lut_ffn_per_channel, _ = parse_lut_value(LUT_FFN)
    lut_lmh_bits, lut_lmh_per_channel, _ = parse_lut_value(LUT_LMH)

    # Check which files actually exist and adjust LUT values accordingly (use only bits for filenames)
    embeddings_base = f'{PREFIX}_embeddings'
    embeddings_name, lut_emb_actual, _ = check_file_exists(OUTPUT_DIR, embeddings_base, lut_emb_bits)

    lmhead_base = f'{PREFIX}_lm_head'
    lmhead_name, lut_lmh_actual, _ = check_file_exists(OUTPUT_DIR, lmhead_base, lut_lmh_bits)

    # Check FFN (always use LUT if specified, as it's required for ANE) - use only bits for filename
    ffn_base = f'{PREFIX}_FFN_PF'
    ffn_name = f'{ffn_base}_lut{lut_ffn_bits}' if lut_ffn_bits != 'none' else ffn_base
    
    # Add .mlmodelc extension to model paths
    embeddings_path = f'{embeddings_name}.mlmodelc'
    lmhead_path = f'{lmhead_name}.mlmodelc'
    ffn_path = f'{ffn_name}.mlmodelc'
    
    # Set split_lm_head based on architecture
    split_lm_head = 16 if ARCH.startswith('qwen') else 8

    # Build metadata with per_channel info if available
    meta_parts = [f'''model_info:
  name: anemll-{MODEL_NAME}-ctx{CONTEXT}
  version: 0.3.4
  description: |
    Demonstarates running {MODEL_NAME} on Apple Neural Engine
    Context length: {CONTEXT}
    Batch size: {BATCH}
    Chunks: {NUM_CHUNKS}
  license: MIT
  author: Anemll
  framework: Core ML
  language: Python
  architecture: {ARCH}
  parameters:
    context_length: {CONTEXT}
    batch_size: {BATCH}
    lut_embeddings: {lut_emb_actual}''']

    if lut_emb_per_channel is not None:
        meta_parts.append(f'    lut_embeddings_per_channel: {lut_emb_per_channel}')

    meta_parts.append(f'    lut_ffn: {lut_ffn_bits}')
    if lut_ffn_per_channel is not None:
        meta_parts.append(f'    lut_ffn_per_channel: {lut_ffn_per_channel}')

    meta_parts.append(f'    lut_lmhead: {lut_lmh_actual}')
    if lut_lmh_per_channel is not None:
        meta_parts.append(f'    lut_lmhead_per_channel: {lut_lmh_per_channel}')

    meta_parts.append(f'''    num_chunks: {NUM_CHUNKS}
    model_prefix: {PREFIX}
    embeddings: {embeddings_path}
    lm_head: {lmhead_path}
    ffn: {ffn_path}
    split_lm_head: {split_lm_head}
''')

    meta = '\n'.join(meta_parts)

    output_file = os.path.join(OUTPUT_DIR, 'meta.yaml')
    with open(output_file, 'w') as f:
        f.write(meta)

    print(f"Generated meta.yaml at: {output_file}")
    print(f"  lut_embeddings: {lut_emb_actual}" + (f" (per_channel: {lut_emb_per_channel})" if lut_emb_per_channel else ""))
    print(f"  lut_ffn: {lut_ffn_bits}" + (f" (per_channel: {lut_ffn_per_channel})" if lut_ffn_per_channel else ""))
    print(f"  lut_lmhead: {lut_lmh_actual}" + (f" (per_channel: {lut_lmh_per_channel})" if lut_lmh_per_channel else ""))

if __name__ == "__main__":
    main() 