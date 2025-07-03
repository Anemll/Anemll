#!/usr/bin/env python3
"""
Generate meta.yaml with correct LUT values based on actual file existence
"""

import sys
import os

def check_file_exists(output_dir, base_name, lut_value):
    """Check if file exists and return actual name and LUT value"""
    if lut_value == 'none':
        return base_name, 'none'
    
    # Check if LUT version exists
    lut_name = f'{base_name}_lut{lut_value}'
    if os.path.exists(os.path.join(output_dir, f'{lut_name}.mlmodelc')):
        return lut_name, lut_value
    else:
        # Fallback to non-LUT version
        print(f"Warning: {lut_name}.mlmodelc not found, using {base_name}.mlmodelc instead")
        return base_name, 'none'

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
    
    # Check which files actually exist and adjust LUT values accordingly
    embeddings_base = f'{PREFIX}_embeddings'
    embeddings_name, lut_emb_actual = check_file_exists(OUTPUT_DIR, embeddings_base, LUT_EMB)
    
    lmhead_base = f'{PREFIX}_lm_head'
    lmhead_name, lut_lmh_actual = check_file_exists(OUTPUT_DIR, lmhead_base, LUT_LMH)
    
    # Check FFN (always use LUT if specified, as it's required for ANE)
    ffn_base = f'{PREFIX}_FFN_PF'
    ffn_name = f'{ffn_base}_lut{LUT_FFN}' if LUT_FFN != 'none' else ffn_base
    
    # Add .mlmodelc extension to model paths
    embeddings_path = f'{embeddings_name}.mlmodelc'
    lmhead_path = f'{lmhead_name}.mlmodelc'
    ffn_path = f'{ffn_name}.mlmodelc'
    
    # Set split_lm_head based on architecture
    split_lm_head = 16 if ARCH.startswith('qwen') else 8
    
    meta = f'''model_info:
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
    lut_embeddings: {lut_emb_actual}
    lut_ffn: {LUT_FFN}
    lut_lmhead: {lut_lmh_actual}
    num_chunks: {NUM_CHUNKS}
    model_prefix: {PREFIX}
    embeddings: {embeddings_path}
    lm_head: {lmhead_path}
    ffn: {ffn_path}
    split_lm_head: {split_lm_head}
'''
    
    output_file = os.path.join(OUTPUT_DIR, 'meta.yaml')
    with open(output_file, 'w') as f:
        f.write(meta)
    
    print(f"Generated meta.yaml at: {output_file}")
    print(f"  lut_embeddings: {lut_emb_actual}")
    print(f"  lut_lmhead: {lut_lmh_actual}")

if __name__ == "__main__":
    main() 