#!/usr/bin/env python3
"""
Generate meta.yaml with correct LUT values based on actual file existence
"""

import sys
import os
import json
import re
from datetime import datetime


def escape_yaml_value(value):
    """Escape a value for safe inclusion in YAML.

    Handles special characters that could break YAML parsing:
    - Quotes (single and double)
    - Colons, hashes, and other YAML special chars
    - Newlines and control characters

    Args:
        value: The value to escape (will be converted to string)

    Returns:
        str: Safely escaped string for YAML
    """
    if value is None:
        return 'null'

    s = str(value)

    # Check if the string needs quoting
    # YAML special characters and patterns that need quoting
    needs_quoting = (
        s == '' or
        s.startswith((' ', '\t')) or
        s.endswith((' ', '\t')) or
        re.search(r'[\n\r\t\x00-\x1f]', s) or  # Control characters
        re.search(r'^[#!&*?|>\[\]{}@`]', s) or  # YAML indicators at start
        ':' in s or
        '#' in s or
        '"' in s or
        "'" in s or
        '\\' in s or
        s.lower() in ('true', 'false', 'null', 'yes', 'no', 'on', 'off')
    )

    if not needs_quoting:
        return s

    # Use double quotes and escape special characters
    escaped = s.replace('\\', '\\\\')  # Escape backslashes first
    escaped = escaped.replace('"', '\\"')  # Escape double quotes
    escaped = escaped.replace('\n', '\\n')  # Escape newlines
    escaped = escaped.replace('\r', '\\r')  # Escape carriage returns
    escaped = escaped.replace('\t', '\\t')  # Escape tabs

    return f'"{escaped}"'


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

def check_chunk_file_exists(output_dir, base_prefix, lut_value, chunk_suffix, rot_suffix=''):
    """Check chunked FFN/PF file existence with optional LUT and rotate suffix."""
    if lut_value == 'none':
        name = f'{base_prefix}{chunk_suffix}{rot_suffix}'
        return name, 'none'

    lut_name = f'{base_prefix}_lut{lut_value}{chunk_suffix}{rot_suffix}'
    if (os.path.exists(os.path.join(output_dir, f'{lut_name}.mlmodelc')) or
            os.path.exists(os.path.join(output_dir, f'{lut_name}.mlpackage'))):
        return lut_name, lut_value

    fallback = f'{base_prefix}{chunk_suffix}{rot_suffix}'
    print(f"Warning: {lut_name}.mlmodelc/.mlpackage not found, using {fallback}.mlmodelc instead")
    return fallback, 'none'

def generate_monolithic_meta(
    model_name,
    context,
    batch,
    lut_bits,
    prefix,
    arch,
    output_dir,
    lut_embeddings=None,
    lut_lmhead=None,
    argmax_in_model=False,
    rotate=False,
    sliding_window=None,
    update_mask_prefill=False,
    prefill_dynamic_slice=False,
    single_cache=False,
    vocab_size=None,
    lm_head_chunk_sizes=None,
):
    """Generate meta.yaml for monolithic model format.

    Args:
        model_name: Name of the model
        context: Context length
        batch: Batch size
        lut_bits: LUT quantization bits (or 'none')
        prefix: Model name prefix
        arch: Architecture type
        output_dir: Output directory
        argmax_in_model: If True, model computes argmax internally
        rotate: If True, include rotation functions (4-function model for context >= 512)
        sliding_window: Sliding window size for Gemma3 models (or None)
    """
    # Parse LUT value
    lut_value, lut_per_channel, _ = parse_lut_value(lut_bits)

    # Build model filename
    if lut_value != 'none':
        model_name_file = f'{prefix}_monolithic_full_lut{lut_value}.mlmodelc'
    else:
        model_name_file = f'{prefix}_monolithic_full.mlmodelc'

    # Check if file exists
    model_path = os.path.join(output_dir, model_name_file)
    if not os.path.exists(model_path):
        print(f"Warning: Monolithic model not found: {model_path}")

    # Set split_lm_head based on architecture
    if arch.startswith('qwen') or arch.startswith('gemma'):
        split_lm_head = 16
    else:
        split_lm_head = 8

    # Build meta.yaml content
    meta_parts = [f'''model_info:
  name: anemll-{model_name}-ctx{context}-monolithic
  version: 0.3.5
  description: |
    Monolithic model running {model_name} on Apple Neural Engine
    Context length: {context}
    Batch size: {batch}
    Type: Monolithic (single file with embed+FFN+lm_head)
  license: MIT
  author: Anemll
  framework: Core ML
  language: Python
  architecture: {arch}
  model_type: monolithic
  parameters:
    context_length: {context}
    batch_size: {batch}
    lut_bits: {lut_value}''']

    if lut_per_channel is not None:
        meta_parts.append(f'    lut_per_channel: {lut_per_channel}')

    # Per-component LUT overrides (only if different from main lut_bits)
    if lut_embeddings is not None:
        emb_val, emb_pc, _ = parse_lut_value(lut_embeddings)
        meta_parts.append(f'    lut_embeddings: {emb_val}')
        if emb_pc is not None:
            meta_parts.append(f'    lut_embeddings_per_channel: {emb_pc}')
    if lut_lmhead is not None:
        lmh_val, lmh_pc, _ = parse_lut_value(lut_lmhead)
        meta_parts.append(f'    lut_lmhead: {lmh_val}')
        if lmh_pc is not None:
            meta_parts.append(f'    lut_lmhead_per_channel: {lmh_pc}')

    argmax_line = f'\n    argmax_in_model: true' if argmax_in_model else ''
    vocab_line = f'\n    vocab_size: {vocab_size}' if vocab_size is not None else ''
    chunk_sizes_line = f'\n    lm_head_chunk_sizes: [{", ".join(str(x) for x in lm_head_chunk_sizes)}]' if lm_head_chunk_sizes else ''
    sliding_window_line = f'\n    sliding_window: {sliding_window}' if sliding_window is not None else ''
    update_mask_line = '\n    update_mask_prefill: true' if update_mask_prefill else ''
    prefill_dynamic_slice_line = '\n    prefill_dynamic_slice: true' if prefill_dynamic_slice else ''
    single_cache_line = '\n    single_cache: true' if single_cache else ''

    # Build functions list based on rotate flag
    if rotate:
        functions_yaml = '''    functions:
      - infer
      - infer_rotate
      - prefill
      - prefill_rotate'''
    else:
        functions_yaml = '''    functions:
      - infer
      - prefill'''

    meta_parts.append(f'''    model_prefix: {prefix}
    monolithic_model: {model_name_file}
    split_lm_head: {split_lm_head}{argmax_line}{vocab_line}{chunk_sizes_line}{sliding_window_line}{update_mask_line}{prefill_dynamic_slice_line}{single_cache_line}
{functions_yaml}
''')

    meta = '\n'.join(meta_parts)

    # Add conversion info as comments if provided
    conversion_info = os.environ.get('ANEMLL_CONVERSION_INFO')
    if conversion_info:
        meta += generate_conversion_comments(conversion_info)

    output_file = os.path.join(output_dir, 'meta.yaml')
    with open(output_file, 'w') as f:
        f.write(meta)

    print(f"Generated monolithic meta.yaml at: {output_file}")
    print(f"  model: {model_name_file}")
    print(f"  lut_bits: {lut_value}" + (f" (per_channel: {lut_per_channel})" if lut_per_channel else ""))


def generate_conversion_comments(conversion_info_json):
    """Generate YAML comments with conversion parameters for troubleshooting.

    Args:
        conversion_info_json: JSON string with conversion parameters

    Returns:
        str: YAML comment block with conversion info
    """
    try:
        info = json.loads(conversion_info_json)
    except json.JSONDecodeError:
        return ""

    lines = [
        "",
        "# =============================================================================",
        "# Conversion Parameters (for troubleshooting)",
        "# =============================================================================",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    # Add each parameter as a comment
    param_order = [
        'model_path', 'output_dir', 'command_line', 'context_length', 'batch_size',
        'num_chunks', 'lut_part1', 'lut_part2', 'lut_part3',
        'prefix', 'architecture', 'argmax_in_model', 'split_rotate',
        'sliding_window', 'fp16_scale', 'clamp', 'single_cache',
        'dynamic_prefill_slice', 'update_mask_prefill', 'converter',
        'monolithic', 'anemll_version'
    ]

    lines.append("#")
    for key in param_order:
        if key in info and info[key] is not None:
            value = info[key]
            # Format boolean values
            if isinstance(value, bool):
                value = 'true' if value else 'false'
            else:
                # Escape special characters for YAML safety
                value = escape_yaml_value(value)
            lines.append(f"# {key}: {value}")

    # Add any extra keys not in param_order
    for key, value in info.items():
        if key not in param_order and value is not None:
            if isinstance(value, bool):
                value = 'true' if value else 'false'
            else:
                # Escape special characters for YAML safety
                value = escape_yaml_value(value)
            lines.append(f"# {key}: {value}")

    lines.append("# =============================================================================")
    lines.append("")

    return '\n'.join(lines)


def main():
    conversion_info = os.environ.get('ANEMLL_CONVERSION_INFO')
    conversion_info_dict = None
    if conversion_info:
        try:
            conversion_info_dict = json.loads(conversion_info)
        except json.JSONDecodeError:
            conversion_info_dict = None

    # Check for --monolithic flag
    is_monolithic = '--monolithic' in sys.argv
    if is_monolithic:
        sys.argv.remove('--monolithic')

    # Check for --argmax flag
    argmax_in_model = '--argmax' in sys.argv
    if argmax_in_model:
        sys.argv.remove('--argmax')

    # Check for --rotate flag (4-function model for context >= 512)
    rotate = '--rotate' in sys.argv
    if rotate:
        sys.argv.remove('--rotate')

    # Check for --update-mask-prefill flag
    update_mask_prefill = '--update-mask-prefill' in sys.argv
    if update_mask_prefill:
        sys.argv.remove('--update-mask-prefill')

    # Check for --single-cache flag
    single_cache = '--single-cache' in sys.argv
    if single_cache:
        sys.argv.remove('--single-cache')

    # Prefill dynamic slice defaults to ON; use --static-prefill-slice to disable.
    prefill_dynamic_slice = True
    if '--static-prefill-slice' in sys.argv:
        prefill_dynamic_slice = False
        sys.argv.remove('--static-prefill-slice')
    if '--dynamic-prefill-slice' in sys.argv:
        prefill_dynamic_slice = True
        sys.argv.remove('--dynamic-prefill-slice')
    if '--prefill-dynamic-slice' in sys.argv:
        prefill_dynamic_slice = True
        sys.argv.remove('--prefill-dynamic-slice')

    # Check for --split-rotate flag (2 files per chunk: FFN and PF)
    split_rotate = '--split-rotate' in sys.argv
    if split_rotate:
        sys.argv.remove('--split-rotate')

    # Check for --sliding-window flag (e.g., --sliding-window 1024)
    sliding_window = None
    for i, arg in enumerate(sys.argv):
        if arg == '--sliding-window' and i + 1 < len(sys.argv):
            sliding_window = int(sys.argv[i + 1])
            sys.argv.pop(i + 1)
            sys.argv.pop(i)
            break

    if is_monolithic:
        if len(sys.argv) != 11:
            print("Usage: python3 generate_meta_yaml.py <model_name> <context> <batch> <lut_bits> <lut_bits> <lut_bits> <num_chunks> <prefix> <arch> <output_dir> --monolithic [--argmax] [--rotate] [--update-mask-prefill] [--dynamic-prefill-slice|--static-prefill-slice] [--single-cache] [--sliding-window N]")
            sys.exit(1)
        # Positional args: [4]=lut_emb, [5]=lut_ffn, [6]=lut_lmh
        # For monolithic, use FFN LUT as the main lut_bits (filename & global config)
        lut_emb_arg = sys.argv[4]
        lut_ffn_arg = sys.argv[5]
        lut_lmh_arg = sys.argv[6]
        # Per-component overrides: only pass if different from FFN default
        emb_override = lut_emb_arg if lut_emb_arg != lut_ffn_arg else None
        lmh_override = lut_lmh_arg if lut_lmh_arg != lut_ffn_arg else None
        generate_monolithic_meta(
            model_name=sys.argv[1],
            context=sys.argv[2],
            batch=sys.argv[3],
            lut_bits=lut_ffn_arg,  # FFN LUT = main lut_bits
            prefix=sys.argv[8],
            arch=sys.argv[9],
            output_dir=sys.argv[10],
            lut_embeddings=emb_override,
            lut_lmhead=lmh_override,
            # Optional metadata from conversion info (if provided)
            vocab_size=conversion_info_dict.get('vocab_size') if conversion_info_dict else None,
            lm_head_chunk_sizes=conversion_info_dict.get('lm_head_chunk_sizes') if conversion_info_dict else None,
            argmax_in_model=argmax_in_model,
            rotate=rotate,
            sliding_window=sliding_window,
            update_mask_prefill=update_mask_prefill,
            prefill_dynamic_slice=prefill_dynamic_slice,
            single_cache=single_cache,
        )
        return

    if len(sys.argv) != 11:
        print("Usage: python3 generate_meta_yaml.py <model_name> <context> <batch> <lut_emb> <lut_ffn> <lut_lmh> <num_chunks> <prefix> <arch> <output_dir> [--argmax] [--split-rotate] [--update-mask-prefill] [--dynamic-prefill-slice|--static-prefill-slice] [--single-cache] [--sliding-window N]")
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

    # Check FFN/PF files (chunked; LUT is optional based on actual files)
    # FFN files include chunk suffix: e.g., qwen_FFN_PF_lut4_chunk_01of01.mlmodelc
    ffn_base = f'{PREFIX}_FFN_PF'
    chunk_suffix = f'_chunk_01of{int(NUM_CHUNKS):02d}'
    ffn_name, lut_ffn_actual = check_chunk_file_exists(
        OUTPUT_DIR, ffn_base, lut_ffn_bits, chunk_suffix, rot_suffix=''
    )
    if split_rotate:
        pf_name, lut_pf_actual = check_chunk_file_exists(
            OUTPUT_DIR, ffn_base, lut_ffn_actual, chunk_suffix, rot_suffix='_rot'
        )
        if lut_pf_actual != lut_ffn_actual:
            print(f"Warning: rotate PF LUT ({lut_pf_actual}) differs from FFN LUT ({lut_ffn_actual})")

    # Add .mlmodelc extension to model paths
    embeddings_path = f'{embeddings_name}.mlmodelc'
    lmhead_path = f'{lmhead_name}.mlmodelc'
    ffn_path = f'{ffn_name}.mlmodelc'
    if split_rotate:
        pf_path = f'{pf_name}.mlmodelc'
    
    # Set split_lm_head based on architecture
    if ARCH.startswith('qwen') or ARCH.startswith('gemma'):
        split_lm_head = 16
    else:
        split_lm_head = 8

    # Build metadata with per_channel info if available
    meta_parts = [f'''model_info:
  name: anemll-{MODEL_NAME}-ctx{CONTEXT}
  version: 0.3.5
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

    meta_parts.append(f'    lut_ffn: {lut_ffn_actual}')
    if lut_ffn_per_channel is not None and lut_ffn_actual != 'none':
        meta_parts.append(f'    lut_ffn_per_channel: {lut_ffn_per_channel}')

    meta_parts.append(f'    lut_lmhead: {lut_lmh_actual}')
    if lut_lmh_per_channel is not None:
        meta_parts.append(f'    lut_lmhead_per_channel: {lut_lmh_per_channel}')

    # Add argmax_in_model if requested
    argmax_line = '\n    argmax_in_model: true' if argmax_in_model else ''

    vocab_size = None
    lm_head_chunk_sizes = None
    if conversion_info_dict:
        vocab_size = conversion_info_dict.get('vocab_size')
        lm_head_chunk_sizes = conversion_info_dict.get('lm_head_chunk_sizes')

    vocab_line = f'\n    vocab_size: {vocab_size}' if vocab_size is not None else ''
    chunk_sizes_line = (
        f'\n    lm_head_chunk_sizes: [{", ".join(str(x) for x in lm_head_chunk_sizes)}]'
        if isinstance(lm_head_chunk_sizes, list) and lm_head_chunk_sizes else ''
    )

    # Add split_rotate field if in split-rotate mode
    split_rotate_line = '\n    split_rotate: true' if split_rotate else ''
    pf_line = f'\n    pf: {pf_path}' if split_rotate else ''

    # Add sliding_window for Gemma3 models
    sliding_window_line = f'\n    sliding_window: {sliding_window}' if sliding_window is not None else ''
    update_mask_line = '\n    update_mask_prefill: true' if update_mask_prefill else ''
    prefill_dynamic_slice_line = '\n    prefill_dynamic_slice: true' if prefill_dynamic_slice else ''
    single_cache_line = '\n    single_cache: true' if single_cache else ''

    meta_parts.append(f'''    num_chunks: {NUM_CHUNKS}
    model_prefix: {PREFIX}
    embeddings: {embeddings_path}
    lm_head: {lmhead_path}
    ffn: {ffn_path}{pf_line}
    split_lm_head: {split_lm_head}{argmax_line}{vocab_line}{chunk_sizes_line}{split_rotate_line}{sliding_window_line}{update_mask_line}{prefill_dynamic_slice_line}{single_cache_line}
''')

    meta = '\n'.join(meta_parts)

    # Add conversion info as comments if provided
    if conversion_info:
        meta += generate_conversion_comments(conversion_info)

    output_file = os.path.join(OUTPUT_DIR, 'meta.yaml')
    with open(output_file, 'w') as f:
        f.write(meta)

    print(f"Generated meta.yaml at: {output_file}")
    print(f"  lut_embeddings: {lut_emb_actual}" + (f" (per_channel: {lut_emb_per_channel})" if lut_emb_per_channel else ""))
    print(f"  lut_ffn: {lut_ffn_actual}" + (f" (per_channel: {lut_ffn_per_channel})" if lut_ffn_per_channel and lut_ffn_actual != 'none' else ""))
    print(f"  lut_lmhead: {lut_lmh_actual}" + (f" (per_channel: {lut_lmh_per_channel})" if lut_lmh_per_channel else ""))

if __name__ == "__main__":
    main() 
