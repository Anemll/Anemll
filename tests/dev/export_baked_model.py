#!/usr/bin/env python3
"""
Export Q4_4 baked model: Complete pipeline for exporting embeddings, FFN+Prefill, and LM Head.

This script exports a quantized checkpoint with baked weights (scales pre-computed at conversion time)
to CoreML format ready for inference.

Usage:
    python tests/dev/export_baked_model.py \
        --checkpoint /path/to/model_state_dict.pt \
        --model Qwen/Qwen3-0.6B \
        --context 512 \
        --output /path/to/output

Arguments:
    --checkpoint: Path to the quantized checkpoint (.pt file)
    --model: HuggingFace model ID (for config and tokenizer)
    --context: Context length (default: 512)
    --output: Output directory for CoreML models
    --lut-lmhead: LUT bits for LM head quantization (default: 6)
    --batch: Batch size for prefill (default: 64)
    --prefix: Model prefix (default: qwen)
"""

import os
import sys
import argparse
import subprocess
import shutil
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_command(cmd, description, env=None, cwd=None):
    """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {cmd[:100]}..." if len(cmd) > 100 else f"  Command: {cmd}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    result = subprocess.run(cmd, shell=True, env=merged_env, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0:
        print(f"  FAILED!")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False

    print(f"  SUCCESS")
    if result.stdout.strip():
        # Print last few lines of output
        lines = result.stdout.strip().split('\n')
        for line in lines[-5:]:
            print(f"    {line}")
    return True


def find_hf_cache_path(model_id):
    """Find the local HuggingFace cache path for a model."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = f"models--{model_id.replace('/', '--')}"
    model_path = os.path.join(cache_dir, model_dir_name)

    if os.path.exists(model_path):
        # Find the snapshot directory
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[0])

    return None


def get_converter_for_model(model_id):
    """Get the appropriate converter module for the model."""
    model_lower = model_id.lower()
    if 'qwen' in model_lower:
        return "python3 -m anemll.ane_converter.qwen_converter"
    elif 'deepseek' in model_lower:
        return "python3 -m anemll.ane_converter.deepseek_converter"
    else:
        return "python3 -m anemll.ane_converter.llama_converter"


def main():
    parser = argparse.ArgumentParser(description="Export Q4_4 baked model to CoreML")
    parser.add_argument("--checkpoint", required=True, help="Path to quantized checkpoint (.pt file)")
    parser.add_argument("--model", required=True, help="HuggingFace model ID (e.g., Qwen/Qwen3-0.6B)")
    parser.add_argument("--context", type=int, default=512, help="Context length (default: 512)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--lut-lmhead", type=int, default=6, help="LUT bits for LM head (default: 6)")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for prefill (default: 64)")
    parser.add_argument("--prefix", default="qwen", help="Model prefix (default: qwen)")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation step")
    parser.add_argument("--skip-test", action="store_true", help="Skip test step")

    args = parser.parse_args()

    # Validate checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Find HF cache path
    hf_path = find_hf_cache_path(args.model)
    if not hf_path:
        print(f"Error: Could not find HuggingFace cache for {args.model}")
        print("Please download the model first: from transformers import AutoConfig; AutoConfig.from_pretrained(model_id)")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  EXPORT Q4_4 BAKED MODEL")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Model: {args.model}")
    print(f"  HF Cache: {hf_path}")
    print(f"  Context: {args.context}")
    print(f"  Output: {args.output}")
    print(f"  LUT LM Head: {args.lut_lmhead}")
    print(f"  Batch: {args.batch}")
    print(f"  Prefix: {args.prefix}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    converter = get_converter_for_model(args.model)

    success = True

    # Step 1: Export Embeddings (using converter directly)
    cmd = (
        f"source env-anemll/bin/activate && "
        f"{converter} "
        f"--part 1 "
        f"--context-length {args.context} "
        f"--batch-size {args.batch} "
        f"--prefix {args.prefix} "
        f"--model {hf_path} "
        f"--output {args.output}"
    )
    if not run_command(cmd, "Step 1: Export Embeddings", cwd=project_root):
        print("Warning: Embeddings export failed")

    # Step 2: Export LM Head (using converter directly with LUT)
    cmd = (
        f"source env-anemll/bin/activate && "
        f"{converter} "
        f"--part 3 "
        f"--lut {args.lut_lmhead} "
        f"--context-length {args.context} "
        f"--prefix {args.prefix} "
        f"--model {hf_path} "
        f"--output {args.output}"
    )
    if not run_command(cmd, "Step 2: Export LM Head", cwd=project_root):
        print("Warning: LM Head export failed")

    # Step 3: Export FFN with baked weights (using test_qwen_a1f_layer.py)
    # Output file: {prefix}_FFN_chunk_01of01.mlpackage
    ffn_output = os.path.join(args.output, f"{args.prefix}_FFN_chunk_01of01.mlpackage")
    a1f_script = os.path.join(script_dir, "test_qwen_a1f_layer.py")
    if os.path.exists(a1f_script):
        cmd = (
            f"source env-anemll/bin/activate && "
            f"python {a1f_script} "
            f"--checkpoint {args.checkpoint} "
            f"--model {args.model} "
            f"--context {args.context} "
            f"--output {ffn_output}"
        )
        if not run_command(cmd, "Step 3: Export FFN (baked weights)", cwd=project_root):
            print("Error: FFN export failed!")
            success = False
    else:
        print(f"Warning: A1F layer script not found: {a1f_script}")
        success = False

    # Step 4: Export Prefill (using converter directly)
    # Output file: {prefix}_prefill_chunk_01of01.mlpackage
    cmd = (
        f"source env-anemll/bin/activate && "
        f"{converter} "
        f"--part 2_prefill "
        f"--chunk 1 "
        f"--context-length {args.context} "
        f"--batch-size {args.batch} "
        f"--prefix {args.prefix} "
        f"--model {hf_path} "
        f"--output {args.output}"
    )
    if not run_command(cmd, "Step 4: Export Prefill", cwd=project_root):
        print("Warning: Prefill export failed")

    # List files before combine
    print(f"\n  Files before combine:")
    for f in sorted(os.listdir(args.output)):
        if f.endswith('.mlpackage'):
            print(f"    {f}")

    # Step 5: Combine FFN and Prefill
    combine_script = os.path.join(project_root, "anemll", "utils", "combine_models.py")
    cmd = (
        f"source env-anemll/bin/activate && "
        f"python {combine_script} "
        f"--input {args.output} "
        f"--prefix {args.prefix} "
        f"--output {args.output} "
        f"--chunk 1"
    )
    if not run_command(cmd, "Step 5: Combine FFN + Prefill", cwd=project_root):
        print("Error: Combine failed!")
        success = False

    # Step 6: Compile models (if not skipped)
    if not args.skip_compile:
        models_to_compile = [
            f"{args.prefix}_embeddings.mlpackage",
            f"{args.prefix}_lm_head_lut{args.lut_lmhead}.mlpackage",
            f"{args.prefix}_FFN_PF_chunk_01of01.mlpackage",
        ]

        for model_name in models_to_compile:
            model_path = os.path.join(args.output, model_name)
            if os.path.exists(model_path):
                cmd = (
                    f"source env-anemll/bin/activate && "
                    f"xcrun coremlcompiler compile {model_path} {args.output}/"
                )
                run_command(cmd, f"Step 6: Compile {model_name}", cwd=project_root)
            else:
                print(f"  Warning: Model not found for compilation: {model_path}")

    # Step 7: Create meta.yaml
    print(f"\n{'='*60}")
    print(f"  Step 7: Create meta.yaml")
    print(f"{'='*60}")

    meta = {
        'model_info': {
            'name': f'anemll-{args.model.split("/")[-1]}-Q4_4-baked-ctx{args.context}',
            'version': '0.3.5',
            'description': f'{args.model.split("/")[-1]} with Q4_4 quantization (4-bit LUT, rank-4 scales, baked)\n'
                          f'Context length: {args.context}\n'
                          f'Batch size: {args.batch}\n'
                          f'Chunks: 1\n'
                          f'FFN: 4-bit LUT with rank-4 scales (baked at conversion time)\n'
                          f'LM Head: {args.lut_lmhead}-bit LUT\n'
                          f'Attention: 4-bit LUT with rank-4 scales (baked at conversion time)',
            'license': 'MIT',
            'author': 'Anemll',
            'framework': 'Core ML',
            'language': 'Python',
            'architecture': args.model.split("/")[-1].lower().split("-")[0],
            'parameters': {
                'context_length': args.context,
                'batch_size': args.batch,
                'lut_embeddings': 'none',
                'lut_ffn': 'none',
                'lut_lmhead': args.lut_lmhead,
                'num_chunks': 1,
                'model_prefix': args.prefix,
                'embeddings': f'{args.prefix}_embeddings.mlmodelc',
                'lm_head': f'{args.prefix}_lm_head_lut{args.lut_lmhead}.mlmodelc',
                'ffn': f'{args.prefix}_FFN_PF_chunk_01of01.mlmodelc',
                'split_lm_head': 16,
            }
        }
    }

    meta_path = os.path.join(args.output, 'meta.yaml')
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {meta_path}")

    # Step 8: Copy tokenizer
    print(f"\n{'='*60}")
    print(f"  Step 8: Copy Tokenizer")
    print(f"{'='*60}")

    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                       'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja']
    copied = 0
    for tf in tokenizer_files:
        src = os.path.join(hf_path, tf)
        dst = os.path.join(args.output, tf)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1
    print(f"  Copied {copied} tokenizer files")

    # Step 9: Test (if not skipped)
    if not args.skip_test and success:
        cmd = (
            f"source env-anemll/bin/activate && "
            f"echo 'Hello, how are you?' | python tests/chat.py --meta {meta_path}"
        )
        run_command(cmd, "Step 9: Test Inference", cwd=project_root)

    # Summary
    print(f"\n{'='*60}")
    print(f"  EXPORT COMPLETE")
    print(f"{'='*60}")

    print(f"\n  Output directory: {args.output}")
    print(f"\n  Files:")
    for f in sorted(os.listdir(args.output)):
        path = os.path.join(args.output, f)
        if os.path.isdir(path):
            size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, dn, fns in os.walk(path) for fn in fns)
            print(f"    {f}: {size / 1024 / 1024:.1f} MB")
        else:
            size = os.path.getsize(path)
            if size > 1024 * 1024:
                print(f"    {f}: {size / 1024 / 1024:.1f} MB")
            else:
                print(f"    {f}: {size / 1024:.1f} KB")

    print(f"\n  To test:")
    print(f"    python tests/chat.py --meta {meta_path}")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
