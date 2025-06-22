#!/usr/bin/env python3
"""
Complete QuaRot quantization workflow for Qwen 2.5
Equivalent to the bash commands you provided.
"""

import os
import sys
import subprocess
import tempfile
import argparse


def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False
    
    print(f"SUCCESS: {description} completed")
    return True


def main():
    parser = argparse.ArgumentParser(description='QuaRot quantization workflow for Qwen 2.5')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B',
                        help='Model name or path')
    parser.add_argument('--output_dir', type=str, default='/tmp/qwen25_quarot_w4',
                        help='Output directory for quantized model')
    parser.add_argument('--calib_tokens', type=int, default=2048,
                        help='Number of tokens for calibration')
    parser.add_argument('--calib_samples', type=int, default=128,
                        help='Number of calibration samples')
    
    args = parser.parse_args()
    
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, 'qwen25_quarot.json')
    calib_file = os.path.join(current_dir, 'calib.json')
    prepare_script = os.path.join(current_dir, 'prepare_calib.py')
    quantize_script = os.path.join(current_dir, 'quantize_qwen25_quarot.py')
    
    print(f"QuaRot Quantization Workflow for {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Config file: {config_file}")
    
    # Step 1: Prepare calibration dataset
    calib_cmd = f"python {prepare_script} --dataset pile --out {calib_file} --tokens {args.calib_tokens} --model {args.model} --nsamples {args.calib_samples}"
    
    if not run_command(calib_cmd, "Prepare calibration dataset"):
        print("Failed to prepare calibration dataset")
        return 1
    
    # Step 2: Run quantization
    quant_cmd = f"""python {quantize_script} \\
  --model_dir {args.model} \\
  --calib_data {calib_file} \\
  --quant_scheme w_int4_per_tensor \\
  --pre_quantization_optimization quarot \\
  --quarot_config {config_file} \\
  --output_dir {args.output_dir}"""
    
    if not run_command(quant_cmd, "Quantize model with QuaRot"):
        print("Failed to quantize model")
        return 1
    
    print(f"\n{'='*60}")
    print("QUANTIZATION WORKFLOW COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Quantized model saved to: {args.output_dir}")
    print(f"Calibration data saved to: {calib_file}")
    print()
    print("To test the quantized model with ANEMLL:")
    print("1. Set environment variable: export ENABLE_SP_QUANT=true")
    print("2. Run ANEMLL conversion on the quantized model directory")
    print(f"3. The quantized model is compatible with per-tensor quantization infrastructure")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())