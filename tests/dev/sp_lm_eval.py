#!/usr/bin/env python3
"""
Command-line interface for SP quantization evaluation, similar to lm_eval.

Usage:
    python sp_lm_eval.py --model hf \
        --model_args pretrained=Qwen/Qwen2.5-0.5B \
        --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag,boolq \
        --device cpu \
        --batch_size auto:2
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from test_sp_quant_eval import run_evaluation


def parse_model_args(model_args_str: str) -> Dict[str, str]:
    """Parse model arguments string into dictionary."""
    args_dict = {}
    if model_args_str:
        for arg in model_args_str.split(','):
            if '=' in arg:
                key, value = arg.split('=', 1)
                args_dict[key.strip()] = value.strip()
    return args_dict


def main():
    parser = argparse.ArgumentParser(
        description="SP Quantization LM Evaluation (sp_lm_eval)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate base model with SP quantization
  python sp_lm_eval.py --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B --tasks arc_easy,boolq
  
  # Evaluate without quantization for comparison
  python sp_lm_eval.py --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B --tasks arc_easy --no_quant
  
  # Evaluate with custom batch size
  python sp_lm_eval.py --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B --batch_size 4
        """
    )
    
    parser.add_argument("--model", type=str, default="hf",
                        help="Model type (currently only 'hf' supported)")
    parser.add_argument("--model_args", type=str, required=True,
                        help="Model arguments (e.g., pretrained=Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--tasks", type=str, required=True,
                        help="Comma-separated list of tasks")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (only cpu supported for ANEMLL)")
    parser.add_argument("--batch_size", type=str, default="1",
                        help="Batch size (use 'auto:N' for automatic or just a number)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--no_quant", action="store_true",
                        help="Disable SP quantization")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Parse model arguments
    model_args = parse_model_args(args.model_args)
    
    if "pretrained" not in model_args:
        print("Error: 'pretrained' must be specified in model_args")
        sys.exit(1)
    
    model_path = model_args["pretrained"]
    
    # Parse batch size
    if args.batch_size.startswith("auto:"):
        batch_size = int(args.batch_size.split(":")[1])
    else:
        batch_size = int(args.batch_size)
    
    # Parse tasks
    tasks = [t.strip() for t in args.tasks.split(",")]
    
    # Set environment variables
    if args.no_quant:
        os.environ['SKIP_SP_FORWARD'] = '1'
        print("Running WITHOUT SP quantization (baseline)")
    else:
        os.environ['ENABLE_SP_QUANT'] = '1'
        print("Running WITH SP quantization")
    
    if args.verbose:
        print(f"\nConfiguration:")
        print(f"  Model: {model_path}")
        print(f"  Tasks: {tasks}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {args.device}")
        print(f"  Quantization: {'Disabled' if args.no_quant else 'Enabled'}")
    
    # Run evaluation
    try:
        results = run_evaluation(model_path, tasks, batch_size)
        
        # Save results if requested
        if args.output_path:
            output_data = {
                "model": model_path,
                "tasks": tasks,
                "batch_size": batch_size,
                "quantization": not args.no_quant,
                "results": results
            }
            with open(args.output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output_path}")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()