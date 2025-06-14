#!/usr/bin/env python3
"""Export CoreML model - IDE debuggable version of the command line converter.

Replicates this command for full model:
python -m anemll.ane_converter.qwen_converter \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/ \
    --prefix "test_qwen" \
    --context-length 256 \
    --output /tmp/qwen-test/full/

Or for prefill model:
python export_coreml.py --part prefill --batch-size 64
"""

import os
import argparse
from anemll.ane_converter.qwen_converter import test_conversion

def main():
    """Export CoreML model using the same parameters as the command line."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Export Qwen CoreML model")
    parser.add_argument("--model", default=None, help="Path to model directory")
    parser.add_argument("--part", choices=["full", "prefill", "embeddings"], default="full", 
                       help="Part to convert: 'full' for complete model, 'prefill' for prefill mode, 'embeddings' for embeddings only")
    parser.add_argument("--prefix", default="test_qwen", help="Model prefix")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for prefill mode (default: 64, not used for full model)")
    parser.add_argument("--context-length", type=int, default=256, help="Context length")
    parser.add_argument("--lut", type=int, default=None, help="Use LUT quantization with N bits (e.g., 4, 6)")
    parser.add_argument("--output", default="/tmp/qwen-test/", help="Output directory")
    
    args = parser.parse_args()
    
    print("🚀 Starting CoreML export (IDE debuggable version)")
    print("=" * 60)
    
    # Use default model path if not provided
    if args.model is None:
        model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    else:
        model_path = os.path.expanduser(args.model)
    
    # Adjust output directory and prefix based on part
    if args.part == "prefill":
        output_dir = os.path.join(args.output, "prefill/")
        prefix = f"{args.prefix}_prefill"
    elif args.part == "embeddings":
        output_dir = os.path.join(args.output, "embeddings/")
        prefix = f"{args.prefix}_embeddings"
    else:
        output_dir = os.path.join(args.output, "full/")
        prefix = args.prefix
    
    # Use LUT bits from command line argument
    lut_bits = args.lut
    
    print("📋 Conversion Parameters:")
    print(f"  Model path: {model_path}")
    print(f"  Part: {args.part}")
    print(f"  Prefix: {prefix}")
    if args.part == "prefill":
        print(f"  Batch size: {args.batch_size} (for prefill processing)")
    elif args.part == "embeddings":
        print(f"  Batch size: {args.batch_size} (flexible: 1 or {args.batch_size} tokens)")
    else:
        print(f"  Batch size: not used (full model conversion)")
    print(f"  Context length: {args.context_length}")
    print(f"  Output directory: {output_dir}")
    print(f"  LUT bits: {lut_bits} {'(quantization enabled)' if lut_bits else '(no quantization)'}")
    print()
    
    try:
        # Call the test_conversion function with part parameter
        result = test_conversion(
            model_path=model_path,
            prefix=prefix,
            context_length=args.context_length,
            lut_bits=lut_bits,
            batch_size=args.batch_size,
            output_dir=output_dir,
            part=args.part,
        )
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"Model type: {type(result)}")
        print(f"Output saved to: {output_dir}{prefix}.mlpackage")
        
        # Show usage examples
        print(f"\n💡 Usage Examples:")
        print(f"  Full model:   python export_coreml.py --part full")
        print(f"  Embeddings:   python export_coreml.py --part embeddings")
        print(f"  Prefill mode: python export_coreml.py --part prefill")
        print(f"  Custom batch: python export_coreml.py --part prefill --batch-size 32")
        
        if args.part == "prefill":
            print(f"\n📝 Note: For prefill workflow, you need both:")
            print(f"  1. Embeddings: python export_coreml.py --part embeddings")
            print(f"  2. Prefill:    python export_coreml.py --part prefill")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 