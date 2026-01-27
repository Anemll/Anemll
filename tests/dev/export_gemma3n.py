#!/usr/bin/env python3
# Copyright (c) 2025 ANEMLL
# Licensed under the MIT License

"""Export Gemma3n CoreML model - IDE debuggable version of the command line converter.


Replicates this command for full model:
python -m anemll.ane_converter.gemma3n_converter \
    --model google/gemma-1.1-2b-it \
    --output /tmp/gemma3n-test/full/ \
    --context 256 \
    --chunk 4

Or for specific parts:
python export_gemma3n.py --part embeddings --batch-size 64
python export_gemma3n.py --part ffn --chunk 4
python export_gemma3n.py --part lm_head
python export_gemma3n.py --part infer_init
python export_gemma3n.py --part combine_streams
"""


import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from anemll.ane_converter.gemma3n_converter import Gemma3nConverter
from transformers import AutoTokenizer, AutoConfig


MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/0330734afc91972a1aa1ba1bc4495e2723666854/"
)
def test_gemma3n_conversion(
    model_path: str,
    output_dir: str,
    context_length: int = 256,
    batch_size: int = 1,
    lut2: int = None,
    lut3: int = None,
    lut_per_channel: int = 8,
    lut_workers: int = 1,
    lut_scope: str = "all",
    lut_include: str = None,
    lut_exclude: str = None,
    lut_report: bool = False,
    chunk_size: int = 2,
    vocab_split_factor: int = 16,
    part: str = "full",
    enable_laurel: bool = True,
    enable_per_layer_embeddings: bool = True,
    text_only_mode: bool = True,
    disable_sparsity: bool = False,
):
    """Test conversion function for Gemma3n model"""
    
    #
    print(f"🔧 Creating Gemma3n converter...")
    converter = Gemma3nConverter(
        model_path=model_path,
        output_dir=output_dir,
        context_length=context_length,
        batch_size=batch_size,
        lut2=lut2,
        lut3=lut3,
        lut_per_channel=lut_per_channel,
        lut_num_workers=lut_workers,
        lut_scope=lut_scope,
        lut_include=lut_include,
        lut_exclude=lut_exclude,
        lut_report=lut_report,
        chunk_size=chunk_size,
        enable_laurel=enable_laurel,
        enable_per_layer_embeddings=enable_per_layer_embeddings,
        text_only_mode=text_only_mode,
        disable_sparsity=disable_sparsity,
    )
    
    print(f"📦 Converting {part} part(s)...")
    
    if part == "full":
        # Convert all parts
        converter.convert()
    elif part == "embeddings":
        converter.convert_embeddings()
    elif part == "ffn":
        # Convert all FFN chunks
        for chunk_idx in range(chunk_size):
            converter.convert_ffn(chunk_idx, chunk_size)
    elif part == "attention":
        converter.convert_attention_prefill()
    elif part == "infer_init":
        converter.convert_infer_init()
    elif part == "combine_streams":
        converter.convert_combine_streams()
    elif part == "infer":
        converter.convert_infer_init()
        for chunk_idx in range(chunk_size):
            converter.convert_infer(chunk_idx, chunk_size)
        converter.convert_combine_streams()
    elif part == "infer_rotate":
        converter.convert_infer_init()
        for chunk_idx in range(chunk_size):
            converter.convert_infer_rotate(chunk_idx, chunk_size)
        converter.convert_combine_streams()
    elif part == "prefill":
        converter.convert_infer_init()
        for chunk_idx in range(chunk_size):
            converter.convert_prefill(chunk_idx, chunk_size)
        converter.convert_combine_streams()
    elif part == "prefill_rotate":
        converter.convert_infer_init()
        for chunk_idx in range(chunk_size):
            converter.convert_prefill_rotate(chunk_idx, chunk_size)
        converter.convert_combine_streams()
    elif part == "lm_head":
        converter.convert_lm_head(vocab_split_factor)
    elif part == "tokenizer":
        converter.copy_tokenizer()
        converter.create_meta_config()
    else:
        raise ValueError(f"Unknown part: {part}")
    
    print(f"✅ Conversion of {part} completed!")
    return converter


def main():
    """Export Gemma3n CoreML model using the same parameters as the command line."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Export Gemma3n CoreML model")
    parser.add_argument("--model", default="google/gemma-3n-E2B-it", 
                       help="Path to model directory or HuggingFace model ID")
    parser.add_argument("--part", choices=["full", "embeddings", "ffn", "attention", "infer_init", "infer", "infer_rotate", "prefill", "prefill_rotate", "combine_streams", "lm_head", "tokenizer"], 
                       default="full", help="Part to convert")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--context-length", type=int, default=256, help="Context length")
    parser.add_argument("--lut", type=int, default=None, help="LUT quantization level for both FFN and LM head")
    parser.add_argument("--lut2", type=int, default=None, help="LUT quantization for Part 2 (FFN)")
    parser.add_argument("--lut3", type=int, default=None, help="LUT quantization for Part 3 (LM head)")
    parser.add_argument("--lut-per-channel", type=int, default=8, help="LUT group size (per-grouped-channel), default 8")
    parser.add_argument("--lut-workers", type=int, default=1, help="LUT k-means worker count (default: 1)")
    parser.add_argument("--lut-scope", choices=["all", "linear", "conv", "none"], default="all",
                        help="Which CoreML op types to palettize (default: all)")
    parser.add_argument("--lut-include", type=str, default=None, help="Regex for weight names to include in LUT")
    parser.add_argument("--lut-exclude", type=str, default=None, help="Regex for weight names to exclude from LUT")
    parser.add_argument("--lut-report", action="store_true", help="Print LUT match + unique-values report")
    parser.add_argument("--chunk", type=int, default=4, help="Number of FFN chunks")
    parser.add_argument("--vocab-split", type=int, default=16, help="Vocabulary split factor for LM head (default: 16)")
    parser.add_argument("--output", default="/tmp/gemma3n-test/", help="Output directory")
    parser.add_argument("--no-subdir", action="store_true",
                       help="Write outputs directly into --output without a part subfolder")
    parser.add_argument("--disable-laurel", action="store_true", help="Disable LAUREL blocks")
    parser.add_argument("--disable-per-layer-embeddings", action="store_true", 
                       help="Disable per-layer embeddings")
    parser.add_argument("--enable-multimodal", action="store_true", 
                       help="Enable multimodal weights (default: text-only)")
    parser.add_argument("--disable-sparsity", action="store_true",
                       help="Disable activation sparsity (conversion-friendly)")
    
    args = parser.parse_args()

    if args.lut is not None:
        if args.lut2 is None:
            args.lut2 = args.lut
        if args.lut3 is None:
            args.lut3 = args.lut
    
    print("🚀 Starting Gemma3n CoreML export (IDE debuggable version)")
    print("=" * 70)
    
    # Resolve model path - prefer local cache if available
    if args.model == "google/gemma-3n-E2B-it" and os.path.exists(MODEL_PATH):
        print(f"  Found local Gemma3n model, using: {MODEL_PATH}")
        model_path = MODEL_PATH
    elif os.path.exists(args.model):
        model_path = os.path.expanduser(args.model)
    else:
        # Assume it's a HuggingFace model ID
        model_path = args.model
    
    # Adjust output directory based on part
    if args.no_subdir:
        output_dir = args.output
    else:
        if args.part == "full":
            output_dir = os.path.join(args.output, "full/")
        else:
            output_dir = os.path.join(args.output, f"{args.part}/")
    
    print("📋 Conversion Parameters:")
    print(f"  Model: {model_path}")
    print(f"  Part: {args.part}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Context length: {args.context_length}")
    print(f"  Chunk size: {args.chunk}")
    print(f"  Vocab split factor: {args.vocab_split} (for LM head memory efficiency)")
    print(f"  Output directory: {output_dir}")
    if args.lut is not None:
        print(f"  LUT quantization: {args.lut}")
    print(f"  LUT2 quantization: {args.lut2 if args.lut2 else 'disabled'}")
    print(f"  LUT3 quantization: {args.lut3 if args.lut3 else 'disabled'}")
    print(f"  LUT per-channel group size: {args.lut_per_channel}")
    print(f"  LUT workers: {args.lut_workers}")
    print(f"  LUT scope: {args.lut_scope}")
    if args.lut_include:
        print(f"  LUT include: {args.lut_include}")
    if args.lut_exclude:
        print(f"  LUT exclude: {args.lut_exclude}")
    if args.lut_report:
        print("  LUT report: enabled")
    print(f"  LAUREL blocks: {'disabled' if args.disable_laurel else 'enabled'}")
    print(f"  Per-layer embeddings: {'disabled' if args.disable_per_layer_embeddings else 'enabled'}")
    print(f"  Multimodal mode: {'enabled' if args.enable_multimodal else 'text-only'}")
    print(f"  Activation sparsity: {'disabled' if args.disable_sparsity else 'enabled'}")
    print()
    
    # Check if model exists or is downloadable
    try:
        print("🔍 Checking model availability...")
        if not os.path.exists(model_path):
            print(f"  Downloading model from HuggingFace: {model_path}")
            # Try to load config to verify model exists
            config = AutoConfig.from_pretrained(model_path)
            print(f"  Model type: {config.model_type}")
            print(f"  Architecture: {config.architectures[0] if config.architectures else 'Unknown'}")
        else:
            print(f"  Using local model: {model_path}")
            if model_path == MODEL_PATH:
                print(f"  ✅ Using cached Gemma3n-E2B-it from local HuggingFace cache")
    except Exception as e:
        print(f"❌ Error accessing model: {e}")
        if args.model == "google/gemma-3n-E2B-it":
            print(f"💡 Local cache not found at: {MODEL_PATH}")
            print("💡 The model will be downloaded automatically from HuggingFace")
        else:
            print("💡 Make sure the model path is correct or you have internet access for HuggingFace models")
        return
    
    try:
        # Call the test conversion function
        result = test_gemma3n_conversion(
            model_path=model_path,
            output_dir=output_dir,
            context_length=args.context_length,
            batch_size=args.batch_size,
            lut2=args.lut2,
            lut3=args.lut3,
            lut_per_channel=args.lut_per_channel,
            lut_workers=args.lut_workers,
            lut_scope=args.lut_scope,
            lut_include=args.lut_include,
            lut_exclude=args.lut_exclude,
            lut_report=args.lut_report,
            chunk_size=args.chunk,
            vocab_split_factor=args.vocab_split,
            part=args.part,
            enable_laurel=not args.disable_laurel,
            enable_per_layer_embeddings=not args.disable_per_layer_embeddings,
            text_only_mode=not args.enable_multimodal,
            disable_sparsity=args.disable_sparsity,
        )
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"Converter type: {type(result)}")
        print(f"Output directory: {output_dir}")
        
        # Show usage examples
        print(f"\n💡 Usage Examples:")
        print(f"  Full model:     python export_gemma3n.py --part full")
        print(f"  Embeddings:     python export_gemma3n.py --part embeddings")
        print(f"  FFN layers:     python export_gemma3n.py --part ffn --chunk 4")
        print(f"  Attention:      python export_gemma3n.py --part attention")
        print(f"  Infer init:     python export_gemma3n.py --part infer_init")
        print(f"  Infer (KV):     python export_gemma3n.py --part infer")
        print(f"  Infer rotate:   python export_gemma3n.py --part infer_rotate")
        print(f"  Prefill:        python export_gemma3n.py --part prefill")
        print(f"  Prefill rotate: python export_gemma3n.py --part prefill_rotate")
        print(f"  Combine:        python export_gemma3n.py --part combine_streams")
        print(f"  LM head:        python export_gemma3n.py --part lm_head")
        print(f"  With LUT:       python export_gemma3n.py --lut 6 --lut-per-channel 8 --lut-scope all")
        print(f"  With LUT (FFN): python export_gemma3n.py --lut 6 --lut-scope linear --lut-include \"gate_proj|up_proj|down_proj\" --lut-exclude \"q_proj|k_proj|v_proj|o_proj|router|per_layer_input_gate\" --lut-report")
        print(f"  No LAUREL:      python export_gemma3n.py --disable-laurel")
        print(f"  Custom model:   python export_gemma3n.py --model /path/to/model")
        
        if args.part == "full":
            print(f"\n📝 Full conversion includes:")
            print(f"  ✓ Embeddings (with per-layer projections)")
            print(f"  ✓ FFN layers ({args.chunk} chunks with LAUREL blocks)")
            print(f"  ✓ Attention (prefill mode)")
            print(f"  ✓ Infer init (token -> hidden_states + per-layer inputs)")
            print(f"  ✓ Infer (KV cache stateful)")
            print(f"  ✓ (Rotation variants not included; export infer_rotate/prefill_rotate separately)")
            print(f"  ✓ Combine streams (AltUp)")
            print(f"  ✓ LM head (with soft-capping)")
            print(f"  ✓ Tokenizer and meta.yaml")
            
        elif args.part in ["embeddings", "ffn", "attention", "infer_init", "infer", "infer_rotate", "prefill", "prefill_rotate", "combine_streams", "lm_head"]:
            print(f"\n📝 For complete workflow, you need all parts:")
            print(f"  1. Embeddings: python export_gemma3n.py --part embeddings")
            print(f"  2. FFN:        python export_gemma3n.py --part ffn")
            print(f"  3. Attention:  python export_gemma3n.py --part attention")
            print(f"  4. Infer init: python export_gemma3n.py --part infer_init")
            print(f"  5. Infer KV:   python export_gemma3n.py --part infer")
            print(f"  6. Infer rot:  python export_gemma3n.py --part infer_rotate (optional)")
            print(f"  7. Prefill:    python export_gemma3n.py --part prefill (optional)")
            print(f"  8. Prefill rot:python export_gemma3n.py --part prefill_rotate (optional)")
            print(f"  9. Combine:    python export_gemma3n.py --part combine_streams")
            print(f"  10. LM head:   python export_gemma3n.py --part lm_head")
            print(f"  11. Tokenizer: python export_gemma3n.py --part tokenizer")
            
        # Show file structure
        print(f"\n📁 Expected output structure:")
        if args.part == "full":
            print(f"  {output_dir}")
            print(f"  ├── gemma3n_embeddings.mlpackage")
            print(f"  ├── gemma3n_FFN_chunk_00of{args.chunk:02d}.mlpackage")
            print(f"  ├── gemma3n_FFN_chunk_01of{args.chunk:02d}.mlpackage")
            print(f"  ├── gemma3n_attention_prefill.mlpackage")
            print(f"  ├── gemma3n_infer_init.mlpackage")
            print(f"  ├── gemma3n_infer_chunk_00of{args.chunk:02d}.mlpackage")
            print(f"  ├── gemma3n_combine_streams.mlpackage")
            print(f"  ├── gemma3n_lm_head.mlpackage")
            print(f"  ├── tokenizer.json")
            print(f"  └── meta.yaml")
        elif args.part == "ffn":
            print(f"  {output_dir}")
            print(f"  ├── gemma3n_FFN_chunk_00of{args.chunk:02d}.mlpackage")
            print(f"  ├── gemma3n_FFN_chunk_01of{args.chunk:02d}.mlpackage")
            print(f"  └── ...")
        elif args.part == "attention":
            print(f"  {output_dir}")
            print(f"  └── gemma3n_attention_prefill.mlpackage")
        elif args.part == "embeddings":
            print(f"  {output_dir}")
            print(f"  └── gemma3n_embeddings.mlpackage")
        elif args.part == "infer_init":
            print(f"  {output_dir}")
            print(f"  └── gemma3n_infer_init.mlpackage")
        elif args.part == "combine_streams":
            print(f"  {output_dir}")
            print(f"  └── gemma3n_combine_streams.mlpackage")
        elif args.part == "infer":
            print(f"  {output_dir}")
            print(f"  ├── gemma3n_infer_init.mlpackage")
            print(f"  ├── gemma3n_infer_chunk_00of{args.chunk:02d}.mlpackage")
            print(f"  └── gemma3n_combine_streams.mlpackage")
        elif args.part == "lm_head":
            print(f"  {output_dir}")
            print(f"  └── gemma3n_lm_head.mlpackage")
        else:
            print(f"  {output_dir}")
            print(f"  └── gemma3n_{args.part}.mlpackage")
            
    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Show common troubleshooting tips
        print(f"\n🔧 Troubleshooting:")
        print(f"  • Check model path/ID is correct")
        print(f"  • Ensure sufficient disk space")
        print(f"  • Verify Python environment has required packages")
        print(f"  • Try smaller context length if out of memory")
        print(f"  • Use --disable-laurel to reduce complexity")
        raise


if __name__ == "__main__":
    main()
