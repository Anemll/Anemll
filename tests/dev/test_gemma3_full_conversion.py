#!/usr/bin/env python3
"""
Full Gemma3 Conversion Pipeline Test

This script runs the complete Gemma3 conversion pipeline:
1. Convert embeddings (part 1)
2. Convert LM head (part 3)
3. Convert FFN chunks (part 2)
4. Convert Prefill chunks (part 2_prefill)
5. Combine FFN and Prefill models
6. Create meta.yaml configuration
7. Test with a simple inference

Usage:
    python tests/dev/test_gemma3_full_conversion.py --model google/gemma-3n-E2B-it --output /tmp/gemma3_test
"""

import argparse
import os
import sys
import shutil
import yaml
from pathlib import Path

# Add package root to path
package_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_root)

import torch
import numpy as np
import coremltools as ct


def parse_args():
    parser = argparse.ArgumentParser(description="Full Gemma3 conversion test")
    parser.add_argument("--model", type=str, default="google/gemma-3n-E2B-it",
                        help="Path to Gemma3 model or HuggingFace model name")
    parser.add_argument("--output", type=str, default="/tmp/gemma3_converted",
                        help="Output directory for converted models")
    parser.add_argument("--context-length", type=int, default=512,
                        help="Context length for conversion")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for prefill")
    parser.add_argument("--lut", type=int, default=None,
                        help="LUT quantization bits (optional)")
    parser.add_argument("--chunks", type=int, default=2,
                        help="Number of chunks for FFN/prefill")
    parser.add_argument("--prefix", type=str, default="gemma3",
                        help="Model prefix for filenames")
    parser.add_argument("--skip-conversion", action="store_true",
                        help="Skip conversion, only combine and create meta.yaml")
    parser.add_argument("--only", type=str, choices=["1", "2", "2_prefill", "3", "combine", "meta", "test"],
                        help="Only run a specific step")
    return parser.parse_args()


def convert_embeddings(model, converter, output_dir, prefix, lut_bits):
    """Convert embeddings (part 1)"""
    print("\n" + "=" * 60)
    print("STEP 1: Converting Embeddings")
    print("=" * 60)

    mlmodel = converter.convert(part="1")

    # Build filename
    fname = f"{prefix}_embeddings"
    if lut_bits:
        fname += f"_lut{lut_bits}"
    fname += ".mlpackage"

    out_path = os.path.join(output_dir, fname)
    print(f"Saving embeddings to: {out_path}")
    mlmodel.save(out_path)

    return out_path


def convert_lm_head(model, converter, output_dir, prefix, lut_bits):
    """Convert LM head (part 3)"""
    print("\n" + "=" * 60)
    print("STEP 2: Converting LM Head")
    print("=" * 60)

    mlmodel = converter.convert(part="3")

    # Build filename
    fname = f"{prefix}_lm_head"
    if lut_bits:
        fname += f"_lut{lut_bits}"
    fname += ".mlpackage"

    out_path = os.path.join(output_dir, fname)
    print(f"Saving LM head to: {out_path}")
    mlmodel.save(out_path)

    return out_path


def convert_ffn(model, converter, output_dir, prefix, lut_bits, num_chunks):
    """Convert FFN chunks (part 2)"""
    print("\n" + "=" * 60)
    print("STEP 3: Converting FFN Chunks")
    print("=" * 60)

    mlmodels = converter.convert(part="2")

    if not isinstance(mlmodels, list):
        mlmodels = [mlmodels]

    paths = []
    for i, mlmodel in enumerate(mlmodels):
        fname = f"{prefix}_FFN"
        if lut_bits:
            fname += f"_lut{lut_bits}"
        fname += f"_chunk_{i+1:02d}of{num_chunks:02d}.mlpackage"

        out_path = os.path.join(output_dir, fname)
        print(f"Saving FFN chunk {i+1} to: {out_path}")
        mlmodel.save(out_path)
        paths.append(out_path)

    return paths


def convert_prefill(model, converter, output_dir, prefix, lut_bits, num_chunks):
    """Convert Prefill chunks (part 2_prefill)"""
    print("\n" + "=" * 60)
    print("STEP 4: Converting Prefill Chunks")
    print("=" * 60)

    mlmodels = converter.convert(part="2_prefill")

    if not isinstance(mlmodels, list):
        mlmodels = [mlmodels]

    paths = []
    for i, mlmodel in enumerate(mlmodels):
        fname = f"{prefix}_prefill"
        if lut_bits:
            fname += f"_lut{lut_bits}"
        fname += f"_chunk_{i+1:02d}of{num_chunks:02d}.mlpackage"

        out_path = os.path.join(output_dir, fname)
        print(f"Saving prefill chunk {i+1} to: {out_path}")
        mlmodel.save(out_path)
        paths.append(out_path)

    return paths


def combine_models(output_dir, prefix, lut_bits, num_chunks):
    """Combine FFN and Prefill models into multi-function models"""
    print("\n" + "=" * 60)
    print("STEP 5: Combining FFN and Prefill Models")
    print("=" * 60)

    from anemll.ane_converter.metadata import AddCombinedMetadata

    combined_paths = []

    for chunk_idx in range(num_chunks):
        # Build paths
        lut_suffix = f"_lut{lut_bits}" if lut_bits else ""
        ffn_fname = f"{prefix}_FFN{lut_suffix}_chunk_{chunk_idx+1:02d}of{num_chunks:02d}.mlpackage"
        prefill_fname = f"{prefix}_prefill{lut_suffix}_chunk_{chunk_idx+1:02d}of{num_chunks:02d}.mlpackage"
        combined_fname = f"{prefix}_FFN_PF{lut_suffix}_chunk_{chunk_idx+1:02d}of{num_chunks:02d}.mlpackage"

        ffn_path = os.path.join(output_dir, ffn_fname)
        prefill_path = os.path.join(output_dir, prefill_fname)
        combined_path = os.path.join(output_dir, combined_fname)
        temp_path = os.path.join(output_dir, f"temp_{combined_fname}")

        print(f"\nProcessing chunk {chunk_idx + 1}:")
        print(f"  FFN: {ffn_fname}")
        print(f"  Prefill: {prefill_fname}")
        print(f"  Output: {combined_fname}")

        if not os.path.exists(ffn_path):
            print(f"  ERROR: FFN file not found: {ffn_path}")
            continue
        if not os.path.exists(prefill_path):
            print(f"  ERROR: Prefill file not found: {prefill_path}")
            continue

        # Load models
        ffn_model = ct.models.MLModel(ffn_path)
        prefill_model = ct.models.MLModel(prefill_path)

        # Create combined model using MultiFunctionDescriptor
        desc = ct.utils.MultiFunctionDescriptor()
        desc.add_function(ffn_path, src_function_name="main", target_function_name="infer")
        desc.add_function(prefill_path, src_function_name="main", target_function_name="prefill")
        desc.default_function_name = "infer"

        print("  Creating combined model...")
        ct.utils.save_multifunction(desc, temp_path)

        # Load the temp model to add metadata
        print("  Adding metadata...")
        combined_model = ct.models.MLModel(temp_path)
        AddCombinedMetadata(combined_model, [ffn_model, prefill_model])

        print(f"  Saving to: {combined_path}")
        combined_model.save(combined_path)

        # Clean up temp file
        shutil.rmtree(temp_path, ignore_errors=True)

        combined_paths.append(combined_path)
        print(f"  Successfully combined chunk {chunk_idx + 1}")

    return combined_paths


def create_meta_yaml(output_dir, prefix, context_length, batch_size, lut_bits, num_chunks, tokenizer_path=None):
    """Create meta.yaml configuration file"""
    print("\n" + "=" * 60)
    print("STEP 6: Creating meta.yaml")
    print("=" * 60)

    meta = {
        "model_info": {
            "name": f"anemll-{prefix}-ctx{context_length}",
            "version": "0.3.5",
            "parameters": {
                "context_length": context_length,
                "batch_size": batch_size,
                "lut_embeddings": lut_bits if lut_bits else "none",
                "lut_ffn": lut_bits if lut_bits else "none",
                "lut_lmhead": lut_bits if lut_bits else "none",
                "num_chunks": num_chunks,
                "model_prefix": prefix,
                "split_lm_head": 16,  # Gemma3 uses 16-way LM head split
                "num_logits": 16,  # Legacy compatibility
            }
        }
    }

    # Add tokenizer path if provided
    if tokenizer_path:
        meta["model_info"]["parameters"]["tokenizer_path"] = tokenizer_path

    meta_path = os.path.join(output_dir, "meta.yaml")
    print(f"Writing meta.yaml to: {meta_path}")

    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    print("meta.yaml content:")
    print("-" * 40)
    with open(meta_path, "r") as f:
        print(f.read())

    return meta_path


def test_inference(output_dir, prefix, lut_bits, num_chunks, context_length):
    """Test basic inference with converted models"""
    print("\n" + "=" * 60)
    print("STEP 7: Testing Inference")
    print("=" * 60)

    lut_suffix = f"_lut{lut_bits}" if lut_bits else ""

    # Load embeddings
    embed_path = os.path.join(output_dir, f"{prefix}_embeddings{lut_suffix}.mlpackage")
    print(f"Loading embeddings: {embed_path}")
    embed_model = ct.models.MLModel(embed_path)

    # Load LM head
    lmhead_path = os.path.join(output_dir, f"{prefix}_lm_head{lut_suffix}.mlpackage")
    print(f"Loading LM head: {lmhead_path}")
    lmhead_model = ct.models.MLModel(lmhead_path)

    # Load combined FFN model (first chunk with 'infer' function)
    combined_path = os.path.join(output_dir, f"{prefix}_FFN_PF{lut_suffix}_chunk_01of{num_chunks:02d}.mlpackage")
    print(f"Loading combined model: {combined_path}")
    ffn_model = ct.models.MLModel(combined_path, function_name="infer")

    # Create test inputs
    print("\nRunning test inference...")

    # Test embeddings
    test_input_ids = np.array([[1]], dtype=np.int32)  # Single token
    embed_output = embed_model.predict({"input_ids": test_input_ids})
    hidden_states = embed_output["hidden_states"]
    print(f"Embeddings output shape: {hidden_states.shape}")

    # Test FFN (infer mode)
    position_ids = np.array([0], dtype=np.int32)
    causal_mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
    current_pos = np.array([0], dtype=np.int32)

    # Create state
    state = ffn_model.make_state()

    ffn_output = ffn_model.predict({
        "hidden_states": hidden_states.astype(np.float16),
        "position_ids": position_ids,
        "causal_mask": causal_mask,
        "current_pos": current_pos,
    }, state)

    ffn_hidden = ffn_output["output_hidden_states"]
    print(f"FFN output shape: {ffn_hidden.shape}")

    # Test LM head
    lmhead_output = lmhead_model.predict({"hidden_states": ffn_hidden.astype(np.float16)})

    # Combine all 16 logit outputs
    logits_parts = []
    for i in range(1, 17):
        key = f"logits{i}"
        if key in lmhead_output:
            logits_parts.append(lmhead_output[key])

    if logits_parts:
        logits = np.concatenate(logits_parts, axis=-1)
        print(f"Combined logits shape: {logits.shape}")

        # Get top prediction
        top_idx = np.argmax(logits[0, 0, :])
        top_val = logits[0, 0, top_idx]
        print(f"Top prediction: token {top_idx} with logit {top_val:.4f}")
    else:
        print("ERROR: No logits outputs found")
        return False

    print("\n✅ Inference test passed!")
    return True


def main():
    args = parse_args()

    print("=" * 60)
    print("GEMMA3 FULL CONVERSION PIPELINE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Context Length: {args.context_length}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LUT Bits: {args.lut if args.lut else 'None (FP16)'}")
    print(f"Chunks: {args.chunks}")
    print(f"Prefix: {args.prefix}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    model = None
    converter = None

    # Load model if needed for conversion
    if not args.skip_conversion and args.only not in ["combine", "meta", "test"]:
        print("\n" + "=" * 60)
        print("Loading Gemma3 Model")
        print("=" * 60)

        from anemll.models.gemma3_model import Gemma3ForCausalLM, Gemma3Config
        from anemll.ane_converter.gemma3_converter import Gemma3Converter

        # Check if model path is local or HuggingFace
        if os.path.exists(args.model):
            config_path = os.path.join(args.model, "config.json")
            print(f"Loading config from: {config_path}")
            config = Gemma3Config.from_json(config_path)
        else:
            # Load from HuggingFace
            print(f"Loading config from HuggingFace: {args.model}")
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            config = Gemma3Config(
                hidden_size=hf_config.hidden_size,
                intermediate_size=hf_config.intermediate_size,
                num_hidden_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_key_value_heads,
                vocab_size=hf_config.vocab_size,
                rms_norm_eps=hf_config.rms_norm_eps,
                head_dim=getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads),
            )

        # Update config for conversion
        config.context_length = args.context_length
        config.state_length = max(config.state_length, args.context_length)

        print(f"Creating model with config:")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_hidden_layers: {config.num_hidden_layers}")
        print(f"  vocab_size: {config.vocab_size}")

        model = Gemma3ForCausalLM(config, enable_coreml=True)

        # Load weights
        if os.path.exists(args.model):
            print(f"Loading weights from: {args.model}")
            model.load_pretrained_weights(args.model)
        else:
            print(f"Loading weights from HuggingFace: {args.model}")
            converter_temp = Gemma3Converter(model, context_length=args.context_length)
            converter_temp.load_weights_from_hf(args.model)

        # Set model to eval mode
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Create converter
        converter = Gemma3Converter(
            model=model,
            context_length=args.context_length,
            batch_size=args.batch_size,
            lut_bits=args.lut,
            num_chunks=args.chunks,
        )

    # Run conversion steps
    try:
        if args.only:
            # Run specific step only
            if args.only == "1":
                convert_embeddings(model, converter, args.output, args.prefix, args.lut)
            elif args.only == "2":
                convert_ffn(model, converter, args.output, args.prefix, args.lut, args.chunks)
            elif args.only == "2_prefill":
                convert_prefill(model, converter, args.output, args.prefix, args.lut, args.chunks)
            elif args.only == "3":
                convert_lm_head(model, converter, args.output, args.prefix, args.lut)
            elif args.only == "combine":
                combine_models(args.output, args.prefix, args.lut, args.chunks)
            elif args.only == "meta":
                create_meta_yaml(args.output, args.prefix, args.context_length, args.batch_size,
                               args.lut, args.chunks, args.model)
            elif args.only == "test":
                test_inference(args.output, args.prefix, args.lut, args.chunks, args.context_length)
        else:
            # Run full pipeline
            if not args.skip_conversion:
                convert_embeddings(model, converter, args.output, args.prefix, args.lut)
                convert_lm_head(model, converter, args.output, args.prefix, args.lut)
                convert_ffn(model, converter, args.output, args.prefix, args.lut, args.chunks)
                convert_prefill(model, converter, args.output, args.prefix, args.lut, args.chunks)

            combine_models(args.output, args.prefix, args.lut, args.chunks)
            create_meta_yaml(args.output, args.prefix, args.context_length, args.batch_size,
                           args.lut, args.chunks, args.model)
            test_inference(args.output, args.prefix, args.lut, args.chunks, args.context_length)

        print("\n" + "=" * 60)
        print("CONVERSION COMPLETE")
        print("=" * 60)
        print(f"\nConverted models saved to: {args.output}")
        print(f"\nTo test with chat.py:")
        print(f"  python tests/chat.py --meta {args.output}/meta.yaml --tokenizer {args.model}")

    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
