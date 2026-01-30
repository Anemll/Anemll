#!/usr/bin/env python3
"""Test that the embeddings fix works - LUT should be skipped for fp16_scaled."""

import sys
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-0.3.5')

import numpy as np
from anemll.ane_converter.gemma3_converter import Gemma3Converter, _apply_fp16_scaling
from anemll.models.gemma3_model import Gemma3ForCausalLM

model_path = "google/gemma-3-4b-it-qat-int4-unquantized"

print("Loading model...")
model = Gemma3ForCausalLM.from_pretrained(model_path, context_length=4096, batch_size=64)
print("Model loaded")

print("\nApplying FP16 scaling (alpha=auto)...")
alpha = _apply_fp16_scaling(model, "auto", model_path)
print(f"Applied with alpha={alpha}")

print("\nCreating converter with fp16_scaled=True...")
converter = Gemma3Converter(
    model,
    context_length=4096,
    batch_size=64,
    lut_bits=4,
    per_channel=0,
    num_chunks=2,
    argmax_in_model=True,
    fp16_scaled=True,
)

print("\nConverting embeddings (should skip LUT)...")
embed_model = converter.convert_embeddings(model)

print("\nTesting embeddings...")
for tok in [2, 108, 573, 1000]:
    out = embed_model.predict({'input_ids': np.array([[tok]], dtype=np.int32)})
    h = np.array(out['hidden_states'])
    print(f'Token {tok}: range=[{h.min():.4f}, {h.max():.4f}]')

# Check if tokens have different values
out1 = embed_model.predict({'input_ids': np.array([[2]], dtype=np.int32)})
out2 = embed_model.predict({'input_ids': np.array([[108]], dtype=np.int32)})
h1 = np.array(out1['hidden_states'])
h2 = np.array(out2['hidden_states'])

if np.allclose(h1, h2):
    print("\n❌ FAIL: Tokens 2 and 108 have same embeddings!")
    sys.exit(1)
else:
    print("\n✅ PASS: Tokens have different embeddings!")
    sys.exit(0)
