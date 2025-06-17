#!/usr/bin/env python3
"""Direct comparison without hooks to resolve discrepancy."""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig
import warnings
warnings.filterwarnings('ignore')

def test_direct_comparison():
    print("🔍 Direct Comparison (No Hooks) - Resolving Discrepancy")
    print("=" * 65)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    
    # Load models
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    
    print("Loading official model...")
    official_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.float32
    )
    official_model = official_model.to("cpu")
    official_model.eval()
    
    print("Loading custom model...")
    config = QwenConfig.from_json(os.path.join(model_path, "config.json"))
    config.context_length = 256
    custom_model = QwenForCausalLM(config, enable_coreml=False)
    success = custom_model.load_pretrained_weights(model_path)
    if not success:
        raise RuntimeError("Failed to load custom model weights")
    custom_model.eval()
    
    # Test identical inputs
    test_token = 3838  # "What"
    CONTEXT_LENGTH = 256
    PAD_TOKEN = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    print(f"\n📊 Test Setup:")
    print(f"  Token: {test_token} ('What')")
    print(f"  Context length: {CONTEXT_LENGTH}")
    print(f"  Pad token: {PAD_TOKEN}")
    
    # === Official Model Test ===
    print(f"\n🔍 Official Model Test:")
    input_ids_official = torch.tensor([[test_token] + [PAD_TOKEN] * (CONTEXT_LENGTH - 1)], dtype=torch.long)
    attention_mask_official = torch.zeros_like(input_ids_official)
    attention_mask_official[0, 0] = 1
    
    with torch.no_grad():
        outputs_official = official_model(
            input_ids=input_ids_official,
            attention_mask=attention_mask_official,
            return_dict=True
        )
        logits_official = outputs_official.logits[0, 0, :]  # Extract position 0
    
    top_indices_official = torch.argsort(logits_official, descending=True)[:5]
    print(f"  Top 5 tokens: {top_indices_official.tolist()}")
    print(f"  Top 5 logits: {[logits_official[idx].item() for idx in top_indices_official]}")
    
    # === Custom Model Test ===
    print(f"\n🔍 Custom Model Test:")
    input_ids_custom = torch.tensor([[test_token] + [PAD_TOKEN] * (CONTEXT_LENGTH - 1)], dtype=torch.long)
    position_ids_custom = torch.arange(CONTEXT_LENGTH, dtype=torch.long)
    
    causal_mask_custom = torch.full((1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH), -float('inf'), dtype=torch.float32)
    for i in range(CONTEXT_LENGTH):
        for j in range(i + 1):
            causal_mask_custom[0, 0, i, j] = 0
    
    current_pos_custom = torch.tensor([0], dtype=torch.long)
    update_mask_custom = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=torch.float32)
    
    with torch.no_grad():
        outputs_custom = custom_model(
            input_ids=input_ids_custom,
            position_ids=position_ids_custom,
            causal_mask=causal_mask_custom,
            current_pos=current_pos_custom,
            update_mask=update_mask_custom,
            IN_PREFILL=False
        )
        
        # Handle output format
        if isinstance(outputs_custom, tuple):
            logits_custom = torch.cat(outputs_custom, dim=-1)[0, 0, :]
        else:
            logits_custom = outputs_custom[0, 0, :]
    
    top_indices_custom = torch.argsort(logits_custom, descending=True)[:5]
    print(f"  Top 5 tokens: {top_indices_custom.tolist()}")
    print(f"  Top 5 logits: {[logits_custom[idx].item() for idx in top_indices_custom]}")
    
    # === Comparison ===
    print(f"\n🆚 DIRECT COMPARISON:")
    print(f"  Official: {top_indices_official.tolist()}")
    print(f"  Custom:   {top_indices_custom.tolist()}")
    print(f"  Match:    {torch.equal(top_indices_official, top_indices_custom)}")
    
    # === Historical Comparison ===
    print(f"\n📚 HISTORICAL RESULTS:")
    print(f"  Previous Official Test: [15846, 21806, 3405, 38297, 13355]")
    print(f"  Previous Custom Test:   [15846, 3405, 21806, 38297, 13355]")
    print(f"  Per-Tensor Comparison:  [15846, 3405, 21806, 38297, 13355] (both)")
    print(f"  Current Official:       {top_indices_official.tolist()}")
    print(f"  Current Custom:         {top_indices_custom.tolist()}")
    
    # Check logit differences for top tokens
    print(f"\n📊 LOGIT ANALYSIS:")
    common_tokens = [15846, 21806, 3405, 38297, 13355]
    for token in common_tokens:
        official_logit = logits_official[token].item()
        custom_logit = logits_custom[token].item()
        diff = abs(official_logit - custom_logit)
        print(f"  Token {token}: Official={official_logit:.4f}, Custom={custom_logit:.4f}, Diff={diff:.4f}")
    
    # Check which logits are very close (might explain ranking differences)
    print(f"\n🔍 CLOSE COMPARISONS (tokens 21806 vs 3405):")
    print(f"  Official - Token 21806: {logits_official[21806].item():.6f}")
    print(f"  Official - Token 3405:  {logits_official[3405].item():.6f}")
    print(f"  Official - Difference:  {abs(logits_official[21806] - logits_official[3405]).item():.6f}")
    print(f"")
    print(f"  Custom - Token 21806:   {logits_custom[21806].item():.6f}")
    print(f"  Custom - Token 3405:    {logits_custom[3405].item():.6f}")
    print(f"  Custom - Difference:    {abs(logits_custom[21806] - logits_custom[3405]).item():.6f}")
    
    return top_indices_official.tolist(), top_indices_custom.tolist()

if __name__ == "__main__":
    try:
        official_tokens, custom_tokens = test_direct_comparison()
        print(f"\n🎯 CONCLUSION:")
        if official_tokens == custom_tokens:
            print("✅ Models produce identical rankings")
        else:
            print("❌ Models produce different rankings")
            print("   → Need to investigate why rankings differ")
            print("   → Previous test discrepancies now explained")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc() 