import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from anemll.models.qwen_model import (
    QwenConfig,
    QwenForCausalLM,
    MODEL_DTYPE,
    TEST_DEVICE,
    CONTEXT_LENGTH,
)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

def make_causal_mask(length, start, device=None, dtype=torch.float16):
    """Create causal attention mask."""
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return torch.tensor(mask, dtype=dtype, device=device)

@torch.no_grad()
def test_qwen_hf_vs_custom_inference():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=MODEL_DTYPE)
    hf_model.eval()
    hf_model = hf_model.to(TEST_DEVICE)

    prompt = "what is capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(TEST_DEVICE)

    hf_logits = hf_model(**inputs).logits
    hf_next = torch.argmax(hf_logits[:, -1, :], dim=-1)

    local_dir = snapshot_download(model_name)
    config = QwenConfig(**hf_model.config.to_dict())
    custom_model = QwenForCausalLM(config).to(TEST_DEVICE)
    assert custom_model.load_pretrained_weights(local_dir)
    custom_model.eval()

    seq_len = inputs["input_ids"].shape[1]
    causal_mask = make_causal_mask(seq_len, 0, device=TEST_DEVICE, dtype=MODEL_DTYPE)
    position_ids = torch.arange(seq_len, device=TEST_DEVICE).unsqueeze(0)
    update_mask = torch.zeros(
        (1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE
    )
    current_pos = torch.tensor([0], device=TEST_DEVICE)

    custom_logits = custom_model(
        input_ids=inputs["input_ids"],
        update_mask=update_mask,
        position_ids=position_ids,
        causal_mask=causal_mask,
        current_pos=current_pos,
        IN_PREFILL=False,
    )

    custom_next = torch.argmax(custom_logits[:, -1, :], dim=-1)

    # Add detailed comparison output
    print(f"\n=== DETAILED COMPARISON ===")
    print(f"Prompt: '{prompt}'")
    print(f"Input tokens: {inputs['input_ids'].tolist()}")
    print(f"Sequence length: {seq_len}")
    
    # Compare logit shapes
    print(f"\nHF logits shape: {hf_logits.shape}")
    print(f"Custom logits shape: {custom_logits.shape}")
    
    # Compare predicted tokens
    hf_token = hf_next.item()
    custom_token = custom_next.item()
    print(f"\nHF predicted token: {hf_token} -> '{tokenizer.decode([hf_token])}'")
    print(f"Custom predicted token: {custom_token} -> '{tokenizer.decode([custom_token])}'")
    print(f"Tokens match: {hf_token == custom_token}")
    
    # Compare top-5 predictions
    hf_top5 = torch.topk(hf_logits[:, -1, :], 5)
    custom_top5 = torch.topk(custom_logits[:, -1, :], 5)
    
    print(f"\nHF top-5 tokens: {hf_top5.indices.tolist()[0]}")
    print(f"HF top-5 logits: {hf_top5.values.tolist()[0]}")
    print(f"Custom top-5 tokens: {custom_top5.indices.tolist()[0]}")
    print(f"Custom top-5 logits: {custom_top5.values.tolist()[0]}")
    
    # Compare logit similarity
    last_hf_logits = hf_logits[:, -1, :]
    last_custom_logits = custom_logits[:, -1, :]
    
    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(last_hf_logits, last_custom_logits, dim=-1)
    print(f"\nCosine similarity: {cos_sim.item():.6f}")
    
    # Calculate max absolute difference
    max_diff = torch.max(torch.abs(last_hf_logits - last_custom_logits)).item()
    print(f"Max absolute difference: {max_diff:.6f}")
    
    # Calculate mean absolute difference
    mean_diff = torch.mean(torch.abs(last_hf_logits - last_custom_logits)).item()
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    print(f"=========================\n")

    assert torch.equal(hf_next, custom_next)
