import torch
from torch.nn import functional as F

from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open

import warnings
warnings.filterwarnings("ignore", message=".*Applying the CPU mse kernel on half-type tensors.*")

DTYPE = torch.float16 # or torch.float32

torch.set_grad_enabled(False)

def snap_to_codebook(x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """
    Replace every element of `x` with the nearest value in the 1-D `cb`
    in O(N log K) time and O(N+K) memory  (N = x.numel(), K = cb.numel()).

    Works by:
      1. sorting the code-book                   – O(K log K)
      2. computing mid-points between entries    – O(K)
      3. bucketising each element of x           – O(N log K)
    """
    if cb.ndim != 1:
        cb = cb.view(-1)              # (K,)

    cb_sorted, _ = torch.sort(cb)     # (K,)  ascending
    mid = (cb_sorted[:-1] + cb_sorted[1:]) * 0.5   # (K-1,)

    # idx ∈ [0 … K-1] gives, for each x element, the index of the nearest code word
    idx = torch.bucketize(x.view(-1), mid)

    return cb_sorted[idx].view_as(x)


def compare_full_precision_to_quantized(base, quant, quant_tensors):
    """
    Compare original unquantized model to fake quantized model (where
    linear.weight is not 16 unique values) and real quantized model
    (where linear.weight is 16 unique values and input/output scales
    are applied).
    """
    print("-> Comparing original unquantized model to fake and real quantized versions.\n")
    input_ids = torch.tensor([9707, 11, 847, 829, 374], dtype=torch.long)

    base_x = base.model.embed_tokens(input_ids)
    base_x = base.model.layers[0].input_layernorm(base_x)
    base_x = base.model.layers[0].self_attn.q_proj(base_x)
    print("Unquantized", base_x)

    fake_quant_x = quant.model.embed_tokens(input_ids)
    fake_quant_x = quant.model.layers[0].input_layernorm(fake_quant_x)
    fake_quant_x = quant.model.layers[0].self_attn.q_proj(fake_quant_x)
    print("Fake Quant", fake_quant_x)
    print("Unquantized <> Fake Quant MSE", F.mse_loss(base_x, fake_quant_x))

    real_quant_x = quant.model.embed_tokens(input_ids)
    real_quant_x = quant.model.layers[0].input_layernorm(real_quant_x)

    # Load relevant tensors for q_proj.
    codebook = quant_tensors.get_tensor("model.layers.0.self_attn.q_proj.codebook")
    input_scales = quant_tensors.get_tensor("model.layers.0.self_attn.q_proj.input_scales")
    output_scales = quant_tensors.get_tensor("model.layers.0.self_attn.q_proj.output_scales")
    fake_weight = quant_tensors.get_tensor("model.layers.0.self_attn.q_proj.weight")
    fake_bias = quant_tensors.get_tensor("model.layers.0.self_attn.q_proj.bias")

    # Same as AutoModel. ("Qwen/Qwen2.5-0.5B" has bias on q_proj already, for other projs need to manually add it.)
    assert torch.equal(quant.model.layers[0].self_attn.q_proj(real_quant_x), \
        F.linear(real_quant_x, fake_weight.to(DTYPE), fake_bias.to(DTYPE))), f"F.linear should equal AutoModel-loaded weights."

    quant_weight = snap_to_codebook(fake_weight / (input_scales*output_scales), codebook).to(DTYPE)
    assert quant_weight.unique().shape[0] == 16, f"Should be 4-bit quantized."
    quant_bias = (fake_bias / output_scales.squeeze()).to(DTYPE)

    real_quant_x *= input_scales.squeeze().to(DTYPE)
    real_quant_x = F.linear(real_quant_x, quant_weight, quant_bias)
    real_quant_x *= output_scales.squeeze().to(DTYPE)

    print("Real Quant", real_quant_x)
    print("Unquantized <> Real Quant MSE", F.mse_loss(base_x, real_quant_x))
    print("Fake Quant <> Real Quant MSE", F.mse_loss(fake_quant_x, real_quant_x))


def compare_fake_to_real_gate_proj(base, quant, quant_tensors):
    """
    Compare fake and real quantized results for a layer that does not
    have bias in the unquantized Qwen model.
    """
    print("-> Comparing fake and real quantized versions for gate_proj, which has no bias in unquantized Qwen model.\n")
    codebook = quant_tensors.get_tensor("model.layers.0.mlp.gate_proj.codebook")
    input_scales = quant_tensors.get_tensor("model.layers.0.mlp.gate_proj.input_scales")
    output_scales = quant_tensors.get_tensor("model.layers.0.mlp.gate_proj.output_scales")
    fake_weight = quant_tensors.get_tensor("model.layers.0.mlp.gate_proj.weight")
    fake_bias = quant_tensors.get_tensor("model.layers.0.mlp.gate_proj.bias")

    quant_weight = snap_to_codebook(fake_weight / (input_scales*output_scales), codebook).to(DTYPE)
    assert quant_weight.unique().shape[0] == 16, f"Should be 4-bit quantized."
    quant_bias = (fake_bias / output_scales.squeeze()).to(DTYPE)

    hidden = torch.randn(5, 896, dtype=DTYPE)
    fake_quant_x = F.linear(hidden, fake_weight, fake_bias)

    real_quant_x = hidden * input_scales.squeeze().to(DTYPE)
    real_quant_x = F.linear(real_quant_x, quant_weight, quant_bias)
    real_quant_x *= output_scales.squeeze().to(DTYPE)

    print("Fake Quant", fake_quant_x)
    print("Real Quant", real_quant_x)
    print("Fake Quant <> Real Quant MSE", F.mse_loss(fake_quant_x, real_quant_x))

if __name__ == "__main__":
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=DTYPE)
    quant = AutoModelForCausalLM.from_pretrained("smpanaro/Qwen2.5-0.5B-4bit-PerTensor", torch_dtype=DTYPE)
    quant_path = snapshot_download(repo_id="smpanaro/Qwen2.5-0.5B-4bit-PerTensor")
    quant_tensors = safe_open(f"{quant_path}/model.safetensors", "pt")

    compare_full_precision_to_quantized(base, quant, quant_tensors)
    print("\n" + "-"*40 + "\n")
    compare_fake_to_real_gate_proj(base, quant, quant_tensors)