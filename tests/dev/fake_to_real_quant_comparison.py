import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open

import warnings
warnings.filterwarnings("ignore", message=".*Applying the CPU mse kernel on half-type tensors.*")

DTYPE = torch.float16 # or torch.float32

torch.set_grad_enabled(False)

max_tokens = 200

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
    print("# Full Precision vs. Quantized")
    print("Compare the original unquantized model to fake and real quantized versions.\n")
    input_ids = torch.tensor([9707, 11, 847, 829, 374], dtype=torch.long)

    base_x = base.model.embed_tokens(input_ids)
    base_x = base.model.layers[0].input_layernorm(base_x)
    base_x = base.model.layers[0].self_attn.q_proj(base_x)
    print("Unquantized:\n```python\n", base_x, "\n```")

    fake_quant_x = quant.model.embed_tokens(input_ids)
    fake_quant_x = quant.model.layers[0].input_layernorm(fake_quant_x)
    fake_quant_x = quant.model.layers[0].self_attn.q_proj(fake_quant_x)
    print("Fake Quant:\n```python\n", fake_quant_x, "\n```")
    print("**Unquantized <> Fake Quant MSE:**", F.mse_loss(base_x, fake_quant_x))

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

    print("\n\nReal Quant:\n```python\n", real_quant_x, "\n```")
    print("\n**Unquantized <> Real Quant MSE:**", F.mse_loss(base_x, real_quant_x))
    print("\n**Fake Quant <> Real Quant MSE:**", F.mse_loss(fake_quant_x, real_quant_x))


def compare_fake_to_real_gate_proj(base, quant, quant_tensors):
    """
    Compare fake and real quantized results for a layer that does not
    have bias in the unquantized Qwen model.
    """
    print("# Fake vs. Real Quantized gate_proj")
    print("Compare the fake and real quantized versions for gate_proj, which has no bias in unquantized Qwen model.\n")
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

    print("Fake Quant:\n```python\n", fake_quant_x, "\n```")
    print("Real Quant:\n```python\n", real_quant_x, "\n```")
    print("**Fake Quant <> Real Quant MSE:**", F.mse_loss(fake_quant_x, real_quant_x))

def end_to_end_comparison(base, quant, quant_tensors, tokenizer):
    class ScaledLinear(nn.Module):
        def __init__(self, weight, bias, input_scales, output_scales):
            super().__init__()
            self.weight = weight
            self.bias = bias
            self.input_scales = input_scales
            self.output_scales = output_scales
        def forward(self, x):
            x = x * self.input_scales
            x = F.linear(x, self.weight, self.bias)
            x = x * self.output_scales
            return x

    def fake_to_real(key: str):
        codebook = quant_tensors.get_tensor(f"{key}.codebook")
        input_scales = quant_tensors.get_tensor(f"{key}.input_scales")
        output_scales = quant_tensors.get_tensor(f"{key}.output_scales")
        fake_weight = quant_tensors.get_tensor(f"{key}.weight")
        fake_bias = quant_tensors.get_tensor(f"{key}.bias")

        # .to(DTYPE) since scales are float32 in model.safetensors
        quant_weight = snap_to_codebook(fake_weight / (input_scales*output_scales), codebook).to(DTYPE)
        quant_bias = (fake_bias / output_scales.squeeze()).to(DTYPE)

        return ScaledLinear(quant_weight, quant_bias, input_scales.squeeze().to(DTYPE), output_scales.squeeze().to(DTYPE))

    prompts = [
        ("Non-Chat Prompt",
            tokenizer("Apple Neural Engine (ANE), explained in one sentence:", return_tensors="pt")
        )
    ]

    base_logits, base_generation, fake_quant_logits, fake_quant_generation = {}, {}, {}, {}
    for name, prompt in prompts:
        base_logits[name] = base(**prompt).logits
        base_generation[name] = base.generate(**prompt, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
        fake_quant_logits[name] = quant(**prompt).logits
        fake_quant_generation[name] = quant.generate(**prompt, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)

    # Fake -> Real
    for idx, layer in enumerate(quant.model.layers):
        layer.self_attn.q_proj = fake_to_real(f"model.layers.{idx}.self_attn.q_proj")
        layer.self_attn.k_proj = fake_to_real(f"model.layers.{idx}.self_attn.k_proj")
        layer.self_attn.v_proj = fake_to_real(f"model.layers.{idx}.self_attn.v_proj")
        layer.self_attn.o_proj = fake_to_real(f"model.layers.{idx}.self_attn.o_proj")

        layer.mlp.up_proj = fake_to_real(f"model.layers.{idx}.mlp.up_proj")
        layer.mlp.gate_proj = fake_to_real(f"model.layers.{idx}.mlp.gate_proj")
        layer.mlp.down_proj = fake_to_real(f"model.layers.{idx}.mlp.down_proj")

    real_quant_logits, real_quant_generation = {}, {}
    for name, prompt in prompts:
        real_quant_logits[name] = quant(**prompt).logits
        real_quant_generation[name] = quant.generate(**prompt, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)

    print("# End-to-End Generation")
    for name, prompt in prompts:
        print("##", name)

        input_length = len(prompt['input_ids'][0])
        base_length = len(base_generation[name][0])
        fake_quant_length = len(fake_quant_generation[name][0])
        real_quant_length = len(real_quant_generation[name][0])


        print(f"BOS token: {tokenizer.bos_token!r} (ID: {tokenizer.bos_token_id})")
        print(f"EOS token: {tokenizer.eos_token!r} (ID: {tokenizer.eos_token_id})")

        print(f"**Input Prompt** ({input_length} tokens)\n```\n{tokenizer.decode(prompt['input_ids'][0])}\n```")
        print(f"Token IDs: {prompt['input_ids'][0].tolist()}")

        print(f"**Base Generation** ({base_length - input_length} new tokens, {base_length} total)\n```\n{tokenizer.decode(base_generation[name][0])}\n```")
        print(f"**Fake Quant Generation** ({fake_quant_length - input_length} new tokens, {fake_quant_length} total)\n```\n{tokenizer.decode(fake_quant_generation[name][0])}\n```")
        print(f"**Real Quant Generation** ({real_quant_length - input_length} new tokens, {real_quant_length} total)\n```\n{tokenizer.decode(real_quant_generation[name][0])}\n```")
        print("\n**Real Quant Generation == Fake Quant Generation:**", torch.equal(real_quant_generation[name][0], fake_quant_generation[name][0]))

        # Expected that this is high-ish, quantization is lossy.
        print("\n**Unquantized <> Fake Quant KLDiv:**", F.kl_div(
            fake_quant_logits[name].log_softmax(dim=-1), base_logits[name].log_softmax(dim=-1), log_target=True, reduction="batchmean"),
            "(Note: This should be high due to quantization error.)")

        # This should be low.
        print("\n**Fake Quant <> Real Quant KLDiv:**", F.kl_div(
            real_quant_logits[name].log_softmax(dim=-1), fake_quant_logits[name].log_softmax(dim=-1), log_target=True, reduction="batchmean"))


if __name__ == "__main__":
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=DTYPE)
    quant = AutoModelForCausalLM.from_pretrained("smpanaro/Qwen2.5-0.5B-4bit-PerTensor", torch_dtype=DTYPE)
    quant_path = snapshot_download(repo_id="smpanaro/Qwen2.5-0.5B-4bit-PerTensor")
    quant_tensors = safe_open(f"{quant_path}/model.safetensors", "pt")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    compare_full_precision_to_quantized(base, quant, quant_tensors)
    print("\n" + "-"*40 + "\n")
    compare_fake_to_real_gate_proj(base, quant, quant_tensors)
    print("\n" + "-"*40 + "\n")
    end_to_end_comparison(base, quant, quant_tensors, tokenizer)