#!/usr/bin/env python3
"""Compare CoreML vs PyTorch generation step-by-step.

This test runs both models on the same prompt and tracks when their
next-token predictions diverge. It feeds PyTorch's chosen token to both
models to isolate numeric drift from token selection.
"""

import argparse
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from transformers import AutoTokenizer

sys.path.insert(0, "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll")

from anemll.models.gemma3_model import Gemma3ForCausalLM, Gemma3Config, MODEL_DTYPE
from anemll.ane_converter.gemma3_converter import Gemma3Converter


def load_pytorch_model(model_path: str, context_length: int) -> Gemma3ForCausalLM:
    config = Gemma3Config.from_json(f"{model_path}/config.json")
    config.context_length = context_length
    config.state_length = context_length
    model = Gemma3ForCausalLM(config)
    converter = Gemma3Converter(model, context_length=context_length, batch_size=64)
    converter.load_weights_from_hf(model_path)
    model.eval()
    return model


def load_coreml_models(coreml_dir: str):
    embed = ct.models.MLModel(f"{coreml_dir}/gemma3_embeddings.mlpackage")
    ffn = ct.models.MLModel(f"{coreml_dir}/gemma3_FFN_chunk_01of01.mlpackage")
    lmhead = ct.models.MLModel(f"{coreml_dir}/gemma3_lm_head.mlpackage")
    return embed, ffn, lmhead


def make_causal_mask(pos: int, context_length: int) -> torch.Tensor:
    mask = torch.zeros((1, 1, 1, context_length), dtype=MODEL_DTYPE)
    if pos + 1 < context_length:
        mask[0, 0, 0, pos + 1 :] = float("-inf")
    return mask


def pt_step(
    model: Gemma3ForCausalLM,
    token_id: int,
    pos: int,
    context_length: int,
) -> torch.Tensor:
    token = torch.tensor([[token_id]], dtype=torch.long)
    with torch.no_grad():
        embed = model.model.embed_tokens(token)
        embed = (embed * model.model.embedding_scale).to(MODEL_DTYPE)
        position_ids = torch.tensor([pos], dtype=torch.int32)
        causal_mask = make_causal_mask(pos, context_length)
        current_pos = torch.tensor([pos], dtype=torch.int32)
        out = model.model.process_layers(
            embed,
            position_ids,
            causal_mask,
            current_pos,
            start_layer=0,
            end_layer=None,
            IN_PREFILL=False,
        )
        out = model.model.norm(out)
    return out


def cm_step(
    embed_model,
    ffn_model,
    state,
    token_id: int,
    pos: int,
    context_length: int,
) -> np.ndarray:
    token = np.array([[token_id]], dtype=np.int32)
    embed = embed_model.predict({"input_ids": token})["hidden_states"]
    position_ids = np.array([pos], dtype=np.int32)
    causal_mask = make_causal_mask(pos, context_length).numpy().astype(np.float16)
    current_pos = np.array([pos], dtype=np.int32)
    out = ffn_model.predict(
        {
            "hidden_states": embed.astype(np.float16),
            "position_ids": position_ids,
            "causal_mask": causal_mask,
            "current_pos": current_pos,
        },
        state,
    )["output_hidden_states"]
    return out


def pt_logits(model: Gemma3ForCausalLM, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_for_lm = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
    parts = []
    for i in range(1, 17):
        lm_head = getattr(model, f"lm_head16_{i}")
        part = lm_head(hidden_for_lm).squeeze(2).transpose(1, 2)
        parts.append(part)
    return torch.cat(parts, dim=-1).squeeze(1)


def cm_logits(lmhead_model, hidden_states: np.ndarray) -> np.ndarray:
    out = lmhead_model.predict({"hidden_states": hidden_states.astype(np.float16)})
    parts = [out[f"logits{i}"] for i in range(1, 17)]
    return np.concatenate(parts, axis=-1).squeeze()


def compare_generation(
    model_path: str,
    coreml_dir: str,
    context_length: int,
    prompt: str,
    steps: int,
) -> None:
    print("Loading models...")
    pt_model = load_pytorch_model(model_path, context_length)
    cm_embed, cm_ffn, cm_lmhead = load_coreml_models(coreml_dir)
    cm_state = cm_ffn.make_state()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = input_ids.shape[1]

    print(f"Prompt length: {prompt_len} tokens")
    print("Running prompt tokens...")

    pt_model.model.kv_cache_0.zero_()
    last_pt_hidden = None
    last_cm_hidden = None
    for i in range(prompt_len):
        token_id = int(input_ids[0, i].item())
        last_pt_hidden = pt_step(pt_model, token_id, i, context_length)
        last_cm_hidden = cm_step(cm_embed, cm_ffn, cm_state, token_id, i, context_length)

    if last_pt_hidden is None or last_cm_hidden is None:
        print("No prompt tokens processed.")
        return

    # First next-token prediction from the last prompt hidden state.
    pt_log = pt_logits(pt_model, last_pt_hidden)
    cm_log = cm_logits(cm_lmhead, last_cm_hidden)
    pt_next = int(torch.argmax(pt_log).item())
    cm_next = int(np.argmax(cm_log))
    print(f"Step 0 predicted token: PT={pt_next} CM={cm_next} match={pt_next == cm_next}")

    # Generate steps, feeding PT token to both.
    current_token = pt_next
    pos = prompt_len
    first_divergence = None
    for step in range(1, steps + 1):
        pt_hidden = pt_step(pt_model, current_token, pos, context_length)
        cm_hidden = cm_step(cm_embed, cm_ffn, cm_state, current_token, pos, context_length)

        pt_log = pt_logits(pt_model, pt_hidden)
        cm_log = cm_logits(cm_lmhead, cm_hidden)
        pt_next = int(torch.argmax(pt_log).item())
        cm_next = int(np.argmax(cm_log))

        if pt_next != cm_next and first_divergence is None:
            first_divergence = step
            diff = np.max(np.abs(pt_log.detach().cpu().numpy() - cm_log))
            pt_topk = torch.topk(pt_log, k=5)
            pt_top = pt_topk.indices.tolist()
            pt_vals = pt_topk.values.tolist()
            cm_top = np.argsort(cm_log)[-5:][::-1].tolist()
            cm_vals = [float(cm_log[idx]) for idx in cm_top]
            pt_margin = float(pt_vals[0] - pt_vals[1]) if len(pt_vals) > 1 else 0.0
            cm_margin = float(cm_vals[0] - cm_vals[1]) if len(cm_vals) > 1 else 0.0
            print(
                f"Divergence at step {step}: PT={pt_next} CM={cm_next} "
                f"max_logit_diff={diff:.4f} pt_margin={pt_margin:.4f} cm_margin={cm_margin:.4f}"
            )
            print(f"  PT top5: {pt_top}")
            print(f"  CM top5: {cm_top}")

        if step % 25 == 0 or step == steps:
            diff = np.max(np.abs(pt_log.detach().cpu().numpy() - cm_log))
            print(f"Step {step}: max_logit_diff={diff:.4f} PT={pt_next} CM={cm_next}")

        current_token = pt_next
        pos += 1

    if first_divergence is None:
        print("No divergence in argmax tokens across steps.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--coreml-dir", required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()

    compare_generation(
        args.model_path,
        args.coreml_dir,
        args.context_length,
        args.prompt,
        args.steps,
    )


if __name__ == "__main__":
    main()
