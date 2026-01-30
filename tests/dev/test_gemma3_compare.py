#!/usr/bin/env python3
"""Compare CoreML vs HF logits and track divergence.

This is a generic harness (despite the filename) that supports Gemma/Qwen-style
CoreML exports. It can be driven by CoreML or PyTorch tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_meta_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        meta = p / "meta.yaml"
        if meta.exists():
            return meta
    if p.is_file():
        return p
    raise FileNotFoundError(f"meta.yaml not found: {path}")


def _load_meta(meta_path: Path) -> Dict[str, Any]:
    import yaml

    with meta_path.open("r") as f:
        meta = yaml.safe_load(f)
    if "model_info" not in meta or "parameters" not in meta["model_info"]:
        raise ValueError("meta.yaml missing model_info.parameters")
    return meta


def _candidate_paths(base_dir: Path, name: str) -> List[Path]:
    p = Path(name)
    if p.is_absolute():
        base_dir = Path("/")
        name = p.name
    stem = name
    if stem.endswith(".mlmodelc") or stem.endswith(".mlpackage"):
        stem = stem.rsplit(".", 1)[0]
    candidates = [
        base_dir / f"{stem}.mlpackage",
        base_dir / f"{stem}.mlmodelc",
        base_dir / stem,
    ]
    return [c for c in candidates if c.exists()]


def _find_chunk_paths(base_dir: Path, base_name: str) -> List[Path]:
    from glob import glob
    import re

    stem = base_name
    if stem.endswith(".mlmodelc") or stem.endswith(".mlpackage"):
        stem = stem.rsplit(".", 1)[0]

    # If stem contains a specific chunk like _chunk_01of08, replace with wildcard
    # to find all chunks
    stem = re.sub(r"_chunk_\d+of\d+", "_chunk_*of*", stem)

    pattern = str(base_dir / stem)
    if "_chunk_" not in pattern:
        pattern += "_chunk_*of*"

    pkg_paths = [Path(p) for p in glob(pattern + ".mlpackage")]
    mlc_paths = [Path(p) for p in glob(pattern + ".mlmodelc")]

    # Prefer .mlmodelc (compiled) when both exist for the same chunk stem.
    pkg_by_stem = {p.with_suffix("").name: p for p in pkg_paths}
    mlc_by_stem = {p.with_suffix("").name: p for p in mlc_paths}

    combined = []
    for stem_name in sorted(set(pkg_by_stem.keys()) | set(mlc_by_stem.keys())):
        if stem_name in mlc_by_stem:
            combined.append(mlc_by_stem[stem_name])
        else:
            combined.append(pkg_by_stem[stem_name])

    return [p for p in combined if p.exists()]


def _load_coreml_model(path: Path, function_name: str | None = None):
    """Load CoreML model - handles both .mlpackage and .mlmodelc formats."""
    path_str = str(path)
    if path_str.endswith(".mlmodelc"):
        # Compiled model - use CompiledMLModel
        compute_unit = ct.ComputeUnit.CPU_AND_NE
        if function_name:
            return ct.models.CompiledMLModel(path_str, compute_unit, function_name=function_name)
        return ct.models.CompiledMLModel(path_str, compute_unit)
    else:
        # Package model - use MLModel
        if function_name:
            return ct.models.MLModel(path_str, function_name=function_name)
        return ct.models.MLModel(path_str)


def _model_input_names(model: ct.models.MLModel, function_name: str | None = None) -> List[str]:
    spec = model.get_spec()
    if spec.description.input:
        return [i.name for i in spec.description.input]

    # MLProgram functions live under spec.mlProgram.functions.
    fn_name = function_name
    if getattr(spec, "mlProgram", None) is not None:
        functions = getattr(spec.mlProgram, "functions", None)
        if functions:
            if fn_name and fn_name in functions:
                return [i.name for i in functions[fn_name].inputs]
            # Fallback to first function with inputs
            for _, fn in functions.items():
                if fn.inputs:
                    return [i.name for i in fn.inputs]

    return []


def _model_input_names_from_path(path: Path, function_name: str | None = None) -> List[str]:
    model = ct.models.MLModel(str(path))
    return _model_input_names(model, function_name)


def _get_output_key(output: Dict[str, np.ndarray]) -> str:
    if "output_hidden_states" in output:
        return "output_hidden_states"
    return list(output.keys())[0]


def _combine_logits(lm_out: Dict[str, np.ndarray], split_lm_head: int | None) -> np.ndarray:
    if "logits1" in lm_out:
        parts = []
        max_parts = split_lm_head or 16
        for i in range(1, max_parts + 1):
            key = f"logits{i}"
            if key in lm_out:
                parts.append(lm_out[key])
        if not parts:
            raise ValueError("split_lm_head outputs not found")
        logits = np.concatenate(parts, axis=-1)
    else:
        logits = lm_out.get("output_logits")
        if logits is None:
            logits = lm_out[list(lm_out.keys())[0]]
    return logits.squeeze()


def _build_causal_mask(context_length: int) -> np.ndarray:
    mask = np.full((1, 1, context_length, context_length), -np.inf, dtype=np.float16)
    for i in range(context_length):
        mask[0, 0, i, : i + 1] = 0.0
    return mask


class CoreMLRunner:
    def __init__(self, meta_path: Path):
        meta = _load_meta(meta_path)
        params = meta["model_info"]["parameters"]
        base_dir = meta_path.parent

        self.context_length = int(params["context_length"])
        self.batch_size = int(params.get("batch_size", 64))
        self.split_lm_head = params.get("split_lm_head")
        self.model_prefix = params.get("model_prefix", "model")
        self.num_chunks = int(params.get("num_chunks", 1))

        embed_name = params.get("embeddings", f"{self.model_prefix}_embeddings")
        lm_head_name = params.get("lm_head", f"{self.model_prefix}_lm_head")
        ffn_name = params.get("ffn", f"{self.model_prefix}_FFN_PF_chunk_01of{self.num_chunks:02d}")

        embed_paths = _candidate_paths(base_dir, embed_name)
        if not embed_paths:
            raise FileNotFoundError(f"Embeddings model not found: {embed_name}")
        self.embed_model = _load_coreml_model(embed_paths[0])

        lm_paths = _candidate_paths(base_dir, lm_head_name)
        if not lm_paths:
            raise FileNotFoundError(f"LM head model not found: {lm_head_name}")
        self.lm_head = _load_coreml_model(lm_paths[0])

        chunk_paths = _find_chunk_paths(base_dir, ffn_name)
        if not chunk_paths:
            raise FileNotFoundError(f"FFN model not found: {ffn_name}")

        self.ffn_models: List[Dict[str, ct.models.MLModel]] = []
        for chunk in chunk_paths:
            try:
                infer = _load_coreml_model(chunk, function_name="infer")
            except Exception:
                infer = _load_coreml_model(chunk)
            self.ffn_models.append({"infer": infer})
        self._causal_mask = _build_causal_mask(self.context_length)
        # Create per-chunk states (each chunk has its own KV cache)
        self._states = [chunk["infer"].make_state() for chunk in self.ffn_models]

    def reset(self) -> None:
        self._states = [chunk["infer"].make_state() for chunk in self.ffn_models]

    def _predict_infer(
        self, model: ct.models.MLModel, state, inputs: Dict[str, np.ndarray], update_mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        try:
            return model.predict(inputs, state)
        except RuntimeError as e:
            msg = str(e)
            if "update_mask" in msg and "required" in msg:
                inputs_with_mask = dict(inputs)
                inputs_with_mask["update_mask"] = update_mask
                return model.predict(inputs_with_mask, state)
            raise

    def step(self, token_id: int, pos: int) -> np.ndarray:
        # Embeddings
        token_arr = np.array([[token_id]], dtype=np.int32)
        if self.batch_size > 1:
            padded = np.zeros((1, self.batch_size), dtype=np.int32)
            padded[0, 0] = token_id
            token_arr = padded
        embed_out = self.embed_model.predict({"input_ids": token_arr})
        hidden = embed_out["hidden_states"]
        if hidden.shape[1] > 1:
            hidden = hidden[:, :1, :]

        position_ids = np.array([pos], dtype=np.int32)
        current_pos = np.array([pos], dtype=np.int32)
        single_mask = self._causal_mask[:, :, pos : pos + 1, :]

        update_mask = np.zeros((1, 1, self.context_length, 1), dtype=np.float16)
        update_mask[0, 0, pos, 0] = 1.0

        for chunk_idx, chunk in enumerate(self.ffn_models):
            inputs = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": position_ids,
                "causal_mask": single_mask.astype(np.float16),
                "current_pos": current_pos,
            }
            out = self._predict_infer(chunk["infer"], self._states[chunk_idx], inputs, update_mask)
            hidden = out[_get_output_key(out)]

        lm_out = self.lm_head.predict({"hidden_states": hidden.astype(np.float16)})
        return _combine_logits(lm_out, self.split_lm_head)


def _tokenize_prompt(tokenizer, prompt: str, use_template: bool, no_think: bool) -> List[int]:
    if use_template and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if no_think:
            messages.append({"role": "system", "content": "Respond without reasoning."})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    return tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()


def _compute_metrics(hf_logits: np.ndarray, cm_logits: np.ndarray) -> Dict[str, float]:
    hf = hf_logits.astype(np.float64)
    cm = cm_logits.astype(np.float64)
    hf_probs = F.softmax(torch.from_numpy(hf), dim=-1).numpy()
    cm_probs = F.softmax(torch.from_numpy(cm), dim=-1).numpy()

    eps = 1e-12
    kl = float(np.sum(hf_probs * (np.log(hf_probs + eps) - np.log(cm_probs + eps))))

    hf_mean = hf.mean()
    cm_mean = cm.mean()
    hf_std = hf.std()
    cm_std = cm.std()
    if hf_std < 1e-12 or cm_std < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.mean((hf - hf_mean) * (cm - cm_mean)) / (hf_std * cm_std))

    entropy = float(-np.sum(cm_probs * np.log(cm_probs + eps)))
    match = int(np.argmax(hf) == np.argmax(cm))

    return {"kl": kl, "corr": corr, "entropy": entropy, "match": match}


def compare_prompt(
    meta_path: Path,
    hf_reference: str,
    prompt: str,
    max_tokens: int,
    driver: str,
    no_think: bool,
    use_template: bool,
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(hf_reference, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_reference, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    hf_model.eval()

    tokens = _tokenize_prompt(tokenizer, prompt, use_template, no_think)
    if len(tokens) < 1:
        raise ValueError("Prompt produced no tokens")

    cm = CoreMLRunner(meta_path)
    cm.reset()

    # Prime state with all but last token
    for pos in range(len(tokens) - 1):
        cm.step(tokens[pos], pos)

    metrics: Dict[str, List[float]] = {"kl": [], "corr": [], "entropy": [], "match": []}
    first_div = None

    for step in range(max_tokens):
        pos = len(tokens) - 1
        cm_logits = cm.step(tokens[-1], pos)

        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            hf_out = hf_model(input_ids=input_ids)
            hf_logits = hf_out.logits[0, -1, :].float().cpu().numpy()

        m = _compute_metrics(hf_logits, cm_logits)
        for k in metrics:
            metrics[k].append(m[k])

        if m["match"] == 0 and first_div is None:
            first_div = step

        if driver == "coreml":
            next_token = int(np.argmax(cm_logits))
        else:
            next_token = int(np.argmax(hf_logits))

        tokens.append(next_token)

    gen_tokens = tokens
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=False)

    summary = {
        "prompt": prompt,
        "generated_text": decoded,
        "steps": max_tokens,
        "driver": driver,
        "first_divergence": first_div,
        "metrics": {
            "kl_mean": float(np.mean(metrics["kl"])) if metrics["kl"] else float("nan"),
            "corr_mean": float(np.nanmean(metrics["corr"])) if metrics["corr"] else float("nan"),
            "entropy_mean": float(np.mean(metrics["entropy"])) if metrics["entropy"] else float("nan"),
            "match_rate": float(np.mean(metrics["match"])) if metrics["match"] else float("nan"),
        },
    }
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare CoreML vs HF divergence.")
    parser.add_argument("coreml", type=str, help="Path to CoreML model dir or meta.yaml")
    parser.add_argument("--hf-reference", type=str, required=True, help="HF model ID or path")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--driver", choices=["coreml", "pt"], default="coreml", help="Token driver")
    parser.add_argument("--no-think", action="store_true", help="Disable reasoning via system prompt")
    parser.add_argument("--no-template", action="store_true", help="Disable chat template")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    meta_path = _resolve_meta_path(args.coreml)
    summary = compare_prompt(
        meta_path=meta_path,
        hf_reference=args.hf_reference,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        driver=args.driver,
        no_think=args.no_think,
        use_template=not args.no_template,
    )

    print(json.dumps(summary["metrics"], indent=2))
    print(f"\nFirst divergence: {summary['first_divergence']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
