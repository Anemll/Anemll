#!/usr/bin/env python3
"""
Debug version of official MLX evaluation to see exactly what it's doing.
Based on mlx_lm.evaluate but with extensive debug output.
"""

import argparse
import collections
from pathlib import Path
from typing import Optional
import logging

import lm_eval
import mlx.core as mx
import mlx.nn as nn
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.base import create_causal_mask

def _pad_inputs(inputs):
    import numpy as np
    lengths = np.array([len(x) for x in inputs])
    maxlen = lengths.max()
    padded = np.stack(
        [np.pad(x, (0, maxlen - len(x))) for x in inputs],
        axis=0,
    )
    return mx.array(padded), mx.array(lengths)

class DebugMLXLM(LM):
    """Debug version of MLXLM with extensive logging"""

    def __init__(
        self,
        path_or_hf_repo: str,
        max_tokens: Optional[int] = None,
        use_chat_template: Optional[bool] = None,
        debug_limit: int = 5,
        **kwargs  # Ignore unexpected arguments
    ) -> None:
        super().__init__()
        self._model, self.tokenizer = load(path_or_hf_repo)
        self._max_tokens = max_tokens or self.tokenizer.model_max_length
        self._batch_size = 8
        self.use_chat_template = use_chat_template
        self.debug_limit = debug_limit
        self.debug_count = 0
        
        if use_chat_template is None:
            self.use_chat_template = self.tokenizer.chat_template is not None

    def _tokenize(self, prompts):
        """Tokenize prompts using the model's tokenizer."""
        return [
            self.tokenizer.encode(
                prompt, add_special_tokens=not self.use_chat_template
            )
            for prompt in prompts
        ]

    def _score_fn(self, inputs, cache=None, step_size: int = 2048):
        inputs, lengths = _pad_inputs(inputs)
        inputs, targets = inputs[..., :-1], inputs[..., 1:]

        cache = cache or make_prompt_cache(self._model)
        lengths += cache[0].offset

        scores, is_greedy = [], []
        for i in range(0, inputs.shape[1], step_size):
            inp = inputs[:, i : i + step_size]
            T = inp.shape[1]

            offset = cache[0].offset
            mask = create_causal_mask(T, offset, lengths=lengths)

            logits = self._model(inp, cache=cache, mask=mask)
            log_probs = nn.log_softmax(logits.astype(mx.float32))

            score = mx.take_along_axis(
                log_probs, targets[:, i : i + step_size, mx.newaxis], axis=-1
            )[..., 0]

            probs = mx.softmax(logits.astype(mx.float32))
            greedy = mx.argmax(logits, axis=-1) == targets[:, i : i + step_size]

            scores.append(score)
            is_greedy.append(greedy)

        scores = mx.concatenate(scores, axis=1)
        is_greedy = mx.concatenate(is_greedy, axis=1)

        return scores, is_greedy

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """Debug version of loglikelihood with extensive logging"""
        print(f"\n🔍 DEBUG: loglikelihood called with {len(requests)} requests")
        
        # Group by common prefix
        group_reqs = collections.defaultdict(list)
        for idx, req in enumerate(requests):
            context, continuation = req.args[0], req.args[1]
            group_reqs[context].append((idx, continuation))
            
            # Debug first few requests
            if self.debug_count < self.debug_limit:
                print(f"\n📝 REQUEST {idx} (debug #{self.debug_count + 1}):")
                print(f"   Context: {repr(context)}")
                print(f"   Continuation: {repr(continuation)}")
                print(f"   Full: {repr(context + continuation)}")
                
                # Tokenize and show details
                context_tokens = self._tokenize([context])[0]
                continuation_tokens = self._tokenize([continuation])[0]
                full_tokens = self._tokenize([context + continuation])[0]
                
                print(f"   Context tokens ({len(context_tokens)}): {context_tokens}")
                print(f"   Continuation tokens ({len(continuation_tokens)}): {continuation_tokens}")
                print(f"   Full tokens ({len(full_tokens)}): {full_tokens}")
                print(f"   Tokenizer settings: add_special_tokens={not self.use_chat_template}")
                
                self.debug_count += 1

        questions = list(group_reqs.keys())
        responses = []
        indices = []
        for v in group_reqs.values():
            idx, resp = zip(*v)
            indices.extend(idx)
            responses.append(resp)

        print(f"\n📊 GROUPED: {len(questions)} unique contexts, {len(requests)} total requests")

        scores, is_greedy = [], []
        for q_idx, (q, rs) in enumerate(tqdm(zip(questions, responses), total=len(questions))):
            if q_idx < 3:  # Debug first few questions
                print(f"\n🎯 PROCESSING QUESTION {q_idx}:")
                print(f"   Context: {repr(q)}")
                print(f"   Responses: {rs}")
            
            prefix = self._tokenize([q])[0]
            full_sequences = self._tokenize([q + r for r in rs])
            max_completed_l = max(len(s) for s in full_sequences)

            # compute truncation length
            truncation = max(0, max_completed_l - self._max_tokens - 1)
            orig_prefix_l = len(prefix)
            
            if q_idx < 3:
                print(f"   Prefix length: {orig_prefix_l}")
                print(f"   Max completion length: {max_completed_l}")
                print(f"   Truncation: {truncation}")

            if truncation > 0:
                prefix = prefix[truncation:]
                full_sequences = [s[truncation:] for s in full_sequences]

            prefix_l = len(prefix)
            inputs = [seq[: prefix_l + 1] for seq in full_sequences]
            
            if q_idx < 3:
                print(f"   After truncation - prefix length: {prefix_l}")
                print(f"   Input lengths: {[len(inp) for inp in inputs]}")

            # Score the sequences
            q_scores, q_is_greedy = self._score_fn(inputs)
            
            for i, (s, g) in enumerate(zip(q_scores, q_is_greedy)):
                continuation_l = len(full_sequences[i]) - prefix_l
                if continuation_l > 0:
                    score = float(s[prefix_l - 1 : prefix_l - 1 + continuation_l].sum())
                    greedy = bool(g[prefix_l - 1 : prefix_l - 1 + continuation_l].all())
                else:
                    score = 0.0
                    greedy = True
                    
                if q_idx < 3:
                    print(f"   Response {i} ({repr(rs[i])}): score={score:.4f}, greedy={greedy}")
                
                scores.append(score)
                is_greedy.append(greedy)

        # Reorder to match original request order
        reordered_scores = [0] * len(scores)
        reordered_is_greedy = [False] * len(is_greedy)
        for i, (score, greedy) in enumerate(zip(scores, is_greedy)):
            reordered_scores[indices[i]] = score
            reordered_is_greedy[indices[i]] = greedy

        print(f"\n✅ FINAL: Processed {len(requests)} requests")
        return list(zip(reordered_scores, reordered_is_greedy))
    
    def generate_until(self, requests) -> list[str]:
        """Minimal implementation of generate_until (not used for BoolQ)"""
        return [""] * len(requests)
    
    def loglikelihood_rolling(self, requests) -> list[float]:
        """Minimal implementation of loglikelihood_rolling (not used for BoolQ)"""
        return [0.0] * len(requests)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Model to debug")
    parser.add_argument("--limit", type=int, default=5, help="Number of questions to debug")
    parser.add_argument("--debug-requests", type=int, default=10, help="Number of requests to debug in detail")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting debug evaluation of {args.model}")
    print(f"   Limiting to {args.limit} questions")
    print(f"   Debugging first {args.debug_requests} requests in detail")
    
    # Create debug model
    debug_model = DebugMLXLM(
        args.model, 
        use_chat_template=False,
        debug_limit=args.debug_requests
    )
    
    # Register with lm-eval
    register_model("debug_mlx")(DebugMLXLM)
    
    # Run evaluation
    results = lm_eval.simple_evaluate(
        model="debug_mlx",
        model_args=f"path_or_hf_repo={args.model},use_chat_template=false,debug_limit={args.debug_requests}",
        tasks=["boolq"],
        limit=args.limit,
        batch_size=1,
    )
    
    print(f"\n🎯 RESULTS:")
    for task, result in results["results"].items():
        print(f"   {task}: {result}")

if __name__ == "__main__":
    main()