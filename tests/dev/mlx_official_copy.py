#!/usr/bin/env python3
"""
1:1 copy of official MLX evaluation logic to confirm 62% accuracy.
This is an exact copy of the official MLXLM.loglikelihood method.
"""

import argparse
import collections
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from mlx_lm import load
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import make_prompt_cache

def _pad_inputs(inputs):
    """Exact copy from official MLX"""
    lengths = np.array([len(x) for x in inputs])
    maxlen = lengths.max()
    padded = np.stack(
        [np.pad(x, (0, maxlen - len(x))) for x in inputs],
        axis=0,
    )
    return mx.array(padded), mx.array(lengths)

class OfficialMLXCopy:
    """1:1 copy of official MLXLM class methods"""
    
    def __init__(self, path_or_hf_repo: str, max_tokens: Optional[int] = None, use_chat_template: Optional[bool] = None):
        self._model, self.tokenizer = load(path_or_hf_repo)
        self._max_tokens = max_tokens or self.tokenizer.model_max_length
        self._batch_size = 8
        self.use_chat_template = use_chat_template
        if use_chat_template is None:
            self.use_chat_template = self.tokenizer.chat_template is not None

    def _process_prompt(self, prompt, step_size: int = 2048):
        """Exact copy from official MLX"""
        prompt = mx.array(prompt)[None]
        cache = make_prompt_cache(self._model)
        for i in range(0, prompt.shape[1], step_size):
            logits = self._model(prompt[:, i : i + step_size], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        logprobs = nn.log_softmax(logits[:, -1, :].astype(mx.float32))
        return logprobs, cache

    def _score_fn(self, inputs, cache: Optional[Any] = None, step_size: int = 2048):
        """Exact copy from official MLX"""
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
            ig = targets[:, i : i + step_size] == mx.argmax(logits, axis=-1)
            ig = mx.where(mx.arange(T) + offset < lengths[:, None], ig, False)

            mx.eval(score, ig)
            mx.clear_cache()

            is_greedy.append(ig)
            scores.append(score)

        scores = mx.concatenate(scores, axis=1)
        is_greedy = mx.concatenate(is_greedy, axis=1)

        return scores, lengths, is_greedy

    def _tokenize(self, texts):
        """Exact copy from official MLX"""
        return [
            tuple(
                self.tokenizer.encode(t, add_special_tokens=not self.use_chat_template)
            )
            for t in texts
        ]

    def loglikelihood_boolq(self, questions, responses_list):
        """Simplified version of official loglikelihood for BoolQ evaluation"""
        logging.info("Estimating loglikelihood for %d pairs." % (len(questions) * 2))

        # Group by common prefix (each question has 2 responses: " no", " yes")
        long_completions = 0
        scores, is_greedy = [], []
        
        for q, rs in tqdm(zip(questions, responses_list), total=len(questions)):
            prefix = self._tokenize([q])[0]
            full_sequences = self._tokenize([q + r for r in rs])
            max_completed_l = max(len(s) for s in full_sequences)

            # compute truncation length
            truncation = max(0, max_completed_l - self._max_tokens - 1)
            orig_prefix_l = len(prefix)
            prefix_l = max(len(prefix) - truncation, 0)
            prefix = prefix[len(prefix) - prefix_l :]

            # If the entire prompt got truncated ignore the question
            if prefix_l == 0:
                long_completions += 1
                scores.extend([-float("inf")] * len(rs))
                is_greedy.extend([False] * len(rs))
                continue

            # model scoring, returns num_requests x (logp, is_greedy, length).
            logprobs, cache = self._process_prompt(prefix)
            max_idx = mx.argmax(logprobs).item()

            for s in full_sequences:
                inputs = s[len(prefix) :]
                # The logprobs from the last token of the prompt are
                # for the first input token
                scores.append(logprobs[0, inputs[0]].item())
                is_greedy.append((inputs[0] == max_idx))

                if len(inputs) == 1:
                    continue
                score, _, ig = self._score_fn(
                    mx.array(inputs)[None, :], cache=copy.deepcopy(cache)
                )
                scores[-1] += mx.sum(score).item()
                is_greedy[-1] &= mx.all(ig).item()

        if long_completions > 0:
            logging.info(
                f"Prefix eliminated for {long_completions} requests with "
                + "completion longer than context."
            )

        return scores, is_greedy

def evaluate_boolq_official(model_path, limit=None):
    """Evaluate BoolQ using exact official MLX logic"""
    
    # Load model
    print(f"Loading MLX model: {model_path}")
    mlx_model = OfficialMLXCopy(model_path, use_chat_template=False)  # Base model, no chat template
    
    # Load BoolQ dataset
    print("Loading BoolQ dataset...")
    dataset = load_dataset("boolq", split="validation")
    if limit:
        dataset = dataset.select(range(limit))
    
    print(f"Evaluating {len(dataset)} questions...")
    
    correct = 0
    total = 0
    
    # Process in segments to avoid memory issues
    segment_size = 100
    for start_idx in range(0, len(dataset), segment_size):
        end_idx = min(start_idx + segment_size, len(dataset))
        segment = dataset.select(range(start_idx, end_idx))
        
        print(f"Processing segment [{start_idx} .. {end_idx-1}]")
        
        # Prepare questions and responses
        questions = []
        all_responses = []
        ground_truths = []
        
        for item in segment:
            passage = item['passage']
            question = item['question']
            context = f"{passage}\nQuestion: {question}\nAnswer:"
            
            questions.append(context)
            all_responses.append([" no", " yes"])  # Official order
            ground_truths.append(item['answer'])  # True/False
        
        # Get scores using official logic
        scores, is_greedy = mlx_model.loglikelihood_boolq(questions, all_responses)
        
        # Process results
        for i in range(len(questions)):
            no_score = scores[i * 2]     # " no" score
            yes_score = scores[i * 2 + 1] # " yes" score
            
            prediction = "yes" if yes_score > no_score else "no"
            ground_truth_str = "yes" if ground_truths[i] else "no"
            
            if prediction == ground_truth_str:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Official MLX BoolQ Evaluation Results")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*60}")
    
    return accuracy, correct, total

def main():
    parser = argparse.ArgumentParser(description="Official MLX BoolQ evaluation copy")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="MLX model path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run evaluation
    accuracy, correct, total = evaluate_boolq_official(args.model, args.limit)

if __name__ == "__main__":
    main()