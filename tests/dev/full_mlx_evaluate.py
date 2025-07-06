# Copyright © 2024 Apple Inc.

"""
Adapted from a PyTorch implementation by David Grangier
"""

import argparse
import collections
import copy
import json
import logging
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional

import lm_eval
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from mlx_lm.generate import stream_generate
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import common_prefix_len, load


def _rstrip_until(s, untils):
    """Limit a string <s> to the first occurrence of any substring in untils."""
    l = len(s)
    f = [s.find(u) for u in untils]
    f = [l if x < 0 else x for x in f]
    return s[: min(f)]


def _pad_inputs(inputs):
    lengths = np.array([len(x) for x in inputs])
    maxlen = lengths.max()
    padded = np.stack(
        [np.pad(x, (0, maxlen - len(x))) for x in inputs],
        axis=0,
    )
    return mx.array(padded), mx.array(lengths)


def gold_idx(doc, opts):
    """Extract the correct answer index from the document."""
    if doc is None:
        return None
    # Unwrap nested 'doc' field if present (newer harness may nest the original doc)
    if isinstance(doc.get('doc', None), dict):
        return gold_idx(doc['doc'], opts)
    
    # BoolQ: answer field contains True/False -> 0/1
    if "answer" in doc:
        return 0 if doc["answer"] else 1
    
    # ARC easy/challenge: answerKey contains "A"-"D" -> 0-3
    if "answerKey" in doc:
        answer_key = doc["answerKey"]
        if isinstance(answer_key, str) and len(answer_key) == 1:
            return "ABCD".index(answer_key.upper())
    
    # HellaSwag and others: label contains integer index
    if "label" in doc:
        return int(doc["label"])
    
    # Other common patterns
    if "gold" in doc:
        return int(doc["gold"])

    # New-style API uses 'target' for ground truth
    if "target" in doc:
        val = doc["target"]
        try:
            return int(val)
        except (TypeError, ValueError):
            if isinstance(val, str):
                low = val.lower()
                if low in ("true", "yes"):
                    return 1
                if low in ("false", "no"):
                    return 0
        return None

    return None


def chat_template_fn(**extra_kwargs):
    def apply_chat_template(self, chat_history, add_generation_prompt=True) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            **extra_kwargs,
        )

    return apply_chat_template


@register_model("mlxlm")
class MLXLM(LM):

    tokenizer_name = lm_eval.models.huggingface.HFLM.tokenizer_name
    apply_chat_template = chat_template_fn()

    def __init__(
        self,
        path_or_hf_repo: str,
        max_tokens: Optional[int] = None,
        use_chat_template: Optional[bool] = None,
        debug: bool = False,
        log_incorrect_answers: bool = False,
        skip: int = 0,
    ) -> None:
        super().__init__()
        self._model, self.tokenizer = load(path_or_hf_repo)
        self._max_tokens = max_tokens or self.tokenizer.model_max_length
        self._batch_size = 8
        self.debug = debug
        self.log_incorrect_answers = log_incorrect_answers
        self.skip = skip
        # Default to False for base models unless explicitly requested
        if use_chat_template is None:
            self.use_chat_template = False
        else:
            self.use_chat_template = use_chat_template

    def _process_prompt(self, prompt, step_size: int = 2048):
        prompt = mx.array(prompt)[None]
        cache = make_prompt_cache(self._model)
        for i in range(0, prompt.shape[1], step_size):
            logits = self._model(prompt[:, i : i + step_size], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        logprobs = nn.log_softmax(logits[:, -1, :].astype(mx.float32))
        return logprobs, cache

    def _score_fn(self, inputs, cache: Optional[Any] = None, step_size: int = 2048):
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
        return [
            tuple(
                self.tokenizer.encode(t, add_special_tokens=not self.use_chat_template)
            )
            for t in texts
        ]

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.
        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.
        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """
        logging.info("Estimating loglikelihood for %d pairs." % len(requests))

        group = mx.distributed.init()

        # Group by common prefix
        group_reqs = collections.defaultdict(list)
        request_metadata = {}  # Store additional metadata about each request
        
        for idx, req in enumerate(requests):
            # Extract request data and try to find ground truth
            context = None
            continuation = None
            doc = None
            
            if hasattr(req, 'arguments') and len(req.arguments) >= 2:
                # New API with req.arguments and req.doc
                context, continuation = req.arguments[0], req.arguments[1]
                doc = req.doc if hasattr(req, 'doc') else None
            elif hasattr(req, 'args') and len(req.args) >= 2:
                # Fallback: Old-style API with req.args
                context, continuation = req.args[0], req.args[1]
                doc = req.doc if hasattr(req, 'doc') else None
            elif hasattr(req, 'kwargs') and 'doc' in req.kwargs:
                # Fallback: kwargs-style API
                doc = req.kwargs['doc']
                context, continuation = doc['context'], doc['continuation']
            else:
                print(f"Warning: Unknown request format: {req}")
                continue
            
            # Store metadata about this request for ground truth detection
            request_metadata[idx] = {
                'doc': doc,
                'req_obj': req,
                'context': context,
                'continuation': continuation
            }
            
            # Store doc alongside for ground truth access
            group_reqs[context].append((idx, continuation, doc))
            
        questions = list(group_reqs.keys())
        responses = []
        indices = []
        docs = []  # Store docs for ground truth
        for v in group_reqs.values():
            idx, resp, doc_list = zip(*v)
            indices.extend(idx)
            responses.append(resp)
            docs.append(doc_list)

        # split data accross ranks
        questions = questions[group.rank() :: group.size()]
        responses = responses[group.rank() :: group.size()]
        docs = docs[group.rank() :: group.size()]

        long_completions = 0
        scores, is_greedy = [], []
        first_pair_printed = False
        
        # Track results for incorrect answer logging
        if self.log_incorrect_answers:
            self._question_results = []
            
        for q_idx, (q, rs, question_docs) in enumerate(tqdm(zip(questions, responses, docs), total=len(questions))):
            prefix = self._tokenize([q])[0]
            full_sequences = self._tokenize([q + r for r in rs])
            max_completed_l = max(len(s) for s in full_sequences)
            
            # Debug output for first question
            if self.debug and q_idx == 0:
                print(f"\n[DEBUG] Context: {repr(q)}")
                print(f"[DEBUG] Continuations: {[repr(r) for r in rs]}")
                print(f"[DEBUG] Context tokens length: {len(prefix)}")
                print(f"[DEBUG] First 5 context tokens: {prefix[:5]}")
                print(f"[DEBUG] Last 5 context tokens: {prefix[-5:]}")
                print(f"[DEBUG] Full sequences lengths: {[len(s) for s in full_sequences]}")

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

            # Debug output for first question
            if self.debug and len(scores) == 0:
                print(f"\n[DEBUG] Context: {repr(q)}")
                print(f"[DEBUG] Continuations: {rs}")
                print(f"[DEBUG] Context tokens length: {len(prefix)}")
                print(f"[DEBUG] First 5 context tokens: {prefix[:5]}")
                print(f"[DEBUG] Last 5 context tokens: {prefix[-5:]}")
                cont_tokens = [self._tokenize([r])[0] for r in rs]
                print(f"[DEBUG] Continuation token IDs: {[t[0] if len(t) > 0 else 'EMPTY' for t in cont_tokens]}")

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

            # Debug first pair scores (for BoolQ typically " no" and " yes")
            if not first_pair_printed and len(rs) == 2 and len(scores) >= 2:
                print(f"  First pair scores: {rs[0].strip()}={scores[-2]:.4f}, {rs[1].strip()}={scores[-1]:.4f}")
                first_pair_printed = True
                
            # Store information for incorrect answer logging
            if self.log_incorrect_answers and len(rs) > 0:
                # Get the scores for just this question (last len(rs) scores)
                current_question_scores = scores[-len(rs):]
                
                # Find the answer with the highest score (least negative log probability)
                selected_idx = max(range(len(current_question_scores)), key=lambda i: current_question_scores[i])
                selected_answer = rs[selected_idx]
                selected_score = current_question_scores[selected_idx]
                
                # Extract ground truth using the first doc
                first_doc = question_docs[0] if question_docs else None
                correct_idx = gold_idx(first_doc, rs)
                
                self._question_results.append({
                    'question_idx': q_idx,
                    'context': q,
                    'options': list(rs),
                    'selected_idx': selected_idx,
                    'selected_answer': selected_answer,
                    'selected_score': selected_score,
                    'all_scores': current_question_scores.copy(),
                    'correct_idx': correct_idx,
                    'ground_truth_doc': first_doc
                })

        scores = mx.array(scores)
        is_greedy = mx.array(is_greedy)

        if long_completions > 0:
            logging.info(
                f"Prefix eliminated for {long_completions} requests with "
                + "completion longer than context."
            )

        num_results = len(requests)

        # all gather the results across groups
        if group.size() > 1:
            per_group = int(np.ceil(num_results / group.size()))
            scores = mx.pad(scores, ((0, per_group - len(scores)),))
            is_greedy = mx.pad(is_greedy, ((0, per_group - len(is_greedy))))
            scores = mx.distributed.all_gather(scores[mx.newaxis], stream=mx.cpu)
            is_greedy = mx.distributed.all_gather(is_greedy[mx.newaxis], stream=mx.cpu)
            mx.eval(scores, is_greedy)
            scores = scores.T.reshape(-1)
            is_greedy = is_greedy.T.reshape(-1)

        inv_sort = mx.argsort(mx.array(indices))
        scores = scores[:num_results][inv_sort]
        is_greedy = is_greedy[:num_results][inv_sort]
        
        result = list(zip(scores.tolist(), is_greedy.tolist()))
        
        # Log incorrect answers if requested
        if self.log_incorrect_answers and hasattr(self, '_question_results'):
            self._log_incorrect_answers(result)
        
        return result

    def loglikelihood_rolling(self, requests) -> list[float]:
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: EOT
            Max context length: 4
            Resulting input/prediction pairs:
                INPUT:  EOT   0   1   2
                PRED:     0   1   2   3
                INPUT:    3   4   5   6
                PRED:     4   5   6   7
                INPUT:    5   6   7   8
                PRED:             8   9
          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens
        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context,).
            string: str
                String for which we are computing overall loglikelihood
        :return: list[tuple[float]]
            A list of tuples (logprob,)
            logprob: float
                The log probability of `context` conditioned on the EOT token.
        """
        logging.info(
            "Estimating loglikelihood rolling for %d sequences." % len(requests)
        )
        texts = self._tokenize([req.args[0] for req in requests])
        all_scores = []
        for i in tqdm(range(0, len(texts), self._batch_size)):
            batch = texts[i : i + self._batch_size]
            scores, lengths, _ = self._score_fn(batch)
            mask = mx.arange(scores.shape[-1]) < lengths[:, None]
            all_scores.extend((mask * scores).sum(axis=-1).tolist())

        return all_scores

    def _log_incorrect_answers(self, loglikelihood_results):
        """Log detailed information about incorrect answers using proper ground truth."""
        print(f"\n[INCORRECT ANSWER ANALYSIS] Analyzing {len(self._question_results)} questions...")
        
        incorrect_count = 0
        total_questions = len(self._question_results)
        
        # Open log file for writing
        import os
        import time
        log_file = os.path.join(os.getcwd(), "incorrect_answers_mlx.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== INCORRECT ANSWER LOG (MLX) ===\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for q_info in self._question_results:
                question_idx = q_info['question_idx']
                context = q_info['context']
                options = q_info['options']
                selected_idx = q_info['selected_idx']
                selected_answer = q_info['selected_answer']
                selected_score = q_info['selected_score']
                all_scores = q_info['all_scores']
                correct_idx = q_info.get('correct_idx')
                ground_truth_doc = q_info.get('ground_truth_doc')
                
                # Determine if the answer was incorrect
                if correct_idx is None:
                    is_incorrect = not loglikelihood_results[question_idx][1]
                else:
                    is_incorrect = (selected_idx != correct_idx)
                
                if is_incorrect:
                    incorrect_count += 1
                    # Prepare correct-answer info if available
                    if correct_idx is not None and 0 <= correct_idx < len(options):
                        correct_answer = options[correct_idx]
                        correct_score = all_scores[correct_idx]
                        score_diff = selected_score - correct_score
                    else:
                        # Fallback: display raw ground-truth fields if available
                        if ground_truth_doc is not None:
                            if 'answer' in ground_truth_doc:
                                correct_answer = ground_truth_doc['answer']
                            elif 'answerKey' in ground_truth_doc:
                                correct_answer = ground_truth_doc['answerKey']
                            elif 'label' in ground_truth_doc:
                                correct_answer = ground_truth_doc['label']
                            elif 'gold' in ground_truth_doc:
                                correct_answer = ground_truth_doc['gold']
                            elif 'target' in ground_truth_doc:
                                correct_answer = ground_truth_doc['target']
                            else:
                                correct_answer = '<unknown>'
                        else:
                            correct_answer = '<unknown>'
                        correct_score = None
                        score_diff = None

                    # Calculate absolute question number
                    absolute_question_num = self.skip + question_idx + 1
                    
                    print(f"\n[INCORRECT] Question {absolute_question_num}:")
                    print(f"  Context: {repr(context[:200])}{'...' if len(context) > 200 else ''}")
                    print(f"  Options: {options}")
                    print(f"  Selected: '{selected_answer}' (index {selected_idx}, score: {selected_score:.4f})")
                    if correct_score is not None:
                        print(f"  Correct: '{correct_answer}' (index {correct_idx}, score: {correct_score:.4f})")
                        print(f"  Score difference: {score_diff:.4f}")
                    else:
                        print(f"  Correct: {correct_answer}")

                    # Write to log file
                    f.write(f"QUESTION {absolute_question_num} (INCORRECT):\n")
                    f.write(f"Context: {context}\n")
                    f.write(f"Options: {options}\n")
                    f.write(f"Selected Answer: '{selected_answer}' (index {selected_idx})\n")
                    if correct_score is not None:
                        f.write(f"Correct Answer: '{correct_answer}' (index {correct_idx})\n")
                        f.write(f"Selected Score: {selected_score:.4f}\n")
                        f.write(f"Correct Score: {correct_score:.4f}\n")
                        f.write(f"Score Difference: {score_diff:.4f}\n")
                    else:
                        f.write(f"Correct Answer: {correct_answer}\n")
                    f.write(f"All Scores: {[f'{s:.4f}' for s in all_scores]}\n")

                    # Log ground truth source information if available
                    if ground_truth_doc:
                        gt_fields = []
                        for field in ['answer', 'answerKey', 'label', 'gold', 'target']:
                            if field in ground_truth_doc:
                                gt_fields.append(f"{field}={ground_truth_doc[field]}")
                        f.write(f"Ground Truth: {', '.join(gt_fields) if gt_fields else 'Unknown'}\n")
                    f.write("=" * 50 + "\n\n")
        
        print(f"\n[SUMMARY] Found {incorrect_count} incorrect answers out of {total_questions} questions")
        if total_questions > 0:
            print(f"Accuracy: {((total_questions - incorrect_count) / total_questions * 100):.1f}%")
        print(f"Detailed log saved to: {log_file}")

    def generate_until(self, requests) -> list[str]:
        """Generate greedily until a stopping sequence
        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, until).
            context: str
                Context string
            until: [str]
                The string sequences to generate until. These string sequences
                may each span across multiple tokens, or may be part of one token.
        :return: list[str]
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        logging.info("Generating continuation for %d sequences." % len(requests))
        contexts, options = zip(*[req.args for req in requests])
        # contrary to the doc the second element of the tuple contains
        # {'do_sample': False, 'until': ['\n\n'], 'temperature': 0}
        completions = []

        for context, opt in tqdm(zip(contexts, options), total=len(contexts)):
            until = opt["until"]
            context = self.tokenizer.encode(
                context, add_special_tokens=not self.use_chat_template
            )
            max_tokens = min(
                opt.get("max_gen_tokens", self._max_tokens),
                self.tokenizer.model_max_length - len(context),
            )
            text = ""
            for response in stream_generate(
                self._model, self.tokenizer, prompt=context, max_tokens=max_tokens
            ):
                text += response.text
                if any(u in text for u in until):
                    text = _rstrip_until(text, until)
                    completions.append(text)
                    break
            else:
                completions.append(text)
        return completions


def main():
    parser = argparse.ArgumentParser(
        "Evaluate an MLX model using lm-evaluation-harness."
    )
    parser.add_argument("--model", help="Model to evaluate", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", default=["boolq"])
    parser.add_argument(
        "--output-dir", default=".", help="Output directory for result files."
    )
    parser.add_argument(
        "--output-path", default=None, help="Specific output file path (overrides auto-generated path)."
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-shots", type=int, default=None, help="Number of shots")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum nunber of tokens to generate. Defaults to the model's max context length.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="Limit the number of examples per task.",
        type=int,
    )
    parser.add_argument(
        "--skip",
        default=0,
        help="Skip the first N examples per task.",
        type=int,
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--fewshot-as-multiturn",
        action="store_true",
        help="Whether to provide the fewshot examples as a multiturn "
        "conversation or a single user turn.",
        default=False,
    )
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        help="Specifies whether to apply a chat template to the prompt. If "
        "the model has a chat template, this defaults to `True`, "
        "otherwise `False`.",
        default=None,
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="""A JSON formatted string of arguments for the tokenizer's "
        "apply_chat_template, e.g. '{"enable_thinking":false}'""",
        default="{}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output with detailed prompt information",
        default=False,
    )
    parser.add_argument(
        "--log-incorrect-answers",
        action="store_true",
        help="Log detailed information for incorrect answers including context, options, selected answer, and correct answer",
        default=False,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Silence tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    mx.random.seed(args.seed)

    lm = MLXLM(
        args.model,
        max_tokens=args.max_tokens,
        use_chat_template=args.apply_chat_template,
        debug=args.debug,
        log_incorrect_answers=args.log_incorrect_answers,
        skip=args.skip,
    )
    MLXLM.apply_chat_template = chat_template_fn(**args.chat_template_args)

    # Handle skip parameter by creating custom samples dict
    samples = None
    if args.skip > 0:
        # Create samples dict that skips the first N examples
        samples = {}
        for task in args.tasks:
            start_idx = args.skip
            end_idx = start_idx + args.limit if args.limit else start_idx + 100
            samples[task] = list(range(start_idx, end_idx))
        # When using samples, don't use limit
        eval_limit = None
    else:
        eval_limit = args.limit

    # For base models, force apply_chat_template=False
    use_chat_template = lm.use_chat_template
    if args.apply_chat_template is False:
        use_chat_template = False
    
    print(f"Using chat template: {use_chat_template}")
    print(f"Skip: {args.skip}, Limit: {args.limit}")
    if samples:
        print(f"Samples dict: {samples}")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        apply_chat_template=use_chat_template,
        num_fewshot=args.num_shots,
        limit=eval_limit,
        samples=samples,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    file_keys = ["eval", args.model.replace("/", "_"), version("lm_eval")]
    if args.num_shots is not None:
        file_keys += [f"{args.num_shots:02d}"]
    file_keys += args.tasks
    filename = "_".join(file_keys)
    if mx.distributed.init().rank() == 0:
        if args.output_path:
            output_path = Path(args.output_path)
        else:
            output_path = output_dir / filename
        output_path.write_text(json.dumps(results["results"], indent=4))
        print("Results:")
        for result in results["results"].values():
            print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
