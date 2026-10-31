"""In-model argmax over LM-head vocab chunks (--argmax)."""

import torch


def chunked_argmax(logits_list):
    """Return (argmax_idx, argmax_val), each [num_chunks]: the LOCAL winner index (int32)
    and its logit for every LM-head chunk.

    The chunks are stacked into one [num_chunks, chunk_size] tensor and reduced by a
    single argmax, so no op runs on the int32 indices afterwards. Core ML can place shape
    ops on an int32 tensor on the ANE, which carries it as fp16 and rounds indices above
    2048. A single argmax after all LM-head convs also keeps the convs on the ANE, where a
    per-chunk argmax interleaves ANE and CPU work.
    """
    width = max(logits.shape[-1] for logits in logits_list)
    rows = []
    for logits in logits_list:
        assert tuple(logits.shape[:-1]) == (1, 1), (
            f"argmax expects single-token logits [1, 1, chunk_size], got {tuple(logits.shape)}"
        )
        if logits.shape[-1] < width:  # uneven vocab split: pad with a logit that can never win
            pad = width - logits.shape[-1]
            logits = torch.nn.functional.pad(logits, (0, pad), value=torch.finfo(logits.dtype).min)
        rows.append(logits)
    stacked = torch.cat(rows, dim=1).squeeze(0)  # [num_chunks, width]
    return torch.argmax(stacked, dim=-1).to(torch.int32), torch.amax(stacked, dim=-1)
