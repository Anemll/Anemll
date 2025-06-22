#!/usr/bin/env python3
"""
Prepare calibration dataset for QuaRot quantization.
Based on QuaRot's data preparation utilities.
"""

import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer


def get_wikitext_data(tokenizer, nsamples=128, seqlen=2048):
    """Get calibration data from WikiText dataset."""
    print("Loading WikiText-2 dataset...")
    
    # Load WikiText dataset (more reliable than Pile)
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    trainloader = []
    for i, sample in enumerate(traindata):
        if len(trainloader) >= nsamples:
            break
            
        text = sample['text']
        # Skip empty texts
        if not text or len(text.strip()) < 50:
            continue
            
        trainenc = tokenizer(text, return_tensors='pt', max_length=seqlen, truncation=True)
        
        # More flexible: accept sequences that are at least 80% of target length
        min_length = max(128, int(seqlen * 0.8))
        if trainenc.input_ids.shape[1] >= min_length:
            # Pad to exact sequence length if needed
            actual_ids = trainenc.input_ids[:, :seqlen].tolist()[0]
            if len(actual_ids) < seqlen:
                # Pad with tokenizer.pad_token_id or eos_token_id
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                actual_ids.extend([pad_id] * (seqlen - len(actual_ids)))
            
            trainloader.append({
                'input_ids': actual_ids,
                'text': text[:500]  # Store first 500 chars for reference
            })
            
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} samples, collected {len(trainloader)} valid sequences")
    
    print(f"Collected {len(trainloader)} calibration samples")
    return trainloader


def main():
    parser = argparse.ArgumentParser(description='Prepare calibration dataset for QuaRot')
    parser.add_argument('--dataset', type=str, default='wikitext', choices=['wikitext', 'pile'], 
                        help='Dataset to use for calibration')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B',
                        help='Model name for tokenizer')
    parser.add_argument('--out', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--tokens', type=int, default=2048,
                        help='Sequence length for calibration')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration samples')
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get calibration data
    if args.dataset == 'wikitext':
        calib_data = get_wikitext_data(tokenizer, nsamples=args.nsamples, seqlen=args.tokens)
    elif args.dataset == 'pile':
        # Keep pile option for compatibility but use a fallback
        try:
            from datasets import load_dataset
            print("Attempting to load Pile dataset...")
            calib_data = get_pile_data(tokenizer, nsamples=args.nsamples, seqlen=args.tokens)
        except Exception as e:
            print(f"Pile dataset failed ({e}), falling back to WikiText")
            calib_data = get_wikitext_data(tokenizer, nsamples=args.nsamples, seqlen=args.tokens)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Save calibration data
    print(f"Saving calibration data to {args.out}...")
    with open(args.out, 'w') as f:
        json.dump(calib_data, f, indent=2)
    
    print(f"Calibration dataset saved with {len(calib_data)} samples")


if __name__ == '__main__':
    main()