#!/usr/bin/env python3
"""Create a simple calibration dataset by combining WikiText samples"""

import json
from datasets import load_dataset
from transformers import AutoTokenizer

def create_calibration_data(seqlen=512, nsamples=128):
    print("Loading WikiText-2 dataset...")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Creating {nsamples} calibration samples of {seqlen} tokens each...")
    
    # Collect all text
    all_texts = []
    for sample in traindata:
        text = sample['text']
        if text and len(text.strip()) > 10:
            all_texts.append(text.strip())
    
    print(f"Collected {len(all_texts)} text samples")
    
    # Combine texts to create longer sequences
    calib_data = []
    current_text = ""
    
    for i, text in enumerate(all_texts):
        current_text += " " + text
        
        # Tokenize current combined text
        tokens = tokenizer(current_text, return_tensors='pt', max_length=seqlen * 2, truncation=True)
        
        if tokens.input_ids.shape[1] >= seqlen:
            # Extract exactly seqlen tokens
            calib_sample = {
                'input_ids': tokens.input_ids[:, :seqlen].tolist()[0],
                'text': current_text[:500]  # Store preview
            }
            calib_data.append(calib_sample)
            
            # Start fresh for next sample
            current_text = ""
            
            if len(calib_data) >= nsamples:
                break
            
            if len(calib_data) % 10 == 0:
                print(f"Created {len(calib_data)}/{nsamples} calibration samples")
    
    print(f"Created {len(calib_data)} calibration samples")
    return calib_data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create calibration data for quantization')
    parser.add_argument('--nsamples', type=int, default=128, 
                       help='Number of calibration samples to generate (default: 128)')
    parser.add_argument('--seqlen', type=int, default=512,
                       help='Sequence length for each sample (default: 512)')
    parser.add_argument('--output', type=str, default='calib.json',
                       help='Output file for calibration data (default: calib.json)')
    
    args = parser.parse_args()
    
    print(f"Creating {args.nsamples} calibration samples of {args.seqlen} tokens each...")
    calib_data = create_calibration_data(seqlen=args.seqlen, nsamples=args.nsamples)
    
    with open(args.output, 'w') as f:
        json.dump(calib_data, f, indent=2)
    
    print(f"Saved {len(calib_data)} calibration samples to {args.output}")

if __name__ == '__main__':
    main()