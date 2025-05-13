#!/usr/bin/env python3
# Perplexity test script with longer default text
# Usage: python perplexity_test.py [--model MODEL_PATH] [--debug]

import os
import sys
import argparse
import math
import torch
from tqdm import tqdm
from pathlib import Path

# Add current directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import ANE_Model
try:
    from ane_model import ANE_Model
except ImportError:
    print(f"Error: Failed to import ane_model from {current_dir}")
    sys.exit(1)

# Import tokenizer
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers package not found. Please install with: pip install transformers")
    sys.exit(1)

# A longer sample text approaching context window size
# This combines multiple technical and literary passages to create a diverse, lengthy text
LONG_SAMPLE_TEXT = """
In the fields of machine learning and artificial intelligence, the concept of neural networks has evolved significantly over the decades. Early neural networks, inspired by the biological structure of the human brain, consisted of simple perceptrons capable of linear classification. As computational resources expanded, so did the complexity of these networks, leading to the development of deep learning architectures that have revolutionized the field.

Modern neural networks typically comprise multiple layers: an input layer that receives data, hidden layers that perform computations, and an output layer that produces predictions. Each layer contains nodes, or "neurons," connected by weights that are adjusted during training. The training process involves optimizing these weights to minimize a loss function, often using gradient descent algorithms.

The Apple Neural Engine (ANE) represents a specialized hardware implementation designed to accelerate neural network operations on Apple devices. By offloading computations from the CPU and GPU to a dedicated neural processing unit, the ANE significantly improves the efficiency of machine learning tasks, enabling complex models to run with lower power consumption and higher speed.

Language models present unique challenges in neural network design. Unlike computer vision tasks, which process spatial data, language models must process sequential data with complex dependencies across time. This led to the development of recurrent neural networks (RNNs), which maintain an internal state that can capture information from previous timesteps. However, RNNs struggled with long-range dependencies due to vanishing and exploding gradient problems.

The introduction of attention mechanisms, particularly the Transformer architecture, marked a significant breakthrough in natural language processing. Transformers use self-attention to weigh the importance of different words in a sequence, allowing the model to focus on relevant context regardless of distance. This innovation enabled the development of powerful language models like BERT, GPT, and T5, which have achieved state-of-the-art results across a wide range of language tasks.

Tokenization represents a critical preprocessing step for language models. This process converts raw text into numerical vectors that neural networks can process. Different tokenization approaches include word-level, character-level, and subword-level methods, each with its own advantages and limitations. Byte Pair Encoding (BPE) and WordPiece have emerged as popular subword tokenization methods, striking a balance between the vocabulary size and the ability to handle out-of-vocabulary words.

The evaluation of language models typically involves multiple metrics. Perplexity measures how well a probability model predicts a sample, with lower values indicating better performance. This metric is calculated as the exponential of the average negative log-likelihood of the sequence. Other common evaluation metrics include BLEU score for translation tasks, ROUGE for summarization, and accuracy for classification tasks.

Transformer architectures have scaled dramatically in recent years, from models with millions of parameters to those with hundreds of billions. This scaling has generally led to improved performance across tasks, though with diminishing returns and increasing computational costs. Research continues to explore more efficient architectures and training methods to maximize performance while minimizing resource requirements.

One fascinating aspect of large language models is their emergent abilities—capabilities that were not explicitly trained for but arise from scale and diverse training data. These include in-context learning, where models can adapt to new tasks based on a few examples provided in the prompt, and chain-of-thought reasoning, where models can work through complex problems step by step.

The deployment of language models on edge devices presents unique challenges. These include memory constraints, computational limitations, and power consumption concerns. Techniques such as quantization, pruning, and knowledge distillation have been developed to compress models while maintaining acceptable performance. The Apple Neural Engine plays a crucial role in this context, enabling more efficient execution of compressed models on Apple devices.

The field of neural networks continues to evolve at a rapid pace. Recent innovations include mixture-of-experts architectures, which activate only a subset of parameters for each input, reducing computational costs while maintaining model capacity. Research into retrieval-augmented generation combines parametric knowledge stored in weights with non-parametric knowledge accessed through retrieval systems, potentially improving factual accuracy and reducing hallucinations.

Despite their impressive capabilities, language models face significant challenges. These include bias inherited from training data, potential for generating harmful content, and difficulties with factual accuracy and reasoning. Researchers are actively working on addressing these issues through techniques such as reinforcement learning from human feedback, red-teaming exercises, and various safety alignment methods.

The evaluation of language models across different hardware platforms requires careful consideration of trade-offs between accuracy, speed, and resource utilization. Metrics like inference time, memory footprint, and energy consumption become increasingly important when deploying models on resource-constrained devices. Standardized benchmarks help in comparing different models and optimizations across various deployment scenarios.

As we look to the future, the integration of neuromorphic computing principles—inspired by the brain's biological processes—may lead to more efficient neural network architectures. These approaches could potentially overcome current limitations in power consumption and computational efficiency, enabling more powerful AI systems on edge devices.

The democratization of AI through improved tools, frameworks, and pre-trained models has lowered the barrier to entry for developers interested in implementing machine learning solutions. However, this accessibility also brings responsibilities regarding the ethical use and deployment of these technologies, particularly in applications that directly impact human lives and society.

In conclusion, the field of neural networks and language models represents a dynamic area of research and application, with continuous innovations driving improvements in both capabilities and efficiency. The Apple Neural Engine exemplifies the trend towards specialized hardware accelerators designed to meet the unique computational demands of these models, enabling more powerful AI experiences on consumer devices while managing resource constraints.
"""

def setup_argparse():
    """Configure argument parser for script"""
    parser = argparse.ArgumentParser(description="Calculate perplexity on a text sample")
    parser.add_argument("--model", type=str, 
                        default="/Users/anemll/Models/ANE/anemll-Llama-3.2-1B-FP16-b64-ctx1024",
                        help="Path to the ANE model directory")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug output")
    parser.add_argument("--safety-margin", type=int, default=100,
                        help="Safety margin for context length (default: 100)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    return parser

def calculate_perplexity(model_path, debug=False, safety_margin=100, output_dir="results"):
    """Calculate perplexity on the long sample text"""
    print(f"Loading model from {model_path}")
    
    # Initialize model
    model = ANE_Model(model_path)
    model.debug = 1 if debug else 0
    
    # Get context length
    ctx_length = model.metadata.get('context_length', 2048)
    safe_length = ctx_length - safety_margin
    print(f"Model context length: {ctx_length}")
    print(f"Safe length with margin: {safe_length}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer is None:
            raise ValueError("Tokenizer not found")
    except Exception as e:
        print(f"Error loading tokenizer from {model_path}: {str(e)}")
        print("Falling back to Llama tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Tokenize the long sample text
    tokens = tokenizer.encode(LONG_SAMPLE_TEXT)
    print(f"Long sample text tokenized to {len(tokens)} tokens")
    
    # If text exceeds safe length, truncate but warn
    if len(tokens) > safe_length:
        print(f"Warning: Text exceeds safe length ({len(tokens)} > {safe_length}). Truncating.")
        tokens = tokens[:safe_length]
    
    # Split into input and targets (shifting by 1)
    inputs, targets = tokens[:-1], tokens[1:]
    
    # Reset model state
    model.reset_state()
    
    # Track scores and tokens
    total_log_likelihood = 0.0
    total_tokens = 0
    
    try:
        # Prefill with input tokens
        print("Running prefill...")
        input_tensor = torch.tensor([inputs], dtype=torch.int32)
        _ = model.prefill(input_tensor)
        
        # Score each target token with progress bar
        print("Scoring tokens...")
        pbar = tqdm(total=len(targets), desc="Token scoring", unit="token")
        
        for i, target in enumerate(targets):
            # Get current token
            current_token = inputs[i]
            
            try:
                # Get log probabilities for current token
                token_tensor = torch.tensor([[current_token]], dtype=torch.int32)
                log_probs = model.compute_logprobs(token_tensor)
                
                if log_probs is None:
                    if debug:
                        print(f"Warning: No log probabilities for token at position {i}")
                    pbar.update(1)
                    continue
                
                # Score target token
                token_score = log_probs[target].item()
                total_log_likelihood += token_score
                total_tokens += 1
                
                # Show token info in debug mode for first few tokens
                if debug and i < 10:
                    token_text = tokenizer.decode([current_token])
                    target_text = tokenizer.decode([target])
                    print(f"Token {i}: '{token_text}' → '{target_text}', score: {token_score:.4f}")
                
                # Update state for next token
                if i < len(targets) - 1:
                    _ = model.predict(token_tensor)
                
            except Exception as e:
                if debug:
                    print(f"Error at token {i}: {str(e)}")
                # Continue with next token
            
            pbar.update(1)
        
        pbar.close()
        
        # Calculate perplexity
        if total_tokens > 0:
            avg_log_likelihood = total_log_likelihood / total_tokens
            perplexity = math.exp(-avg_log_likelihood)
            print(f"\nTotal tokens scored: {total_tokens}")
            print(f"Average log likelihood: {avg_log_likelihood:.4f}")
            print(f"Perplexity: {perplexity:.4f}")
            
            # Save results to file
            os.makedirs(output_dir, exist_ok=True)
            result_file = os.path.join(output_dir, "long_text_perplexity.txt")
            with open(result_file, "w") as f:
                f.write(f"Long sample text perplexity: {perplexity:.4f}\n")
                f.write(f"Total tokens: {total_tokens}\n")
                f.write(f"Average log likelihood: {avg_log_likelihood:.4f}\n")
                f.write(f"Model: {model_path}\n")
            
            print(f"Results saved to {result_file}")
            return perplexity
        else:
            print("Error: No tokens were successfully scored")
            return float('inf')
            
    except Exception as e:
        print(f"Error in perplexity calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return float('inf')

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    calculate_perplexity(
        model_path=args.model,
        debug=args.debug,
        safety_margin=args.safety_margin,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 