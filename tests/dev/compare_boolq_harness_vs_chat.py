#!/usr/bin/env python3
"""
Compare BoolQ evaluation between lm-evaluation-harness and direct chat.py calls.
This will help identify if there are discrepancies in how the two methods handle BoolQ.
"""

import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_HUB_OFFLINE"] = "1"

# Add paths for imports
evaluate_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evaluate", "ane")
sys.path.append(evaluate_dir)

from evaluate_with_harness import ANELM

def load_boolq_samples(num_samples=10):
    """Load BoolQ samples from dataset."""
    import datasets
    dataset = datasets.load_dataset('super_glue', 'boolq', split=f'validation[:{num_samples}]')
    return dataset

def test_with_harness(model_path, num_samples=10):
    """Test with lm-evaluation-harness."""
    print("=" * 60)
    print("TESTING WITH LM-EVALUATION-HARNESS")
    print("=" * 60)
    
    # Run harness evaluation
    cmd = [
        "python", "evaluate/ane/evaluate_with_harness.py",
        "--model", model_path,
        "--tasks", "boolq",
        "--batch-size", "1",
        "--limit", str(num_samples)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll")
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Try to extract accuracy from output
        lines = result.stdout.split('\n')
        accuracy = None
        for line in lines:
            if 'acc,none' in line or 'boolq' in line.lower():
                print(f"Harness result line: {line}")
                # Try to parse JSON-like output
                try:
                    if '{' in line and '}' in line:
                        # Extract JSON part
                        json_start = line.find('{')
                        json_part = line[json_start:]
                        data = json.loads(json_part)
                        if 'boolq' in data and 'acc,none' in data['boolq']:
                            accuracy = data['boolq']['acc,none']
                except:
                    pass
        
        print(f"Harness accuracy: {accuracy}")
        return accuracy
        
    except Exception as e:
        print(f"Error running harness: {e}")
        return None

def test_with_chat(model_path, samples):
    """Test with direct chat.py calls."""
    print("\n" + "=" * 60)
    print("TESTING WITH DIRECT CHAT.PY CALLS")
    print("=" * 60)
    
    results = []
    
    for i, sample in enumerate(samples):
        passage = sample['passage']
        question = sample['question']
        label = sample['label']  # True=1, False=0
        correct_answer = "True" if label else "False"
        
        print(f"\nQuestion {i+1}/10:")
        print(f"PASSAGE: {passage[:100]}...")
        print(f"QUESTION: {question}")
        print(f"CORRECT: {correct_answer}")
        
        # Create prompt like harness does
        prompt = f"{passage}\nQuestion: {question}?\nAnswer:"
        
        # Test both "True" and "False" continuations using chat.py
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            true_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            false_file = f.name
        
        try:
            # Test " True" continuation
            true_cmd = [
                "python", "tests/chat.py",
                "--meta", f"{model_path}/meta.yaml",
                "--no-template",
                "--prompt", prompt + " True",
                "--eval",
                "--max-tokens", "1"
            ]
            
            # Test " False" continuation  
            false_cmd = [
                "python", "tests/chat.py", 
                "--meta", f"{model_path}/meta.yaml",
                "--no-template", 
                "--prompt", prompt + " False",
                "--eval",
                "--max-tokens", "1"
            ]
            
            print(f"Testing True continuation...")
            true_result = subprocess.run(true_cmd, capture_output=True, text=True, cwd="/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll")
            print(f"Testing False continuation...")
            false_result = subprocess.run(false_cmd, capture_output=True, text=True, cwd="/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll")
            
            true_output = true_result.stdout.strip()
            false_output = false_result.stdout.strip()
            
            print(f"True continuation output: '{true_output}'")
            print(f"False continuation output: '{false_output}'")
            
            # For this simple test, we'll just see what the model generates
            # when given the context and asked to continue
            simple_cmd = [
                "python", "tests/chat.py",
                "--meta", f"{model_path}/meta.yaml", 
                "--no-template",
                "--prompt", prompt,
                "--eval",
                "--max-tokens", "5"
            ]
            
            print(f"Testing free generation...")
            simple_result = subprocess.run(simple_cmd, capture_output=True, text=True, cwd="/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll")
            simple_output = simple_result.stdout.strip()
            
            print(f"Free generation: '{simple_output}'")
            
            # Determine model's answer
            model_answer = None
            if "true" in simple_output.lower():
                model_answer = "True"
            elif "false" in simple_output.lower():
                model_answer = "False"
            else:
                # Try to analyze which continuation was more likely based on context
                # This is a simplified approach
                if len(true_output) > len(false_output):
                    model_answer = "True"
                elif len(false_output) > len(true_output):
                    model_answer = "False"
                else:
                    model_answer = "Unknown"
            
            is_correct = model_answer == correct_answer
            results.append({
                'question': question,
                'correct': correct_answer,
                'model_answer': model_answer,
                'is_correct': is_correct,
                'free_generation': simple_output,
                'true_continuation': true_output,
                'false_continuation': false_output
            })
            
            print(f"MODEL ANSWER: {model_answer}")
            print(f"CORRECT: {is_correct}")
            
        except Exception as e:
            print(f"Error testing question {i+1}: {e}")
            results.append({
                'question': question,
                'correct': correct_answer,
                'model_answer': 'Error',
                'is_correct': False,
                'error': str(e)
            })
        
        finally:
            # Clean up temp files
            try:
                os.unlink(true_file)
                os.unlink(false_file)
            except:
                pass
    
    # Calculate accuracy
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = correct_count / len(results)
    
    print(f"\n" + "=" * 60)
    print("CHAT.PY RESULTS SUMMARY")
    print("=" * 60)
    print(f"Correct: {correct_count}/{len(results)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy, results

def main():
    model_path = "/tmp/test-qwen25-0.5b-base"
    num_samples = 10
    
    print("Loading BoolQ samples...")
    samples = load_boolq_samples(num_samples)
    
    print(f"Testing {num_samples} BoolQ questions with model: {model_path}")
    
    # Test with harness
    harness_accuracy = test_with_harness(model_path, num_samples)
    
    # Test with chat
    chat_accuracy, chat_results = test_with_chat(model_path, samples)
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Harness accuracy: {harness_accuracy}")
    print(f"Chat accuracy: {chat_accuracy}")
    
    if harness_accuracy is not None and chat_accuracy is not None:
        diff = abs(harness_accuracy - chat_accuracy)
        print(f"Difference: {diff:.4f}")
        
        if diff < 0.1:
            print("✓ Results are similar - evaluation methods are consistent")
        else:
            print("⚠ Results differ significantly - there may be an issue with one method")
    
    print("\nDetailed chat results:")
    for i, result in enumerate(chat_results):
        print(f"{i+1}. {result['question'][:50]}...")
        print(f"   Correct: {result['correct']}, Model: {result['model_answer']}, Match: {result['is_correct']}")
        if 'free_generation' in result:
            print(f"   Generated: '{result['free_generation']}'")

if __name__ == "__main__":
    main()