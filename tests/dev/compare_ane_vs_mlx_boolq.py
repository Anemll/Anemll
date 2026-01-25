#!/usr/bin/env python3
"""
Compare ANE vs MLX model predictions on specific BoolQ questions.
This script evaluates the same questions on both models to identify discrepancies.
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path

def run_ane_evaluation(model_path, skip, limit, output_file):
    """Run ANE evaluation and return results"""
    cmd = [
        sys.executable,
        "./evaluate/ane/evaluate_with_harness.py",
        "--model", model_path,
        "--tasks", "boolq",
        "--batch-size", "1",
        "--limit", str(limit),
        "--skip", str(skip),
        "--log-incorrect-answers",
        "--verbose-output"
    ]
    
    print(f"Running ANE evaluation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll")
    
    # Save raw output
    with open(f"{output_file}_ane_raw.log", "w") as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
    
    return result.stdout, result.stderr

def run_mlx_evaluation(model_path, skip, limit, output_file):
    """Run MLX evaluation and return results"""
    # Use our modified MLX evaluate script with skip support
    cmd = [
        sys.executable, "./tests/dev/mlx_evaluate_with_skip.py",
        "--model", model_path,
        "--tasks", "boolq",
        "--batch-size", "1", 
        "--limit", str(limit),
        "--skip", str(skip),
        "--apply-chat-template", "false"  # Force no chat template for base model
    ]
    
    print(f"Running MLX evaluation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save raw output  
    with open(f"{output_file}_mlx_raw.log", "w") as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
    
    return result.stdout, result.stderr

def parse_ane_incorrect_answers(log_file_path):
    """Parse incorrect answers from ANE log file"""
    incorrect_answers = []
    
    if not os.path.exists(log_file_path):
        print(f"Warning: ANE log file not found: {log_file_path}")
        return incorrect_answers
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Parse questions from the log
    questions = content.split("QUESTION ")[1:]  # Skip header
    
    for q_text in questions:
        if "(INCORRECT)" not in q_text:
            continue
            
        lines = q_text.strip().split('\n')
        question_data = {}
        
        # Extract question number
        first_line = lines[0]
        if " (INCORRECT):" in first_line:
            question_data['question_num'] = int(first_line.split(" (INCORRECT):")[0])
        
        # Parse the structured data
        for line in lines:
            if line.startswith("Question: "):
                question_data['question'] = line[10:]
            elif line.startswith("Selected Answer: "):
                # Format: "Selected Answer: ' no' (index 0)"
                parts = line[17:].split(" (index ")
                question_data['selected_answer'] = parts[0].strip("'\"")
                question_data['selected_index'] = int(parts[1].rstrip(")"))
            elif line.startswith("Correct Answer: "):
                # Format: "Correct Answer: ' yes' (index 1)"
                parts = line[16:].split(" (index ")
                question_data['correct_answer'] = parts[0].strip("'\"")
                question_data['correct_index'] = int(parts[1].rstrip(")"))
            elif line.startswith("Selected Score: "):
                question_data['selected_score'] = float(line[16:])
            elif line.startswith("Correct Score: "):
                question_data['correct_score'] = float(line[15:])
            elif line.startswith("Context: "):
                question_data['context'] = line[9:]
        
        if 'question' in question_data:
            incorrect_answers.append(question_data)
    
    return incorrect_answers

def compare_results(ane_incorrect, mlx_stdout, output_file):
    """Compare ANE vs MLX results and generate report"""
    
    # Parse MLX results to get accuracy
    mlx_accuracy = None
    mlx_lines = mlx_stdout.split('\n')
    for line in mlx_lines:
        if '"acc,none":' in line:
            # Extract accuracy from JSON-like output
            import re
            match = re.search(r'"acc,none":\s*([\d.]+)', line)
            if match:
                mlx_accuracy = float(match.group(1))
                break
    
    # Generate comparison report
    report = []
    report.append("=" * 80)
    report.append("ANE vs MLX BoolQ Comparison Report")
    report.append("=" * 80)
    report.append("")
    
    if mlx_accuracy:
        report.append(f"MLX Accuracy: {mlx_accuracy:.4f}")
    
    report.append(f"ANE Incorrect Answers: {len(ane_incorrect)}")
    report.append("")
    
    report.append("ANE INCORRECT PREDICTIONS:")
    report.append("-" * 40)
    
    for i, qa in enumerate(ane_incorrect, 1):
        report.append(f"{i}. Question {qa.get('question_num', '?')}: {qa.get('question', 'Unknown')}")
        report.append(f"   Context: {qa.get('context', 'Unknown')[:100]}...")
        report.append(f"   ANE predicted: '{qa.get('selected_answer', '?')}' (score: {qa.get('selected_score', 'N/A')})")
        report.append(f"   Correct answer: '{qa.get('correct_answer', '?')}' (score: {qa.get('correct_score', 'N/A')})")
        report.append("")
    
    report.append("=" * 80)
    report.append("NEXT STEPS:")
    report.append("1. Run same questions on MLX model")
    report.append("2. Compare specific predictions for each question")
    report.append("3. Identify pattern in discrepancies")
    report.append("=" * 80)
    
    # Save report
    report_text = '\n'.join(report)
    with open(f"{output_file}_comparison_report.txt", "w") as f:
        f.write(report_text)
    
    print("\n" + report_text)
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description="Compare ANE vs MLX model predictions on BoolQ")
    parser.add_argument("--ane-model", required=True, help="Path to ANE model")
    parser.add_argument("--mlx-model", required=True, help="Path to MLX model") 
    parser.add_argument("--skip", type=int, default=1702, help="Skip N questions")
    parser.add_argument("--limit", type=int, default=10, help="Evaluate N questions")
    parser.add_argument("--output", default="ane_vs_mlx_comparison", help="Output file prefix")
    
    args = parser.parse_args()
    
    print(f"Comparing ANE model: {args.ane_model}")
    print(f"      vs MLX model: {args.mlx_model}")
    print(f"Questions {args.skip} to {args.skip + args.limit - 1}")
    print()
    
    # Run ANE evaluation
    print("Running ANE evaluation...")
    ane_stdout, ane_stderr = run_ane_evaluation(args.ane_model, args.skip, args.limit, args.output)
    
    # Run MLX evaluation  
    print("Running MLX evaluation...")
    mlx_stdout, mlx_stderr = run_mlx_evaluation(args.mlx_model, args.skip, args.limit, args.output)
    
    # Parse ANE incorrect answers
    log_file = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/incorrect_answers.log"
    ane_incorrect = parse_ane_incorrect_answers(log_file)
    
    # Generate comparison report
    compare_results(ane_incorrect, mlx_stdout, args.output)
    
    print(f"\nComparison complete. Check {args.output}_comparison_report.txt for details.")

if __name__ == "__main__":
    main()