#!/usr/bin/env python3
"""
Compare PyTorch vs MLX incorrect answers to find differences between the two implementations.
"""

import re
from pathlib import Path

def extract_question_numbers(log_file):
    """Extract question numbers from incorrect answers log"""
    if not Path(log_file).exists():
        print(f"File not found: {log_file}")
        return set()
    
    question_numbers = set()
    with open(log_file, 'r') as f:
        content = f.read()
        # Find all "QUESTION X (INCORRECT):" patterns
        matches = re.findall(r'QUESTION (\d+) \(INCORRECT\):', content)
        question_numbers = {int(match) for match in matches}
    
    return question_numbers

def extract_score_data(log_file):
    """Extract question data with scores from log"""
    if not Path(log_file).exists():
        print(f"File not found: {log_file}")
        return {}
    
    question_data = {}
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Split into question blocks
        question_blocks = content.split('QUESTION ')[1:]  # Skip first empty part
        
        for block in question_blocks:
            lines = block.strip().split('\n')
            if not lines:
                continue
                
            # Extract question number
            first_line = lines[0]
            match = re.match(r'(\d+) \(INCORRECT\):', first_line)
            if not match:
                continue
                
            question_num = int(match.group(1))
            
            # Extract scores
            selected_score = None
            correct_score = None
            score_diff = None
            
            for line in lines:
                if line.startswith('Selected Score:'):
                    selected_score = float(line.split(': ')[1])
                elif line.startswith('Correct Score:'):
                    correct_score = float(line.split(': ')[1])
                elif line.startswith('Score Difference:'):
                    score_diff = float(line.split(': ')[1])
            
            if selected_score is not None and correct_score is not None:
                question_data[question_num] = {
                    'selected_score': selected_score,
                    'correct_score': correct_score,
                    'score_difference': score_diff,
                    'score_range': abs(selected_score - correct_score)
                }
    
    return question_data

def compare_pytorch_mlx():
    """Compare PyTorch and MLX incorrect answers"""
    
    pytorch_incorrect = extract_question_numbers('incorrect_answers_pytorch.log')
    mlx_incorrect = extract_question_numbers('incorrect_answers_mlx.log')
    
    print("=== PyTorch vs MLX Incorrect Answer Comparison ===")
    print(f"PyTorch incorrect: {len(pytorch_incorrect)} questions")
    print(f"MLX incorrect: {len(mlx_incorrect)} questions")
    
    # Questions that PyTorch got wrong but MLX got right
    pytorch_wrong_mlx_right = pytorch_incorrect - mlx_incorrect
    
    # Questions that MLX got wrong but PyTorch got right  
    mlx_wrong_pytorch_right = mlx_incorrect - pytorch_incorrect
    
    # Questions both got wrong
    both_wrong = pytorch_incorrect & mlx_incorrect
    
    print(f"\nQuestions PyTorch got wrong but MLX got RIGHT: {len(pytorch_wrong_mlx_right)}")
    if pytorch_wrong_mlx_right:
        print(f"  Questions: {sorted(pytorch_wrong_mlx_right)}")
    
    print(f"\nQuestions MLX got wrong but PyTorch got RIGHT: {len(mlx_wrong_pytorch_right)}")
    if mlx_wrong_pytorch_right:
        print(f"  Questions: {sorted(mlx_wrong_pytorch_right)}")
    
    print(f"\nQuestions BOTH got wrong: {len(both_wrong)}")
    if both_wrong:
        print(f"  Questions: {sorted(both_wrong)}")
    
    print(f"\n=== Summary ===")
    print(f"MLX advantage: {len(pytorch_wrong_mlx_right)} questions")
    print(f"PyTorch advantage: {len(mlx_wrong_pytorch_right)} questions")
    print(f"Net MLX advantage: {len(pytorch_wrong_mlx_right) - len(mlx_wrong_pytorch_right)} questions")
    
    # If there are differences, analyze score ranges
    if pytorch_wrong_mlx_right:
        print(f"\n=== Score Analysis for Questions MLX got RIGHT but PyTorch got WRONG ===")
        pytorch_scores = extract_score_data('incorrect_answers_pytorch.log')
        
        # Create list with score data for MLX wins
        mlx_win_data = []
        for q_num in pytorch_wrong_mlx_right:
            if q_num in pytorch_scores:
                data = pytorch_scores[q_num]
                mlx_win_data.append((q_num, data))
        
        # Sort by score range (largest difference first)
        mlx_win_data.sort(key=lambda x: x[1]['score_range'], reverse=True)
        
        print(f"{'Question':<10} {'Selected':<10} {'Correct':<10} {'Range':<10} {'Difference':<12}")
        print("-" * 65)
        
        for q_num, data in mlx_win_data:
            print(f"{q_num:<10} {data['selected_score']:<10.4f} {data['correct_score']:<10.4f} "
                  f"{data['score_range']:<10.4f} {data['score_difference']:<12.4f}")

if __name__ == "__main__":
    compare_pytorch_mlx()