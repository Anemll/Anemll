#!/usr/bin/env python3
"""
Analyze score differences for questions where MLX got them right but ANE got them wrong.
Focus on questions with larger yes/no score ranges.
"""

import re
from pathlib import Path

def extract_ane_score_data(log_file):
    """Extract question data with scores from ANE log"""
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

def analyze_mlx_wins():
    """Analyze questions where MLX won with focus on score differences"""
    
    # Questions where ANE failed but MLX succeeded
    mlx_wins = [58, 78, 85, 86, 87, 91, 92]
    
    # Get ANE score data
    ane_scores = extract_ane_score_data('incorrect_answers_ane.log')
    
    print("=== Questions where MLX got RIGHT but ANE got WRONG ===")
    print("(Sorted by yes/no score range - largest differences first)\n")
    
    # Create list with score data for MLX wins
    mlx_win_data = []
    for q_num in mlx_wins:
        if q_num in ane_scores:
            data = ane_scores[q_num]
            mlx_win_data.append((q_num, data))
        else:
            print(f"Warning: No score data found for question {q_num}")
    
    # Sort by score range (largest difference first)
    mlx_win_data.sort(key=lambda x: x[1]['score_range'], reverse=True)
    
    print(f"{'Question':<10} {'Selected':<10} {'Correct':<10} {'Range':<10} {'Difference':<12}")
    print("-" * 65)
    
    for q_num, data in mlx_win_data:
        print(f"{q_num:<10} {data['selected_score']:<10.4f} {data['correct_score']:<10.4f} "
              f"{data['score_range']:<10.4f} {data['score_difference']:<12.4f}")
    
    print(f"\n=== Top 3 questions with largest yes/no score ranges ===")
    for i, (q_num, data) in enumerate(mlx_win_data[:3]):
        print(f"{i+1}. Question {q_num}: Range = {data['score_range']:.4f}")
        print(f"   Selected: {data['selected_score']:.4f}, Correct: {data['correct_score']:.4f}")
        print(f"   Score difference: {data['score_difference']:.4f}")
        print()

if __name__ == "__main__":
    analyze_mlx_wins()