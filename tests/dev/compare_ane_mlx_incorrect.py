#!/usr/bin/env python3
"""
Compare incorrect answers between two log files to find differences.
"""

import argparse
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

def compare_incorrect_answers(file1_path, file2_path, file1_name="File1", file2_name="File2"):
    """Compare incorrect answers between two log files"""
    
    file1_incorrect = extract_question_numbers(file1_path)
    file2_incorrect = extract_question_numbers(file2_path)
    
    print(f"=== {file1_name} vs {file2_name} Incorrect Answer Comparison ===")
    print(f"{file1_name} incorrect: {len(file1_incorrect)} questions")
    print(f"{file2_name} incorrect: {len(file2_incorrect)} questions")
    
    # Questions that File1 got wrong but File2 got right
    file1_wrong_file2_right = file1_incorrect - file2_incorrect
    
    # Questions that File2 got wrong but File1 got right  
    file2_wrong_file1_right = file2_incorrect - file1_incorrect
    
    # Questions both got wrong
    both_wrong = file1_incorrect & file2_incorrect
    
    print(f"\nQuestions {file1_name} got wrong but {file2_name} got RIGHT: {len(file1_wrong_file2_right)}")
    if file1_wrong_file2_right:
        print(f"  Questions: {sorted(file1_wrong_file2_right)}")
    
    print(f"\nQuestions {file2_name} got wrong but {file1_name} got RIGHT: {len(file2_wrong_file1_right)}")
    if file2_wrong_file1_right:
        print(f"  Questions: {sorted(file2_wrong_file1_right)}")
    
    print(f"\nQuestions BOTH got wrong: {len(both_wrong)}")
    if both_wrong:
        print(f"  Questions: {sorted(both_wrong)}")
    
    print(f"\n=== Summary ===")
    print(f"{file2_name} advantage: {len(file1_wrong_file2_right)} questions")
    print(f"{file1_name} advantage: {len(file2_wrong_file1_right)} questions")
    print(f"Net {file2_name} advantage: {len(file1_wrong_file2_right) - len(file2_wrong_file1_right)} questions")

def main():
    parser = argparse.ArgumentParser(description='Compare incorrect answers between two log files')
    parser.add_argument('file1', nargs='?', default='incorrect_answers_ane.log', 
                       help='First log file (default: incorrect_answers_ane.log)')
    parser.add_argument('file2', nargs='?', default='incorrect_answers_mlx.log',
                       help='Second log file (default: incorrect_answers_mlx.log)')
    parser.add_argument('--name1', default=None, help='Name for first file in output')
    parser.add_argument('--name2', default=None, help='Name for second file in output')
    
    args = parser.parse_args()
    
    # Auto-detect names from filenames if not provided
    if args.name1 is None:
        args.name1 = Path(args.file1).stem.replace('incorrect_answers_', '').upper()
    if args.name2 is None:
        args.name2 = Path(args.file2).stem.replace('incorrect_answers_', '').upper()
    
    compare_incorrect_answers(args.file1, args.file2, args.name1, args.name2)

if __name__ == "__main__":
    main()