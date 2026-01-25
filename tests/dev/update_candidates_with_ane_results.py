#!/usr/bin/env python3
"""
Update the candidates file with ANE vs MLX comparison results
"""

import json
from pathlib import Path

def update_candidates_file():
    # Load the ANE vs MLX comparison data
    comparison_file = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev/segment_analysis/ane_vs_mlx_detailed_scores_comparison.json")
    candidates_file = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev/segment_analysis/segment_1800_mlx_vs_ane_candidates.txt")
    
    with open(comparison_file, 'r') as f:
        comparison_data = json.load(f)
    
    # Create lookup by question index
    comparison_lookup = {item['question_idx']: item for item in comparison_data}
    
    # Read the original file
    with open(candidates_file, 'r') as f:
        content = f.read()
    
    # Update the content
    lines = content.split('\n')
    new_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for question lines (format: " 1. Q1835: question text")
        if line.strip().startswith(tuple(f"{j}." for j in range(1, 21))) and "Q" in line and ":" in line:
            # Extract question number
            try:
                q_part = line.split("Q")[1].split(":")[0]
                q_idx = int(q_part)
                
                if q_idx in comparison_lookup:
                    comp = comparison_lookup[q_idx]
                    
                    # Add the updated question line
                    question_text = comp['question_text']
                    new_lines.append(f" {len([x for x in comparison_lookup.keys() if x <= q_idx])}. Q{q_idx}: {question_text}")
                    
                    # Add ground truth and predictions
                    gt = comp['ground_truth'].upper()
                    mlx_pred = comp['mlx_prediction'].upper()
                    ane_pred = comp['ane_prediction'].upper()
                    
                    mlx_check = "✓" if comp['mlx_correct'] else "✗"
                    ane_check = "✓" if comp['ane_correct'] else "✗"
                    
                    new_lines.append(f"    Ground truth: {gt} ✓")
                    new_lines.append(f"    MLX prediction: {mlx_pred} {mlx_check} (confidence: {comp['mlx_confidence']:.3f}) - MLX scores: no={comp['mlx_no_score']:.3f}, yes={comp['mlx_yes_score']:.3f}")
                    new_lines.append(f"    ANE prediction: {ane_pred} {ane_check} (confidence: {comp['ane_confidence']:.3f}) - ANE scores: no={comp['ane_no_score']:.3f}, yes={comp['ane_yes_score']:.3f}")
                    new_lines.append(f"    OUTCOME: {comp['outcome']}")
                    
                    # Skip the old format lines and find the context
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("FULL PREFILL CONTEXT:"):
                        i += 1
                    
                    # Add the context and following lines until next question or end
                    while i < len(lines):
                        if (i + 1 < len(lines) and 
                            lines[i + 1].strip().startswith(tuple(f"{j}." for j in range(1, 21))) and 
                            "Q" in lines[i + 1]):
                            new_lines.append(lines[i])
                            break
                        new_lines.append(lines[i])
                        i += 1
                else:
                    new_lines.append(line)
            except (ValueError, IndexError):
                new_lines.append(line)
        else:
            new_lines.append(line)
        
        i += 1
    
    # Add summary at the end
    new_lines.extend([
        "",
        "=" * 80,
        "FINAL SUMMARY",
        "=" * 80,
        f"Questions analyzed: {len(comparison_data)}",
        f"MLX wins (MLX correct, ANE wrong): {sum(1 for item in comparison_data if item['outcome'] == 'MLX WINS')}",
        f"ANE wins (ANE correct, MLX wrong): {sum(1 for item in comparison_data if item['outcome'] == 'ANE WINS')}",
        f"Both models correct: {sum(1 for item in comparison_data if item['outcome'] == 'BOTH CORRECT')}",
        f"Both models wrong: {sum(1 for item in comparison_data if item['outcome'] == 'BOTH WRONG')}",
        "",
        "This analysis shows exactly how ANE quantization affects performance on borderline questions",
        "where MLX had low confidence. These represent the core differences between the models."
    ])
    
    # Write the updated content
    updated_content = '\n'.join(new_lines)
    with open(candidates_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated {candidates_file} with ANE vs MLX comparison results")
    print(f"Added direct score comparisons for {len(comparison_data)} questions")

if __name__ == "__main__":
    update_candidates_file()