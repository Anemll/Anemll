#!/usr/bin/env python3
"""
Compare MLX vs ANE segment results using the stored JSON files.
This script analyzes the segment-by-segment differences.
"""

import json
import argparse
from pathlib import Path

def load_segment_results(results_file):
    """Load segment results from JSON file"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def compare_segments(mlx_file, ane_file, output_dir):
    """Compare MLX and ANE segment results"""
    
    # Load results
    mlx_data = load_segment_results(mlx_file)
    ane_data = load_segment_results(ane_file)
    
    mlx_segments = {seg['start']: seg for seg in mlx_data['segments']}
    ane_segments = {seg['start']: seg for seg in ane_data['segments']}
    
    # Create comparison data
    comparison_data = []
    
    for start_idx in sorted(set(mlx_segments.keys()) & set(ane_segments.keys())):
        mlx_seg = mlx_segments[start_idx]
        ane_seg = ane_segments[start_idx]
        
        diff = mlx_seg['accuracy'] - ane_seg['accuracy']
        
        comparison_data.append({
            'start': start_idx,
            'end': mlx_seg['end'],
            'size': mlx_seg['size'],
            'mlx_accuracy': mlx_seg['accuracy'],
            'ane_accuracy': ane_seg['accuracy'],
            'mlx_correct': mlx_seg['correct'],
            'ane_correct': ane_seg['correct'],
            'difference': diff,
            'abs_difference': abs(diff)
        })
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save comparison TSV
    comparison_file = output_path / "mlx_vs_ane_segment_comparison.tsv"
    with open(comparison_file, 'w') as f:
        f.write("start\tend\tsize\tmlx_acc\tane_acc\tmlx_correct\tane_correct\tdifference\tabs_diff\n")
        for row in comparison_data:
            f.write(f"{row['start']}\t{row['end']}\t{row['size']}\t{row['mlx_accuracy']:.4f}\t{row['ane_accuracy']:.4f}\t{row['mlx_correct']}\t{row['ane_correct']}\t{row['difference']:.4f}\t{row['abs_difference']:.4f}\n")
    
    # Analysis
    print("=" * 80)
    print("MLX vs ANE Segment Comparison")
    print("=" * 80)
    
    print(f"Total segments compared: {len(comparison_data)}")
    print(f"MLX overall accuracy: {mlx_data['overall_accuracy']:.4f}")
    print(f"ANE overall accuracy: {ane_data['overall_accuracy']:.4f}")
    print(f"Overall difference (MLX - ANE): {mlx_data['overall_accuracy'] - ane_data['overall_accuracy']:.4f}")
    
    # Sort by difference
    comparison_data.sort(key=lambda x: x['difference'])
    
    print(f"\nBiggest ANE advantages (ANE > MLX):")
    ane_better_count = 0
    for row in comparison_data[:10]:
        if row['difference'] < 0:
            ane_better_count += 1
            print(f"  [{row['start']:4d}..{row['end']:4d}]: MLX {row['mlx_accuracy']:.4f} vs ANE {row['ane_accuracy']:.4f} (diff: {row['difference']:+.4f})")
    
    print(f"\nBiggest MLX advantages (MLX > ANE):")
    mlx_better_count = 0
    for row in reversed(comparison_data[-10:]):
        if row['difference'] > 0:
            mlx_better_count += 1
            print(f"  [{row['start']:4d}..{row['end']:4d}]: MLX {row['mlx_accuracy']:.4f} vs ANE {row['ane_accuracy']:.4f} (diff: {row['difference']:+.4f})")
    
    # Find segments where both models struggle
    comparison_data.sort(key=lambda x: (x['mlx_accuracy'] + x['ane_accuracy']) / 2)
    print(f"\nWorst performing segments (both models):")
    for row in comparison_data[:10]:
        avg_acc = (row['mlx_accuracy'] + row['ane_accuracy']) / 2
        print(f"  [{row['start']:4d}..{row['end']:4d}]: MLX {row['mlx_accuracy']:.4f}, ANE {row['ane_accuracy']:.4f} (avg: {avg_acc:.4f})")
    
    # Find segments where both models excel
    comparison_data.sort(key=lambda x: (x['mlx_accuracy'] + x['ane_accuracy']) / 2, reverse=True)
    print(f"\nBest performing segments (both models):")
    for row in comparison_data[:10]:
        avg_acc = (row['mlx_accuracy'] + row['ane_accuracy']) / 2
        print(f"  [{row['start']:4d}..{row['end']:4d}]: MLX {row['mlx_accuracy']:.4f}, ANE {row['ane_accuracy']:.4f} (avg: {avg_acc:.4f})")
    
    # Statistics
    equal_segments = sum(1 for row in comparison_data if abs(row['difference']) < 0.01)
    mlx_better_total = sum(1 for row in comparison_data if row['difference'] > 0.01)
    ane_better_total = sum(1 for row in comparison_data if row['difference'] < -0.01)
    
    print(f"\nSegment Performance Summary:")
    print(f"  Equal performance (±0.01): {equal_segments}")
    print(f"  MLX better segments: {mlx_better_total}")
    print(f"  ANE better segments: {ane_better_total}")
    
    # Average difference statistics
    differences = [row['difference'] for row in comparison_data]
    abs_differences = [row['abs_difference'] for row in comparison_data]
    
    print(f"\nDifference Statistics:")
    print(f"  Mean difference (MLX - ANE): {sum(differences) / len(differences):+.4f}")
    print(f"  Mean absolute difference: {sum(abs_differences) / len(abs_differences):.4f}")
    print(f"  Max difference (MLX favor): {max(differences):+.4f}")
    print(f"  Max difference (ANE favor): {min(differences):+.4f}")
    
    print(f"\nComparison saved to: {comparison_file}")
    
    return comparison_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-results", required=True, help="MLX results JSON file")
    parser.add_argument("--ane-results", required=True, help="ANE results JSON file") 
    parser.add_argument("--output-dir", default="./tests/dev/segmented_results", help="Output directory")
    
    args = parser.parse_args()
    
    compare_segments(args.mlx_results, args.ane_results, args.output_dir)

if __name__ == "__main__":
    main()