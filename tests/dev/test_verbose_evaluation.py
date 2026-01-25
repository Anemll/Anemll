#!/usr/bin/env python3
"""Test script to demonstrate verbose output in evaluation."""

import subprocess
import sys

def run_evaluation_with_verbose():
    """Run evaluation with verbose output enabled."""
    
    # Command to run
    cmd = [
        sys.executable,
        "evaluate/ane/evaluate_with_harness.py",
        "--model", "/tmp/test-qwen25-0.5b-base",
        "--tasks", "boolq",
        "--batch-size", "1",
        "--apply-chat-template",
        "--verbose-output",  # Enable verbose output
        "--limit", "5"  # Limit to 5 examples for testing
    ]
    
    print("Running evaluation with verbose output...")
    print("Command:", " ".join(cmd))
    print("-" * 80)
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_evaluation_with_verbose())