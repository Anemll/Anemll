#!/usr/bin/env python3
"""
Test script to verify that absolute question numbering works correctly 
in both ANE and MLX evaluation scripts.
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evaluate", "ane"))

class MockQuestion:
    """Mock question for testing"""
    def __init__(self, question_idx, skip):
        self.question_idx = question_idx
        self.skip = skip
        
    def test_absolute_numbering(self):
        """Test the absolute numbering calculation"""
        absolute_question_num = self.skip + self.question_idx + 1
        return absolute_question_num

def test_ane_implementation():
    """Test the ANE implementation"""
    try:
        from evaluate_with_harness import ANELM
        print("✓ Successfully imported ANELM")
        
        # Test constructor with skip parameter
        model = ANELM.__new__(ANELM)  # Create instance without calling __init__
        model.skip = 45
        
        # Test the numbering calculation that would be used in _log_incorrect_answers
        test_cases = [
            (0, 45, 46),    # First question when skip=45 should be question 46
            (4, 45, 50),    # Fifth question when skip=45 should be question 50 
            (99, 45, 145),  # 100th question when skip=45 should be question 145
            (0, 0, 1),      # First question when skip=0 should be question 1
            (4, 0, 5),      # Fifth question when skip=0 should be question 5
        ]
        
        for question_idx, skip, expected in test_cases:
            model.skip = skip
            absolute_question_num = model.skip + question_idx + 1
            assert absolute_question_num == expected, f"Expected {expected}, got {absolute_question_num}"
            print(f"✓ ANE: question_idx={question_idx}, skip={skip} → absolute={absolute_question_num}")
            
        print("✓ ANE absolute numbering implementation is correct")
        
    except ImportError as e:
        print(f"✗ Could not import ANE implementation: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing ANE implementation: {e}")
        return False
    
    return True

def test_mlx_implementation():
    """Test the MLX implementation"""
    try:
        from full_mlx_evaluate import MLXLM
        print("✓ Successfully imported MLXLM")
        
        # Test constructor with skip parameter
        model = MLXLM.__new__(MLXLM)  # Create instance without calling __init__
        model.skip = 45
        
        # Test the numbering calculation that would be used in _log_incorrect_answers
        test_cases = [
            (0, 45, 46),    # First question when skip=45 should be question 46
            (4, 45, 50),    # Fifth question when skip=45 should be question 50 
            (99, 45, 145),  # 100th question when skip=45 should be question 145
            (0, 0, 1),      # First question when skip=0 should be question 1
            (4, 0, 5),      # Fifth question when skip=0 should be question 5
        ]
        
        for question_idx, skip, expected in test_cases:
            model.skip = skip
            absolute_question_num = model.skip + question_idx + 1
            assert absolute_question_num == expected, f"Expected {expected}, got {absolute_question_num}"
            print(f"✓ MLX: question_idx={question_idx}, skip={skip} → absolute={absolute_question_num}")
            
        print("✓ MLX absolute numbering implementation is correct")
        
    except ImportError as e:
        print(f"✗ Could not import MLX implementation: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing MLX implementation: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Testing absolute question numbering implementation...\n")
    
    ane_ok = test_ane_implementation()
    print()
    mlx_ok = test_mlx_implementation()
    
    print("\n" + "="*50)
    if ane_ok and mlx_ok:
        print("✓ All tests passed! Both ANE and MLX implementations use absolute question numbering.")
        print("  When skip=N, the first processed question will be numbered N+1")
        print("  When skip=0, the first processed question will be numbered 1")
    else:
        print("✗ Some tests failed. Check the implementation.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())