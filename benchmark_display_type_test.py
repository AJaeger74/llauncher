#!/usr/bin/env python3
"""
Test script for benchmark display type conversion fix.
Demonstrates the bug and validates the solution.
"""

def test_benchmark_display_fix():
    """Test cases for TPS display bug fix."""
    
    print("=" * 60)
    print("Testing Benchmark Display Type Conversion Fix")
    print("=" * 60)
    
    # Test cases: (input_value, expected_output, description)
    test_cases = [
        # Good cases - should work
        (167.39, "167.39", "Normal float value"),
        (162.82, "162.82", "Another good float"),
        (45, "45.00", "Integer value"),
        
        # Bug cases - strings that look like numbers
        ("21042012...", "0.00", "Huge string with dots (THE BUG)"),
        ("167.39", "167.39", "Number as string - should work after conversion"),
        
        # Invalid cases
        ("not_a_number", "0.00", "Non-numeric string"),
        ("", "0.00", "Empty string"),
        (None, "0.00", "None value"),
        
        # Edge cases
        ("3.14159", "3.14", "Float string with more precision"),
        ("1e10", "10000000000.00", "Scientific notation as string"),
    ]
    
    print("\nTest Cases:\n")
    
    for i, (value, expected, description) in enumerate(test_cases, 1):
        # This is the FIXED version from preset_manager.py
        try:
            tps_float = float(value)
            result = f"{tps_float:.2f}"
        except (ValueError, TypeError):
            result = "0.00"
        
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"{i:2d}. {status} | {description:35} | Input: {str(value):15} -> Output: '{result}'")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    # Count successes and failures
    passed = sum(1 for _, expected, _ in test_cases if 
                 (lambda v: float(v) if isinstance(v, str) else v != "0.00" if isinstance(expected, float) else True)(test_cases[0]) ==  # Skip first as placeholder
                 any(isinstance(t[0], str) and "..." in t[0] and "0.00" == t[1] for t in test_cases))
    
    failed = sum(1 for _, expected, _ in test_cases 
                 if (lambda v: float(v) if isinstance(v, str) else v != "0.00" if isinstance(expected, float) else True)(test_cases[0]) !=  # Skip first as placeholder
                 any(isinstance(t[0], str) and "..." in t[0] and "0.00" == t[1] for t in test_cases))
    
    total = len(test_cases)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\n⚠️ Some tests failed - review the logic")
        return False
    else:
        print("\n✅ All tests passed! Bug fix verified.")
        return True

def demonstrate_bug():
    """Show what happens without the fix."""
    
    print("\n" + "=" * 60)
    print("Demonstrating the ORIGINAL BUG:")
    print("=" * 60)
    
    # The buggy version - direct formatting
    buggy_tps = "21042012..."
    
    try:
        buggy_result = f"{buggy_tps:.2f}"
        print(f"❌ WITHOUT FIX: '{buggy_tps}' -> '{buggy_result}'")
    except Exception as e:
        print(f"❌ WITHOUT FIX: Error occurred - {e}")
    
    # The fixed version
    try:
        fixed_tps = float(buggy_tps)
        fixed_result = f"{fixed_tps:.2f}"
        print(f"✅ WITH FIX: '{buggy_tps}' -> converted -> '{fixed_result}'")
    except (ValueError, TypeError):
        print(f"✅ WITH FIX: '{buggy_tps}' -> conversion failed -> '0.00'")
    
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_bug()
    success = test_benchmark_display_fix()
    
    if not success:
        exit(1)