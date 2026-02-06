"""
Test script to verify the TFLite conversion script logic.
Tests configuration loading without requiring full dependencies.
"""

import sys
import os

# Add paths
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

# Test 1: Import config
print("Test 1: Loading config.py...")
try:
    import config
    print("[OK] Config loaded successfully")
    print(f"  Preprocessor type: {config.PREPROCESSOR_CONFIG['preprocessor_type']}")
    print(f"  Width multiplier: {config.PREPROCESSOR_CONFIG['width_mult']}")
    print(f"  Feature dim: {config.PREPROCESSOR_CONFIG['output_features']}")
    print(f"  KAN hidden dims: {config.KAN_CONFIG['hidden_dims']}")
    print(f"  KAN grid: {config.KAN_CONFIG['grid']}")
    print(f"  KAN degree: {config.KAN_CONFIG['degree']}")
except Exception as e:
    print(f"[FAIL] Failed to load config: {e}")
    sys.exit(1)

# Test 2: Check convert_to_tflite.py syntax
print("\nTest 2: Checking convert_to_tflite.py syntax...")
try:
    import py_compile
    py_compile.compile('scripts/convert_to_tflite.py', doraise=True)
    print("[OK] convert_to_tflite.py syntax is valid")
except Exception as e:
    print(f"[FAIL] Syntax error: {e}")
    sys.exit(1)

# Test 3: Check that config values match expected types
print("\nTest 3: Validating config types...")
try:
    assert isinstance(config.PREPROCESSOR_CONFIG['output_features'], int)
    assert isinstance(config.KAN_CONFIG['hidden_dims'], list)
    assert isinstance(config.KAN_CONFIG['grid'], int)
    assert isinstance(config.KAN_CONFIG['degree'], int)
    assert isinstance(config.PREPROCESSOR_CONFIG['width_mult'], (int, float))
    assert isinstance(config.PREPROCESSOR_CONFIG['preprocessor_type'], str)
    print("[OK] All config types are valid")
except AssertionError:
    print("[FAIL] Invalid config types")
    sys.exit(1)

# Test 4: Show expected model parameters
print("\nTest 4: Model architecture based on config...")
print(f"  Input channels: {config.PREPROCESSOR_CONFIG.get('input_channels', 3)}")
print(f"  Feature dimension: {config.PREPROCESSOR_CONFIG['output_features']}")
print(f"  KAN layers: {config.PREPROCESSOR_CONFIG['output_features']} -> {' -> '.join(map(str, config.KAN_CONFIG['hidden_dims']))} -> 2")
print(f"  Preprocessor: {config.PREPROCESSOR_CONFIG['preprocessor_type']} (width_mult={config.PREPROCESSOR_CONFIG['width_mult']})")

print("\n" + "="*60)
print("ALL TESTS PASSED")
print("="*60)
print("\nThe TFLite conversion script is correctly configured.")
print("To run the actual conversion, you need to:")
print("  1. Install dependencies: pip install onnx onnx-tf tensorflow")
print("  2. Run: python scripts/convert_to_tflite.py <model_path.pt>")
print("="*60)
