"""
Alternative TFLite Conversion Script using onnx2tf.
Converts ONNX model to TensorFlow Lite format.

This script uses onnx2tf instead of onnx-tf for better compatibility.

Author: Daniele Faggi
Date: February 2026
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

def convert_onnx_to_tflite(onnx_path, tflite_output_path, quantize='none'):
    """
    Convert ONNX model to TFLite using onnx2tf.
    
    Args:
        onnx_path: Path to ONNX model
        tflite_output_path: Path to save TFLite model
        quantize: Quantization type ('none', 'float16', 'int8')
    """
    print("\n" + "="*60)
    print(f"Converting ONNX to TFLite (quantize={quantize})")
    print("="*60)
    
    try:
        import tensorflow as tf
        import onnx2tf
        
        # Create temporary directory for SavedModel
        saved_model_dir = Path(onnx_path).parent / "temp_saved_model"
        saved_model_dir.mkdir(exist_ok=True)
        
        print(f"\nStep 1: Converting ONNX to TensorFlow SavedModel...")
        print(f"  Input: {onnx_path}")
        print(f"  Output: {saved_model_dir}")
        
        # Convert ONNX to TensorFlow SavedModel
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(saved_model_dir),
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True
        )
        
        print("[OK] ONNX to TensorFlow conversion completed")
        
        # Find the SavedModel directory
        saved_model_path = saved_model_dir / "saved_model"
        if not saved_model_path.exists():
            # Try alternative location
            saved_model_path = saved_model_dir
        
        print(f"\nStep 2: Converting TensorFlow to TFLite...")
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
        
        # Apply quantization
        if quantize == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            print("  Applying float16 quantization...")
        elif quantize == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("  Applying int8 quantization...")
        else:
            print("  No quantization (float32)")
        
        # Convert to TFLite
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(tflite_output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get file size
        file_size_mb = os.path.getsize(tflite_output_path) / (1024 * 1024)
        
        print(f"[OK] TFLite conversion completed")
        print(f"  Output: {tflite_output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        
        # Cleanup temporary directory
        import shutil
        shutil.rmtree(saved_model_dir, ignore_errors=True)
        
        return True, file_size_mb
        
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        print("\nPlease install onnx2tf:")
        print("  pip install onnx2tf")
        return False, 0
    except Exception as e:
        print(f"[FAIL] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    parser = argparse.ArgumentParser(
        description='Convert ONNX model to TFLite using onnx2tf'
    )
    parser.add_argument('onnx_path', type=str,
                        help='Path to ONNX model file')
    parser.add_argument('--output', type=str, default='model.tflite',
                        help='Output TFLite filename (default: model.tflite)')
    parser.add_argument('--quantize', type=str, default='none',
                        choices=['none', 'float16', 'int8'],
                        help='Quantization type (default: none)')
    
    args = parser.parse_args()
    
    # Validate ONNX path
    if not os.path.exists(args.onnx_path):
        print(f"Error: ONNX file not found: {args.onnx_path}")
        sys.exit(1)
    
    print("="*60)
    print("ONNX to TFLite Conversion (using onnx2tf)")
    print("="*60)
    print(f"Input ONNX: {args.onnx_path}")
    print(f"Output TFLite: {args.output}")
    print(f"Quantization: {args.quantize}")
    
    # Convert
    success, file_size = convert_onnx_to_tflite(
        args.onnx_path,
        args.output,
        args.quantize
    )
    
    if not success:
        print("\n[FAIL] Conversion failed")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"[OK] TFLite model saved to: {args.output}")
    print(f"  Model size: {file_size:.2f} MB")
    print(f"  Quantization: {args.quantize}")
    
    if file_size <= 1.5:
        print(f"\n[OK] Model size is IoT-ready (<= 1.5 MB)")
    elif file_size <= 3.0:
        print(f"\n[WARN] Model size is acceptable for some IoT devices (<= 3 MB)")
    else:
        print(f"\n[WARN] Model may be too large for resource-constrained IoT devices")
        print(f"  Consider using quantization (--quantize float16 or int8)")
    
    print("\nNext steps:")
    print("  1. Test the TFLite model on your target IoT device")
    print("  2. Measure inference latency and memory usage")
    print("="*60)


if __name__ == '__main__':
    main()
