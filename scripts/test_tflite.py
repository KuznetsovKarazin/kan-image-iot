import argparse
import time
import os
import sys
import numpy as np
from pathlib import Path

# Important: import TF after settings but before use
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Attempt to handle potential import errors gracefully or mock if needed
# (Assuming standard environment has these, but windows led to valid warnings before)
try:
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"Error importing torch/torchvision: {e}")
    sys.exit(1)

# Add project root to path for config import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Define parser for use in main
parser = argparse.ArgumentParser(description='Test TFLite model on validation set')
parser.add_argument('model_path', type=str, nargs='?', default=None,
                    help='Path to TFLite model file (optional, defaults to experiment model)')
parser.add_argument('--data_dir', type=str, default=r'data/processed/vww_subset/val', 
                    help='Path to validation dataset')
parser.add_argument('--img_size', type=int, default=224, help='Image size')
parser.add_argument('--limit', type=int, default=None, help='Limit number of batches for quick testing')
parser.add_argument('--disable_xnnpack', action='store_true', help='Disable XNNPACK delegate')
args, _ = parser.parse_known_args()

def load_tflite_model(model_path, use_xnnpack=True):
    """Load TFLite model and allocate tensors."""
    # Note: On some Windows builds, TF_LITE_DISABLE_XNNPACK=1 and experimental_delegates=[]
    # are ignored by the standard tf.lite.Interpreter.
    
    try:
        # 1. If XNNPACK is allowed, try standard TFLite first for best performance
        if use_xnnpack:
            try:
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                return interpreter
            except Exception as e:
                print(f"[WARN] Standard TFLite failed (likely XNNPACK crash): {e}")
                # Fall through to modern LiteRT or no-delegate mode

        # 2. Try modern LiteRT if available (usually more stable and respects no-delegate better)
        try:
            from ai_edge_litert.interpreter import Interpreter
            print("[INFO] Attempting to use modern LiteRT interpreter...")
            interpreter = Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARN] LiteRT also failed: {e}")

        # 3. Last resort: Standard TFLite with EXPLICIT delegation off (trying again just in case)
        print("[INFO] Loading with standard TFLite and experimental_delegates=[]...")
        # Force the environment variable just in case another build respects it
        if not use_xnnpack:
            os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"
            
        interpreter = tf.lite.Interpreter(
            model_path=model_path, 
            experimental_delegates=[],
            num_threads=1 # Sometimes helps debugging
        )
        interpreter.allocate_tensors()
        return interpreter
            
    except Exception as e:
        print(f"[FAIL] All interpreter loading attempts failed: {e}")
        return None

def evaluate_model(model_path, data_dir, img_size=224, batch_size=1, limit=None, use_xnnpack=True, save_report=True):
    """
    Evaluate TFLite model on dataset.
    
    Args:
        model_path: Path to TFLite model file
        data_dir: Path to validation dataset
        img_size: Image size for input
        batch_size: Batch size (usually 1 for TFLite)
        limit: Limit number of batches for quick testing
        use_xnnpack: Whether to use XNNPACK delegate
        save_report: Whether to save report to experiment analysis directory
    """
    print(f"Loading model: {model_path}")
    interpreter = load_tflite_model(model_path, use_xnnpack=use_xnnpack)
    if not interpreter:
        return
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check if model expects float or int8 (quantized) input
    input_dtype = input_details[0]['dtype']
    print(f"Model Input Shape: {input_details[0]['shape']}")
    print(f"Model Input Type: {input_dtype}")
    
    # Setup Data Loading
    print(f"Loading dataset from: {data_dir}")
    
    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return

    print(f"Found {len(dataset)} images. Classes: {dataset.classes}")
    
    correct = 0
    total = 0
    latencies = []
    
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    all_preds = []
    all_labels = []
    
    print("Starting inference...")
    
    for i, (images, labels) in enumerate(tqdm(dataloader)):
        if limit and i >= limit:
            break
            
        # Process each image in batch (though TFLite usually runs batch=1 on edge)
        for j in range(len(images)):
            image = images[j] # [3, H, W]
            label = labels[j].item()
            
            # Prepare input
            # TFLite expects [B, H, W, C] usually, but let's check input_details shape
            input_shape = input_details[0]['shape']
            
            # Transpose to NHWC if needed (PyTorch is NCHW)
            # Most TFLite models converted from PyTorch via ONNX might still be NCHW 
            # OR they effectively transpose during conversion.
            # ai_edge_torch conversion usually produces NHWC or preserves semantics.
            # Standard TF conversion expects NHWC.
            
            # We must adhere to input_details shape. 
            # If shape is [1, 224, 224, 3], we transpose.
            # If shape is [1, 3, 224, 224], we keep as is.
            
            if input_shape[3] == 3 and input_shape[1] != 3:
                # NHWC
                input_data = image.permute(1, 2, 0).numpy() # [H, W, C]
                input_data = np.expand_dims(input_data, axis=0) # [1, H, W, C]
            else:
                # NCHW
                input_data = image.numpy()
                input_data = np.expand_dims(input_data, axis=0)
            
            # quantization handling if input expects int8/uint8
            if input_dtype == np.uint8:
                input_scale, input_zero_point = input_details[0]['quantization']
                input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
            elif input_dtype == np.int8:
                input_scale, input_zero_point = input_details[0]['quantization']
                input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
            else:
                input_data = input_data.astype(np.float32)
                
            # Inference
            start_time = time.time()
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_index)
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000) # ms
            
            # Get Prediction
            # Assuming classification output [1, NumClasses]
            prediction = np.argmax(output_data)
            
            if prediction == label:
                correct += 1
            total += 1
            
            all_preds.append(prediction)
            all_labels.append(label)
            
    if total == 0:
        print("No images processed.")
        return

    accuracy = 100 * correct / total
    avg_latency = np.mean(latencies)
    
    # Prepare report text
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=dataset.classes)
    
    report_text = [
        "="*60,
        "TFLite Model Evaluation Report",
        "="*60,
        f"Model: {model_path}",
        f"Dataset: {data_dir}",
        f"Total Images: {total}",
        "",
        "="*60,
        "Results",
        "="*60,
        f"Accuracy: {accuracy:.2f}% ({correct}/{total})",
        f"Avg Latency: {avg_latency:.2f} ms/sample",
        f"Min Latency: {np.min(latencies):.2f} ms",
        f"Max Latency: {np.max(latencies):.2f} ms",
        f"Std Latency: {np.std(latencies):.2f} ms",
        "="*60,
        "",
        "Confusion Matrix:",
        str(cm),
        "",
        "Classification Report:",
        report,
        "="*60,
    ]
    
    report_output = "\n".join(report_text)
    
    # Print to console
    print("\n" + report_output)
    
    # Save report if requested
    if save_report:
        try:
            # Determine analysis directory from model path
            model_dir = Path(model_path).parent
            analysis_dir = model_dir.parent / 'analysis'
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            report_path = analysis_dir / 'report_tflite.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_output)
            
            print(f"\n[INFO] Report saved to: {report_path}")
        except Exception as e:
            print(f"\n[WARN] Failed to save report: {e}")
    
    return accuracy, avg_latency

if __name__ == "__main__":
    # If no model path specified, use default from experiment
    if args.model_path is None:
        # Get experiment paths from config
        exp_paths = config.get_experiment_paths()
        model_dir = exp_paths['model_dir']
        tflite_path = Path(model_dir) / 'model.tflite'
        
        if not tflite_path.exists():
            print(f"[ERROR] No TFLite model found at default location: {tflite_path}")
            print("Please specify a model path or run convert_to_tflite.py first.")
            sys.exit(1)
        
        args.model_path = str(tflite_path)
        print(f"[INFO] Using model from experiment: {args.model_path}")
    
    # Verify model exists
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model not found at {args.model_path}")
        sys.exit(1)
        
    # Verify data directory exists
    if not os.path.exists(args.data_dir):
        # Fallback to absolute path check if relative fails
        abs_data_dir = os.path.join(os.getcwd(), args.data_dir)
        if os.path.exists(abs_data_dir):
            args.data_dir = abs_data_dir
        else:
            print(f"[ERROR] Data directory not found at {args.data_dir}")
            sys.exit(1)
    
    print("="*60)
    print("TFLite Model Testing")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.data_dir}")
    print(f"XNNPACK: {'Disabled' if args.disable_xnnpack else 'Enabled'}")
    print("="*60)
    print()
            
    evaluate_model(args.model_path, args.data_dir, args.img_size, 
                   limit=args.limit, use_xnnpack=not args.disable_xnnpack, save_report=True)
