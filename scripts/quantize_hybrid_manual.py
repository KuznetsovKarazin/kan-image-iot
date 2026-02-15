
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys
import os
from pathlib import Path
import argparse

sys.path.insert(0, 'src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from models.kan_model import KANImageClassifier
import config

def evaluate_model(model, test_loader, dataset_classes):
    """
    Evaluate model and return detailed metrics.
    Similar to test_tflite.py evaluation.
    """
    import time
    
    all_preds = []
    all_targets = []
    latencies = []
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            
            # Calculate latency in ms per sample
            batch_latency = (end_time - start_time) * 1000 / len(inputs)
            latencies.extend([batch_latency] * len(inputs))
            
            _, preds = outputs.max(1)
            all_preds.extend(preds.numpy())
            all_targets.extend(targets.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = 100 * np.mean(all_preds == all_targets)
    
    # Calculate latency statistics
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)
    
    # Generate confusion matrix and classification report
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=dataset_classes)
    
    return accuracy, cm, report, all_preds, all_targets, avg_latency, min_latency, max_latency, std_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Hybrid INT8 Quantization for KAN Image Classifier'
    )
    parser.add_argument("model_path", type=str, nargs='?', default=None,
                        help="Path to FP32 checkpoint (defaults to experiment best model)")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output filename (defaults to kan_hybrid.pt in experiment models dir)")
    parser.add_argument("--force_config", action="store_true",
                        help="Force configuration from config.py (ignore checkpoint config)")
    args = parser.parse_args()

    # Handle default paths
    exp_paths = config.get_experiment_paths()
    
    if args.model_path is None:
        MODEL_PATH = exp_paths['model_dir'] / 'kan_person_detector_best.pt'
        print(f"[INFO] No model path provided. Using default: {MODEL_PATH}")
    else:
        MODEL_PATH = Path(args.model_path)
    
    if args.output_name is None:
        OUTPUT_PATH = exp_paths['model_dir'] / 'kan_hybrid.pt'
        print(f"[INFO] No output name provided. Using default: {OUTPUT_PATH}")
    else:
        OUTPUT_PATH = Path(args.output_name)
    
    # Validate input exists
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        sys.exit(1)
    
    print("=" * 60)
    print("KAN Hybrid INT8 Quantization")
    print("=" * 60)
    print(f"Input model:  {MODEL_PATH}")
    print(f"Output model: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Load original model
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Try to get configuration from checkpoint first, fallback to config.py
    saved_config = None
    if isinstance(checkpoint, dict):
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
        elif 'model_info' in checkpoint and 'config' in checkpoint['model_info']:
            saved_config = checkpoint['model_info']['config']

    if saved_config and not args.force_config:
        print("\nUsing configuration from checkpoint...")
        
        if 'preprocessor' in saved_config:
            # Checkpoint contains preprocessor config
            preproc_config = saved_config['preprocessor']
            kan_config = saved_config['kan']
            
            feature_dim = preproc_config['output_features']
            hidden_dims = kan_config.get('hidden_dims', [feature_dim // 2])
            grid = kan_config.get('grid', 5)
            degree = kan_config.get('degree', 3)
            
            # Infer preprocessor type
            if 'preprocessor_type' in preproc_config:
                preprocessor_type = preproc_config['preprocessor_type']
            else:
                print("[INFO] preprocessor_type missing in config, assuming 'mobilenetv3_small'")
                preprocessor_type = 'mobilenetv3_small'
                
            width_mult = preproc_config.get('width_mult', 1.0)
        else:
            # Config format not recognized, falling back to config.py
            print("[WARN] Config format not recognized, falling back to config.py")
            feature_dim = config.PREPROCESSOR_CONFIG['output_features']
            hidden_dims = config.KAN_CONFIG['hidden_dims']
            grid = config.KAN_CONFIG['grid']
            degree = config.KAN_CONFIG['degree']
            preprocessor_type = config.PREPROCESSOR_CONFIG['preprocessor_type']
            width_mult = config.PREPROCESSOR_CONFIG['width_mult']
    else:
        print("\nUsing configuration from config.py...")
        feature_dim = config.PREPROCESSOR_CONFIG['output_features']
        hidden_dims = config.KAN_CONFIG['hidden_dims']
        grid = config.KAN_CONFIG['grid']
        degree = config.KAN_CONFIG['degree']
        preprocessor_type = config.PREPROCESSOR_CONFIG['preprocessor_type']
        width_mult = config.PREPROCESSOR_CONFIG['width_mult']
    
    print(f"\nModel Configuration:")
    print(f"  Preprocessor: {preprocessor_type}")
    print(f"  Width mult: {width_mult}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  KAN hidden dims: {hidden_dims}")
    print(f"  KAN grid: {grid}")
    print(f"  KAN degree: {degree}")
    print(f"  width mult: {width_mult}")
    
    model = KANImageClassifier(
        input_channels=3,
        img_size=224,
        num_classes=2,
        feature_dim=feature_dim,
        kan_hidden_dims=hidden_dims,
        kan_grid=grid,
        kan_degree=degree,
        preprocessor_type=preprocessor_type,
        width_mult=width_mult,
        preprocessor_pretrained=False,  # Don't load ImageNet weights (we load from checkpoint)
    )
    
    # Load weights - handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Standard training checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            print("[OK] Loaded weights from training checkpoint")
        elif 'quantized_state' in checkpoint:
            # Quantized model checkpoint
            model.load_state_dict(checkpoint['quantized_state'])
            print("[OK] Loaded weights from quantized checkpoint")
        elif 'state_dict' in checkpoint:
            # Alternative checkpoint format
            model.load_state_dict(checkpoint['state_dict'])
            print("[OK] Loaded weights from checkpoint")
        else:
            # Try loading the checkpoint directly as state_dict
            try:
                model.load_state_dict(checkpoint)
                print("[OK] Loaded weights directly from state_dict")
            except Exception as e:
                print(f"[FAIL] Could not load checkpoint. Available keys: {list(checkpoint.keys())}")
                raise e
    else:
        # Checkpoint is directly a state_dict
        model.load_state_dict(checkpoint)
        print("[OK] Loaded weights directly from state_dict")
    model.eval()
    
    # Original size
    torch.save(model.state_dict(), 'temp.pt')
    fp32_size = os.path.getsize('temp.pt') / (1024**2)
    os.remove('temp.pt')
    
    print(f"\nOriginal FP32 size: {fp32_size:.2f} MB")
    
    # Step 1: Quantize MobileNet weights manually
    print("\nStep 1: Quantizing MobileNet weights to INT8...")
    
    quantized_state = {}
    scales = {}
    total_params = 0
    mobilenet_params = 0
    
    for name, param in model.state_dict().items():
        if 'mobilenet' in name and ('weight' in name or 'bias' in name):
            mobilenet_params += param.numel()
            max_val = param.abs().max().item()
            scale = max_val / 127.0 if max_val > 0 else 1.0
            quantized = torch.round(param / scale).clamp(-128, 127).to(torch.int8)
            quantized_state[name] = quantized
            scales[name + '_scale'] = torch.tensor(scale)
        else:
            quantized_state[name] = param
        total_params += param.numel()
    
    for scale_name, scale_val in scales.items():
        quantized_state[scale_name] = scale_val
    
    print(f"  Quantized {mobilenet_params:,} / {total_params:,} parameters ({mobilenet_params/total_params*100:.1f}%)")
    
    # Step 2: Dequantize and load
    print("\nStep 2: Loading INT8 weights as FP32 for computation...")
    state_dict = model.state_dict()
    
    for name, param in state_dict.items():
        if name in quantized_state:
            if quantized_state[name].dtype == torch.int8:
                scale = scales[name + '_scale'].item()
                dequantized = quantized_state[name].float() * scale
                state_dict[name].copy_(dequantized)
            else:
                state_dict[name].copy_(quantized_state[name])
    
    # Check size
    torch.save(quantized_state, 'temp.pt')
    hybrid_size = os.path.getsize('temp.pt') / (1024**2)
    os.remove('temp.pt')
    
    print(f"\nSize comparison:")
    print(f"  FP32:   {fp32_size:.2f} MB")
    print(f"  Hybrid: {hybrid_size:.2f} MB")
    print(f"  Reduction: {(1-hybrid_size/fp32_size)*100:.1f}%")
    print(f"  Compression: {fp32_size/hybrid_size:.2f}x")
    
    # Step 3: Test inference
    print("\nStep 3: Testing inference...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder('data/processed/vww_subset/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    inputs, _ = next(iter(test_loader))
    
    try:
        with torch.no_grad():
            outputs = model(inputs)
        print("[OK] Hybrid inference works!")
    except Exception as e:
        print(f"[FAIL] Failed: {e}")
        sys.exit(1)
    
    # Step 4: Full evaluation
    print("\nStep 4: Evaluating hybrid model...")
    accuracy, cm, report, all_preds, all_targets, avg_latency, min_latency, max_latency, std_latency = evaluate_model(model, test_loader, test_dataset.classes)
    
    print(f"\nHybrid Accuracy: {accuracy:.2f}%")
    print(f"Avg Latency: {avg_latency:.2f} ms/sample")
    print(f"Min Latency: {min_latency:.2f} ms")
    print(f"Max Latency: {max_latency:.2f} ms")
    print(f"Std Latency: {std_latency:.2f} ms")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # Prepare report text
    report_text = [
        "=" * 60,
        "KAN Hybrid INT8 Quantization Report",
        "=" * 60,
        f"Input model:  {MODEL_PATH}",
        f"Output model: {OUTPUT_PATH}",
        "",
        "Model Configuration:",
        f"  Preprocessor: {preprocessor_type}",
        f"  Width mult: {width_mult}",
        f"  Feature dim: {feature_dim}",
        f"  KAN hidden dims: {hidden_dims}",
        f"  KAN grid: {grid}",
        f"  KAN degree: {degree}",
        "",
        "Size Comparison:",
        f"  FP32:   {fp32_size:.2f} MB",
        f"  Hybrid: {hybrid_size:.2f} MB",
        f"  Reduction: {(1-hybrid_size/fp32_size)*100:.1f}%",
        f"  Compression: {fp32_size/hybrid_size:.2f}x",
        "",
        "=" * 60,
        "Evaluation Results",
        "=" * 60,
        f"Accuracy: {accuracy:.2f}%",
        f"Total samples: {len(all_targets)}",
        "",
        "Inference Performance:",
        f"  Avg Latency: {avg_latency:.2f} ms/sample",
        f"  Min Latency: {min_latency:.2f} ms",
        f"  Max Latency: {max_latency:.2f} ms",
        f"  Std Latency: {std_latency:.2f} ms",
        "",
        "Confusion Matrix:",
        str(cm),
        "",
        "Classification Report:",
        report,
        "=" * 60,
    ]
    
    report_output = "\n".join(report_text)
    
    # Save report
    analysis_dir = exp_paths['analysis_dir']
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = analysis_dir / 'report_kan_hybrid.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_output)
    
    print(f"\n[INFO] Report saved to: {report_path}")
    
    # Step 5: Save quantized model
    print("\nStep 5: Saving...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "quantized_state": quantized_state,
        "scales": scales,
        "fp32_size_mb": fp32_size,
        "hybrid_size_mb": hybrid_size,
        "accuracy": accuracy,
        "method": "hybrid_manual_quantization",
        "config": saved_config,
    }, OUTPUT_PATH)

    print(f"[OK] Saved hybrid model to: {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("HYBRID QUANTIZATION RESULTS")
    print("=" * 60)
    print(f"Model Size: {hybrid_size:.2f} MB ({(1-hybrid_size/fp32_size)*100:.1f}% reduction)")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Inference: WORKS [OK]")
    
    if hybrid_size <= 1.2:
        print(f"\n[OK] SUCCESS: Model < 1.2 MB (IoT ready!)")
    elif hybrid_size <= 1.5:
        print(f"\n[GOOD] Model < 1.5 MB (acceptable)")
    else:
        print(f"\n[WARNING] Model > 1.5 MB (may be large for IoT)")
    
    print("=" * 60)