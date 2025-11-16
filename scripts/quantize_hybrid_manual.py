
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
import sys
import os
sys.path.insert(0, 'src')

from models.kan_model import KANImageClassifier

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to FP32 checkpoint")
    parser.add_argument("--output_name", type=str, default="kan_hybrid_manual.pt",
                        help="Output filename in quantized_models/")
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    OUTPUT_NAME = args.output_name
    
    # Load original model
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model_info = checkpoint.get('model_info', {})
    config = model_info.get('config', {})
    kan_config = config.get('kan', {})
    
    model = KANImageClassifier(
        input_channels=3,
        img_size=224,
        num_classes=2,
        feature_dim=64,
        kan_hidden_dims=[32, 24, 16],
        kan_grid=5,
        kan_degree=3,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Original size
    torch.save(model.state_dict(), 'temp.pt')
    fp32_size = os.path.getsize('temp.pt') / (1024**2)
    os.remove('temp.pt')
    
    print(f"Original FP32: {fp32_size:.2f} MB")
    
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
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)  # 0 for Windows
    
    inputs, _ = next(iter(test_loader))
    
    try:
        with torch.no_grad():
            outputs = model(inputs)
        print("✓ Hybrid inference works!")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)
    
    # Step 4: Full evaluation
    print("\nStep 4: Evaluating hybrid model...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.numpy())
            all_targets.extend(targets.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = 100 * np.mean(all_preds == all_targets)
    
    print(f"Hybrid Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=test_dataset.classes))
    
    # Save
    # Save
    print("\nStep 5: Saving...")
    os.makedirs("quantized_models", exist_ok=True)
    output_path = os.path.join("quantized_models", OUTPUT_NAME)

    torch.save({
        "quantized_state": quantized_state,      # INT8 + FP32
        "scales": scales,              
        "fp32_size_mb": fp32_size,
        "hybrid_size_mb": hybrid_size,
        "accuracy": accuracy,
        "method": "hybrid_manual_quantization",
        "config": config,                        
    }, output_path)

    print(f"✓ Saved hybrid model to: {output_path}")
 
    
    print("\n" + "="*60)
    print("HYBRID QUANTIZATION RESULTS")
    print("="*60)
    print(f"Model Size: {hybrid_size:.2f} MB ({(1-hybrid_size/fp32_size)*100:.1f}% reduction)")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Inference: WORKS ✓")
    
    if hybrid_size <= 1.2:
        print(f"\n✓ SUCCESS: Model < 1.2 MB (IoT ready!)")
    elif hybrid_size <= 1.5:
        print(f"\n⚠ GOOD: Model < 1.5 MB (acceptable)")