"""Pruning with actual compression"""
import torch
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
import sys
import os
sys.path.insert(0, 'src')

from models.kan_model import KANImageClassifier

def compress_sparse_model(model):
    """Remove pruned weights to actually reduce size"""
    compressed_state = {}
    
    for name, param in model.state_dict().items():
        if 'weight' in name:
            # Count zeros
            zeros = (param == 0).sum().item()
            total = param.numel()
            sparsity = zeros / total
            
            if sparsity > 0.1:  # If >10% sparse, compress
                # Store only non-zero values
                mask = param != 0
                values = param[mask]
                indices = torch.nonzero(mask, as_tuple=False)
                
                compressed_state[name] = {
                    'values': values,
                    'indices': indices,
                    'shape': param.shape,
                    'sparse': True,
                    'sparsity': sparsity
                }
            else:
                compressed_state[name] = param
        else:
            compressed_state[name] = param
    
    return compressed_state

if __name__ == '__main__':
    MODEL_PATH = sys.argv[1]
    PRUNE_AMOUNT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1  # Start small!
    
    print(f"Structured pruning {PRUNE_AMOUNT*100:.0f}% of weights...")
    
    # Load model
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
    
    # Original
    torch.save(model.state_dict(), 'temp.pt')
    orig_size = os.path.getsize('temp.pt') / (1024**2)
    os.remove('temp.pt')
    
    # Prune only MobileNet (safer)
    print("Pruning MobileNet layers only...")
    pruned_count = 0
    for name, module in model.named_modules():
        if 'mobilenet' in name and isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=PRUNE_AMOUNT)
            prune.remove(module, 'weight')
            pruned_count += 1
    
    print(f"Pruned {pruned_count} Conv2d layers")
    
    # Check sparsity
    total_params = 0
    zero_params = 0
    for name, param in model.state_dict().items():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    actual_sparsity = zero_params / total_params * 100
    print(f"Actual sparsity: {actual_sparsity:.1f}%")
    
    # Regular save (no compression)
    torch.save(model.state_dict(), 'temp.pt')
    regular_size = os.path.getsize('temp.pt') / (1024**2)
    os.remove('temp.pt')
    
    print(f"\nSize (regular save): {orig_size:.2f} MB → {regular_size:.2f} MB (no compression)")
    
    # Evaluate first
    print("\nEvaluating pruned model...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder('data/processed/vww_subset/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
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
    
    print(f"Accuracy: {accuracy:.2f}%")
    print(classification_report(all_targets, all_preds, target_names=test_dataset.classes))
    
    # Save
    output_name = f'kan_pruned_{int(PRUNE_AMOUNT*100)}pct.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'prune_amount': PRUNE_AMOUNT,
        'actual_sparsity': actual_sparsity,
        'orig_size_mb': orig_size,
        'size_mb': regular_size,
        'accuracy': accuracy,
        'config': config
    }, f'quantized_models/{output_name}')
    
    print(f"\n✓ Saved to: quantized_models/{output_name}")
    
    print("\n" + "="*60)
    print("PRUNING RESULTS")
    print("="*60)
    print(f"Target pruning: {PRUNE_AMOUNT*100:.0f}%")
    print(f"Actual sparsity: {actual_sparsity:.1f}%")
    print(f"Size: {regular_size:.2f} MB (zeros not compressed)")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if accuracy < 85:
        print("\n⚠ WARNING: Accuracy too low!")
        print("  Pruning without fine-tuning degrades performance")
        print("  Recommendation: Use smaller prune amount or fine-tune")