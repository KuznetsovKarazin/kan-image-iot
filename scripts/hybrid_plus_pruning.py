"""Hybrid quantization + Pruning"""

import torch
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
import os
import argparse
from pathlib import Path
import sys

sys.path.insert(0, 'src')
from models.kan_model import KANImageClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid INT8 + Pruning compression")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to FP32 checkpoint (.pt)")
    parser.add_argument("--prune_amount", type=float, default=0.3,
                        help="Fraction of weights to prune (0.0–0.9)")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output filename inside quantized_models/")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    MODEL_PATH = args.model_path
    PRUNE_AMOUNT = args.prune_amount
    OUTPUT_NAME = args.output_name

    print(f"Combo: Hybrid INT8 + Pruning {PRUNE_AMOUNT*100:.0f}%")
    print(f"Model: {MODEL_PATH}")

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model_info = checkpoint.get("model_info", {})
    config      = model_info.get("config", {})
    kan_config  = config.get("kan", {})

    # Build model
    model = KANImageClassifier(
        input_channels=3,
        img_size=224,
        num_classes=2,
        feature_dim=kan_config.get("feature_dim", 64),
        kan_hidden_dims=kan_config.get("hidden_dims", [32, 24, 16]),
        kan_grid=kan_config.get("grid", 5),
        kan_degree=kan_config.get("degree", 3),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Original size
    torch.save(model.state_dict(), "temp.pt")
    orig_size = os.path.getsize("temp.pt") / (1024**2)
    os.remove("temp.pt")

    print(f"\nOriginal size: {orig_size:.2f} MB")

    # --------------------------
    # Step 1: PRUNING
    # --------------------------
    print(f"\nStep 1: Pruning {PRUNE_AMOUNT*100:.0f}% ...")

    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    for module, param_name in parameters_to_prune:
        prune.l1_unstructured(module, name=param_name, amount=PRUNE_AMOUNT)
        prune.remove(module, param_name)

    # --------------------------
    # Step 2: HYBRID QUANTIZATION
    # --------------------------
    print("\nStep 2: Hybrid quantization ...")

    quantized_state = {}
    scales = {}

    for name, param in model.state_dict().items():
        if "mobilenet" in name and ("weight" in name or "bias" in name):
            max_val = param.abs().max().item()
            scale = max_val / 127.0 if max_val > 0 else 1.0

            q = torch.round(param / scale).clamp(-128, 127).to(torch.int8)

            quantized_state[name] = q
            scales[name + "_scale"] = torch.tensor(scale)

        else:
            quantized_state[name] = param

    for name, value in scales.items():
        quantized_state[name] = value

    # final size
    torch.save(quantized_state, "temp.pt")
    final_size = os.path.getsize("temp.pt") / (1024**2)
    os.remove("temp.pt")

    print(f"\nFinal size: {final_size:.2f} MB")
    print(f"Reduction: {(1 - final_size/orig_size) * 100:.1f}%")

    # --------------------------
    # Step 3: EVALUATE
    # --------------------------
    print("\nStep 3: Evaluating compressed model...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder("data/processed/vww_subset/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            _, pred = out.max(1)
            all_preds.extend(pred.numpy())
            all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = 100 * np.mean(all_preds == all_targets)

    print(f"Accuracy: {accuracy:.2f}%")
    print(classification_report(all_targets, all_preds, target_names=test_dataset.classes))

    # --------------------------
    # Step 4: SAVE
    # --------------------------
    os.makedirs("quantized_models", exist_ok=True)

    if OUTPUT_NAME is None:
        OUTPUT_NAME = f"kan_hybrid_pruned_{int(PRUNE_AMOUNT*100)}pct.pt"

    output_path = Path("quantized_models") / OUTPUT_NAME

    torch.save({
        "quantized_state": quantized_state,
        "scales": scales,
        "orig_size_mb": orig_size,
        "final_size_mb": final_size,
        "accuracy": accuracy,
        "prune_amount": PRUNE_AMOUNT,
        "method": "hybrid_int8_pruning",
        "config": config
    }, output_path)

    print(f"\n✓ Saved to: {output_path}")

    print("\n" + "="*60)
    print("COMBO RESULTS")
    print("="*60)
    print(f"Size:     {final_size:.2f} MB")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Method:   Hybrid INT8 + {PRUNE_AMOUNT*100:.0f}% pruning")
