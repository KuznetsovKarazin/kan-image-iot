import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- paths & imports из src --- #
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.kan_model import KANImageClassifier
from analyze import analyze_inference_time  


# ============ helpers ============ #

def build_model_from_checkpoint(checkpoint):
    model_info = checkpoint.get("model_info", {})
    config = model_info.get("config", {})
    kan_cfg = config.get("kan", {})
    pre_cfg = config.get("preprocessor", {})

    img_size = config.get("img_size", 224)
    feature_dim = kan_cfg.get("feature_dim", 64)
    hidden_dims = kan_cfg.get("hidden_dims", [32, 24, 16])
    grid = kan_cfg.get("grid", 5)
    degree = kan_cfg.get("degree", 3)
    conv_channels = pre_cfg.get("conv_channels", [16, 24, 40])
    use_batch_norm = pre_cfg.get("use_batch_norm", True)

    model = KANImageClassifier(
        input_channels=3,
        img_size=img_size,
        num_classes=2,
        feature_dim=feature_dim,
        kan_hidden_dims=hidden_dims,
        kan_grid=grid,
        kan_degree=degree,
        conv_channels=conv_channels,
        use_batch_norm=use_batch_norm,
    )

    kan_config = {
        "img_size": img_size,
        "feature_dim": feature_dim,
        "hidden_dims": hidden_dims,
        "grid": grid,
        "degree": degree,
    }
    return model, config, kan_config


def get_experiment_name(model_path: str) -> str:
    p = Path(model_path)
    return p.parent.parent.name


def append_log(
    log_path: Path,
    model_path: str,
    exp_name: str,
    fp32_size_mb: float,
    hybrid_size_mb: float,
    accuracy: float,
    macro_f1: float,
    person_precision: float,
    person_recall: float,
    roc_auc: float,
    single_ms_96: float,
    batch16_ms_96: float,
    batch32_ms_96: float,
    method: str,
):
    header = (
        "model_path,exp_name,fp32_size_mb,hybrid_size_mb,reduction_pct,"
        "accuracy,macro_f1,person_precision,person_recall,roc_auc,"
        "single_ms_96,batch16_ms_96,batch32_ms_96,method\n"
    )
    reduction_pct = 100.0 * (1.0 - hybrid_size_mb / fp32_size_mb)

    line = (
        f"{model_path},{exp_name},{fp32_size_mb:.4f},{hybrid_size_mb:.4f},"
        f"{reduction_pct:.2f},{accuracy:.2f},{macro_f1:.4f},"
        f"{person_precision:.4f},{person_recall:.4f},{roc_auc:.4f},"
        f"{single_ms_96:.4f},{batch16_ms_96:.4f},{batch32_ms_96:.4f},{method}\n"
    )

    if not log_path.exists():
        with log_path.open("w", encoding="utf-8", newline="") as f:
            f.write(header)

    with log_path.open("a", encoding="utf-8", newline="") as f:
        f.write(line)


# ============ main ============ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid INT8 quantization (MobileNet INT8 storage + KAN FP32 inference)"
    )
    parser.add_argument("model_path", type=str, help="Path to FP32 checkpoint (.pt)")
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    exp_name = get_experiment_name(MODEL_PATH)

    # --- load FP32 checkpoint ---
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model, config, kan_config = build_model_from_checkpoint(checkpoint)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # --- FP32 size (state_dict) ---
    torch.save(model.state_dict(), "temp_fp32.pt")
    fp32_size = os.path.getsize("temp_fp32.pt") / (1024**2)
    os.remove("temp_fp32.pt")
    print(f"Original FP32 size: {fp32_size:.2f} MB")

    # ========== Step 1: Manual hybrid quantization (storage-only) ========== #
    print("\nStep 1: Quantizing MobileNet weights to INT8 (storage-only)...")

    quantized_state = {}
    scales = {}
    total_params = 0
    mobilenet_params = 0

    for name, param in model.state_dict().items():
        total_params += param.numel()
        if "mobilenet" in name and ("weight" in name or "bias" in name):
            mobilenet_params += param.numel()
            max_val = param.abs().max().item()
            scale = max_val / 127.0 if max_val > 0 else 1.0
            q = torch.round(param / scale).clamp(-128, 127).to(torch.int8)
            quantized_state[name] = q
            scales[name + "_scale"] = torch.tensor(scale)
        else:
            quantized_state[name] = param

    for sname, sval in scales.items():
        quantized_state[sname] = sval

    print(
        f"  Quantized {mobilenet_params:,} / {total_params:,} "
        f"parameters ({mobilenet_params / total_params * 100:.1f}%)"
    )

    # ========== Step 2: Dequantize back to FP32 for actual inference ========== #
    print("\nStep 2: Loading INT8 weights as FP32 for computation...")

    sd = model.state_dict()
    for name, param in sd.items():
        if name in quantized_state:
            qparam = quantized_state[name]
            if qparam.dtype == torch.int8:
                scale = scales[name + "_scale"].item()
                deq = qparam.float() * scale
                param.copy_(deq)
            else:
                param.copy_(qparam)

    # --- hybrid storage size (INT8 state_dict) ---
    torch.save(quantized_state, "temp_hybrid.pt")
    hybrid_size = os.path.getsize("temp_hybrid.pt") / (1024**2)
    os.remove("temp_hybrid.pt")

    reduction_pct = (1.0 - hybrid_size / fp32_size) * 100.0
    compression_ratio = fp32_size / hybrid_size

    print("\nSize comparison:")
    print(f"  FP32:   {fp32_size:.2f} MB")
    print(f"  Hybrid: {hybrid_size:.2f} MB")
    print(f"  Reduction: {reduction_pct:.1f}%")
    print(f"  Compression: {compression_ratio:.2f}x")

    # ========== Step 3: Dataset + basic smoke test ========== #
    print("\nStep 3: Testing inference & preparing test data...")

    img_size = config.get("img_size", 224)
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_dataset = datasets.ImageFolder(
        "data/processed/vww_subset/test", transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # smoke test
    inputs, _ = next(iter(test_loader))
    with torch.no_grad():
        _ = model(inputs)
    print("✓ Hybrid inference works on a test batch.")

    # ========== Step 4: Full evaluation (metrics + confusion + ROC AUC) ========== #
    print("\nStep 4: Evaluating hybrid model on test set...")

    all_preds = []
    all_targets = []
    all_scores = []

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1]  # score for "person"
            _, pred = out.max(1)
            all_preds.extend(pred.numpy())
            all_targets.extend(y.numpy())
            all_scores.extend(probs.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)

    accuracy = 100.0 * np.mean(all_preds == all_targets)
    print(f"Hybrid Accuracy: {accuracy:.2f}%\n")

    class_names = test_dataset.classes
    report_text = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
    )
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
    )

    print("Classification Report:")
    print(report_text)

    macro_f1 = report_dict["macro avg"]["f1-score"]
    person_key = "person" if "person" in class_names else class_names[-1]
    person_precision = report_dict[person_key]["precision"]
    person_recall = report_dict[person_key]["recall"]

    try:
        roc_auc = roc_auc_score(all_targets, all_scores)
    except Exception:
        roc_auc = float("nan")

    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix (rows = true, cols = pred):")
    print(cm)

    # ========== Step 5: Reuse analyze_inference_time (same methodology as FP32) ========== #
    print("\nStep 5: Measuring inference time with analyze_inference_time (same as FP32)...")

    device = torch.device("cpu")
    inference_results = analyze_inference_time(
        model,
        image_sizes=[96, img_size, 224],
        batch_sizes=[1, 4, 16, 32],
        device=device,
    )

    # 0: img=96, bs=1; 1: img=96, bs=4; 2: img=96, bs=16; 3: img=96, bs=32
    single_ms_96 = inference_results[0]["per_image_time_ms"]
    batch16_ms_96 = inference_results[2]["per_image_time_ms"]
    batch32_ms_96 = inference_results[3]["per_image_time_ms"]

    # ========== Step 6: Save everything by folders ========== #
    print("\nStep 6: Saving model, reports and logs...")

    base_dir = Path("quantized_models")
    exp_dir = base_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    model_path_out = exp_dir / "hybrid_int8.pt"
    cm_path = exp_dir / "hybrid_int8.confusion.csv"
    report_path = exp_dir / "hybrid_int8.report.txt"
    json_path = exp_dir / "hybrid_int8.analysis.json"
    md_path = exp_dir / "hybrid_int8.analysis.md"
    infer_json_path = exp_dir / "hybrid_int8.inference_time.json"

    torch.save(
        {
            "quantized_state": quantized_state,
            "scales": scales,
            "fp32_size_mb": fp32_size,
            "hybrid_size_mb": hybrid_size,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "person_precision": person_precision,
            "person_recall": person_recall,
            "roc_auc": roc_auc,
            "inference_results": inference_results,
            "method": "hybrid_manual_int8",
            "config": config,
            "kan_config": kan_config,
            "source_model_path": MODEL_PATH,
            "confusion_matrix": cm,
            "classes": class_names,
        },
        model_path_out,
    )

    # 2) confusion matrix CSV
    np.savetxt(cm_path, cm.astype(int), fmt="%d", delimiter=",")

    # 3) classification report
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    # 4) inference_time JSON 
    with infer_json_path.open("w", encoding="utf-8") as f:
        json.dump(inference_results, f, indent=2)

    # 5) JSON summary
    analysis = {
        "experiment_name": exp_name,
        "source_model_path": MODEL_PATH,
        "quantization": {
            "method": "hybrid_manual_int8",
            "fp32_size_mb": fp32_size,
            "hybrid_size_mb": hybrid_size,
            "reduction_pct": reduction_pct,
            "compression_ratio": compression_ratio,
        },
        "kan_config": kan_config,
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "person_precision": person_precision,
            "person_recall": person_recall,
            "roc_auc": roc_auc,
        },
        "latency_summary_96": {
            "single_bs1_ms": single_ms_96,
            "batch16_ms": batch16_ms_96,
            "batch32_ms": batch32_ms_96,
        },
        "inference_results": inference_results,
        "confusion_matrix": cm.tolist(),
        "classes": class_names,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    # 6) Markdown summary
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Hybrid INT8 Model Analysis\n\n")
        f.write(f"**Experiment:** `{exp_name}`  \n")
        f.write(f"**Source FP32 checkpoint:** `{MODEL_PATH}`  \n\n")

        f.write("## Quantization\n")
        f.write(
            "- Method: hybrid_manual_int8 (MobileNet INT8 storage, KAN FP32 inference)\n"
        )
        f.write(f"- FP32 size: {fp32_size:.2f} MB\n")
        f.write(f"- Hybrid size: {hybrid_size:.2f} MB\n")
        f.write(f"- Reduction: {reduction_pct:.1f}%  \n")
        f.write(f"- Compression ratio: {compression_ratio:.2f}x\n\n")

        f.write("## KAN Configuration\n")
        f.write(f"- img_size: {kan_config['img_size']}\n")
        f.write(f"- feature_dim: {kan_config['feature_dim']}\n")
        f.write(f"- hidden_dims: {kan_config['hidden_dims']}\n")
        f.write(f"- grid: {kan_config['grid']}\n")
        f.write(f"- degree: {kan_config['degree']}\n\n")

        f.write("## Classification Metrics (test set)\n")
        f.write(f"- Accuracy: {accuracy:.2f}%\n")
        f.write(f"- Macro F1: {macro_f1:.4f}\n")
        f.write(f"- Person precision: {person_precision:.4f}\n")
        f.write(f"- Person recall: {person_recall:.4f}\n")
        f.write(f"- ROC AUC (person vs no_person): {roc_auc:.4f}\n\n")

        f.write("## Inference Performance (96×96, CPU)\n")
        f.write(
            f"- Single image (bs=1): {single_ms_96:.2f} ms\n"
            f"- Batch inference (16 images): {batch16_ms_96:.2f} ms per image\n"
            f"- Batch inference (32 images): {batch32_ms_96:.2f} ms per image\n\n"
        )

        f.write("## Confusion Matrix\n\n")
        f.write("rows = true class, columns = predicted class\n\n")
        f.write("| true\\pred | " + " | ".join(class_names) + " |\n")
        f.write("|-----------|" + "|".join(["-----------"] * len(class_names)) + "|\n")
        for i, row in enumerate(cm):
            f.write(
                f"| {class_names[i]} | "
                + " | ".join(str(int(v)) for v in row)
                + " |\n"
            )

        f.write("\n\n_Classification report saved in `hybrid_int8.report.txt`._\n")

    print(f"✓ Saved hybrid model to: {model_path_out}")
    print(f"✓ Confusion matrix saved to: {cm_path}")
    print(f"✓ Classification report saved to: {report_path}")
    print(f"✓ Inference-time JSON saved to: {infer_json_path}")
    print(f"✓ JSON analysis saved to:        {json_path}")
    print(f"✓ Markdown analysis saved to:    {md_path}")

    # 7) general CSV log (for all models, at the root quantized_models)
    log_path = exp_dir / "quantization_log_hybrid_int8.csv"
    append_log(
        log_path,
        MODEL_PATH,
        exp_name,
        fp32_size,
        hybrid_size,
        accuracy,
        macro_f1,
        person_precision,
        person_recall,
        roc_auc,
        single_ms_96,
        batch16_ms_96,
        batch32_ms_96,
        "hybrid_manual_int8",
    )

    print(f"✓ Logged results to: {log_path}")

    print("\n" + "=" * 60)
    print("HYBRID QUANTIZATION RESULTS")
    print("=" * 60)
    print(
        f"Model Size: {hybrid_size:.2f} MB "
        f"({reduction_pct:.1f}% reduction)"
    )
    print(f"Accuracy:            {accuracy:.2f}%")
    print(f"Macro F1:            {macro_f1:.4f}")
    print(f"Person P/R:          {person_precision:.4f} / {person_recall:.4f}")
    print(f"Single (96×96, bs=1):   {single_ms_96:.2f} ms")
    print(f"Batch16 (96×96):        {batch16_ms_96:.2f} ms/image")
    print(f"Batch32 (96×96):        {batch32_ms_96:.2f} ms/image")
    print("Status:              OK ✓")
