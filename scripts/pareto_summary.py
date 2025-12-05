"""
Generate Pareto Frontier Summary (FP32 vs Hybrid INT8 for all experiments)

Script:
    - goes through quantized_models/<exp_name>/,
    - reads hybrid_int8.analysis.json,
    - gets: 
        - size FP32 and Hybrid, 
        - accuracy Hybrid,
    - tries to find the accuracy of FP32 (if there is a model analysis),
    - builds a chart "Size (MB) vs Accuracy (%)" with categories: 
        - Baseline (FP32), 
        - Quantization (Hybrid INT8), 
        - optional manual points (Static INT8, Pruning).
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(".")
QUANT_DIR = BASE_DIR / "quantized_models"

def short_experiment_name(exp_name: str) -> str:
    """ 
    Makes a compact name from a long experiment name. 
    Example: 
    'kan_64_32-24-16_grid5_deg3_img224_bs64_lr0.002_wd1e-05_do0.05' 
    -> 'KAN-64-32-24-16' 
    If the format is different, we return the original name. 
    """
    if exp_name.startswith("kan_"):
        parts = exp_name.split("_")
        # ['kan', '64', '32-24-16', 'grid5', ...]
        if len(parts) >= 3:
            return f"KAN-{parts[1]}-{parts[2]}"
    return exp_name

def load_json(path: Path):

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load JSON from {path}: {e}")
        return None


def load_fp32_accuracy(exp_name: str) -> float | None:

    candidates = [
        BASE_DIR / "experiment_data" / exp_name / "analysis" / "model_analysis.json",
        BASE_DIR / "experiment_data" / exp_name / "model_analysis.json",
    ]

    for path in candidates:
        if path.exists():
            data = load_json(path)
            if not data:
                continue


            for key_path in [
                ("metrics", "test_accuracy"),
                ("metrics", "accuracy"),
                ("test_metrics", "accuracy"),
            ]:
                node = data
                ok = True
                for k in key_path:
                    if isinstance(node, dict) and k in node:
                        node = node[k]
                    else:
                        ok = False
                        break
                if ok and isinstance(node, (int, float)):
                    return float(node)

    print(f"[WARN] FP32 accuracy for experiment '{exp_name}' not found.")
    return None


def collect_models():

    rows = []

    if not QUANT_DIR.exists():
        print(f"[ERROR] Directory {QUANT_DIR} does not exist.")
        return pd.DataFrame(columns=["Model", "Size (MB)", "Accuracy (%)",
                                     "Category", "Status", "Path"])

    for exp_dir in sorted(QUANT_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        analysis_path = exp_dir / "hybrid_int8.analysis.json"
        if not analysis_path.exists():

            continue

        analysis = load_json(analysis_path)
        if not analysis:
            continue

        exp_name = analysis.get("experiment_name", exp_dir.name)
        quant_info = analysis.get("quantization", {})
        metrics = analysis.get("metrics", {})

        fp32_size = quant_info.get("fp32_size_mb", None)
        hybrid_size = quant_info.get("hybrid_size_mb", None)
        hybrid_acc = metrics.get("accuracy", None)

        if fp32_size is None or hybrid_size is None or hybrid_acc is None:
            print(f"[WARN] Missing data in {analysis_path}, skipping.")
            continue

        short_name = short_experiment_name(exp_name)

        # 1) FP32 baseline
        fp32_acc = load_fp32_accuracy(exp_name)
        if fp32_acc is not None:
            rows.append(
                {
                    "Model": f"FP32 {short_name}",
                    "Size (MB)": fp32_size,
                    "Accuracy (%)": fp32_acc,
                    "Category": "Baseline",
                    "Status": "Baseline (reference)",
                    "Path": f"experiment_data/{exp_name}/models/...",
                }
            )
        else:
            rows.append(
                {
                    "Model": f"FP32 {short_name}",
                    "Size (MB)": fp32_size,
                    "Accuracy (%)": hybrid_acc,  # грубая оценка
                    "Category": "Baseline",
                    "Status": "Baseline (approx)",
                    "Path": f"experiment_data/{exp_name}/models/...",
                }
            )

        # 2) Hybrid INT8
        status = "Production ready" if hybrid_acc >= 85.0 else "Degraded but usable"
        rows.append(
            {
                "Model": f"INT8 {short_name}",
                "Size (MB)": hybrid_size,
                "Accuracy (%)": hybrid_acc,
                "Category": "Quantization",
                "Status": status,
                "Path": f"quantized_models/{exp_name}/hybrid_int8.pt",
            }
        )


    rows.append(
        {
            "Model": "Hybrid + 30% Pruning (Supervisor)",
            "Size (MB)": 1.40,
            "Accuracy (%)": 56.55,
            "Category": "Pruning",
            "Status": "Too degraded (not recommended)",
            "Path": "quantized_models/.../kan_hybrid_pruned_30_supervisor.pt",
        }
    )
    # Example: Static INT8 (storage-only, no inference)
    rows.append(
        {
            "Model": "Static INT8 (broken inference)",
            "Size (MB)": 1.53,
            "Accuracy (%)": 88.35, 
            "Category": "Quantization",
            "Status": "Storage only (inference broken)",
            "Path": "quantized_models/.../kan_static_int8.pt",
        }
    )

    df = pd.DataFrame(rows)
    return df


def plot_pareto(df: pd.DataFrame, out_path: Path):
    print("=" * 80)
    print("PARETO FRONTIER - SIZE vs ACCURACY (FP32 vs Hybrid INT8)")
    print("=" * 80)
    print(df[["Model", "Size (MB)", "Accuracy (%)", "Category", "Status", "Path"]].to_string(index=False))
    print("=" * 80)

    plt.figure(figsize=(10, 6))

    colors_map = {
        "Baseline": "blue",
        "Quantization": "green",
        "Pruning": "orange",
    }
    markers_map = {
        "Baseline": "o",
        "Quantization": "s",
        "Pruning": "D",
    }

    for _, row in df.iterrows():
        x = row["Size (MB)"]
        y = row["Accuracy (%)"]
        cat = row["Category"]
        label = row["Model"]

        plt.scatter(
            x,
            y,
            s=200,
            c=colors_map.get(cat, "gray"),
            marker=markers_map.get(cat, "o"),
            alpha=0.8,
            edgecolors="k",
            linewidths=0.7,
        )

        label = row["Model"]
        label = label.replace(" ", "\n")  

        plt.annotate(
            label,
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=8,
            ha="center",
        )


    plt.xlabel("Model Size (MB)", fontsize=13)
    plt.ylabel("Accuracy (%)", fontsize=13)
    plt.title("Size vs Accuracy\nFP32 Baselines vs Hybrid INT8", fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3)


    plt.axhline(y=85, color="gray", linestyle="--", alpha=0.4, label="Min target (85%)")
    plt.axvline(x=2.0, color="gray", linestyle="--", alpha=0.4, label="IoT limit (2 MB)")


    handles = []
    labels = []
    for cat, color in colors_map.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=markers_map[cat],
                color="w",
                markerfacecolor=color,
                markersize=9,
                markeredgecolor="k",
                label=cat,
            )
        )
        labels.append(cat)
    plt.legend(handles, labels, title="Category")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"\n✓ Plot saved to: {out_path}")


if __name__ == "__main__":
    df = collect_models()
    if df.empty:
        print("[ERROR] No models found. Check quantized_models/ and JSON files.")
    else:
        out_path = QUANT_DIR / "pareto_frontier_all_models.png"
        plot_pareto(df, out_path)
