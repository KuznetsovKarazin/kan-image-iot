"""Generate Pareto Frontier Summary (Supervisor + Student + Pruning)"""

import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("quantized_models", exist_ok=True)


data = {
    "Model": [
        "FP32 Baseline (Supervisor)",
        "FP32 Baseline (Student)",
        "Static INT8 (broken)",
        "Hybrid INT8 (Supervisor)",
        "Hybrid INT8 (Student)",
        "Hybrid + 30% Pruning (Supervisor)",
    ],
    "Size (MB)": [
        4.02,   # FP32 supervisor
        4.02,   # FP32 student
        1.53,   # static INT8
        1.40,   # hybrid supervisor
        1.40,   # hybrid student
        1.40,   # hybrid + pruning supervisor
    ],
    "Accuracy (%)": [
        88.35,  # FP32 supervisor (test)
        86.95,  # FP32 student (test)
        88.35,  # static INT8 - nominally, but the inference is not working
        86.35,  # hybrid supervisor (test)
        83.75,  # hybrid student (test)
        56.55,  # hybrid + 30% pruning (test)
    ],
    "Inference": [
        "Works",
        "Works",
        "Broken (KAN)",
        "Works",
        "Works",
        "Works (but degraded)",
    ],
    "Status": [
        "Baseline (reference)",
        "Baseline (student)",
        "Storage only",
        "Production ready",
        "Production ready (student)",
        "Too degraded (not recommended)",
    ],
    "Category": [
        "Baseline",
        "Baseline",
        "Quantization",
        "Quantization",
        "Quantization",
        "Pruning",
    ],
}

df = pd.DataFrame(data)

print("=" * 80)
print("PARETO FRONTIER - SIZE vs ACCURACY (Supervisor + Student + Pruning)")
print("=" * 80)
print(df.to_string(index=False))
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
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    plt.annotate(
        label,
        (x, y),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=9,
    )

plt.xlabel("Model Size (MB)", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.title("Pareto Frontier - Size vs Accuracy\nSupervisor vs Student vs Pruning", fontsize=16, fontweight="bold")
plt.grid(True, alpha=0.3)

plt.xlim(1.0, 4.5)
plt.ylim(55, 90)

plt.axhline(y=85, color="gray", linestyle="--", alpha=0.5, label="Min target (85%)")
plt.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, label="IoT limit (2 MB)")

handles = []
labels = []
for cat, color in colors_map.items():
    handles.append(plt.Line2D([0], [0], marker=markers_map[cat], color="w",
                              markerfacecolor=color, markersize=10,
                              markeredgecolor="k", label=cat))
    labels.append(cat)
plt.legend(handles, labels, title="Category")

plt.tight_layout()
out_path = "quantized_models/pareto_frontier_supervisor_student.png"
plt.savefig(out_path, dpi=300)
print(f"\n✓ Plot saved to: {out_path}")
