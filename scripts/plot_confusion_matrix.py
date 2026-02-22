"""
plot_confusion_matrix.py
========================
Generate confusion matrix figures (absolute counts + normalized %)
from raw TP/TN/FP/FN values.

Reproduces the same style as src/utils/visualization.plot_confusion_matrix().

Author: Daniele Faggi
Date: February 2026

Usage
-----
    python scripts/plot_confusion_matrix.py --tn 120 --fp 8 --fn 5 --tp 110
    python scripts/plot_confusion_matrix.py --tn 120 --fp 8 --fn 5 --tp 110 \\
        --output_dir results/figures --title "KAN 32-16 WM0.5 TFLite"
    python scripts/plot_confusion_matrix.py --tn 120 --fp 8 --fn 5 --tp 110 \\
        --neg_label background --pos_label person

Arguments
---------
    --tn            True Negatives  (required)
    --fp            False Positives (required)
    --fn            False Negatives (required)
    --tp            True Positives  (required)
    --neg_label     Label for the negative class  (default: "no_person")
    --pos_label     Label for the positive class  (default: "person")
    --title         Optional figure title prefix
    --output_dir    Directory where PNGs are saved (default: current dir)
    --dpi           Output DPI (default: 300)
    --no_show       Do not open an interactive window (default: always headless)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


# ─────────────────────────────────────────────
# Core plotting function (mirrors visualization.py)
# ─────────────────────────────────────────────
def _plot_cm(cm: np.ndarray,
             classes: list[str],
             normalize: bool,
             title_prefix: str,
             output_dir: Path,
             dpi: int) -> Path:
    """Plot and save a single confusion matrix (absolute or normalized)."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = cm.astype('float') / np.where(row_sums == 0, 1, row_sums)
        fmt = '.2%'
        suffix = '_normalized'
        sub_title = 'Normalized Confusion Matrix (%)'
        # Convert to percentage strings manually so seaborn shows e.g. "92.50%"
        annot_data = cm_display
    else:
        cm_display = cm
        fmt = 'd'
        suffix = ''
        sub_title = 'Confusion Matrix (counts)'
        annot_data = cm

    full_title = f'{title_prefix}\n{sub_title}' if title_prefix else sub_title

    fig, ax = plt.subplots(figsize=(10, 8))

    if _HAS_SNS:
        if normalize:
            # Format as percentage strings for readability
            annot_labels = np.array(
                [[f'{v:.2%}' for v in row] for row in cm_display]
            )
            sns.heatmap(
                cm_display, annot=annot_labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, linecolor='lightgrey', ax=ax,
                annot_kws={'size': 18, 'weight': 'bold'},
            )
        else:
            sns.heatmap(
                cm_display, annot=True, fmt=fmt, cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, linecolor='lightgrey', ax=ax,
                annot_kws={'size': 18, 'weight': 'bold'},
            )
    else:
        # Fallback: plain matplotlib imshow
        im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
        thresh = cm_display.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = f'{cm_display[i, j]:.2%}' if normalize else str(int(cm[i, j]))
                ax.text(j, i, val, ha='center', va='center',
                        fontsize=18, fontweight='bold',
                        color='white' if cm_display[i, j] > thresh else 'black')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)

    ax.set_title(full_title, fontsize=14, pad=14)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()

    filename = f'confusion_matrix{suffix}.png'
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix figures from TP/TN/FP/FN values'
    )
    parser.add_argument('--tn', type=int, required=True, help='True Negatives')
    parser.add_argument('--fp', type=int, required=True, help='False Positives')
    parser.add_argument('--fn', type=int, required=True, help='False Negatives')
    parser.add_argument('--tp', type=int, required=True, help='True Positives')
    parser.add_argument('--neg_label', type=str, default='no_person',
                        help='Label for negative class (default: no_person)')
    parser.add_argument('--pos_label', type=str, default='person',
                        help='Label for positive class (default: person)')
    parser.add_argument('--title', type=str, default='',
                        help='Optional title prefix for the figure')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save PNG files (default: current dir)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (default: 300)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build 2×2 confusion matrix:
    #            Predicted no_person  |  Predicted person
    # True no_person       TN         |       FP
    # True person          FN         |       TP
    cm = np.array([
        [args.tn, args.fp],
        [args.fn, args.tp],
    ])

    classes = [args.neg_label, args.pos_label]
    title = args.title

    total = cm.sum()
    accuracy = (args.tp + args.tn) / total * 100 if total > 0 else 0.0
    precision = args.tp / (args.tp + args.fp) * 100 if (args.tp + args.fp) > 0 else 0.0
    recall    = args.tp / (args.tp + args.fn) * 100 if (args.tp + args.fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"  Confusion Matrix Summary")
    print(f"{'='*50}")
    print(f"  {'TN':>6}  {'FP':>6}  |  {'FN':>6}  {'TP':>6}")
    print(f"  {args.tn:>6}  {args.fp:>6}  |  {args.fn:>6}  {args.tp:>6}")
    print(f"{'─'*50}")
    print(f"  Total samples : {total}")
    print(f"  Accuracy      : {accuracy:.2f}%")
    print(f"  Precision     : {precision:.2f}%")
    print(f"  Recall        : {recall:.2f}%")
    print(f"  F1 Score      : {f1:.2f}%")
    print(f"{'='*50}\n")

    # Generate both figures
    p_abs  = _plot_cm(cm, classes, normalize=False, title_prefix=title,
                      output_dir=output_dir, dpi=args.dpi)
    p_norm = _plot_cm(cm, classes, normalize=True,  title_prefix=title,
                      output_dir=output_dir, dpi=args.dpi)

    print(f"  [OK] Absolute  → {p_abs}")
    print(f"  [OK] Normalized→ {p_norm}\n")


if __name__ == '__main__':
    main()
