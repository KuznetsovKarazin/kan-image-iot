"""
plot_training_curves.py
=======================
Generate a 3-panel training curves figure (PNG) from a model_analysis.json file.

Author: Daniele Faggi
Date: February 2026

Usage
-----
    python scripts/plot_training_curves.py <json_path> <output_png>

    python scripts/plot_training_curves.py \\
        experiment_data/kan_16_4_.../analysis/model_analysis.json \\
        figures/training_curves.png

Panels
------
  1. Loss During Training  — train_loss / val_loss   vs epoch
  2. Accuracy During Training — train_acc / val_acc (%) vs epoch
  3. Learning Rate Schedule   — lr vs epoch
"""

import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe on servers/Windows)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Plot training curves from a model_analysis.json file.'
    )
    p.add_argument('json_path', type=str,
                   help='Path to model_analysis.json')
    p.add_argument('output_png', type=str,
                   help='Output PNG file path')
    p.add_argument('--dpi', type=int, default=300,
                   help='Figure DPI (default: 300 -> 4471x2966 px output)')
    p.add_argument('--title', type=str, default=None,
                   help='Optional figure suptitle (auto-inferred from directory name if omitted)')
    p.add_argument('--export_csv', type=str, default=None, metavar='CSV_PATH',
                   help='If specified, also export training data to this CSV file. '
                        'If the value is "auto", the CSV is saved next to the PNG '
                        'with the same base name.')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def infer_title(json_path: Path) -> str:
    """Build a readable title from the experiment directory name."""
    # .../kan_16_4_grid5_.../analysis/model_analysis.json
    # pick the experiment dir (two levels up from the file)
    exp_dir = json_path.parent.parent.name
    # Shorten: kan_16_4_grid5_deg3_img224_bs256_lr0.003_wd1e-05_do0.05_mobilenetv3_small_wm1.0
    # -> KAN 16-4 | MobileNetV3-Small WM 1.0
    import re
    m = re.match(r'kan_(.+?)_grid(\d+)_deg(\d+)_img\d+_bs\d+_lr[\d.e+-]+_wd[\d.e+-]+_do[\d.]+_(\w+)(?:_wm([\d.]+))?', exp_dir)
    if m:
        dims = m.group(1).replace('_', '-')
        grid = m.group(2)
        deg  = m.group(3)
        prep = m.group(4).replace('_', ' ').title()
        wm   = f' WM {m.group(5)}' if m.group(5) else ''
        return f'KAN {dims} | Grid {grid} Deg {deg} | {prep}{wm}'
    return exp_dir


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    json_path = Path(args.json_path)
    output_png = Path(args.output_png)

    if not json_path.exists():
        print(f'[ERROR] File not found: {json_path}')
        sys.exit(1)

    output_png.parent.mkdir(parents=True, exist_ok=True)

    # ── Load JSON ────────────────────────────────────────────────
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cfg = data.get('model_configuration', data)
    history = cfg.get('history', cfg)

    epochs     = history.get('epochs',     list(range(1, len(history.get('train_loss', [])) + 1)))
    train_loss = history.get('train_loss', [])
    val_loss   = history.get('val_loss',   [])
    train_acc  = history.get('train_acc',  [])
    val_acc    = history.get('val_acc',    [])
    lr         = history.get('lr',         [])

    best_epoch = cfg.get('best_epoch', None)
    best_acc   = cfg.get('best_accuracy', None)

    n = len(epochs)
    if n == 0:
        print('[ERROR] No epoch data found in JSON.')
        sys.exit(1)

    # ── Figure setup ─────────────────────────────────────────────
    fig_title = args.title if args.title else infer_title(json_path)

    TRAIN_CLR  = '#1f77b4'   # matplotlib blue
    VAL_CLR    = '#d62728'   # matplotlib red
    LR_CLR     = '#2ca02c'   # matplotlib green
    BEST_CLR   = '#ff7f0e'   # matplotlib orange

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor':   'white',
        'axes.edgecolor':   '#cccccc',
        'axes.labelcolor':  '#333333',
        'xtick.color':      '#555555',
        'ytick.color':      '#555555',
        'text.color':       '#222222',
        'legend.facecolor': 'white',
        'legend.edgecolor': '#cccccc',
        'grid.color':       '#e0e0e0',
        'grid.linestyle':   '--',
        'grid.alpha':       0.8,
        'font.family':      'DejaVu Sans',
        'font.size':        10,
    })

    has_loss = bool(train_loss or val_loss)
    has_acc  = bool(train_acc  or val_acc)
    has_lr   = bool(lr)

    # Layout: Loss (row0,col0) | Accuracy (row0,col1)
    #         LR   (row1,col0) | [empty]  (row1,col1) hidden
    # figsize calibrated so bbox_inches='tight' output = 4471x2966 px @ 300 DPI
    # (raw figsize 14.903x9.887 produced 4437x3010 -> scaled by 4471/4437 x 2966/3010)
    fig, axes2d = plt.subplots(2, 2, figsize=(14.903 * 4471/4437, 9.887 * 2966/3010))
    ax_loss = axes2d[0, 0]
    ax_acc  = axes2d[0, 1]
    ax_lr   = axes2d[1, 0]
    ax_empty= axes2d[1, 1]
    ax_empty.set_visible(False)   # hide unused bottom-right cell

    fig.suptitle(fig_title, fontsize=13, fontweight='bold', y=1.01)

    def _best_vline(ax, color=BEST_CLR):
        if best_epoch is not None:
            ax.axvline(best_epoch, color=color, linestyle=':', linewidth=1.5,
                       label=f'Best epoch {best_epoch}')

    # ── Panel 1: Loss ─────────────────────────────────────────────
    if has_loss:
        if train_loss:
            ax_loss.plot(epochs[:len(train_loss)], train_loss,
                         color=TRAIN_CLR, linewidth=2, marker='o', markersize=3,
                         label='Train Loss')
        if val_loss:
            ax_loss.plot(epochs[:len(val_loss)], val_loss,
                         color=VAL_CLR, linewidth=2, marker='s', markersize=3,
                         label='Val Loss')
        _best_vline(ax_loss)
        ax_loss.set_title('Loss During Training', fontweight='bold')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(loc='upper right', framealpha=0.9)
        ax_loss.grid(True)
        ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    else:
        ax_loss.set_visible(False)

    # ── Panel 2: Accuracy ─────────────────────────────────────────
    if has_acc:
        if train_acc:
            ax_acc.plot(epochs[:len(train_acc)], train_acc,
                        color=TRAIN_CLR, linewidth=2, marker='o', markersize=3,
                        label='Train Acc')
        if val_acc:
            ax_acc.plot(epochs[:len(val_acc)], val_acc,
                        color=VAL_CLR, linewidth=2, marker='s', markersize=3,
                        label='Val Acc')
        if best_epoch is not None:
            lbl = f'Best epoch {best_epoch}'
            if best_acc is not None:
                lbl += f'\n({best_acc:.2f}%)'
            ax_acc.axvline(best_epoch, color=BEST_CLR, linestyle=':', linewidth=1.5,
                           label=lbl)
            if best_acc is not None:
                ax_acc.axhline(best_acc, color=BEST_CLR, linestyle=':', linewidth=1,
                               alpha=0.6)
        ax_acc.set_title('Accuracy During Training', fontweight='bold')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.legend(loc='lower right', framealpha=0.9)
        ax_acc.grid(True)
        ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    else:
        ax_acc.set_visible(False)

    # ── Panel 3: Learning Rate (below Loss) ───────────────────────
    if has_lr:
        ax_lr.plot(epochs[:len(lr)], lr,
                   color=LR_CLR, linewidth=2, marker='D', markersize=3,
                   label='Learning Rate')
        _best_vline(ax_lr)
        ax_lr.set_title('Learning Rate Schedule', fontweight='bold')
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax_lr.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))
        ax_lr.legend(framealpha=0.9)
        ax_lr.grid(True)
        ax_lr.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    else:
        ax_lr.set_visible(False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'[OK] Figure saved to: {output_png.resolve()}')
    print(f'     Epochs: {n}  |  Best epoch: {best_epoch}  |  Best acc: {best_acc:.2f}%' if best_acc else '')

    # ── Optional CSV export ───────────────────────────────────────
    if args.export_csv:
        import csv
        csv_path = Path(args.export_csv)
        if args.export_csv.lower() == 'auto':
            csv_path = output_png.with_suffix('.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Pad shorter lists with empty string so all rows have the same length
        rows_data = []
        for i, ep in enumerate(epochs):
            rows_data.append({
                'epoch':      ep,
                'train_loss': train_loss[i] if i < len(train_loss) else '',
                'val_loss':   val_loss[i]   if i < len(val_loss)   else '',
                'train_acc':  train_acc[i]  if i < len(train_acc)  else '',
                'val_acc':    val_acc[i]    if i < len(val_acc)    else '',
                'lr':         lr[i]         if i < len(lr)         else '',
            })

        fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_data)

        print(f'[OK] CSV  saved to:   {csv_path.resolve()}')


if __name__ == '__main__':
    main()
