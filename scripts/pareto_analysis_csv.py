"""
Generate Pareto Frontier Analysis from models.csv

This script reads model performance data from models.csv and generates
three Pareto frontier plots:
1. Size vs Accuracy
2. Size vs Latency
3. Accuracy vs Latency

Author: AI Assistant
Date: February 2026
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Output directory
os.makedirs("analysis", exist_ok=True)

# Read CSV data
print("Reading models.csv...")
df = pd.read_csv("models.csv", skipinitialspace=True)

# Clean column names (remove quotes and spaces)
df.columns = df.columns.str.strip().str.replace('"', '')

# Clean data (remove quotes from string columns)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip().str.replace('"', '')

# Convert numeric columns
df['size_MB'] = pd.to_numeric(df['size_MB'])
df['accuracy_%'] = pd.to_numeric(df['accuracy_%'])
df['latency_ms'] = pd.to_numeric(df['latency_ms'])

print(f"\nLoaded {len(df)} models")
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)
print(df.to_string(index=False))
print("="*80)

# Define color mapping by type
type_colors = {
    'orginal': '#3498db',    # Blue for original
    'hybrid': '#2ecc71',     # Green for hybrid
    'tflite': '#e74c3c',     # Red for tflite
}

type_markers = {
    'orginal': 'o',   # Circle
    'hybrid': 's',    # Square
    'tflite': '^',    # Triangle
}

def plot_pareto_frontier(df, x_col, y_col, maximize_x=False, maximize_y=True,
                         xlabel='X', ylabel='Y', title='Pareto Frontier',
                         output_file='pareto.png', x_lim=None, y_lim=None,
                         reference_lines=None):
    """
    Generate a Pareto frontier plot.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        maximize_x: If True, prefer higher x values. If False, prefer lower.
        maximize_y: If True, prefer higher y values. If False, prefer lower.
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        output_file: Output filename
        x_lim: Tuple (min, max) for x-axis limits
        y_lim: Tuple (min, max) for y-axis limits
        reference_lines: Dict with 'x' and 'y' for reference line positions
    """
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    for model_type in df['type'].unique():
        subset = df[df['type'] == model_type]
        color = type_colors.get(model_type, 'gray')
        marker = type_markers.get(model_type, 'o')
        
        plt.scatter(
            subset[x_col],
            subset[y_col],
            s=200,
            c=color,
            marker=marker,
            alpha=0.7,
            edgecolors='k',
            linewidths=1.5,
            label=model_type.capitalize()
        )
        
        # Add annotations
        for _, row in subset.iterrows():
            x = row[x_col]
            y = row[y_col]
            label = row['Model_ID']
            
            plt.annotate(
                label,
                (x, y),
                xytext=(8, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
            )
    
    # Calculate and plot Pareto frontier
    # For each unique type, find Pareto optimal points
    pareto_points = []
    
    for model_type in df['type'].unique():
        subset = df[df['type'] == model_type].copy()
        
        # Sort by x-axis
        subset = subset.sort_values(by=x_col)
        
        # Find Pareto frontier
        frontier = []
        best_y = -np.inf if maximize_y else np.inf
        
        for _, row in subset.iterrows():
            y_val = row[y_col]
            
            if maximize_y:
                if y_val >= best_y:
                    frontier.append(row)
                    best_y = y_val
            else:
                if y_val <= best_y:
                    frontier.append(row)
                    best_y = y_val
        
        if len(frontier) > 1:
            frontier_df = pd.DataFrame(frontier)
            color = type_colors.get(model_type, 'gray')
            plt.plot(
                frontier_df[x_col],
                frontier_df[y_col],
                '--',
                color=color,
                alpha=0.5,
                linewidth=2,
                label=f'{model_type.capitalize()} Frontier'
            )
    
    # Add reference lines if provided
    if reference_lines:
        if 'x' in reference_lines:
            plt.axvline(x=reference_lines['x']['value'], 
                       color='gray', linestyle='--', alpha=0.5,
                       label=reference_lines['x'].get('label', ''))
        if 'y' in reference_lines:
            plt.axhline(y=reference_lines['y']['value'],
                       color='gray', linestyle='--', alpha=0.5,
                       label=reference_lines['y'].get('label', ''))
    
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    
    plt.tight_layout()
    output_path = f"analysis/{output_file}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


# Plot 1: Size vs Accuracy (minimize size, maximize accuracy)
print("\nGenerating Size vs Accuracy plot...")
plot_pareto_frontier(
    df,
    x_col='size_MB',
    y_col='accuracy_%',
    maximize_x=False,  # Prefer smaller size
    maximize_y=True,   # Prefer higher accuracy
    xlabel='Model Size (MB)',
    ylabel='Accuracy (%)',
    title='Pareto Frontier: Model Size vs Accuracy',
    output_file='pareto_size_vs_accuracy.png',
    x_lim=(0, df['size_MB'].max() * 1.1),
    y_lim=(df['accuracy_%'].min() * 0.98, df['accuracy_%'].max() * 1.02),
    reference_lines={
        'x': {'value': 1.2, 'label': 'IoT Target (1.2 MB)'},
        'y': {'value': 85.0, 'label': 'Min Accuracy (85%)'}
    }
)

# Plot 2: Size vs Latency (minimize both)
print("Generating Size vs Latency plot...")
plot_pareto_frontier(
    df,
    x_col='size_MB',
    y_col='latency_ms',
    maximize_x=False,  # Prefer smaller size
    maximize_y=False,  # Prefer lower latency
    xlabel='Model Size (MB)',
    ylabel='Inference Latency (ms)',
    title='Pareto Frontier: Model Size vs Latency',
    output_file='pareto_size_vs_latency.png',
    x_lim=(0, df['size_MB'].max() * 1.1),
    y_lim=(0, df['latency_ms'].max() * 1.1),
    reference_lines={
        'x': {'value': 1.2, 'label': 'IoT Target (1.2 MB)'},
        'y': {'value': 5.0, 'label': 'Real-time Target (5 ms)'}
    }
)

# Plot 3: Accuracy vs Latency (maximize accuracy, minimize latency)
print("Generating Accuracy vs Latency plot...")
plot_pareto_frontier(
    df,
    x_col='latency_ms',
    y_col='accuracy_%',
    maximize_x=False,  # Prefer lower latency
    maximize_y=True,   # Prefer higher accuracy
    xlabel='Inference Latency (ms)',
    ylabel='Accuracy (%)',
    title='Pareto Frontier: Accuracy vs Latency',
    output_file='pareto_accuracy_vs_latency.png',
    x_lim=(0, df['latency_ms'].max() * 1.1),
    y_lim=(df['accuracy_%'].min() * 0.98, df['accuracy_%'].max() * 1.02),
    reference_lines={
        'x': {'value': 5.0, 'label': 'Real-time Target (5 ms)'},
        'y': {'value': 85.0, 'label': 'Min Accuracy (85%)'}
    }
)

print("\n" + "="*80)
print("PARETO ANALYSIS COMPLETE")
print("="*80)
print("Generated plots:")
print("  - analysis/pareto_size_vs_accuracy.png")
print("  - analysis/pareto_size_vs_latency.png")
print("  - analysis/pareto_accuracy_vs_latency.png")
print("="*80)
