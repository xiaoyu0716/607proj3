import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

def plot_bias_comparison(metrics_df, setting, save_path):
    """
    Plot bias comparison across methods.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    setting_name = "Nonlongitudinal model" if setting == "A" else "Treatment-effect model"
    methods = metrics_df['method'].tolist()
    bias = metrics_df['bias'].tolist()
    
    colors = ['#2ecc71' if 'DR' in m and '.fal' not in m else 
              '#e74c3c' if '.fal' in m else 
              '#3498db' for m in methods]
    
    bars = ax.bar(range(len(methods)), bias, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Bias', fontsize=12)
    ax.set_title(f'Bias Comparison - {setting_name}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, bias)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_boxplot(estimates_df, setting, truth_value, save_path):
    """
    Plot boxplot of estimates across methods.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    setting_name = "Nonlongitudinal model" if setting == "A" else "Treatment-effect model"
    methods = sorted(estimates_df['method'].unique())
    data_to_plot = [estimates_df[estimates_df['method'] == m]['estimate'].values for m in methods]
    
    bp = ax.boxplot(data_to_plot, labels=methods, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.6),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1))
    
    # Color boxes by method type
    for i, (patch, method) in enumerate(zip(bp['boxes'], methods)):
        if 'DR' in method and '.fal' not in method:
            patch.set_facecolor('#2ecc71')
        elif '.fal' in method:
            patch.set_facecolor('#e74c3c')
        else:
            patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    
    # Add truth line
    ax.axhline(truth_value, color='black', linestyle='--', linewidth=2, 
               label=f'Truth = {truth_value:.3f}', alpha=0.7)
    
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Estimate', fontsize=12)
    ax.set_title(f'Distribution of Estimates - {setting_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_combined_bias(metrics_A, metrics_B, save_path):
    """
    Plot bias comparison for both settings side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    for ax, metrics_df, setting in zip(axes, [metrics_A, metrics_B], ['A', 'B']):
        methods = metrics_df['method'].tolist()
        bias = metrics_df['bias'].tolist()
        
        colors = ['#2ecc71' if 'DR' in m and '.fal' not in m else 
                  '#e74c3c' if '.fal' in m else 
                  '#3498db' for m in methods]
        
        bars = ax.bar(range(len(methods)), bias, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Bias', fontsize=12)
        ax.set_title(f'Setting {setting}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, bias):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=7)
    
    fig.suptitle('Bias Comparison Across Settings', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load data
    metrics_A = pd.read_csv(os.path.join(results_dir, 'metrics_A.csv'))
    metrics_B = pd.read_csv(os.path.join(results_dir, 'metrics_B.csv'))
    estimates_A = pd.read_csv(os.path.join(results_dir, 'estimates_A.csv'))
    estimates_B = pd.read_csv(os.path.join(results_dir, 'estimates_B.csv'))
    
    # Get truth values
    truth_A = estimates_A['truth'].iloc[0]
    truth_B = estimates_B['truth'].iloc[0]
    
    print("Generating visualizations...")
    
    # Individual bias plots
    plot_bias_comparison(metrics_A, 'A', os.path.join(figures_dir, 'bias_comparison_A.png'))
    plot_bias_comparison(metrics_B, 'B', os.path.join(figures_dir, 'bias_comparison_B.png'))
    
    # Combined bias plot
    plot_combined_bias(metrics_A, metrics_B, os.path.join(figures_dir, 'bias_comparison_combined.png'))
    
    # Boxplots
    plot_boxplot(estimates_A, 'A', truth_A, os.path.join(figures_dir, 'boxplot_A.png'))
    plot_boxplot(estimates_B, 'B', truth_B, os.path.join(figures_dir, 'boxplot_B.png'))
    
    print("\nAll visualizations saved to:", figures_dir)
    print("\nSummary:")
    print(f"  Setting A - Truth: {truth_A:.3f}")
    print(f"  Setting B - Truth: {truth_B:.3f}")
    print(f"\n  Files generated:")
    print(f"    - bias_comparison_A.png")
    print(f"    - bias_comparison_B.png")
    print(f"    - bias_comparison_combined.png")
    print(f"    - boxplot_A.png")
    print(f"    - boxplot_B.png")

if __name__ == "__main__":
    main()
