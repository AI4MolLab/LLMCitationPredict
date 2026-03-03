#MseR2.py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# Global Academic Style
academic_params = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.facecolor': 'white',
    'axes.edgecolor': '#404040', 
    'axes.linewidth': 1.0,
    'grid.color': '#EAEAEA',
    'grid.linestyle': '--',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'legend.fontsize': 13, 
    'legend.frameon': False, 
    'mathtext.fontset': 'stix',
}
plt.rcParams.update(academic_params)

# Global Data & Configuration
models = ['SciBERT', 'Llama-3-8B', 'Gemma-7B', 'Mistral-7B']
x = np.arange(len(models)) 
width = 0.28 

# Data
mse_base = [0.6539, 0.5426, 0.5530, 0.5387]
mse_full = [0.6382, 0.5378, 0.5526, 0.5285]

r2_base = [0.4598, 0.5517, 0.5431, 0.5549]
r2_full = [0.4727, 0.5557, 0.5435, 0.5634]

ndcg5_base = [0.7671, 0.8419, 0.8358, 0.8858]
ndcg5_full = [0.8478, 0.7525, 0.8409, 0.8798]

ndcg20_base = [0.7698, 0.8191, 0.8478, 0.8558]
ndcg20_full = [0.8268, 0.8321, 0.8522, 0.8252]

# Color scheme
color_base = '#85C1E9' 
color_full = '#F1948A'
color_base_ndcg = '#9EAAD1'
color_full_ndcg = '#F59790' 
bar_edge_color = '#505050'

OUTPUT_DIR = '/beagle/Figure/Fig3'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_combined_figure():
    """
    Draw a combined figure containing regression (MSE, R²) and ranking (NDCG@5, NDCG@20) metrics.
    """
    print("Creating combined comparison figure...")


    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5))
    
    # MSE
    ax = axes[0, 0]
    ax.bar(x - width/2, mse_base, width, label='Baseline', color=color_base, edgecolor=bar_edge_color, linewidth=0.8)
    ax.bar(x + width/2, mse_full, width, label='Full Method', color=color_full, edgecolor=bar_edge_color, linewidth=0.8)
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('Regression Error (MSE)', pad=12, fontweight='bold', color='#2C3E50', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0.50, 0.68)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) 
    ax.legend(loc='best') 

    # R2
    ax = axes[0, 1]
    ax.bar(x - width/2, r2_base, width, label='Baseline', color=color_base, edgecolor=bar_edge_color, linewidth=0.8)
    ax.bar(x + width/2, r2_full, width, label='Full Method', color=color_full, edgecolor=bar_edge_color, linewidth=0.8)
    ax.set_ylabel(r'R-Squared ($R^2$)')
    ax.set_title(r'Goodness-of-Fit ($R^2$)', pad=12, fontweight='bold', color='#2C3E50', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0.40, 0.60)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='best')

    # NDCG@5
    ax = axes[1, 0]
    ax.bar(x - width/2, ndcg5_base, width, label='Baseline', color=color_base_ndcg, edgecolor=bar_edge_color, linewidth=0.8)
    ax.bar(x + width/2, ndcg5_full, width, label='Full Method', color=color_full_ndcg, edgecolor=bar_edge_color, linewidth=0.8)
    ax.set_ylabel('NDCG@5 Score')
    ax.set_title('Top-5 Ranking Quality (NDCG@5)', pad=12, fontweight='bold', color='#2C3E50', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.65, 0.94) 
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='best')

    # NDCG@20
    ax = axes[1, 1]
    ax.bar(x - width/2, ndcg20_base, width, label='Baseline', color=color_base_ndcg, edgecolor=bar_edge_color, linewidth=0.8)
    ax.bar(x + width/2, ndcg20_full, width, label='Full Method', color=color_full_ndcg, edgecolor=bar_edge_color, linewidth=0.8)
    ax.set_ylabel('NDCG@20 Score')
    ax.set_title('Top-20 Ranking Quality (NDCG@20)', pad=12, fontweight='bold', color='#2C3E50', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.70, 0.90) 
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='best')

    # Layout Adjustment
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.2) 
    
    save_path = os.path.join(OUTPUT_DIR, 'model_comparison_combined_2x2.png')
    plt.savefig(save_path, dpi=600)
    print(f"Combined comparison figure saved to: {save_path}")
    plt.close()

# Auxiliary Functions: Draw Individual Figures
def plot_regression_metrics_single():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MSE
    ax = axes[0]
    ax.bar(x - width/2, mse_base, width, label='Baseline', color=color_base, edgecolor=bar_edge_color, linewidth=0.8)
    ax.bar(x + width/2, mse_full, width, label='Full Method', color=color_full, edgecolor=bar_edge_color, linewidth=0.8)
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('MSE', pad=15, fontweight='bold', color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0.50, 0.68)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) 
    ax.legend(loc='best') 

    # R2
    ax = axes[1]
    ax.bar(x - width/2, r2_base, width, label='Baseline', color=color_base, edgecolor=bar_edge_color, linewidth=0.8)
    ax.bar(x + width/2, r2_full, width, label='Full Method', color=color_full, edgecolor=bar_edge_color, linewidth=0.8)
    ax.set_ylabel(r'R-Squared ($R^2$)')
    ax.set_title(r'$R^2$', pad=15, fontweight='bold', color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0.40, 0.60)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='best')

    save_path = os.path.join(OUTPUT_DIR, 'model_comparison_regression_strict_single.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Individual regression figure saved to: {save_path}")
    plt.close()

def plot_ranking_metrics_single():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # NDCG@5
    ax = axes[0]
    ax.bar(x - width/2, ndcg5_base, width, label='Baseline', color=color_base_ndcg, edgecolor=bar_edge_color, linewidth=0.8)
    ax.bar(x + width/2, ndcg5_full, width, label='Full Method', color=color_full_ndcg, edgecolor=bar_edge_color, linewidth=0.8)
    ax.set_ylabel('NDCG@5 Score')
    ax.set_title('Top-5 Ranking Quality', pad=15, fontweight='bold', color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.65, 0.94) 
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='best')

    # NDCG@20
    ax = axes[1]
    ax.bar(x - width/2, ndcg20_base, width, label='Baseline', color=color_base_ndcg, edgecolor=bar_edge_color, linewidth=0.8)
    ax.bar(x + width/2, ndcg20_full, width, label='Full Method', color=color_full_ndcg, edgecolor=bar_edge_color, linewidth=0.8)
    ax.set_ylabel('NDCG@20 Score')
    ax.set_title('Top-20 Ranking Quality', pad=15, fontweight='bold', color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.70, 0.90)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='best')
    
    save_path = os.path.join(OUTPUT_DIR, 'model_comparison_ranking_strict_single.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    print(f"Individual ranking figure saved to: {save_path}")
    plt.close()

if __name__ == '__main__':
    # Main execution: Draw combined figure
    plot_combined_figure()
    
    # Draw individual figures
    plot_regression_metrics_single()
    plot_ranking_metrics_single()
    
    print("🎉 Plotting completed!")