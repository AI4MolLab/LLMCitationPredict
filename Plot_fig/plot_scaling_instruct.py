#plot_scaling_instruct.py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import matplotlib.patches as mpatches

# Global academic style configuration for consistent visualization
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333', 
    'axes.linewidth': 1.2,
    'grid.color': '#EAEAEA', 
    'grid.linestyle': '--',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'legend.fontsize': 12,
    'legend.frameon': False,
    'mathtext.fontset': 'stix',
})

OUTPUT_DIR = '/beagle/Figure/Fig5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

models = ['Llama3.2\n1B', 'Llama3.2\n3B', 'Llama3\n8B', 'Llama3\n8B-Instruct']
x = np.arange(len(models))

mse_full   = [0.5953, 0.5819, 0.5378, 0.5445]
r2_full    = [0.5082, 0.5192, 0.5557, 0.5501]
ndcg5_full = [0.6812, 0.8646, 0.7525, 0.7594]
ndcg20_full= [0.7599, 0.8336, 0.8321, 0.8230]

model_colors = ['#99BADB', '#F9C699', '#C1C2E0', '#C1DCAF']
edge_color = '#505050'

def draw_chart_on_ax(ax, data, title, ylabel, ylim=None):
    """
    Helper function: Draw bar chart on specified axes
    
    Args:
        ax (matplotlib.axes.Axes): Target axes to plot on
        data (list/np.array): Numerical data to plot
        title (str): Plot title
        ylabel (str): Y-axis label
        ylim (tuple, optional): Y-axis limits (min, max). Defaults to None.
    """

    bars = ax.bar(x, data, width=0.6, color=model_colors, edgecolor=edge_color, linewidth=1.0, alpha=1.0, zorder=3)
    
    ax.set_title(title, pad=15, fontweight='bold', color='#2C3E50')
    ax.set_ylabel(ylabel, color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.tick_params(axis='x', colors='#333333')
    ax.tick_params(axis='y', colors='#333333')
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    if ylim:
        ax.set_ylim(ylim)
        
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    legend_patches = [mpatches.Patch(color=model_colors[i], label=models[i].replace('\n', ' ')) for i in range(len(models))]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=12, ncol=1) 

def save_single_chart(data, title, ylabel, filename, ylim=None):
    """
    Plot and save individual bar chart (Full Method performance across 4 models only)
    
    Args:
        data (list/np.array): Numerical data to plot
        title (str): Plot title
        ylabel (str): Y-axis label
        filename (str): Output filename
        ylim (tuple, optional): Y-axis limits (min, max). Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))
    draw_chart_on_ax(ax, data, title, ylabel, ylim)
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    print(f"Individual chart saved to: {save_path}")
    plt.close() 

def save_combined_chart():
    """
    Plot and save combined chart of all performance metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MSE
    draw_chart_on_ax(axes[0, 0], mse_full, 
                     title='Regression Error (MSE)', 
                     ylabel='Mean Squared Error', 
                     ylim=(0.50, 0.62))

    # R2
    draw_chart_on_ax(axes[0, 1], r2_full, 
                     title=r'Goodness-of-Fit ($R^2$)', 
                     ylabel=r'$R^2$ Score', 
                     ylim=(0.48, 0.60))

    # NDCG@5
    draw_chart_on_ax(axes[1, 0], ndcg5_full, 
                     title='Ranking Quality (NDCG@5)', 
                     ylabel='NDCG@5 Score', 
                     ylim=(0.60, 0.90))

    # NDCG@20
    draw_chart_on_ax(axes[1, 1], ndcg20_full, 
                     title='Ranking Quality (NDCG@20)', 
                     ylabel='NDCG@20 Score', 
                     ylim=(0.70, 0.88))
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    save_path = os.path.join(OUTPUT_DIR, 'Scaling_Instruct_Combined_Premium.png')
    plt.savefig(save_path)
    print(f"Combined chart saved to: {save_path}")
    plt.close()

def main():
    print("Starting to plot Full Method model comparison charts...")

    # Save 4 individual charts for each metric
    save_single_chart(
        mse_full, 
        title='Regression Error (MSE)', 
        ylabel='Mean Squared Error',
        filename='Scaling_Instruct_MSE_FullOnly.png',
        ylim=(0.50, 0.62) 
    )

    save_single_chart(
        r2_full, 
        title=r'Goodness-of-Fit ($R^2$)', 
        ylabel=r'$R^2$ Score',
        filename='Scaling_Instruct_R2_FullOnly.png',
        ylim=(0.48, 0.58)
    )

    save_single_chart(
        ndcg5_full, 
        title='Ranking Quality (NDCG@5)', 
        ylabel='NDCG@5 Score',
        filename='Scaling_Instruct_NDCG5_FullOnly.png',
        ylim=(0.60, 0.90)
    )

    save_single_chart(
        ndcg20_full, 
        title='Ranking Quality (NDCG@20)', 
        ylabel='NDCG@20 Score',
        filename='Scaling_Instruct_NDCG20_FullOnly.png',
        ylim=(0.70, 0.85)
    )

    # Save combined chart
    save_combined_chart()

    print("全部完成！")

if __name__ == '__main__':
    main()