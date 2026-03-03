#plot_cold_start.py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# Global Academic Style Configuration
academic_params = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'grid.color': '#E0E0E0',
    'grid.linestyle': '--',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'legend.fontsize': 12,
    'legend.frameon': False,
    'mathtext.fontset': 'stix',
}
plt.rcParams.update(academic_params)

OUTPUT_DIR = '/beagle/Figure/Fig4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

colors = {
    'base':  '#B3DDD1',  
    'topic': '#D1DCE2', 
    'div':   '#F5B994',  
    'full':  '#EE9C6C'   
}

bar_edge_color = '#455A64'

def plot_cold_start_detailed():
    """
    Draw detailed analysis plot of data scale and cold start effect (Llama3-8B)
    Comparison: Performance of Baseline, +Topic, +Div, Full under 1k vs 10k dataset sizes
    """
    # Data Preparation
    scenarios = ['Dataset Size (1k)', 'Dataset Size (10k+)']
    x = np.arange(len(scenarios))
    width = 0.18
    
    base_scores  = [0.4986, 0.5517]
    topic_scores = [0.5452, 0.5623]
    div_scores   = [0.5124, 0.5528]
    full_scores  = [0.5121, 0.5557]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Plot four groups of bar charts
    ax.bar(x - 1.5*width, base_scores,  width, label='Baseline',    color=colors['base'],  edgecolor=bar_edge_color, linewidth=0.8, alpha=0.9, zorder=3)
    ax.bar(x - 0.5*width, topic_scores, width, label='TP-Only',  color=colors['topic'], edgecolor=bar_edge_color, linewidth=0.8, alpha=0.9, zorder=3)
    ax.bar(x + 0.5*width, div_scores,   width, label='TD-Only',  color=colors['div'],   edgecolor=bar_edge_color, linewidth=0.8, alpha=0.9, zorder=3)
    ax.bar(x + 1.5*width, full_scores,  width, label='Full Method', color=colors['full'],  edgecolor=bar_edge_color, linewidth=0.8, alpha=1.0, zorder=3)

    # Set axis labels and title
    ax.set_ylabel(r'Regression Accuracy ($R^2$)')
    ax.set_title('Impact of Data Scale: Component Analysis', pad=15, fontweight='bold', color='#263238')
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    
    ax.set_ylim(0.45, 0.60)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Add numerical value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, rotation=0, color='#37474F')

    autolabel(ax.containers[0]) 
    autolabel(ax.containers[1])
    autolabel(ax.containers[2])
    autolabel(ax.containers[3])

    # Legend configuration
    ax.legend(loc='best', ncol=2, columnspacing=1.0)

    # Save figure
    save_path = os.path.join(OUTPUT_DIR, 'Data_Scale_Cold_Start_1-17.png')
    plt.savefig(save_path)
    print(f"Cold start detailed comparison plot (premium color scheme - legend inside) saved to: {save_path}")
    plt.close()  

if __name__ == '__main__':
    print("Start...")
    plot_cold_start_detailed()
    print("Plotting completed!")