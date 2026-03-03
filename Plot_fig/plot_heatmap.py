# plot_heatmap.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'lines.linewidth': 1,
    'axes.linewidth': 1,
    'axes.labelsize': 22,       
    'axes.titlesize': 22,       
    'xtick.labelsize': 22,      
    'ytick.labelsize': 22,      
    'xtick.major.size': 5,      
    'xtick.major.width': 1,     
    'ytick.major.size': 5,
    'ytick.major.width': 1,
    'axes.unicode_minus': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = '/beagle/Figure/Fig3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_feature_contribution_heatmap():
    """
    Plot high-contrast emerald-style heatmap for feature contribution analysis
    Visualizes R² scores across different LLMs and feature configurations
    """
    # Data Preparation
    models = ['Llama-3-8B', 'Gemma-7B', 'Mistral-7B']
    features = ['Baseline', 'TP-Only', 'TD-Only', 'Full Method']
    data = [
        [0.5517, 0.5431, 0.5549],  
        [0.5623, 0.5606, 0.5531],  
        [0.5528, 0.5277, 0.5467],  
        [0.5557, 0.5435, 0.5634]   
    ]
    df = pd.DataFrame(data, index=features, columns=models)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 9))
    colors_emerald = ["#FFFFE0", "#B2DFDB", "#4DB6AC", "#00897B", "#004D40"]
    cmap_high_contrast = mcolors.LinearSegmentedColormap.from_list("HighContrastEmerald", colors_emerald, N=256)

    # Set color scale limits (vmin/vmax) for consistent visualization
    vmin = 0.525  
    vmax = 0.560  
    
    # Plot heatmap with custom styling
    sns.heatmap(df, 
                annot=True, 
                fmt=".4f", 
                cmap=cmap_high_contrast, 
                vmin=vmin, 
                vmax=vmax,
                linewidths=1.5,       
                linecolor='white',    
                annot_kws={"size": 24, "weight": "bold", "color": "black"}, 
                cbar_kws={'label': r'R-Squared ($R^2$) Score', 'shrink': 0.8}
               )

    # Configure axis labels, title, and spines
    font_style = {'fontsize': 24, 'color': '#333333', 'fontweight': 'bold'}
    
    plt.title(r'Feature Contribution Analysis ($R^2$)', pad=25, **font_style)
    plt.xlabel('Large Language Models', labelpad=15, **font_style)
    plt.ylabel('Feature Configurations', labelpad=15, **font_style)
    plt.yticks(rotation=0, fontsize=22, color='#333333') 
    plt.xticks(fontsize=22, color='#333333')
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('#333333')

    # Save
    save_path = os.path.join(OUTPUT_DIR, 'Feature_Contribution_Heatmap.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    print(f"High-contrast emerald-style heatmap saved to: {save_path}")
    plt.close()

if __name__ == '__main__':
    plot_feature_contribution_heatmap()