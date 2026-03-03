# visualize_distributions.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.colors as mcolors

# Configuration
DATA_PATH = "/beagle/llama3-8b/data/10000/metadata_10000_with_similarity.csv" 
OUTPUT_DIR = '/beagle/Figure/Fig2'

# Global Style Settings
academic_params = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.facecolor': 'white',
    'axes.edgecolor': '#404040', 
    'axes.linewidth': 1.2,
    'grid.color': '#EAEAEA', 
    'grid.linestyle': '--',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
}

plt.rcParams.update(academic_params)

def plot_feature_distributions():
    """
    Plot distribution histograms for topic popularity rank and topic diversity (entropy)
    Creates side-by-side histograms with custom color schemes and academic styling
    """
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("Error: File not found.")
        return

    df = pd.read_csv(DATA_PATH)
    if 'diversity_raw' not in df.columns or 'topic_rank_raw' not in df.columns:
        print("Error: Required columns not found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    rank_cmap = mcolors.LinearSegmentedColormap.from_list("teal_mint_grad", ["#00897B", "#B2DFDB"])
    div_fill_color = '#EC8D61' 
    div_line_color = '#641E16'
    ax1 = axes[0]
    sns.histplot(
        data=df, 
        x='topic_rank_raw', 
        bins=25, 
        kde=False, 
        color='grey', 
        edgecolor='white',
        alpha=1.0,    
        ax=ax1,
        discrete=True
    )
    
    x_min, x_max = df['topic_rank_raw'].min(), df['topic_rank_raw'].max()
    norm = mcolors.Normalize(vmin=x_min, vmax=x_max)
    
    for patch in ax1.patches:
        x = patch.get_x() + patch.get_width() / 2
        color = rank_cmap(norm(x))
        patch.set_facecolor(color)
        patch.set_edgecolor('white') 

    ax1.set_title('Distribution of Topic Popularity', pad=20, fontweight='bold', color='#2C3E50')
    ax1.set_xlabel('Topic Popularity Rank', labelpad=12, color='#2C3E50')
    ax1.set_ylabel('Number of Papers', labelpad=12, color='#2C3E50')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax2 = axes[1]
    sns.histplot(
        data=df, 
        x='diversity_raw', 
        kde=True, 
        color=div_fill_color, 
        edgecolor='white',    
        alpha=0.9,            
        ax=ax2,
        line_kws={'linewidth': 2.5, 'color': div_line_color} 
    )
    
    ax2.set_title('Distribution of Topic Diversity', pad=20, fontweight='bold', color='#2C3E50')
    ax2.set_xlabel('Topic Entropy', labelpad=12, color='#2C3E50')
    ax2.set_ylabel('Frequency / Density', labelpad=12, color='#2C3E50')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Save the final figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'feature_distribution_histograms_premium_v2.png')
    plt.tight_layout()
    plt.savefig(save_path)
    
    print(f"Premium color scheme distribution histograms (right plot enhanced) saved to: {save_path}")
    print("\n--- Statistical Summary ---")
    print(df[['topic_rank_raw', 'diversity_raw']].describe())

if __name__ == '__main__':
    plot_feature_distributions()