#visualize_topics.py
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from wordcloud import WordCloud
import numpy as np

# Configuration
ARTIFACTS_DIR = '/beagle/LDA/10825'
OUTPUT_IMG_DIR = '/beagle/Figure/Fig2/Topic_Visualizations'
TOP_N_TOPICS_TO_SHOW = 20

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
def identity_tokenizer(text):
    return text

def get_target_topic_ids():
    """Get list of topic IDs to visualize 
    
    Retrieves topic IDs ordered by their impact/popularity rank from the 
    precomputed topic_rank_map. If rank file is not found, returns default 
    sequential topic IDs.
    
    Returns:
        list: List of topic IDs to visualize 
    """
    rank_map_path = os.path.join(ARTIFACTS_DIR, 'topic_rank_map.joblib')
    if os.path.exists(rank_map_path):
        rank_map = joblib.load(rank_map_path)
        rank_to_topic = {v: k for k, v in rank_map.items()}
        available_n = min(TOP_N_TOPICS_TO_SHOW, len(rank_to_topic))
        target_ids = [rank_to_topic[i] for i in range(1, available_n + 1)]
        print(f"Processing top {available_n} topics by impact rank: {target_ids}")
        return target_ids
    else:
        print("Rank file not found, showing first N topics by default.")
        return list(range(TOP_N_TOPICS_TO_SHOW))

def save_individual_bar_charts(model, feature_names, n_top_words=10):

    target_topic_ids = get_target_topic_ids()
    bar_dir = os.path.join(OUTPUT_IMG_DIR, "BarCharts")
    os.makedirs(bar_dir, exist_ok=True)

    premium_cmap = mcolors.LinearSegmentedColormap.from_list("fresh_academic_grad", ["#A3E4D7", "#2980B9"])

    for i, topic_idx in enumerate(target_topic_ids):
        # Prepare topic data
        topic = model.components_[topic_idx]
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[j].replace('_', ' ') for j in top_features_ind]
        weights = topic[top_features_ind]

        norm = mcolors.Normalize(vmin=weights[-1], vmax=weights[0])
        bar_colors = premium_cmap(norm(weights))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_features, weights, height=0.7, color=bar_colors, zorder=3, alpha=0.95)
        ax.set_title(f'Topic {topic_idx} (Impact Rank: {i+1})', pad=20, fontweight='bold', fontsize=20, color='#2C3E50')
        ax.set_xlabel('Topic Weight / Keyword Importance', labelpad=12, fontsize=18, color='#2C3E50')
        ax.tick_params(axis='both', which='major', labelsize=16, colors='#404040')
        ax.grid(axis='x', linestyle='--', alpha=0.6, zorder=0) 
        ax.grid(axis='y', visible=False)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        filename = f"Rank_{i+1:02d}_Topic_{topic_idx}_Bar.png"
        save_path = os.path.join(bar_dir, filename)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close() 
        print(f"  - Bar chart saved (Light & Bright Style): {filename}")

def save_individual_word_clouds(model, feature_names):

    target_topic_ids = get_target_topic_ids()
    wc_dir = os.path.join(OUTPUT_IMG_DIR, "WordClouds")
    os.makedirs(wc_dir, exist_ok=True)
    
    for i, topic_idx in enumerate(target_topic_ids):
        topic = model.components_[topic_idx]
        top_indices = topic.argsort()[:-50:-1] 
        freqs = {feature_names[j].replace('_', ' '): topic[j] for j in top_indices}
        
        if not freqs:
            print(f"Warning: Topic {topic_idx} is empty, skipping.")
            continue
        wc = WordCloud(background_color="white", max_words=50, width=1600, height=1200, margin=5, colormap='GnBu')
        wc.generate_from_frequencies(freqs)
        
        plt.figure(figsize=(20, 6))
        
        try:
            pil_img = wc.to_image()
            image_array = np.array(pil_img) 
            
            plt.imshow(image_array, interpolation='bilinear')
            plt.axis("off")
            plt.title(f'Topic {topic_idx} (Rank {i+1})', fontsize=20, pad=15, color='#2C3E50')
            filename = f"Rank_{i+1:02d}_Topic_{topic_idx}_WordCloud.png"
            save_path = os.path.join(wc_dir, filename)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"  - Word cloud saved: {filename}")
            
        except Exception as e:
            print(f"Error plotting Topic {topic_idx}: {e}")
        
        plt.close()

def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    print("Loading LDA model and Vectorizer...")
    try:
        vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, 'vectorizer.joblib'))
        lda = joblib.load(os.path.join(ARTIFACTS_DIR, 'lda_model.joblib'))
    except FileNotFoundError:
        print(f"Error: Files not found, please check path: {ARTIFACTS_DIR}")
        return

    feature_names = vectorizer.get_feature_names_out()

    print("Generating bar charts")
    save_individual_bar_charts(lda, feature_names, n_top_words=10)
    
    # print("Generating word cloud visualizations")
    # save_individual_word_clouds(lda, feature_names)
    
    print(f"\nAll visualizations saved to directory: {OUTPUT_IMG_DIR}")

if __name__ == "__main__":
    main()