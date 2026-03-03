#plot_fig4c_bias.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.feature_extraction import text as sklearn_text

# Global Academic Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 600,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix',
    'axes.edgecolor': '#404040',
    'axes.linewidth': 1.2,
    'grid.color': '#EAEAEA',
})

# Path Configuration
ARTIFACTS_DIR = "/beagle/llama3-8b/data/10000/topic_model_artifacts"
DATA_PATH = "/beagle/llama3-8b/data/10000/metadata_10000_with_similarity.csv"
CSV_DIR = "/beagle/Figure/Fig4"
PATH_BASELINE_CSV = os.path.join(CSV_DIR, "results_baseline.csv")
PATH_FULL_CSV = os.path.join(CSV_DIR, "results_base_lora.csv")
OUTPUT_PLOT_DIR = "/beagle/Figure/Fig4"
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# Academic stop words list
ACADEMIC_STOP_WORDS = [
    'review', 'overview', 'summary', 'paper', 'article', 'work', 'research', 'study', 'studies',
    'experiment', 'experiments', 'experimental', 'development', 'developments', 'developed', 'developing',
    'present', 'presented', 'presents', 'role', 'roles', 'number', 'numbers', 'field', 'fields',
    'data', 'analysis', 'information', 'model', 'models', 'modeling', 'design', 'designs', 'designed',
    'performance', 'performances', 'application', 'applications', 'properties', 'property',
    'characteristics', 'characterization', 'structure', 'structures', 'structural', 'complex', 'complexes',
    'species', 'individual', 'group', 'groups', 'type', 'types',
    'new', 'novel', 'proposed', 'propose', 'proposes', 'observed', 'observation', 'related', 'associated', 'potential',
    'various', 'different', 'similar',
    'using', 'used', 'use', 'uses', 'utilizing', 'utilized', 'via', 'based', 'basing', 'bases',
    'showing', 'shown', 'show', 'shows', 'suggesting', 'suggested', 'suggest', 'suggests',
    'reporting', 'reported', 'report', 'reports', 'demonstrating', 'demonstrated', 'demonstrate',
    'providing', 'provided', 'provide', 'including', 'included', 'include',
    'method', 'methods', 'methodology', 'result', 'results', 'finding', 'findings', 'conclusion', 'conclusions',
    'approach', 'approaches', 'technique', 'techniques', 'system', 'systems', 'process', 'processes',
    'case', 'order', 'time', 'effect', 'effects',
    'high', 'low', 'higher', 'lower', 'highest', 'lowest', 'large', 'small', 'significantly', 'significant',
    'respectively', 'corresponding', 'furthermore', 'moreover', 'addition', 'contrast', 'however', 'therefore', 'thus', 'hence',
    's'
]
FINAL_STOP_WORDS = list(sklearn_text.ENGLISH_STOP_WORDS.union(ACADEMIC_STOP_WORDS))

def custom_tokenizer(text):
    if not isinstance(text, str): return []
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+(?:-[a-z0-9]+)*\b', text)
    return [t for t in tokens if not t.isdigit() and t not in FINAL_STOP_WORDS and len(t) > 1]

def identity_tokenizer(text): return text


def get_custom_rank_labels(series, total_ranks):
    series = series.astype(float)
    bins = [0, total_ranks * 0.05, total_ranks * 0.20, total_ranks * 0.60, total_ranks]
    
    labels = [
        "Top Rank\n(Top 5%)", 
        "High Rank\n(Top 20%)", 
        "Medium Rank\n(Mid 40%)", 
        "Low Rank\n(Bottom 40%)"
    ]
    
    bins = sorted(list(set(bins)))
    if len(bins) != 5:
        print(f"Warning: Bins collapsed: {bins}. Using default labels subset.")
        labels = labels[:len(bins)-1]

    return pd.cut(series, bins=bins, labels=labels, right=True, include_lowest=True)

def get_test_set_with_groups():
    """Load test set metadata and assign topic popularity rank groups
    
    Loads LDA model artifacts, processes test set text, assigns primary topics,
    and creates topic popularity rank groups for bias analysis.
    
    Returns:
        pd.DataFrame: Test set dataframe with topic rank and group labels, or None if error
    """
    print("Loading LDA model artifacts...")

    try:
        phraser = joblib.load(os.path.join(ARTIFACTS_DIR, 'bigram_phraser.joblib'))
        vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, 'vectorizer.joblib'))
        lda = joblib.load(os.path.join(ARTIFACTS_DIR, 'lda_model.joblib'))
        topic_rank_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'topic_rank_map.joblib'))
        test_indices = joblib.load(os.path.join(ARTIFACTS_DIR, 'test_indices.joblib'))
        
        with open(os.path.join(ARTIFACTS_DIR, 'num_topics.txt'), 'r') as f:
            num_topics = int(f.read())
            
    except FileNotFoundError as e:
        print(f"Error: LDA model artifacts not found: {e}")
        return None

    print(f"Loading metadata from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    test_df = df.loc[test_indices].copy()
    
    # --- 计算 Rank ---
    raw_texts = (test_df['title'].fillna('') + ' ' + test_df['abstract'].fillna('')).tolist()
    tokens = [custom_tokenizer(t) for t in raw_texts]
    final_tokens = [phraser[doc] for doc in tokens]
    
    X = vectorizer.transform(final_tokens)
    topic_distr = lda.transform(X)
    primary_topic_ids = np.argmax(topic_distr, axis=1)
    
    median_rank = np.median(list(topic_rank_map.values()))
    ranks = [topic_rank_map.get(tid, median_rank) for tid in primary_topic_ids]
    
    test_df['topic_rank_raw'] = ranks
    test_df['group_label'] = get_custom_rank_labels(test_df['topic_rank_raw'], num_topics)
    
    return test_df.reset_index(drop=True)

def plot_bias_analysis():
    """Main function to plot topic popularity bias analysis boxplots
    
    Compares absolute prediction error between Baseline and Full methods
    across different topic popularity rank groups.
    """

    # Prepare test set data with topic rank groups 
    df_meta = get_test_set_with_groups()
    if df_meta is None: return

    # Load prediction results from CSV files 
    if not os.path.exists(PATH_BASELINE_CSV) or not os.path.exists(PATH_FULL_CSV):
        print("Error: Prediction CSV files not found.")
        return

    df_base = pd.read_csv(PATH_BASELINE_CSV)
    df_full = pd.read_csv(PATH_FULL_CSV)

    # Align data lengths
    min_len = min(len(df_meta), len(df_base), len(df_full))
    df_meta = df_meta.iloc[:min_len]
    df_base = df_base.iloc[:min_len]
    df_full = df_full.iloc[:min_len]

    # Calculate absolute prediction error for both models
    ae_base = np.abs(df_base['true_log_citations'] - df_base['pred_log_citations'])
    ae_full = np.abs(df_full['true_log_citations'] - df_full['pred_log_citations'])

    # Build plotting dataframe
    plot_data = []
    group_categories = df_meta['group_label'].cat.categories
    
    for i in range(min_len):
        group = df_meta['group_label'].iloc[i]
        plot_data.append({
            'Group': group,
            'Model': 'Baseline',
            'AbsError': ae_base[i]
        })
        plot_data.append({
            'Group': group,
            'Model': 'Full Method',
            'AbsError': ae_full[i]
        })
    
    df_plot = pd.DataFrame(plot_data)

    # Create boxplot for bias analysis
    print("Starting plot generation...")
    plt.figure(figsize=(12, 8))
    
    palette = {'Baseline': '#80DEEA', 'Full Method': '#FFAB91'}
    ax = sns.boxplot(x='Group', y='AbsError', hue='Model', data=df_plot, 
                     order=group_categories, 
                     palette=palette, 
                     showfliers=False, 
                     width=0.7, linewidth=1.5,
                     boxprops=dict(alpha=0.95, edgecolor='#505050'))

    plt.title('Topic Popularity Bias Analysis', pad=20, fontweight='bold', color='#2C3E50')
    plt.ylabel('Absolute Prediction Error', labelpad=12, color='#2C3E50')
    plt.xlabel('Topic Popularity Group', labelpad=12, color='#2C3E50')
    
    ax.tick_params(colors='#333333')

    plt.legend(loc='upper left', title=None, framealpha=0.95, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    save_path = os.path.join(OUTPUT_PLOT_DIR, "Fig4c_Topic_Bias_Boxplot_Premium-128.png")
    plt.savefig(save_path)
    print(f"Strict grouping bias analysis plot (premium comparison color scheme) saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_bias_analysis()