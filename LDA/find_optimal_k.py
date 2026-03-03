#find_optimal_k.py
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import time
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text as sklearn_text 
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.phrases import Phrases, Phraser

#Configuration parameters
INPUT_CSV_PATH = '/beagle/LDA/data/10000/metadata_10000_with_similarity.csv' 
TITLE_COL = 'title'
ABSTRACT_COL = 'abstract'
OUTPUT_PLOT_PATH = '/beagle/LDA/topic_coherence_plot-10000.png'

START_K = 5
END_K = 50       
STEP_K = 5 
RANDOM_STATE = 42

# Configure matplotlib parameters for academic plottin
academic_params = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.labelsize': 18,      
    'axes.titlesize': 20,    
    'xtick.labelsize': 16,     
    'ytick.labelsize': 16,     
    'axes.linewidth': 1.2,     
    'figure.dpi': 600,
    'savefig.bbox': 'tight',
    'grid.color': '#E0E0E0',  
    'grid.linestyle': '--',
}
plt.rcParams.update(academic_params)

# Academic stop words list 
ACADEMIC_STOP_WORDS = [
    'review', 'overview', 'summary', 'paper', 'article', 'work', 'research', 'study', 'studies',
    'experiment', 'experiments', 'experimental', 'development', 'developments', 'developed', 'developing',
    'present', 'presented', 'presents', 'role', 'roles', 'number', 'numbers', 'field', 'fields',
    'data', 'analysis', 'information', 'model', 'models', 'modeling', 'design', 'designs', 'designed',
    'performance', 'performances', 'application', 'applications', 'properties', 'property',
    'characteristics', 'characterization', 'structure', 'structures', 'structural', 'complex', 'complexes',
    'species', 'individual', 'group', 'groups', 'type', 'types',
    'new', 'novel', 'proposed', 'propose', 'proposes','observed', 'observation', 'related', 'associated', 'potential',
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

# Merge stop word lists
FINAL_STOP_WORDS = list(sklearn_text.ENGLISH_STOP_WORDS.union(ACADEMIC_STOP_WORDS))

def custom_tokenizer(text):
    """
    Regex tokenizing + stopword filter + short word filter
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+(?:-[a-z0-9]+)*\b', text)
    
    valid_tokens = [
        t for t in tokens 
        if not t.isdigit() 
        and t not in FINAL_STOP_WORDS
        and len(t) > 1 
    ]
    return valid_tokens

def identity_tokenizer(text):
    return text

def find_optimal_k():

    """Main function to find optimal number of LDA topics (k) via coherence score grid search"""

    print(f"Loading data from {INPUT_CSV_PATH}...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found.")
        return

    print("Step 0: Splitting data (80% Training Set)...")
    train_df, _ = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    raw_texts = (train_df[TITLE_COL].fillna('') + ' ' + train_df[ABSTRACT_COL].fillna('')).tolist()

    print("Step 1: Tokenizing and Pre-filtering Stopwords...")
    tokenized_texts = [custom_tokenizer(text) for text in raw_texts]

    print("Step 2: Training Automated Phraser...")
    phrases_model = Phrases(tokenized_texts, min_count=5, threshold=10)
    bigram_phraser = Phraser(phrases_model)

    print("Step 3: Merging Phrases...")
    final_tokens = [bigram_phraser[doc] for doc in tokenized_texts]

    print("Step 4: Vectorizing (Identity Mode)...")
    vectorizer = CountVectorizer(
        tokenizer=identity_tokenizer,    
        preprocessor=identity_tokenizer,
        token_pattern=None,              
        max_df=0.8, 
        min_df=2, 
        stop_words=None,               
        ngram_range=(1, 1),              
        max_features=10000
    )
    
    X = vectorizer.fit_transform(final_tokens)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Feature set size: {len(feature_names)}")

    print("Preparing corpus for Coherence calculation...")

    processed_texts_for_coherence = vectorizer.inverse_transform(X)
    processed_texts_for_coherence = [list(doc) for doc in processed_texts_for_coherence]
    dictionary = Dictionary(processed_texts_for_coherence)
    
    coherence_scores = []
    k_values = range(START_K, END_K + 1, STEP_K)

    print(f"\nStarting Grid Search for K ({START_K} to {END_K})...")
    # Grid search over k values to find optimal number of topics
    for k in k_values:
        start_time = time.time()
        lda_model = LatentDirichletAllocation(
            n_components=k, 
            random_state=RANDOM_STATE,
            learning_method='batch', 
            max_iter=50,
            doc_topic_prior=0.1,    
            topic_word_prior=0.01,      
            n_jobs=-1 
        )
        lda_model.fit(X)
        
        topics = []
        for topic_idx, topic_weights in enumerate(lda_model.components_):
            top_indices = topic_weights.argsort()[:-11:-1]
            topic_words = [feature_names[i] for i in top_indices]
            topics.append(topic_words)
        # Calculate c_v coherence score
        coherence_model = CoherenceModel(
            topics=topics, 
            texts=processed_texts_for_coherence, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        score = coherence_model.get_coherence()
        coherence_scores.append(score)
        
        print(f"k={k}: Coherence={score:.4f} (Time: {time.time() - start_time:.1f}s)")


    print("\nPlotting coherence scores...")
    plt.figure(figsize=(10, 6)) 
    plt.plot(k_values, coherence_scores, marker='o', linestyle='--', color='black', linewidth=2, markersize=8)
    plt.title("LDA Topic Coherence Score", fontweight='bold', pad=20) 
    plt.xlabel("Number of Topics (k)", labelpad=15)
    plt.ylabel("Coherence Score", labelpad=15) 
    plt.xticks(list(k_values))
    plt.grid(True, which='both', linestyle='--', linewidth=1.0, color='#E0E0E0', alpha=0.8)
    
    max_score = max(coherence_scores)
    best_k = k_values[coherence_scores.index(max_score)]
    # plt.annotate(f'Optimal k={best_k}', xy=(best_k, max_score), ...)

    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=600, bbox_inches='tight')
    print(f"✅ Chart saved to {OUTPUT_PLOT_PATH}")
    print(f"✅ Recommended K: {best_k}")

if __name__ == '__main__':
    find_optimal_k()