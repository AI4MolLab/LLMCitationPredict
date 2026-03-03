#train_topic_model-fit.py
import pandas as pd
import numpy as np
import re
import os
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text as sklearn_text
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from gensim.models.phrases import Phrases, Phraser

# Configuration
RAW_DATA_PATH = '/beagle/LDA/data/10000/metadata_10000_with_similarity.csv' 
OUTPUT_ARTIFACTS_DIR = '/beagle/LDA/10825'
OPTIMAL_K = 20
RANDOM_STATE = 42
TITLE_COL = 'title'
ABSTRACT_COL = 'abstract'
TARGET_COL = 'citations_within_3_years'

# Stop Word List
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
FINAL_STOP_WORDS = list(sklearn_text.ENGLISH_STOP_WORDS.union(ACADEMIC_STOP_WORDS))

def custom_tokenizer(text):
    """Custom tokenizer for academic text preprocessing
    
    Performs:
    1. Lowercasing of input text
    2. Regex-based token extraction (supports hyphenated words)
    3. Filtering of:
       - Non-string inputs
       - Pure numeric tokens
       - Stop words (from FINAL_STOP_WORDS)
       - Tokens with length ≤ 1
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of valid preprocessed tokens
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+(?:-[a-z0-9]+)*\b', text)
    valid_tokens = [
        t for t in tokens 
        if not t.isdigit() and t not in FINAL_STOP_WORDS
        and len(t) > 1
    ]
    return valid_tokens

def identity_tokenizer(text):
    return text

def train_and_save_lda_artifacts():
    os.makedirs(OUTPUT_ARTIFACTS_DIR, exist_ok=True)
    
    print(f"Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Split Data into Train/Test Sets and Capture Indices
    print("Step 0: Splitting Data and Capturing Indices...")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    train_indices = train_df.index.values
    test_indices = test_df.index.values
    
    raw_texts_train = (train_df[TITLE_COL].fillna('') + ' ' + train_df[ABSTRACT_COL].fillna('')).tolist()
    raw_texts_test = (test_df[TITLE_COL].fillna('') + ' ' + test_df[ABSTRACT_COL].fillna('')).tolist()
    
    # Train Bigram Phraser on Training Set Only
    print("Step 1: Learning Phrases on Training Set...")
    tokens_train = [custom_tokenizer(text) for text in raw_texts_train]
    phrases_model = Phrases(tokens_train, min_count=5, threshold=10)
    bigram_phraser = Phraser(phrases_model)
    final_tokens_train = [bigram_phraser[doc] for doc in tokens_train]

    # Vectorize Training Set
    print("Step 2: Vectorizing Training Set...")
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
    X_train = vectorizer.fit_transform(final_tokens_train)
    
    # Train LDA Model
    print(f"Step 3: Training LDA model with k={OPTIMAL_K} (max_iter=100)...")
    lda = LatentDirichletAllocation(
        n_components=OPTIMAL_K, 
        random_state=RANDOM_STATE, 
        n_jobs=-1,
        learning_method='batch', 
        max_iter=50,
        doc_topic_prior=0.1,   
        topic_word_prior=0.01       
    )
    lda.fit(X_train) 
    
    # Calculate Topic Ranks & Diversity
    print("Step 4: Calculating Topic Ranks & Diversity Bins based on TRAIN set...")
    
    train_topic_distr = lda.transform(X_train)
    train_df_stats = train_df.copy()
    train_df_stats['topic_id'] = np.argmax(train_topic_distr, axis=1)
    
    topic_value = train_df_stats.groupby('topic_id')[TARGET_COL].mean().sort_values(ascending=False).reset_index()
    topic_value['topic_rank_raw'] = topic_value.index + 1
    topic_rank_map = topic_value.set_index('topic_id')['topic_rank_raw'].to_dict()
    
    print("  - Topic Ranks (Derived from Train):", topic_rank_map)
    diversity_raw_train = -np.sum(train_topic_distr * np.log2(train_topic_distr + 1e-10), axis=1)
    _, diversity_bins = pd.qcut(diversity_raw_train, q=3, labels=False, retbins=True, duplicates='drop')
    
    print("  - Diversity Bins (Derived from Train):", diversity_bins)

    # Validate Model on Test Set
    print("Step 5: Verifying model on Test Set (For observation only)...")
    tokens_test = [custom_tokenizer(text) for text in raw_texts_test]
    final_tokens_test = [bigram_phraser[doc] for doc in tokens_test] 
    X_test = vectorizer.transform(final_tokens_test) 
    test_topic_distr = lda.transform(X_test)

    print(f"  - Train Perplexity: {lda.perplexity(X_train):.2f}")
    print(f"  - Test Perplexity: {lda.perplexity(X_test):.2f}")

    # Save All Model Artifacts
    print("Step 6: Saving Artifacts (Clean & Leak-free)...")
    joblib.dump(bigram_phraser, os.path.join(OUTPUT_ARTIFACTS_DIR, 'bigram_phraser.joblib'))
    joblib.dump(vectorizer, os.path.join(OUTPUT_ARTIFACTS_DIR, 'vectorizer.joblib'))
    joblib.dump(lda, os.path.join(OUTPUT_ARTIFACTS_DIR, 'lda_model.joblib'))
    joblib.dump(topic_rank_map, os.path.join(OUTPUT_ARTIFACTS_DIR, 'topic_rank_map.joblib'))
    joblib.dump(diversity_bins, os.path.join(OUTPUT_ARTIFACTS_DIR, 'diversity_bins.joblib'))
    joblib.dump(train_indices, os.path.join(OUTPUT_ARTIFACTS_DIR, 'train_indices.joblib'))
    joblib.dump(test_indices, os.path.join(OUTPUT_ARTIFACTS_DIR, 'test_indices.joblib'))
    
    with open(os.path.join(OUTPUT_ARTIFACTS_DIR, 'num_topics.txt'), 'w') as f:
        f.write(str(OPTIMAL_K))
        

if __name__ == '__main__':
    train_and_save_lda_artifacts()