#dataloader.py
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
import config
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text as sklearn_text
import logging

#Definition of stop words
ACADEMIC_STOP_WORDS = [
    'review', 'overview', 'summary', 'paper', 'article', 'work', 'research', 'study', 'studies',
    'experiment', 'experiments', 'experimental', 'development', 'developments', 'developed', 'developing',
    'present', 'presented', 'presents', 'role', 'roles', 'number', 'numbers', 'field', 'fields',
    'data', 'analysis', 'information', 'model', 'models', 'modeling', 'design', 'designs', 'designed',
    'performance', 'performances', 'application', 'applications', 'properties', 'property',
    'characteristics', 'characterization', 'structure', 'structures', 'structural', 'complex', 'complexes',
    'species', 'individual', 'group', 'groups', 'type', 'types',
    'new', 'novel', 'proposed', 'propose', 'proposes', 'single', 'breaking', 'break',
    'control', 'controlling', 'observed', 'observation', 'related', 'associated', 'potential',
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

#Custom tokenizer
def custom_tokenizer(text):
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

def get_custom_rank_labels(series, total_ranks):

    """Assign categorical rank labels based on numerical values
    
    Args:
        series (pd.Series): Numerical series to be ranked
        total_ranks (int): Total number of ranks for normalization
        
    Returns:
        pd.Series: Categorical rank labels
    """

    series = series.astype(float)
    bins = [0, total_ranks * 0.05, total_ranks * 0.20, total_ranks * 0.60, total_ranks]
    labels = ["Top Rank", "High Rank", "Medium Rank", "Low Rank"]
    return pd.cut(series, bins=bins, labels=labels, right=True, include_lowest=True)

#Calculate topic popularity and diversity features
def generate_features_for_df(df, lda, vectorizer, phraser, topic_rank_map, diversity_bins, num_topics):
    print(f"  - Generating features for {len(df)} samples...")
    title_col = config.DATA_COLUMNS['title']
    abstract_col = config.DATA_COLUMNS['abstract']
    
    raw_texts = (df[title_col].fillna('') + ' ' + df[abstract_col].fillna('')).tolist()
    
    print("    Processing: Tokenizing...")
    tokens = [custom_tokenizer(text) for text in raw_texts]
    
    print("    Processing: Applying Bigrams...")
    final_tokens = [phraser[doc] for doc in tokens]
    
    print("    Processing: Vectorizing...")
    X = vectorizer.transform(final_tokens)
    
    topic_distr = lda.transform(X)
    
    df['diversity_raw'] = -np.sum(topic_distr * np.log2(topic_distr + 1e-10), axis=1)
    df['topic_id'] = np.argmax(topic_distr, axis=1)
    
    median_rank = np.median(list(topic_rank_map.values()))
    df['topic_rank_raw'] = df['topic_id'].map(topic_rank_map).fillna(median_rank)
    
    df['diversity_soft'] = pd.cut(df['diversity_raw'], bins=diversity_bins, labels=["Narrow", "Medium", "Broad"], include_lowest=True, right=False)
    
    df['diversity_soft'] = df['diversity_soft'].astype(object).fillna("Broad")

    df['topic_rank_soft'] = get_custom_rank_labels(df['topic_rank_raw'], total_ranks=num_topics)
    
    return df

# Main Function
def load_and_prepare_data(tokenizer):

    """Main function to load, preprocess data and prepare tokenized datasets for model training
    
    Args:
        tokenizer (transformers.PreTrainedTokenizer): Hugging Face tokenizer
        
    Returns:
        datasets.DatasetDict: Tokenized train/eval datasets ready for training
    """

    artifacts_path = config.OUTPUT_ARTIFACTS_DIR
    print("Loading pre-trained topic model artifacts...")
    phraser = joblib.load(os.path.join(artifacts_path, 'bigram_phraser.joblib'))
    vectorizer = joblib.load(os.path.join(artifacts_path, 'vectorizer.joblib'))
    lda = joblib.load(os.path.join(artifacts_path, 'lda_model.joblib'))
    topic_rank_map = joblib.load(os.path.join(artifacts_path, 'topic_rank_map.joblib'))
    diversity_bins = joblib.load(os.path.join(artifacts_path, 'diversity_bins.joblib'))
    
    with open(os.path.join(artifacts_path, 'num_topics.txt'), 'r') as f:
        num_topics = int(f.read())

    print(f"Loading RAW data from {config.DATA_PATH}...")
    df = pd.read_csv(config.DATA_PATH)
    
    print("Loading pre-defined split indices (Ensuring consistency with LDA training)...")
    try:
        train_indices = joblib.load(os.path.join(artifacts_path, 'train_indices.joblib'))
        test_indices = joblib.load(os.path.join(artifacts_path, 'test_indices.joblib'))
    except FileNotFoundError:
        raise FileNotFoundError(f"Index files not found in {artifacts_path}. Please re-run train_topic_model-fit.py first.")

    print(f"Restoring exact split: Train={len(train_indices)}, Eval={len(test_indices)}")
    
    train_df = df.loc[train_indices].copy()
    eval_df = df.loc[test_indices].copy()

    print("Dynamically generating topic features for the TRAIN set...")
    train_df = generate_features_for_df(train_df, lda, vectorizer, phraser, topic_rank_map, diversity_bins, num_topics)
    
    print("Dynamically generating topic features for the EVAL set...")
    eval_df = generate_features_for_df(eval_df, lda, vectorizer, phraser, topic_rank_map, diversity_bins, num_topics)
    
    #Prompt assembly
    required_features = re.findall(r'\{(\w+)(?::.*?)?\}', config.PROMPT_TEMPLATE)
    calculated_features = ['num_topics']
    
    def create_input_text(row):
        """Create formatted input text using prompt template and row features"""
        feature_dict = {feature: row.get(feature, "") for feature in required_features if feature not in calculated_features}
        feature_dict['num_topics'] = num_topics
        
        if 'topic_rank_raw' in feature_dict and pd.isna(feature_dict.get('topic_rank_raw')):
             feature_dict['topic_rank_raw'] = "N/A"
             feature_dict['topic_rank_soft'] = "Unknown"
        
        try:
            return config.PROMPT_TEMPLATE.format(**feature_dict)
        except KeyError as e:
            print(f"Error formatting prompt: Missing key {e}")
            return "" 
    
    print("Constructing input text for all sets...")
    train_df['text'] = train_df.apply(create_input_text, axis=1)
    eval_df['text'] = eval_df.apply(create_input_text, axis=1)

    log_message = (
        f"\n{'='*50}\n"
        f"Sample of the FIRST data point (Llama-3 Format):\n"
        f"{'='*50}\n"
        f"{train_df['text'].iloc[0]}\n"
        f"{'='*50}"
    )
    logging.info(log_message)

    # Apply log transformation to predictions (log1p = log(x+1) to handle zero values)
    target_col_name = config.DATA_COLUMNS.get(config.TARGET_COLUMN_KEY)
    train_df['label'] = np.log1p(pd.to_numeric(train_df[target_col_name], errors='coerce').fillna(0)).astype(np.float32)
    eval_df['label'] = np.log1p(pd.to_numeric(eval_df[target_col_name], errors='coerce').fillna(0)).astype(np.float32)

    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'label']])
    tokenized_datasets = DatasetDict({'train': train_dataset, 'eval': eval_dataset})

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=config.MAX_LENGTH)

    print("Tokenizing datasets...")
    tokenized_datasets = tokenized_datasets.map(preprocess_function, batched=True, remove_columns=['text'])
    
    return tokenized_datasets