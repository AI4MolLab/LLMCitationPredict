# config.py
MODEL_PATH = "/beagle/llama3-8b/model"
DATA_PATH = "/beagle/llama3-8b/data/10000/metadata_10000_with_similarity.csv" 
OUTPUT_DIR = "/beagle/llama3-8b/results/10000"
OUTPUT_ARTIFACTS_DIR = "/beagle/llama3-8b/data/10000/topic_model_artifacts"

DATA_COLUMNS = {
    'title': 'title',
    'abstract': 'abstract',
    'target': 'citations_within_3_years',
    'year': 'publicationYear',
}
TEXT_FEATURES = ['title', 'abstract']
TARGET_COLUMN_KEY = 'target'

# Prompt
PROMPT_base = """[Paper Content]
Title: {title}
Abstract: {abstract}

[Prediction Task]
Please provide the predicted number of citations the paper will receive within 3 years.
Predicted 3-Year Citations:"""

PROMPT_full = """[Paper Content]
Title: {title}
Abstract: {abstract}

[Topic Analysis]
Topical Breadth: {diversity_soft} (Value: {diversity_raw:.2f})
Primary Topic Rank: {topic_rank_soft} (Rank: {topic_rank_raw:.2f})

[Prediction Task]
Please provide the predicted number of citations the paper will receive within 3 years.
Predicted 3-Year Citations:"""

PROMPT_rank = """[Paper Content]
Title: {title}
Abstract: {abstract}

[Topic Analysis]
Primary Topic Rank: {topic_rank_soft} (Rank: {topic_rank_raw:.2f})

[Prediction Task]
Please provide the predicted number of citations the paper will receive within 3 years.
Predicted 3-Year Citations:"""

PROMPT_diversity = """[Paper Content]
Title: {title}
Abstract: {abstract}

[Topic Analysis]
Topical Breadth: {diversity_soft} (Value: {diversity_raw:.2f})

[Prediction Task]
Please provide the predicted number of citations the paper will receive within 3 years.
Predicted 3-Year Citations:"""



PROMPT_TEMPLATE = PROMPT_base
MAX_LENGTH = 1024 
LORA_R = 16 
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01

METRIC_FOR_BEST_MODEL = "eval_log_mse"
GREATER_IS_BETTER = False 

SEED = 42
