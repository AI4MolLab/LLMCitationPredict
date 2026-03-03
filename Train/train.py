#train.py
import os
import logging
from transformers import TrainingArguments, Trainer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
from datetime import datetime 
import random
import torch
import config
from data_loader import load_and_prepare_data
from model_setup import setup_model_and_tokenizer

def set_seed(seed_value):

    """Set random seeds for reproducibility across libraries"""
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_file_path):

    """Configure logging to file and console"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging setup complete. Log file at: {log_file_path}")

def compute_metrics(eval_pred):

    """Calculate regression and ranking metrics for model evaluation
    
    Args:
        eval_pred (tuple): Tuple containing (predictions, labels)
        
    Returns:
        dict: Dictionary of evaluation metrics
    """

    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Standard regression metrics
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    
    # Correlation coefficients 
    pearson_corr, _ = pearsonr(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)
    
    # NDCG 
    true_relevance = np.asarray([labels])
    prediction_scores = np.asarray([predictions])

    ndcg_at_5 = ndcg_score(true_relevance, prediction_scores, k=5)
    ndcg_at_10 = ndcg_score(true_relevance, prediction_scores, k=10)
    ndcg_at_15 = ndcg_score(true_relevance, prediction_scores, k=15)
    ndcg_at_20 = ndcg_score(true_relevance, prediction_scores, k=20)
    ndcg_full = ndcg_score(true_relevance, prediction_scores, k=None)

    return {
        "log_mae": mae,
        "log_mse": mse,
        "log_r2": r2,
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
        "ndcg_at_5": ndcg_at_5,
        "ndcg_at_10": ndcg_at_10,
        "ndcg_at_15": ndcg_at_15,
        "ndcg_at_20": ndcg_at_20,
        "ndcg_full": ndcg_full
    }
    
def run_training():

    """Main function to execute model training pipeline"""

    set_seed(config.SEED)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(config.OUTPUT_DIR, f"run_{run_timestamp}")
    
    os.makedirs(run_output_dir, exist_ok=True)
    log_file = os.path.join(run_output_dir, "training_log.log")
    setup_logging(log_file)


    logging.info(f"--- Starting Llama-3 Fine-tuning for Citation Prediction ---")
    logging.info(f"Results for this run will be saved in: {run_output_dir}")

    logging.info("Configuration Parameters:")
    for key, value in vars(config).items():
        if not key.startswith('__'):
            logging.info(f"{key}: {value}")
    

    lora_params = { 'LORA_R': config.LORA_R, 'LORA_ALPHA': config.LORA_ALPHA, 'LORA_DROPOUT': config.LORA_DROPOUT, 'LORA_TARGET_MODULES': config.LORA_TARGET_MODULES }
    model, tokenizer = setup_model_and_tokenizer(config.MODEL_PATH, lora_params)
    tokenized_datasets = load_and_prepare_data(tokenizer)
    train_dataset, eval_dataset = tokenized_datasets['train'], tokenized_datasets['eval']
    logging.info(f"Training data size: {len(train_dataset)}")
    logging.info(f"Evaluation data size: {len(eval_dataset)}")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=run_output_dir, 
        seed=config.SEED,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,

        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="epoch", 
        save_total_limit=2, 

        logging_steps=50,
        load_best_model_at_end=True, 
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL, 
        greater_is_better=config.GREATER_IS_BETTER,       
        bf16=True,
        report_to="none",
    )

    # Initialize Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logging.info("--- Starting Training ---")
    trainer.train()
    logging.info("--- Training Finished ---")

    final_model_path = os.path.join(run_output_dir, "final_best_model")
    trainer.save_model(final_model_path)

    logging.info(f"Best model saved to {final_model_path}")
    tokenizer.save_pretrained(final_model_path)
    logging.info(f"Tokenizer saved to {final_model_path}")

    logging.info("--- Final Evaluation on Eval Set ---")
    final_metrics = trainer.evaluate()
    
    logging.info("Final Evaluation Metrics:")
    for key, value in final_metrics.items():
        logging.info(f"{key}: {value}")
        
    logging.info("--- Process Complete ---")