# inference_base.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import sys
import config
from data_loader import load_and_prepare_data

def identity_tokenizer(text):
    return text

def run():
    # Configuration for model and output paths
    FOUNDATION_PATH = "/beagle/llama3-8b/model" 
    ADAPTER_PATH = "/beagle/llama3-8b/results/10000/full/final_best_model"
    CSV_NAME = "/beagle/Figure/Fig4/results_base_lora.csv"

    print(f"\n{'='*20} Starting Base Model Inference {'='*20}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        FOUNDATION_PATH,
        num_labels=1,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(FOUNDATION_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"✅ Loading Base LoRA Adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    print("Loading evaluation dataset...")
    tokenized_datasets = load_and_prepare_data(tokenizer)
    eval_dataset = tokenized_datasets['eval'] 
    
    try:
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    except ValueError:
        print("⚠️ Warning: Failed to set dataset column format. Please check Dataset column names.")

    dataloader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE * 2)

    true_labels, pred_labels = [], []
    print("Starting inference/prediction...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.squeeze(-1).float().cpu().numpy()
            
            true_labels.extend(labels)
            pred_labels.extend(preds)

    df = pd.DataFrame({'true_log_citations': true_labels, 'pred_log_citations': pred_labels})
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(config.OUTPUT_DIR, CSV_NAME)
    df.to_csv(save_path, index=False)
    print(f"✅ Base Model Inference Results Saved to: {save_path}")

if __name__ == "__main__":
    run()