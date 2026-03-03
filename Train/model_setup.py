# model_setup.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

def setup_model_and_tokenizer(model_path, lora_config_params):

    """
    Args:
        model_path (str): Local path to the model.
        lora_config_params (dict): Dictionary of LoRA configuration parameters.

    Returns:
        tuple: (peft_model, tokenizer)
    """

    print(f"Loading base model and tokenizer from {model_path}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1, # Regression task
        problem_type="regression",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto" 
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Setting up PEFT LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=lora_config_params['LORA_R'],
        lora_alpha=lora_config_params['LORA_ALPHA'],
        lora_dropout=lora_config_params['LORA_DROPOUT'],
        target_modules=lora_config_params['LORA_TARGET_MODULES'],
        bias="none",
    )

    # Apply LoRA adapter to the base model
    peft_model = get_peft_model(model, peft_config)

    print("PEFT model created. Trainable parameters:")
    peft_model.print_trainable_parameters()

    return peft_model, tokenizer