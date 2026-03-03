#main.py
from train import run_training

def identity_tokenizer(text):
    return text

if __name__ == "__main__":
    print("Starting the fine-tuning process...")
    run_training()
    print("Fine-tuning process has finished. Check results in the output directory.")