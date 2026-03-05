# LLMCitationPredict
Early Prediction of Academic Impact via Large Language Models Enhanced by Structured Topic Prompting.

The entire pipeline is divided into two main phases:
- **Topic Modeling (LDA):** Extracts structured topic features (primary topic rank and topical breadth/diversity) from paper titles and abstracts.
- **LLM Fine-Tuning (LoRA):** Trains a quantized Llama 3 sequence classification model to predict citation counts using both the raw text and the extracted topic features.

## Project Structure

### Phase 1: LDA Topic Modeling

- `find_optimal_k.py`: Performs a grid search by calculating the cvc_v coherence score to find the optimal number of topics (kk) and generates a coherence score trend plot.
- `train_topic_model-fit.py`: Trains the final LDA model using the optimal kk value. This script processes academic texts, applies Bigram phrase extraction, calculates topic ranks and diversity bins, and saves all artifacts (model, vectorizer, dataset split indices).

### Phase 2: Llama 3 Fine-Tuning

- `config.py`: Contains all configuration parameters, including file paths, LoRA hyperparameters, and various Prompt templates (Base, Full, Rank-only, Diversity-only).
- `dataloader.py`: Loads raw data and pre-trained LDA artifacts. Generates topic features for the Prompts and applies a `log1p` transformation to the target citation variable.
- `model_setup.py`: Configures the base Llama 3 regression model using 4-bit quantization (BitsAndBytes) and applies the PEFT/LoRA adapter.
- `train.py`: The core training script utilizing the Hugging Face `Trainer`. It logs the training process and calculates standard regression metrics (MAE, MSE, R^2) as well as ranking metrics (Pearson, Spearman, NDCG).
- `main.py`: The entry point script to execute the fine-tuning pipeline.

## Dependencies
* **Python:** 3.11.2
* `requirements.txt`

## Usage

**1. Determine the optimal number of topics**

```bash
python find_optimal_k.py
```

##### **2. Train the LDA model**

```bash
python train_topic_model-fit.py
```

**3. Fine-tune the LLM**

```bash
python main.py
```

## Author
- Hengzhi Huang
- Ziyang Wang
- Yiheng Zhao
- Zhichao Pan(panzhichao@guet.edu.cn)

