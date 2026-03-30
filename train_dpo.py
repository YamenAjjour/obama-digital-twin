import os
import pandas as pd
import torch
import numpy as np
import evaluate
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOTrainer, DPOConfig
import mlflow

# Configuration
# Note: Using Qwen2.5 as a proxy for Qwen3. Replace with specific model ID if available.
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "qwen_dpo_lora_output"
SPEECH_DATASET_FILE = "speech_prompts.csv"
QUESTION_DATASET_FILE ="turns_prompts.csv"

# Load BERTScore metric
bertscore = evaluate.load("bertscore")

def train_dpo():
    # Setup MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("obama-dpo-training")

    # 1. Load and Format Dataset
    if not os.path.exists(SPEECH_DATASET_FILE) or not os.path.exists(QUESTION_DATASET_FILE):
        raise FileNotFoundError(f"Datasets not found. Please run generate_alignement_dataset.py first.")

    df_speeches = pd.read_csv(SPEECH_DATASET_FILE)
    df_speeches = df_speeches.dropna(subset=["prompt", "speech", "wrong_speech"])
    
    df_turns = pd.read_csv(QUESTION_DATASET_FILE)
    df_turns = df_turns.dropna(subset=["question", "answer", "wrong_answer"])
    
    # Standardize column names for concatenation
    df_speeches = df_speeches.rename(columns={"speech": "chosen", "wrong_speech": "rejected"})
    df_turns = df_turns.rename(columns={"question": "prompt", "answer": "chosen", "wrong_answer": "rejected"})

    # Combine both datasets
    df_combined = pd.concat([
        df_speeches[["prompt", "chosen", "rejected"]],
        df_turns[["prompt", "chosen", "rejected"]]
    ], ignore_index=True)

    full_dataset = Dataset.from_pandas(df_combined)

    # Split dataset into train and test
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # 2. Quantization Configuration (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "eager"
    )
    
    # Prepare model for k-bit training (enables gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            print("reduced")
        print(preds)
        print(labels)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        decoded_preds = tokenizer.decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)

        result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        
        return {
            "bertscore_f1": np.mean(result["f1"]),
            "bertscore_precision": np.mean(result["precision"]),
            "bertscore_recall": np.mean(result["recall"])
        }

    # 5. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 6. Training Arguments
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,  # DPO typically requires a lower learning rate
        num_train_epochs=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        save_steps=50,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        warmup_ratio=0.1,
        report_to="mlflow",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_length=2048,
        max_prompt_length=512
    )

    # 7. Initialize DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # 8. Train and Save
    print("Starting DPO training...")
    dpo_trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    dpo_trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    train_dpo()