import os
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig

# Configuration
# Note: Using Qwen2.5 as a proxy for Qwen3. Replace with specific model ID if available.
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "qwen_dpo_lora_output"
DATASET_FILE = "speech_prompts.csv"

def train_dpo():
    # 1. Load and Format Dataset
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset {DATASET_FILE} not found. Please run generate_alignement_dataset.py first.")

    df = pd.read_csv(DATASET_FILE)
    
    # Drop any rows with missing values to prevent errors
    df = df.dropna(subset=["prompt", "speech", "wrong_speech"])

    # Map columns to DPO format: prompt, chosen, rejected
    dataset = Dataset.from_dict({
        "prompt": df["prompt"].tolist(),
        "chosen": df["speech"].tolist(),
        "rejected": df["wrong_speech"].tolist(),
    })

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
        gradient_accumulation_steps=8,
        learning_rate=5e-6,  # DPO typically requires a lower learning rate
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        warmup_ratio=0.1,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_legnth=2048,
        max_prompt_length=512
    )

    # 7. Initialize DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 8. Train and Save
    print("Starting DPO training...")
    dpo_trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    dpo_trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    train_dpo()