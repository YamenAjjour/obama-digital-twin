import argparse
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "qwen_dpo_lora_output"

parser = argparse.ArgumentParser(description="Deploy Obama Digital Twin")
parser.add_argument("--simple", action="store_true", help="Load base model without quantization and adapter")
args = parser.parse_args()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if args.simple:
    print("Loading base model in full precision...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True
    )
    print("Base model loaded. 1")
    model.eval()
    print("Base model loaded. 2")
    title = "Obama Digital Twin (Base Model)"
    description = "Chat with the base Qwen2.5-7B-Instruct model (no fine-tuning)."
else:
    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    title = "Obama Digital Twin (DPO)"
    description = "Chat with the DPO fine-tuned Barack Obama model."

def predict(message, history):
    # Initialize messages with system prompt
    messages = [{"role": "system", "content": "You are Barack Obama. Speak and respond in his tone and style."}]
    
    # Add history
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
        
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the response synchronously
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # Extract only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    
    # Decode the response
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response

demo = gr.ChatInterface(
    predict,
    title=title,
    description=description,
    examples=["What is your vision for the future?", "Can you tell me about the importance of education?"]
)

if __name__ == "__main__":
    print("Lunching demo")
    demo.launch(server_name="0.0.0.0")
