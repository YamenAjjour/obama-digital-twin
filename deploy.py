import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel
from threading import Thread

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "qwen_dpo_lora_output"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

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
    
    streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # Generate in background thread
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    
    # Yield output as it streams
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

demo = gr.ChatInterface(
    predict,
    title="Obama Digital Twin",
    description="Chat with the fine-tuned Barack Obama model.",
    examples=["What is your vision for the future?", "Can you tell me about the importance of education?"]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")