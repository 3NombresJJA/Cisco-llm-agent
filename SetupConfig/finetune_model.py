from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import torch

model_name = "./models/llama32-3b"

### 1. Cargar dataset
dataset = load_dataset("json", data_files="knowledge/dataset.jsonl")

### 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 3. Tokenización 
def tokenize(sample): 
    text = f"### Instruction:\n{sample['instruction']}\n### Input:\n{sample['input']}\n### Output:\n{sample['output']}" 
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=512) 
# Añadir labels = input_ids para calcular loss 
    enc["labels"] = enc["input_ids"].copy() 
    return enc 

dataset = dataset.map(tokenize).remove_columns(["instruction", "input", "output"])

### 3. Configurar entrenamiento en LoRA REAL 4-bit
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
model = get_peft_model(model, lora)

### 4. Training
training_args = TrainingArguments(
    output_dir="./models/llama32-router-lora",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=8e-5,
    fp16=True,
    logging_steps=20,
    save_steps=200,
    save_total_limit=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()

###  importante → guardar adaptador LoRA final
model.save_pretrained("./models/llama32-router-lora")
tokenizer.save_pretrained("./models/llama32-router-lora")


