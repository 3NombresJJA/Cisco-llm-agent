from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch

# ================================
# 1) Cargar modelo base + LoRA
# ================================
model_base = "./models/llama32-3b"
model_path = "./models/llama32-router-lora"   # tu LoRA ya entrenado

tokenizer = AutoTokenizer.from_pretrained(model_base)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(model_base, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(model, model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
    return_full_text = False
)

llm = HuggingFacePipeline(pipeline=pipe)


# ================================
# 2) Prompt sin que se muestre en salida
# ================================
prompt = PromptTemplate(
    input_variables=["instruction"],
    template=(
        "You are a Cisco IOS-XE and enterprise networking specialist.\n"
        "Answer directly, clearly and only with the configuration/result.\n\n"
        "Instruction:\n{instruction}\n\nResponse:"
    )
)

chain = prompt | llm


# ================================
# 3) Loop interactivo
# ================================
while True:
    q = input("\n Pregunta →  ")

    resp = chain.invoke({"instruction": q})  # ← CORRECTO
    print("\n RESPUESTA DEL MODELO:\n")

    # Mostrar solo la respuesta limpia
    if isinstance(resp, dict) and "text" in resp:
        print(resp["text"].strip())
    else:
        print(resp)
