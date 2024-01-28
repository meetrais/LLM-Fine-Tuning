import transformers  as transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from transformers import BitsAndBytesConfig
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset

model_id = "NousResearch/Nous-Hermes-2-SOLAR-10.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
text = "Capital of USA is"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

config  = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias = 'none',
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"]
)

#print(model)

model = get_peft_model(model, config)
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""
model.push_to_hub("meetrais/finetuned-neural-chat-7b-v3-1",
                  token="Your-Hugging-Face-Token-Here",
                  commit_message="basic training",
                  private=True)
"""


