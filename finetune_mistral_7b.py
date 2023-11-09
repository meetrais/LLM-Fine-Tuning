from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes
import torch

model_id = "mistralai/Mistral-7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto", bnb_4bit_compute_dtype=torch.float16 )
tokenizer = AutoTokenizer.from_pretrained(model_id)

#print(model)

text = "Capital of USA is"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))