import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "What is Capital of USA?"
inputs = tokenizer(text, return_tensors="pt").to(0)

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", device_map="cuda", trust_remote_code=True)
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
