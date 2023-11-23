import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

peft_model_id = "meetrais/finetuned-neural-chat-7b-v3-1"
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(peft_model_id,  quantization_config=bnb_config, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
text = "Capital of USA is"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))