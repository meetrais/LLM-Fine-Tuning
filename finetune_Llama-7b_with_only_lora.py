import transformers  as transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from transformers import BitsAndBytesConfig
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset

model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

config  = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias = 'none',
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"]
)


model = get_peft_model(model, config)
#print_trainable_parameters(model)

total_parameters = 0
for name, param in model.named_parameters():
    total_parameters += param.numel()

print(f"Total parameters: {total_parameters}")

# Freeze the non-Lora parameters
for name, param in model.named_parameters():
    if 'lora' not in name:
        print(f'Freezing non-LoRA parameter {name}')
        param.requires_grad = False

#Training
training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 2,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 1,
    optim = "paged_adamw_8bit",
    save_steps= 100,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= True,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "constant"
)
# Setting sft parameters
dataset = load_dataset("imdb", split="train")
trainer = SFTTrainer(
    train_dataset=dataset,
    model=model,
    max_seq_length= 20,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

model.config.use_cache = False
trainer.train()

total_parameters = 0
for name, param in model.named_parameters():
    total_parameters += param.numel()

print(f"Total parameters after Freeze: {total_parameters}")
