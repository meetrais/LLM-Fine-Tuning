import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers  as transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset


#Setup the model
model_id="meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)

print(model.get_memory_footprint())
#Freezing the original weights
for param in model.parameters():
    param.requires_grad = False
    if param.ndim ==1:
        param.data = param.data.to(torch.float32)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

#Setting up the LoRa Adapters
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    ) 

config  = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias = 'none',
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
data = load_dataset("Abirate/english_quotes")

def merge_colunms(example):
    example['prediction'] = example['quote'] + " ->: " + str(example["tags"])
    return example

data['train'] = data['train'].map(merge_colunms)
print(data['train']["prediction"][:5])
print(data['train'][0])

data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)

print(data)

#Training
trainer =  transformers.Trainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_gpu_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False
trainer.train()

model.push_to_hub("meetrais/meta-Llama-2-7b-hf-finetuned",
                  token="HuggingFace-app-key",
                  commit_message="basic training",
                  private=True)
