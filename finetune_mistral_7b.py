import transformers  as transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from transformers import BitsAndBytesConfig
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset

model_id = "mistralai/Mistral-7B-v0.1"
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
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""
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
dataset_name = "gathnex/Gath_baize"
dataset = load_dataset(dataset_name, split="train[:1000]")
dataset["chat_sample"][0]

def merge_colunms(example):
    example['prediction'] = example['quote'] + " ->: " + str(example["tags"])
    return example


#data['train'] = data['train'].map(merge_colunms)
#print(data['train'][0])

#data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)

#print(data)

#Training
training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 10,
    per_device_train_batch_size= 8,
    gradient_accumulation_steps= 2,
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
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length= 20,
    dataset_text_field="chat_sample",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

model.config.use_cache = False
trainer.train()
"""
config  = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias = 'none',
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.push_to_hub("meetrais/finetuned_mistral_7b",
                  token="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",
                  commit_message="basic training",
                  private=True)


