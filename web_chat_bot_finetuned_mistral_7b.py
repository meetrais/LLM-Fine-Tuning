# Import necessary libraries
from threading import Thread
import argparse
import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel, PeftConfig
from transformers import BitsAndBytesConfig

peft_model_id = "meetrais/finetuned_mistral_7b"
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

usingAdapter = True

device =  "cuda:0"
# Function to run the text generation process
def run_generation(user_text, top_p, temperature, top_k, max_new_tokens):
    #template = "### Text: {}\n### The tone is:\n"
    #model_inputs = tokenizer(template.format(user_text) if usingAdapter else user_text, return_tensors="pt")
    #model_inputs = model_inputs.to(device) 
    model_inputs= tokenizer(user_text, return_tensors="pt").to(device)

    # Generate text in a separate thread
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        **model_inputs,
        max_new_tokens=max_new_tokens, 
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    
    # Retrieve and yield the generated text
    model_output = ""
    for new_text in streamer:
        model_output += new_text
        yield model_output
    return model_output

# Gradio UI setup
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=4):
            user_text = gr.Textbox(placeholder="Write your question here", label="User input")
            model_output = gr.Textbox(label="Model output", lines=10, interactive=False)
            button_submit = gr.Button(value="Submit")

        with gr.Column(scale=1):
            max_new_tokens = gr.Slider(minimum=1, maximum=1000, value=250, step=1, label="Max New Tokens")
            top_p = gr.Slider(minimum=0.05, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
            top_k = gr.Slider(minimum=1, maximum=50, value=50, step=1, label="Top-k")
            temperature = gr.Slider(minimum=0.1, maximum=5.0, value=0.8, step=0.1, label="Temperature")

    user_text.submit(run_generation, [user_text, top_p, temperature, top_k, max_new_tokens], model_output)
    button_submit.click(run_generation, [user_text, top_p, temperature, top_k, max_new_tokens], model_output)

    demo.queue(max_size=32).launch(server_port=8082)
