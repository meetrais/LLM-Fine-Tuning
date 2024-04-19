import transformers
import torch
import gradio as gr

device =  "cuda:0"
# Function to run the text generation process
def run_generation(user_text, top_p, temperature, top_k, max_new_tokens):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    device =  "cuda:0"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    
    messages = [
    {"role": "system", "content": "You are a helpfull assistant."},
    {"role": "user", "content": user_text},
]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    return outputs[0]["generated_text"][len(prompt):]

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