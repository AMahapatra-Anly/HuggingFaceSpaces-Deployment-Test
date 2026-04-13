
import gradio as gr
from transformers import pipeline

# Load model
generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt, temperature, max_length):
    if not prompt:
        return "Please enter a prompt."

    result = generator(
        prompt,
        max_new_tokens=int(max_length),
        temperature=temperature,
        do_sample=True,
        pad_token_id=50256
    )
    return result[0]['generated_text']

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt..."),
        gr.Slider(0.1, 1.5, value=0.7, label="Temperature"),
        gr.Slider(20, 200, value=50, label="Max Tokens")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Text Generator",
    description="Generate text with GPT-2"
)

demo.launch()
