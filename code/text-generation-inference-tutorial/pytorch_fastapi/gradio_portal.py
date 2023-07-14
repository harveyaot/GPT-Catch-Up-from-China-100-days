import json
import requests
import gradio as gr
import multiprocessing

multiprocessing.set_start_method('spawn', True)

llm_url = "http://localhost:8080/generate_stream"

def inference(prompt, 
              max_length=100, 
              min_length=20, 
              repetition_penalty=10, 
              top_k=10, 
              top_p=0.9,
              temperature=0.9):
    d = {
        "prompt": prompt,
        "max_length": max_length,
        "min_length": min_length,
        "repetition_penalty":repetition_penalty,
        "do_sample": False,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature
    }

    response = requests.post(
        llm_url,
        stream=True,
        headers={"accept": "application/json"},
        data= json.dumps(d)
    )

    for chunk in response.iter_content(chunk_size=64):
            if chunk:
                yield str(chunk, encoding="utf-8")

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=inference, 
                    inputs=['text', 'number','number','number',
                            'number', 'number', 'number'],
                    outputs="text")
demo.queue()
demo.launch()   