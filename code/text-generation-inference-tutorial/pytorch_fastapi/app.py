import time
import json
import requests
import gradio as gr

title = "RecoGPT"
llm_url = "http://localhost:8000/generate_stream"

examples = [
    ["Q: Who are you? A:"],
    ["How old are you?"],
]


def inference(prompt, 
              max_length=100, 
              min_length=20, 
              temperature=0.9,
              repetition_penalty=10, 
              top_k=10, 
              top_p=0.9):
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

    res = ""
    for chunk in response.iter_content(chunk_size=64):
            if chunk:
                res += str(chunk, encoding="utf-8")
                yield res

demo = gr.Interface(
    fn=inference,
    inputs=[gr.Textbox(lines=3, max_lines=5, label="Input Text"),
            gr.Slider(value=100, minimum=20, maximum=1000, step=10, label="max_length"), 
            gr.Slider(value=20, minimum=20, maximum=1000, step=10, label="min_length"), 
            gr.Slider(value=0.9, minimum=0, maximum=100, step=0.1, label="temparature")
            ],
    title=title,
    examples=examples,
    outputs=gr.Textbox(label="Output Text",lines=10)
)

if __name__ == "__main__":
    demo.queue(concurrency_count=16).launch(height=800, debug=True, server_port=9092)