import os
import torch
import fastapi 
import transformers

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import pipeline, LlamaTokenizer, LlamaForCausalLM, TextIteratorStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from threading import Thread

#a fastapi server
app = fastapi.FastAPI()
cache_dir = '/data'

# reading from the os environment
model_path = os.environ.get('MODEL_PATH', 'openlm-research/open_llama_3b')
if 'llama' in model_path:
    tokenizer = LlamaTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        load_in_4bit=True
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
        load_in_4bit=True
    )

device = 'cuda'
# enabling cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
)

class Request(BaseModel):
    prompt: str
    max_length: int = 100
    min_length: int = 20
    do_sample: bool = False
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 0.9
    repetition_penalty: float = 1.0


@app.get("/")
def hello():
    return "Hello World"

# defining the root endpoint
@app.post("/generate")
def generate(request: Request):
    kwargs = request.dict()
    kwargs.pop('prompt')

    inputs = tokenizer(request.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = dict(inputs, **kwargs)

    return model.generate(**generation_kwargs)


def _generate_stream(request):
    kwargs = request.dict()
    kwargs.pop('prompt')

    inputs = tokenizer(request.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    ts = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(inputs, **kwargs, streamer=ts)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for chunk in ts:
        yield chunk

@app.post("/generate_stream", response_model=str)
def generate_stream(request: Request):
    return StreamingResponse(_generate_stream(request), media_type="text/event-stream")