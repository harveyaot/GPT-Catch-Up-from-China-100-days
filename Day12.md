## Goal

- Targeting demo and production usage of inference API
- New knowledge about MOE and Microsoft 1B token length
- What is INT4 Quantification
- What is FalshAttension?

## Steps

- Inference API from huggingface, which is fast for demo usage, but short for:
    - limitation on the request tokens
    - not support on the streaming mode

- Run hugging face text-generation-inference from local mahine
- Use the matual AI solution, 
- Deploy using endpoints, but seems not successful.
- Try to understand the ChatGLM2-6B version, and understand how they do the int and deployments.
- check the [chatgpt-next-web project](https://github.com/Yidadaa/ChatGPT-Next-Web) 
- still use the TGI, but find a machine for this purpose.
-  check ali machine? or use a azure machine first
    - use a cheap cpu machine for deployment purpose
    - then switch a GPU machine
    - understand the purpose 
    - install docker
    - 

- research about the model [transformer deployment](https://github.com/ELS-RD/transformer-deploy)

## Showing Demos

- use open-lamma-3B,
- how to demo this with chatbot?

## thinking
- what we must have?
    - deploy it in a GPU?
        - try this still in azure first, with hourly usage.
        - then anable the CI/CD solution for this.
        - has tried that deploying using hugging face endpoints
        - Then a new Gradio app.
        - Why not try the huggingface solution driectly?
    - must enable the streaming mode.
- From practical steps:
    - what I should have for now?
        - using hugging faces for models
        - directly load the models and show the models
        - then loading the models.
        