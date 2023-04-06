### Where to start?
    -  Theory: so many articles/papers discussing the success of ChatGPT, it seems the most key points:
        -   A enough big pre-trained language model, which is now LLAMA
        -   High-quality label data for SFT process, which is Alpaca from stanford and China's side efforts from 
        -   The RLHF (PPO),  is not quite needed from the startpoint, but definatelly usueful, when the data flywheel is full-speed running


### The Plan of 4 days.
    - Day1: 
        -   Get LLama, from open-web, since I didn't get the model from application process.


### Details:

    - Get LLAMA, try its ability.
        - I am using windows, prepare the docker for ubuntu latest
        - :rockt: has successfully download LLaMA model wights 
        - understand the LLaMA model from ref^6
        - revisit some basic transforer techs [Harvard NLP](https://nlp.seas.harvard.edu/2018/04/03/attention.html), learnings:
            1. one learning is pe is not learned and updated during training, but fixed, only 
            has relations with token postiion and embedding'e element postion. 
            2. in multi-head attension, the extra parameters are only d_model * d_model * 4, nothing related with the heads number
        - revisit LLM training techniques.
            1. GPT-3, PaLM didn't mension anything else how the LM loss function, which means it's the traditional loss function.
        - run 7B model
        - how to download LLaMA 7B, 13B, 65B firstly?
        - from git got an 
        - Alpaca is good starting point, reading the introduction from [1]
        - what it looks like? a Bin?
        - seems quite hard, if not get it directlly from 

### Concepts Warm Up
    -  BPE(Byte Pair Encoding)[hugging face](https://huggingface.co/course/chapter6/5?fw=pt)
    -  RMSNorm(Root Mean Square Layer Normalization) [Github](https://github.com/bzhangGo/rmsnorm); Layer Normalization VS Batch Norm
    -  SwiGLU
    -  Rotary Embeddings

### References
    - [1]: [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
    - [2]: [Gpt4all](https://github.com/search?q=Gpt4all)
    - [3]: [Chinese LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) from iFLYTEK
    - [4]: [Chinese LLaMA-Luotuo](https://github.com/harveyaot/Chinese-alpaca-lora) from Sensetime
    - [5]: [fastLLaMA](https://github.com/spv420/fastLLaMA)
    - [6]: [LLaMA](https://arxiv.org/pdf/2302.13971v1.pdf)


    
