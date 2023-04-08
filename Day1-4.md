## Where to start?
 - Theory: so many articles/papers discussing the success of ChatGPT, it seems the most key points:
 - A enough big pre-trained language model, which is now LLAMA
 - High-quality label data for SFT process, which is Alpaca from stanford and China's side efforts from 
 - The RLHF (PPO),  is not quite needed from the startpoint, but definatelly usueful, when the data flywheel is full-speed running


## 4-days Plan

- Day1: Get LLama, from open-web, since I didn't get the model from application process.
- Day2: Run LLama


## Details

 - **Get LLaMA**
    - I am using windows, prepare the docker for ubuntu latest
    - 🚀has successfully download LLaMA model wights from ref5
- **Understand LLaMA**
    - Understand the LLaMA model from ref6
    - 🚀Revisit some basic transforer techs [Harvard NLP](https://nlp.seas.harvard.edu/2018/04/03/attention.html), learnings:
        - one learning is pe is not learned and updated during training, but fixed, only has relations with token postiion and embedding'e element postion. 
        - in multi-head attension, the extra parameters are only d_model * d_model * 4, nothing related with the heads number
    - 🚀Revisit LLM training techniques.
         - PaLM didn't mension anything else how the LM loss function, which means it's the traditional loss function.
- **Run LLaMA**
    - :rocket: run llama using llama.cpp succcessfully through github doc
    - :cherries: research on llama.cpp converting script
    - research on the parameters compare GPT2 VS Bert VS LLaMa the differences are:
        - :cherries: Feed forward: fc LLama implementation is `self.w2(F.silu(self.w1(x)) * self.w3(x))`
        - :cherries: Layer norm: using RMSNorm, that's why only has 4096 params in LN.
    
- **Finetune LLaMA**
    - :cherries: Read the Alpaca paper and document. [`train.py`](https://github.com/tatsu-lab/stanford_alpaca/blob/aa65c492bb788e144712daab42bc5d11c2761591/train.py#L32)
    - :cherries: Read [LoRa](https://arxiv.org/pdf/2106.09685.pdf)
    - :cherries: Read [Prefix-Tuning](https://arxiv.org/pdf/2106.09685.pdf)
- **Demo LLaMA**
    - how to productlize LLaMA

## Concepts
- BPE(Byte Pair Encoding) [hugging face](https://huggingface.co/course/chapter6/5?fw=pt)
- RMSNorm(Root Mean Square Layer Normalization) [Github](https://github.com/bzhangGo/rmsnorm); Layer Normalization VS Batch Norm
- SwiGLU
- Rotary Embeddings

## References
1. [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
2. [Gpt4all](https://github.com/search?q=Gpt4all)
3. [Chinese LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) from iFLYTEK
4. [Chinese LLaMA-Luotuo](https://github.com/harveyaot/Chinese-alpaca-lora) from Sensetime
5. [fastLLaMA](https://github.com/spv420/fastLLaMA)
6. [LLaMA](https://arxiv.org/pdf/2302.13971v1.pdf)


    
