# Day 6-9

## Goal
- Pretarin a model on single card and multiple card
- Publish Model in China
- Using RL and RW to finetune a model
- Data: collect chinese data to retrain the model
- A product demo after 3 days 
## Chain of Thoughts

- What a product looks like?
    - collect ideas
        - :cherries: a chinese peom boat
        - :cherries: a lay related bot, if doing this violating laws or not?
        - :cherries: just a bot to write the next paragragh of an given article
    - use which foudnation model
    - starting from a very small sceanrio. 
    - what data I have?
    - What the finetune proces I can afford?
    - Then scrape GPT-4 for the labeling data then translate it into another style

## Details
- What a product goal to achive?
    - A chinese peom boat or writhing things more clearly?
- Pretrain a lanugage model on single card and multiple GPU cards
    - :cherries: using GPT-2 as the start point
    - :cherries: using what data to train
## Progress
### Task

- A technique roadmap
    - finetune on single card.
        - :cherries: ues GPT2-Large
        - :cherries: find some open datasets, using beyond/chinese_clean_passages_80m
        - :rocket: Locally has run successfully
        - ❓how doese the attension_mask used? [Doc1](https://lukesalamone.github.io/posts/what-are-attention-masks/) [Doc2]()
        - ❓how the gpu parameters calculation (why in reality, using gpt-meidum, which occupies 4G GPU? what the gradient accumulation works for?)
              - []
              - [Efficient Training on a Single GPU](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one)
   
    - How to extend tokenization
        - :cherries: a guidline [Training a new tokenizer from an old one](https://huggingface.co/learn/nlp-course/chapter6/2)
    - findtune on multi cards (hf, huoshan)
    - Train and extend Tokenizer
    - pretrain from scratch (hf, huoshan)
    - collect a datasets on chinese task
    - finetune to resolve a task

- Step 1:
    - overall using hugging face to do this. 
    - warmup hugging face with the models and datasets, 

- Step 2:
    - warm up with Hugging face library and train on the local machine
    - use a model card, download
    - :rocket: use a dataset then start the locally train

- Step 3:
    - After the above 2 steps, then think a about a problem to resove
    - prepare a datasets.
    - fintune the LLM, who is good enought, but can fit into 1 12GB card.
    - Cook a dataset, or just find a dataset, start pretraining.




## Knowledge

- about Zhipu AI and the ChatGLM-6B 

## Citation

- [GLM: General Language Model Pretraining
with Autoregressive Blank Infilling](https://arxiv.org/pdf/2103.10360.pdf)
- [Quantization- HF](https://huggingface.co/docs/optimum/concept_guides/quantization)

