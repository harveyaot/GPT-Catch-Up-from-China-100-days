## Goals

* Go throught the document [State of GPT](https://mp.weixin.qq.com/s/zmEGzm1cdXupNoqZ65h7yg) and figure out the key problems.
* Make the pratical pretrain experiment plan for a personal environmnent.
* Understand the SOTA Stable diffusion model.
* How PPO integrated with the LLM training process.
* Try and successfull use Midjourney
* Google's Palm architechture 

##  Details

* Stat of GPT
    *  Ask for relection, LLMs can often recoginize later when their samples didn't seem to have worked out well
    * LLMs don't want to succeed??? If you want to scucceed, and you should ask for it.


* Understand the Stable Diffusion and VIT
    * An easy essey to understand the whole process, [How does Stable Diffusion work?](https://stable-diffusion-art.com/how-stable-diffusion-work/)
    * forward diffusion process gradually adds Gaussian noise to the input image x₀ step 
    by step, a reference about adding normal [numpy random normal](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html)
    * ViT, Vision Transformer, from the ref 2
    * Understand U-Net

* Google's Model
    * [PaLM](https://arxiv.org/abs/2204.02311v5) and PaLM2 are using decoder only setup

* UnderStand the PPO
    * Still have a lot of questions

* The Pretrain Plan
    * Ramp up the hugging face 
    * Train an causul language model from scratch. 
    :rocket: [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/en/chapter1/1)
    * reading the Paper T5 and BART

* Understanding the Background
    * Why BART is naming Denoising? Because they use many noising approches in generating training data and tasks.
    * The T5 model proposing after BART

## Citations

1. [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/pdf/2211.01910.pdf)
2. [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)
3. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
](https://arxiv.org/pdf/1910.10683v3.pdf)
4. [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf)
5. 


## Concept
