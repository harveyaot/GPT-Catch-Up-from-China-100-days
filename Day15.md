## Goal: Understand LLama2 everything

### Starting points
1. about the [LLaMa2](https://arxiv.org/pdf/2307.09288.pdf)
2. about the [Chinese LLaMa2 Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
3. about the [LLaMa2](https://github.com/FlagAlpha/Llama2-Chinese)

### Steps
- read the paper
  - Not well understanded technical terms:
    - gradient clipping
    - weight decay
    - zero out 
    - Rjection Sampling
  -  Iterative Fine-Tuning (newly proposed RL process in LLaMa2)
- About the Chinese LLaMa2, the context length is from 4K -> 8K, used the NTK method, referred from
  - Using the PI method (postition Interpolation)
- Check the pretraining scripts:
  - using deepspeed, using PEFT
  - check the BERT large training
  - when encounter large model using torch run, and hugging face accelerate, with device_map auto settings, it can help you to decide which device to place the parameters.


### Questions:

1. why LLama2 choose Alibi?
2. how to do the pre-training?
3. what's the LLama2 Chinese Alpaca



### Reference
- [TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION](https://arxiv.org/pdf/2108.12409.pdf)
- [NTK-Aware Scaled RoPE ](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/?rdt=60494)
- [EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION](https://arxiv.org/pdf/2306.15595.pdf)
- [Chinese LLaMa2 Alpaca related training py script](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/blob/main/scripts/training/run_clm_pt_with_peft.py#L253)

