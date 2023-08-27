## Goal
To understand below point:
1. PPO and TRPO, the fundermentals
2. How it trained on LLM
3. research on the RLHF benifits and how to use it


## Details

### About the PPO and TRPO:

1. The `Actor-Critic` is a combination of value-based, and policy-based methods where the Actor controls how our agent behaves using the Policy gradient, and the Critic evaluates how good the action taken by the Agent based on value-function.
2. Understatnd how ChatGPT trained a Rewards model and how the the RL learning objective and it support contious training for rewards and RL, but what's the connect with PPO?
3. reading the paper to understand the PPO process. how then connect with the loss fucntion mension.
4. how the really code looks like in PPO of chatgpt? refer to 5, and I found 6 is a good example to understand the PPO process in LLM








## Reference
- [1.Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)
- [2.ChatGPT RL Process](https://dida.do/blog/chatgpt-reinforcement-learning)
- [3.Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)
- [4.Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/pdf/2204.05862.pdf)
- [5.JoaoLages/RLHF.md](https://gist.github.com/JoaoLages/c6f2dfd13d2484aa8bb0b2d567fbf093)

