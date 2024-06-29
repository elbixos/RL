# RL Repository

some tests about **Reinforcement Learning**.

## Bibliography and sources
I found some materials to understand RL, marked here some sort of bibliography.
It needs a bit of time to get used to notations, so feel free to look at some vulgarisation explanations before...

- some videos in French from this channel : [L4Project video on the subject (actor vs critic)](https://www.youtube.com/watch?v=LtRAgxRb5eQ)
- [key concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- a nice [taxonomy of RL technics](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html#a-taxonomy-of-rl-algorithms)
- a long description of the [details between different technics](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)

- a french lesson about RL from [CNRS Fidle](https://www.youtube.com/watch?v=8gakDXPHC8c)

## Experiments

I, for now, just used the *Gymnasium package* to get environnements.

### Cartpole Environment

My firt attempts were with *cartpole v1* which is continuous.
Many of the code that I found was using cartpole v0 and it needs some adjustments
to work.

for cartpole, I have, working :

- *cartpole1_Qlearning_tabular*. demonstrate the use of **Qlearning in a tabular version**. It was my first attempt. It is based on some code found somewhere (shame on me) but slightly modified to make it work.

- *cartpole1_DQN*. demonstrate the use of a **DQN version of Qlearning**. It is also based on some code found somewhere (shame on me) but really heavily modified to make it work. The DQN has 2 models (1 target), and uses memory and replay.

- *cartpole1_critic* : still a **DQN**, based on based on [L4Project video on the subject (actor vs critic)](https://www.youtube.com/watch?v=LtRAgxRb5eQ)
Start with this one, since it is very fast ! (much more than the previous)
Fast and efficient. There is only one model in this one (no target). Work flawlessly, anyway.

- *cartpole1_actor* : A **policy gradient** method , based on based on [L4Project video on the subject (actor vs critic)](https://www.youtube.com/watch?v=LtRAgxRb5eQ)
Very fast and efficient but some learnings are unstable.

I'm still working on some files :

- *cartpole1_actor_critic* will be my next target : **actor critic method**, and based on [L4Project video on the subject](https://www.youtube.com/watch?v=1okjkEMP79c&t=217s) 
The code runs, but the algorithm doesn't converge to a decent solution

Globally, L4Project is a very good source of information in french, for reinforcement learning