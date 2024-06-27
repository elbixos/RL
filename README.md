# RL Repository

some tests about **Reinforcement Learning**.

I, for now, just used the *Gymnasium package* to get environnements.

## Cartpole Environment

My firt attempts were with *cartpole v1* which is continuous.
Many of the code that I found was using cartpole v0 and it needs some adjustments
to work.

for cartpole, I have, working :

- *cartpole1_Qlearning_tabular*. demonstrate the use of **Qlearning in a tabular version**. It was my first attempt. It is based on some code found somewhere (shame on me) but slightly modified to make it work.

- *cartpole1_DQN*. demonstrate the use of a **DQN version of Qlearning**. It is also based on some code found somewhere (shame on me) but really heavily modified to make it work. The DQN has 2 models (1 target), and uses memory and replay.

- *cartpole1_critic* : still a **DQN**, based on based on [L4Project video on the subject (actor vs critic)](https://www.youtube.com/watch?v=LtRAgxRb5eQ)
Start with this one, since it is very fast ! (much more than the previous)
Fast and efficient

- *cartpole1_actor* : A **policy gradient** method , based on based on [L4Project video on the subject (actor vs critic)](https://www.youtube.com/watch?v=LtRAgxRb5eQ)
Very fast and efficient but some learnings are unstable

I'm still working on some files :

- *cartpole1_actor_critic* will be my next target : **actor critic method**, and based on [L4Project video on the subject](https://www.youtube.com/watch?v=1okjkEMP79c&t=217s)

Globally, L4Project is a very good source of information in french, for reinforcement learning