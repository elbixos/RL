# RL Repository

some tests about **Reinforcement Learning**.

I, for now, just used the *Gymnasium package* to get environnements.

## Cartpole Environment

My firt attempts were with *cartpole v1* which is continuous.
Many of the code that I found was using cartpole v0 and it needs some adjustments
to work.

for cartpole, I have, working :

- *cartpole1_Qlearning_tabular*. demonstrate the use of Qlearning in a tabular version. It was my first attempt. It is based on some code found somewhere (shame on me) but slightly modified to make it work.

- *cartpole1_DQN*. demonstrate the use of a DQN version of Qlearning. It is also based on some code found somewhere (shame on me) but really heavily modified to make it work. The DQN has 2 models (1 target), and uses memory and replay.

